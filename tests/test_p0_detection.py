"""Regression fixtures for recall, edge recovery, auditability and fusion safety."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
from pydantic import ValidationError

from agent.graph import build_graph
from app.config import PipelineConfig, SceneLoopConfig
from app.detection_config import DetectionConfig, DedupConfig
from app.models import ModelConfig
from detection.p0_pipeline import P0DetectionPipeline
from diagnostics.coverage_map import coverage_arrays
from diagnostics.report_generator import generate_report
from fusion.candidate_fusion import fuse_candidates
from fusion.mask_fusion import fuse_segmented_records
from schemas.detection_candidate import DetectionCandidate
from services.grounding_service import GroundingService
from services.runtime import ServiceBundle
from services.sam_service import SAMService
from taxonomy.aliases import normalize_category
from taxonomy.categories import category_group
from tiling.coordinate_mapper import restore_bbox
from tiling.tile_generator import generate_tiles


def settings(**kwargs):
    values = {"expand_categories": False, "prompt_group_size": 4,
              "tiling": {"tile_size": 64, "min_tile_size": 32, "overlap": .25},
              "multi_scale": {"enabled": False}, "truncation": {"edge_threshold": 2, "redetect_padding": 20}}
    values.update(kwargs)
    return DetectionConfig.model_validate(values)


def item(label="stone_lantern", box=(25, 20, 32, 32), score=.7):
    return {"category": label, "bbox": list(box), "bbox_format": "xyxy", "confidence": score}


def candidate(name, box, source="tile", window=(0, 0, 64, 64), score=.7, category="tree", edges=()):
    return DetectionCandidate(id=name, category=category, confidence=score, bbox=box,
                              source=source, window=window, is_truncated=bool(edges), truncated_edges=list(edges),
                              appearance=[1.0], observations=[{"id": name, "confidence": score, "window": window, "source": source}])


class P0GeometryTests(unittest.TestCase):
    def test_overlap_tiles_cover_odd_sizes_and_small_inputs(self):
        cover = np.zeros((103, 155), dtype=int)
        for tile in generate_tiles(155, 103, 64, .25):
            cover[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] += 1
        self.assertTrue((cover > 0).all())
        self.assertGreater(cover.max(), 1)
        self.assertEqual(generate_tiles(12, 9, 64)[0].bbox, (0, 0, 12, 9))
        with self.assertRaises(ValueError):
            generate_tiles(100, 100, 64, 0)

    def test_explicit_normalized_and_fractional_coordinate_restore(self):
        self.assertEqual(restore_bbox([.1, .2, .5, .8], 100, 200, 100, 50, normalized=True),
                         {"x": 110, "y": 210, "w": 40, "h": 30})
        self.assertEqual(restore_bbox([-1, 1.2, 20.2, 30], 100, 200, 20, 30),
                         {"x": 100, "y": 201, "w": 20, "h": 29})
        for box in ([.1, .1, 2, .8], [0, 0, float('nan'), 1]):
            with self.assertRaises(ValueError):
                restore_bbox(box, 0, 0, 10, 10, normalized=True)

    def test_aliases_and_groups_preserve_semantic_distinctions(self):
        for label in ("stone lamp", "garden lantern", "石灯", "寺庙石灯"):
            self.assertEqual(normalize_category(label), "stone_lantern")
        self.assertEqual(normalize_category("war banner"), "flag")
        self.assertEqual(normalize_category("阵旗"), "formation_flag")
        self.assertEqual(category_group("stone_pillar"), "structure")
        self.assertEqual(category_group("spirit_stone"), "cultivation_prop")
        self.assertEqual(category_group("fog"), "fx_environment")

    def test_configuration_rejects_no_overlap_disabled_paths_and_invalid_scales(self):
        for data in ({"tiling": {"overlap": 0}}, {"multi_scale": {"scales": []}},
                     {"multi_scale": {"scales": [1]}}, {"global": {"enabled": False}, "tiling": {"enabled": False}}):
            with self.assertRaises(ValidationError):
                DetectionConfig.model_validate(data)

    def test_independent_windows_fuse_but_adjacent_instances_and_contained_props_survive(self):
        a = candidate("a", (30, 10, 50, 40), score=.31)
        b = candidate("b", (31, 10, 51, 40), window=(20, 0, 84, 64), score=.42)
        neighbour = candidate("n", (52, 10, 72, 40))
        nested = candidate("small", (35, 15, 39, 19))
        fused, rejected = fuse_candidates([a, b, neighbour, nested], DedupConfig())
        self.assertEqual(len(fused), 3)
        merged = next(c for c in fused if len(c.observations) == 2)
        self.assertGreater(merged.confidence, .42)
        self.assertEqual(set(merged.merged_from), {"a", "b"})
        self.assertEqual(rejected[0].reject_reason, "cross_tile_duplicate")

    def test_same_window_repeated_prompts_do_not_boost_confidence(self):
        a = candidate("a", (30, 10, 50, 40), score=.31)
        b = candidate("b", (30, 10, 50, 40), score=.42)
        merged, _ = fuse_candidates([a, b], DedupConfig())
        self.assertEqual(merged[0].confidence, .42)

    def test_complementary_fragments_union_requires_appearance_evidence(self):
        a = candidate("a", (40, 10, 64, 40), edges=["right"])
        b = candidate("b", (48, 10, 110, 40), window=(48, 0, 112, 64), edges=["left"])
        config = DedupConfig(bbox_iou=.6, overlap_ratio=.6)
        merged, _ = fuse_candidates([a, b], config)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, (40, 10, 110, 40))
        self.assertTrue(merged[0].is_truncated)  # Must still be re-observed.


class P0PipelineTests(unittest.TestCase):
    def test_every_image_uses_global_and_tile_grouped_paths(self):
        calls = []
        def infer(crop, group, context):
            calls.append(context)
            return []
        P0DetectionPipeline(settings()).run(Image.new("RGB", (30, 20)), ["tree", "stone_lantern", "altar"], infer)
        self.assertEqual({c["source"] for c in calls}, {"global", "tile"})
        self.assertTrue(all(len(c["categories"]) <= 4 for c in calls))
        self.assertTrue(all("tiny" in c["instruction"] for c in calls if c["source"] == "tile"))

    def test_multiscale_uses_distinct_windows_without_repeating_same_crop(self):
        cfg = settings(multi_scale={"enabled": True, "scales": [64, 96]})
        _, stats = P0DetectionPipeline(cfg).run(Image.new("RGB", (130, 100)), ["tree"], lambda *args: [])
        self.assertEqual({t["scale"] for t in stats["tiles"]}, {64, 96})
        windows = [(t["x"], t["y"], t["width"], t["height"]) for t in stats["tiles"]]
        self.assertEqual(len(windows), len(set(windows)))

    def test_failed_tile_invalid_box_and_unknown_category_are_audited(self):
        def infer(crop, group, ctx):
            if ctx["source"] == "global":
                return [item("unrecognized_xyz"), item(box=(5, 5, 4, 4)), item()]
            if ctx["window"][0] > 0:
                raise RuntimeError("model timeout")
            return []
        records, stats = P0DetectionPipeline(settings()).run(Image.new("RGB", (100, 60)), ["stone_lantern"], infer)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(stats["failed_tiles"]), 1)
        self.assertEqual({r["reason"] for r in stats["filtered"]}, {"invalid_bbox", "category_unknown"})
        self.assertEqual(stats["combined_candidates"], 3)
        self.assertEqual(len(stats["candidates"]), 3)
        json.dumps(stats, allow_nan=False)

    def test_tile_small_low_confidence_is_reviewable_and_not_silently_removed(self):
        def infer(crop, group, ctx):
            return [item(score=.12)] if ctx["source"] == "tile" and ctx["window"][0] == 0 else []
        records, stats = P0DetectionPipeline(settings()).run(Image.new("RGB", (100, 60)), ["stone_lantern"], infer)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["review_required"])
        self.assertEqual(len(stats["review_candidate_pool"]), 1)
        self.assertEqual(stats["filtered"], [])

    def test_expanded_redetection_restores_full_object_and_lineage(self):
        def infer(crop, group, ctx):
            if ctx["source"] == "tile" and ctx["window"][0] == 0:
                return [item(box=(55, 20, 64, 35))]
            if ctx["source"] == "redetection":
                x, y, _, _ = ctx["window"]
                return [item(box=(55-x, 20-y, 74-x, 35-y))]
            return []
        records, stats = P0DetectionPipeline(settings()).run(Image.new("RGB", (100, 60)), ["stone_lantern"], infer)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["bbox"], {"x": 55, "y": 20, "w": 19, "h": 15})
        self.assertFalse(records[0]["is_truncated"])
        self.assertTrue(records[0]["redetected"])
        self.assertEqual(len(records[0]["source_candidates"]), 2)
        self.assertEqual(stats["redetections"][0]["status"], "recovered")

    def test_unresolved_edge_is_review_only_and_redetection_budget_is_explicit(self):
        def infer(crop, group, ctx):
            return [item(box=(55, 20, 64, 35))] if ctx["source"] == "tile" and ctx["window"][0] == 0 else []
        cfg = settings(truncation={"edge_threshold": 2, "max_redetections": 0})
        records, stats = P0DetectionPipeline(cfg).run(Image.new("RGB", (100, 60)), ["stone_lantern"], infer)
        self.assertEqual(records, [])
        self.assertTrue(stats["review_candidate_pool"][0]["is_truncated"])
        self.assertEqual(stats["redetections"][0]["status"], "budget_exhausted")

    def test_small_report_measures_unique_objects_with_tile_only_gain(self):
        cfg = settings(tiling={"tile_size": 1024}, multi_scale={"enabled": False})
        def infer(crop, group, ctx):
            return [item(box=(30, 30, 40, 40), score=.27)] if ctx["source"] == "tile" and ctx["window"][0] == 0 else []
        objects, stats = P0DetectionPipeline(cfg).run(Image.new("RGB", (1600, 1000)), ["stone_lantern"], infer)
        self.assertEqual(len(objects), 1)
        report = stats["small_object_report"]
        self.assertEqual((report["small_object_count"], report["detected_by_global"], report["tile_only"]), (1, 0, 1))

    def test_coverage_does_not_count_failed_scans_as_success(self):
        scans = [{"window": [0, 0, 5, 10], "status": "ok"}, {"window": [5, 0, 10, 10], "status": "failed"}]
        coverage, _, density = coverage_arrays((10, 10), scans, [], [])
        self.assertTrue((coverage[:, :5] > 0).all())
        self.assertFalse(coverage[:, 5:].any())
        self.assertFalse(density.any())


class P0IntegrationTests(unittest.TestCase):
    def test_full_graph_exports_lineage_all_diagnostics_and_unique_assets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            Image.new("RGB", (200, 140), (70, 120, 40)).save(source)
            cfg = PipelineConfig(detection=settings(), scene_loop=SceneLoopConfig(enabled=False))
            state = build_graph(cfg).invoke({"source_path": str(source), "output_dir": str(root / "out")})
            self.assertEqual(len(state["objects"]), 4)
            for obj in state["objects"]:
                self.assertTrue(obj["source_candidates"])
                self.assertTrue(obj["observations"])
                self.assertIn("confidence", obj["observations"][0])
                self.assertIsNotNone(obj["asset_path"])
            diag = root / "out" / "diagnostics"
            for name in ("global_detection.png", "tile_detection.png", "merged_detection.png", "filtered_detection.png",
                         "coverage_map.png", "tile_grid.png", "candidates.json", "filtered.json", "report.html", "small_object_report.json"):
                self.assertTrue((diag / name).is_file(), name)
            summary = json.loads((diag / "summary.json").read_text())
            self.assertEqual(summary["final_objects"], 4)
            json.dumps(state, allow_nan=False)

    def test_mask_fusion_keeps_union_and_records_duplicate_reason(self):
        from cv.mask import save_mask
        with TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, x in enumerate((5, 7)):
                mask = np.zeros((30, 30), dtype=np.uint8)
                mask[5:20, x:x+15] = 255
                path = save_mask(mask, root / f"{index}.png")
                records.append({"id": str(index), "category": "tree", "confidence": .8-index*.1,
                                "bbox": {"x": x, "y": 5, "w": 15, "h": 15}, "mask_path": path,
                                "source_candidates": [f"c{index}"], "observations": []})
            kept, rejected = fuse_segmented_records(records, DedupConfig())
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["bbox"]["w"], 17)
            self.assertEqual(kept[0]["source_candidates"], ["c0", "c1"])
            self.assertEqual(rejected[0]["reason"], "mask_duplicate")

    def test_segmentation_failure_preserves_review_record_and_other_objects(self):
        class FailingSAM(SAMService):
            def segment(self, path, records):
                if any(r["category"] == "tree" for r in records):
                    raise RuntimeError("injected segmentation failure")
                return super().segment(path, records)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.png"
            Image.new("RGB", (200, 140)).save(source)
            bundle = replace(ServiceBundle.create(), sam=FailingSAM())
            cfg = PipelineConfig(max_retry=0, detection=settings(), scene_loop=SceneLoopConfig(enabled=False))
            state = build_graph(cfg, bundle).invoke({"source_path": str(source), "output_dir": str(root / "out")})
            self.assertEqual(len(state["objects"]), 4)
            tree = next(o for o in state["objects"] if o["category"] == "tree")
            self.assertIsNone(tree["asset_path"])
            self.assertIn("segmentation_failed", tree["error"])
            rejected = json.loads((root / "out" / "diagnostics" / "filtered.json").read_text())
            self.assertIn("segmentation_failed", {r["reason"] for r in rejected})


class P0AdapterTests(unittest.TestCase):
    def test_real_adapter_retains_low_confidence_duplicates_and_original_labels(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        service = GroundingService(False)
        inputs = MagicMock()
        inputs.to.return_value = inputs
        inputs.__getitem__.return_value = 'input_ids'
        service._processor = MagicMock(return_value=inputs)
        service._model = MagicMock()
        service._torch = SimpleNamespace(inference_mode=nullcontext)
        service._device = 'cpu'
        boxes, scores = MagicMock(), MagicMock()
        boxes.detach.return_value.cpu.return_value.tolist.return_value = [[1, 1, 6, 8], [1, 1, 6, 8]]
        scores.detach.return_value.cpu.return_value.tolist.return_value = [.12, .14]
        service._processor.post_process_grounded_object_detection.return_value = [
            {'boxes': boxes, 'scores': scores, 'text_labels': ['stone lamp', '石灯']}]
        result = service._infer_image(Image.new('RGB', (20, 20)), ['stone_lantern'], .1, .1, preserve_candidates=True)
        self.assertEqual(len(result), 2)
        self.assertEqual([r['confidence'] for r in result], [.12, .14])
        self.assertTrue(all(r['category'] == 'stone_lantern' for r in result))
        self.assertEqual(result[1]['raw_label'], '石灯')

    def test_full_catalog_is_scheduled_in_semantic_groups(self):
        groups = []
        cfg = settings(expand_categories=True)
        def infer(crop, group, context):
            groups.append(group)
            return []
        P0DetectionPipeline(cfg).run(Image.new('RGB', (20, 20)), [], infer)
        labels = {c for g in groups for c in g}
        self.assertTrue({'stone_lantern', 'stone_pillar', 'flag', 'tombstone', 'box', 'jar', 'roof_decoration',
                         'spirit_stone', 'floating_rock', 'mechanism', 'fog', 'teleport_array'} <= labels)
        self.assertTrue(all(len(g) <= cfg.prompt_group_size for g in groups))

    def test_effects_are_exported_separately_and_diagnostics_can_be_disabled(self):
        class EffectDetector(GroundingService):
            def detect(self, path, categories):
                return [{'category': 'fog', 'confidence': .9, 'bbox': {'x': 4, 'y': 4, 'w': 10, 'h': 10}}]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.png'
            Image.new('RGB', (40, 40)).save(source)
            config = PipelineConfig(scene_loop=SceneLoopConfig(enabled=False),
                                    p1={'enabled': False},
                                    detection=settings(expand_categories=True, diagnostics={'enabled': False}))
            services = replace(ServiceBundle.create(), grounding=EffectDetector())
            state = build_graph(config, services).invoke({'source_path': str(source), 'output_dir': str(root/'out')})
            manifest = json.loads(Path(state['scene_json']).read_text())
            self.assertEqual(manifest['objects'], [])
            self.assertEqual(len(manifest['environment_effects']), 1)
            self.assertTrue(manifest['environment_effects'][0]['asset'].startswith('effects/'))
            self.assertTrue(manifest['environment_effects'][0]['hd_asset'].startswith('effects_hd/'))
            self.assertFalse((root/'out'/'diagnostics').exists())


if __name__ == '__main__':
    unittest.main()
