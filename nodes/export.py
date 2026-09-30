"""Export scene.json and enumerate artifacts referenced by this run."""
from collections.abc import Callable
import json
from pathlib import Path
import shutil

from agent.state import SceneState
from exporters.json_exporter import build_manifest, export_json


def _referenced_paths(value):
    """Include nested ownership masks and retry evidence in the artifact list."""
    if isinstance(value, dict):
        for key, child in value.items():
            if (key.endswith('_path') or key == 'path') and isinstance(child, str) and child:
                yield child
            else:
                yield from _referenced_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from _referenced_paths(child)


def make_export(mock: bool, backends: dict[str, str] | None = None, *, diagnostics_enabled: bool = False, psd_enabled: bool = False, export_config=None) -> Callable[[SceneState], dict]:
    """Bind export provenance; generated manifests explicitly mark mocks."""
    def export(state: SceneState) -> dict:
        """Persist validated metadata with portable asset paths."""
        root = Path(state["output_dir"]).resolve()
        assets = set()
        from cv.completion_metrics import completion_metrics
        metrics = completion_metrics(state)
        updates = {'completion_metrics':metrics, 'pipeline_status':metrics['status']}
        if diagnostics_enabled:
            from diagnostics.report_generator import generate_report
            directory = root / "diagnostics"
            generate_report(state["source_path"], directory, state.get("detection_runs", []), state.get("objects", []))
            assets.update(str(path.resolve()) for path in directory.iterdir() if path.is_file())
        if state.get('ownership'):
            # P0 report uses a scan-coverage map. Keep that separately before restoring semantic coverage.
            from diagnostics.p1_report import export_p1
            directory = root / 'diagnostics'
            semantic = directory/'p1_coverage_map.png'
            if semantic.exists():
                if diagnostics_enabled:
                    shutil.copyfile(directory/'coverage_map.png', directory/'detection_scan_coverage.png')
                    report = directory/'report.html'
                    content = report.read_text(encoding='utf-8').replace('coverage_map.png','detection_scan_coverage.png')
                    report.write_text(content, encoding='utf-8')
                shutil.copyfile(semantic, directory/'coverage_map.png')
            updates['p1_summary'] = export_p1({**state, **updates}, mock=mock)
            for folder in ('metadata','diagnostics','objects','terrain'):
                assets.update(str(p.resolve()) for p in (root/folder).glob('*') if p.is_file())
        # Build after P1 reporting so checkpoints and scene.json share the summary.
        manifest = build_manifest({**state, **updates}, mock, backends)
        if psd_enabled:
            from exporters.psd_exporter import export_psd
            updates["psd_path"] = str(export_psd(manifest, root / "scene.psd", asset_root=root,
                                                memory_budget_mb=getattr(export_config, 'memory_budget_mb', 2048),
                                                low_memory=getattr(export_config, 'low_memory', True)))
            manifest.psd = "scene.psd"
            assets.add(updates["psd_path"])
            assets.add(str((root / "scene.export.json").resolve()))
        scene_json = export_json(manifest, root / "scene.json")
        assets.add(scene_json)
        # The manifest is the portability contract. Include every referenced
        # relative artifact, including terrain masks that are not instance records.
        for referenced in _referenced_paths(manifest.model_dump(mode='json')):
            path = (root / referenced).resolve()
            if path.is_file():
                assets.add(str(path))
        for obj in state.get("objects", []):
            assets.update(obj[key] for key in ("asset_path", "hd_asset_path", "mask_path", 'candidate_mask_path', 'visible_mask_path', 'full_mask_path') if obj.get(key))
        assets.update(layer["asset_path"] for layer in state.get("decomposed_layers", []))
        for edit in state.get("object_edits", []):
            assets.update(edit[key] for key in ("asset_path", "mask_path", "source_asset_path"))
        if state.get("reconstruction_path"):
            assets.add(state["reconstruction_path"])
        if state.get("scene_history"):
            assets.update(state[key] for key in ("working_path", "coverage_mask_path") if state.get(key))
        ownership = state.get("ownership", {})
        assets.update(path for path in (ownership.get("owner_map_path"),
                                        ownership.get("residual_background", {}).get("asset_path"),
                                        ownership.get("residual_background", {}).get("mask_path")) if path)
        for layer in state.get("terrain_layers", []):
            assets.update(path for path in (layer.get("asset_path"), layer.get("mask_path"), layer.get("visible_mask_path"),
                                            layer.get("complete_mask_path"), layer.get("complete_asset_path")) if path)
        for key in ('objects', 'ownership', 'terrain_layers', 'retry_history', 'candidate_registry', 'accepted_ownership'):
            assets.update(_referenced_paths(state.get(key, {})))
        assets.update(str(root / "debug" / "candidates" / f"{ident}_contact_sheet.png")
                      for ident in state.get('candidate_registry', {}))
        if (root / "metadata" / "candidates.json").exists():
            assets.add(str(root / "metadata" / "candidates.json"))
        retry_path = root / "metadata" / "retries.json"
        retry_path.parent.mkdir(parents=True, exist_ok=True)
        from diagnostics.p1_report import portable
        retry_path.write_text(json.dumps(portable(state.get("retry_history", []), root),
                                        ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        assets.add(str(retry_path.resolve()))
        for path in (Path(state["output_dir"]) / "metadata" / "ownership.json",
                     Path(state["output_dir"]) / "diagnostics" / "overlap_heatmap.png",
                     Path(state["output_dir"]) / "diagnostics" / "unassigned_regions.png",
                     Path(state["output_dir"]) / "diagnostics" / "reconstruction_diff.png"):
            if path.exists():
                assets.add(str(path.resolve()))
        return {**updates, "scene_json": scene_json,
                "exported_assets": sorted({str(Path(path).resolve()) for path in assets})}
    return export
