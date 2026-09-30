"""Export scene.json and enumerate artifacts referenced by this run."""
from collections.abc import Callable
from pathlib import Path

from agent.state import SceneState
from exporters.json_exporter import build_manifest, export_json


def make_export(mock: bool, backends: dict[str, str] | None = None, *, diagnostics_enabled: bool = False) -> Callable[[SceneState], dict]:
    """Bind export provenance; generated manifests explicitly mark mocks."""
    def export(state: SceneState) -> dict:
        """Persist validated metadata with portable asset paths."""
        manifest = build_manifest(state, mock, backends)
        scene_json = export_json(manifest, Path(state["output_dir"]) / "scene.json")
        assets = {scene_json}
        if diagnostics_enabled:
            from diagnostics.report_generator import generate_report
            directory = Path(state["output_dir"]) / "diagnostics"
            generate_report(state["source_path"], directory, state.get("detection_runs", []), state.get("objects", []))
            assets.update(str(path.resolve()) for path in directory.iterdir() if path.is_file())
        for obj in state.get("objects", []):
                assets.update(obj[key] for key in ("asset_path", "hd_asset_path", "mask_path") if obj.get(key))
        assets.update(layer["asset_path"] for layer in state.get("decomposed_layers", []))
        for edit in state.get("object_edits", []):
            assets.update(edit[key] for key in ("asset_path", "mask_path", "source_asset_path"))
        if state.get("reconstruction_path"):
            assets.add(state["reconstruction_path"])
        if state.get("scene_history"):
            assets.update(state[key] for key in ("working_path", "coverage_mask_path") if state.get(key))
        return {"scene_json": scene_json, "exported_assets": sorted(assets)}
    return export
