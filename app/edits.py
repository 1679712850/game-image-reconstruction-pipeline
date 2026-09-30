"""Read explicit object edit requests without adding images to graph state."""
import json
from pathlib import Path

from schemas.generation import ObjectEditRequest


def validate_edit_requests(records: list[dict]) -> list[ObjectEditRequest]:
    """Require unique object IDs so multiple requests cannot overwrite one output."""
    requests = [ObjectEditRequest.model_validate(record) for record in records]
    if len({request.object_id for request in requests}) != len(requests):
        raise ValueError("Edit requests must have unique object_id values")
    return requests


def load_edit_requests(path: Path) -> list[dict]:
    """Resolve edit-mask paths relative to the UTF-8 JSON request file."""
    path = path.expanduser().resolve()
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Edit requests must be a JSON list")
    requests = validate_edit_requests(records)
    for request in requests:
        mask = Path(request.mask_path).expanduser()
        mask = (mask if mask.is_absolute() else path.parent / mask).resolve()
        if not mask.is_file():
            raise ValueError(f"Edit mask does not exist: {mask}")
        request.mask_path = str(mask)
    return [request.model_dump(mode="json") for request in requests]
