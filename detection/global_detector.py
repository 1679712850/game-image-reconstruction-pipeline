from detection.grouped_detector import scan_window


def detect_global(image, categories, infer, collect, scans, group_size, *, budget=None, pass_id="global_discovery", max_categories=None):
    if max_categories:
        categories = categories[:max_categories]
    scan_window(image, (0, 0, image.width, image.height), categories, infer, collect, scans,
                source="global", group_size=group_size, budget=budget, pass_id=pass_id)
