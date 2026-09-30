from detection.grouped_detector import scan_window


def detect_global(image, categories, infer, collect, scans, group_size):
    scan_window(image, (0, 0, image.width, image.height), categories, infer, collect, scans,
                source="global", group_size=group_size)
