"""All public prompts are original-image coordinates; transform only at inference."""
import numpy as np


def generate_prompts(record, neighbors, window, scale, attempt=0):
    x, y, r, b = window
    box = record["bbox"]
    left, top = max(x, box["x"]), max(y, box["y"])
    right, bottom = min(r, box["x"]+box["w"]), min(b, box["y"]+box["h"])
    positive = [[(left+right)/2, (top+bottom)/2], *record.get("positive_points", [])]
    if attempt:
        positive += [[left+(right-left)*.4, top+(bottom-top)*.4]]
    negative = list(record.get("negative_points", []))
    for other in neighbors:
        if other["id"] != record["id"] and other.get("element_type") != "terrain":
            o = other["bbox"]
            p = [o["x"]+o["w"]/2, o["y"]+o["h"]/2]
            # A neighbor enclosed by the target may be an occluder, not background.
            if not (left <= p[0] < right and top <= p[1] < bottom):
                negative.append(p)
    negative += [[left-2, (top+bottom)/2], [right+2, (top+bottom)/2],
                 [(left+right)/2, top-2], [(left+right)/2, bottom+2]]
    positive = [p for p in positive if x <= p[0] < r and y <= p[1] < b]
    negative = [p for p in negative if x <= p[0] < r and y <= p[1] < b and p not in positive]
    points = np.array([[(px-x)*scale, (py-y)*scale] for px, py in positive+negative], np.float32)
    return {"box": np.array([(left-x)*scale, (top-y)*scale, (right-x)*scale, (bottom-y)*scale], np.float32),
            "point_coords": points, "point_labels": np.array([1]*len(positive)+[0]*len(negative), np.int32)}, {
                "window": list(window), "scale": scale, "positive_points": positive, "negative_points": negative}
