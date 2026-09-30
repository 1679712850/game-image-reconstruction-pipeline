"""Stable back-to-front ordering with explicit occlusion edges overriding depth."""


def ordered_layers(records):
    pending = sorted(records, key=lambda obj: (obj.get('z_order', 0), obj.get('id', '')))
    output = []
    while pending:
        ids = {o.get('id') for o in pending}
        item = next((o for o in pending if not (set(o.get('occludes', [])) & ids)), None)
        if item is None:
            # Preserve deterministic source depth for a cyclic graph; report the cycle elsewhere.
            output.extend(pending)
            break
        output.append(item)
        pending.remove(item)
    return output
