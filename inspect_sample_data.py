"""
Sample_data.pth dosyasini torch ile acip yapisini (state boyutu, action
formati) inceleyen debug scripti.
"""
import torch


def describe(obj, prefix="", depth=0, max_depth=4):
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}{prefix}: ... (max depth)")
        return

    if torch.is_tensor(obj):
        print(f"{indent}{prefix}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype} "
              f"device={obj.device}")
        if obj.numel() <= 20:
            print(f"{indent}  values={obj.tolist()}")
        else:
            flat = obj.flatten()
            print(f"{indent}  first_values={flat[:10].tolist()} "
                  f"min={obj.min().item():.4f} max={obj.max().item():.4f}")
    elif isinstance(obj, dict):
        print(f"{indent}{prefix}: dict with {len(obj)} keys -> {list(obj.keys())}")
        for k, v in obj.items():
            describe(v, prefix=f"[{k!r}]", depth=depth + 1, max_depth=max_depth)
    elif isinstance(obj, (list, tuple)):
        kind = "list" if isinstance(obj, list) else "tuple"
        print(f"{indent}{prefix}: {kind} with {len(obj)} elements")
        if len(obj) > 0:
            describe(obj[0], prefix="[0] (ilk eleman)", depth=depth + 1, max_depth=max_depth)
    else:
        print(f"{indent}{prefix}: {type(obj).__name__} = {obj!r}"[:300])


def main():
    path = "Sample_data.pth"
    print(f"Yukleniyor: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)

    print("\n=== Ust seviye tip ===")
    print(type(data))

    print("\n=== Yapi ===")
    describe(data, prefix="root")

    # Eger bir liste/tuple ise (orn. [(state, action, reward, next_state, done), ...])
    # ilk ornegi ayrintili gosterelim.
    sample = None
    if isinstance(data, (list, tuple)) and len(data) > 0:
        sample = data[0]
    elif isinstance(data, dict):
        for key in ("data", "samples", "transitions", "buffer"):
            if key in data:
                container = data[key]
                if isinstance(container, (list, tuple)) and len(container) > 0:
                    sample = container[0]
                break

    if sample is not None:
        print("\n=== Ornek (ilk kayit) detayli ===")
        describe(sample, prefix="sample")

        if isinstance(sample, (list, tuple)):
            print(f"\nOrnek {len(sample)} elemandan olusuyor "
                  f"(tipik format: state, action, reward, next_state, done gibi).")
            for i, elem in enumerate(sample):
                if torch.is_tensor(elem):
                    print(f"  eleman[{i}]: shape={tuple(elem.shape)} dtype={elem.dtype}")
                else:
                    print(f"  eleman[{i}]: type={type(elem).__name__} value={elem!r}"[:200])


if __name__ == "__main__":
    main()
