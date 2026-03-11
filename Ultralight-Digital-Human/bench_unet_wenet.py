import time

import torch

from unet import Model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    net = Model(6, mode="wenet").to(device).eval()

    img = torch.randn(1, 6, 160, 160, device=device)
    audio = torch.randn(1, 128, 16, 32, device=device)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = net(img, audio)

    iters = 100
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = net(img, audio)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / iters
    print(f"avg_ms_per_frame: {avg_ms:.4f}")


if __name__ == "__main__":
    main()

