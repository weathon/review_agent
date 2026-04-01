import torch
import time
import sys
import threading


def gpu_stress(matrix_size=8192, duration=60):
    """Stress GPU via MPS/CUDA with matmul, transcendentals, sort, and random alloc."""
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[GPU] Using MPS (Apple Silicon GPU)")
        else:
            print("[GPU] No GPU available, running on CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")

    print(f"[GPU] Matrix size: {matrix_size}x{matrix_size}, duration: {duration}s")

    start = time.time()
    ops = 0
    while time.time() - start < duration:
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        c = torch.mm(a, b)
        c = torch.sin(c)
        c = torch.exp(c / c.max())
        _ = torch.sort(c.view(-1))
        ops += 1
        if ops % 5 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"\r[GPU] {elapsed:.0f}s | {ops} iters", end="", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"\n[GPU] Done: {ops} iters in {elapsed:.1f}s")


def npu_stress(duration=60):
    """Stress Apple Neural Engine via Core ML inference."""
    try:
        import coremltools as ct
        import numpy as np
    except ImportError:
        print("[NPU] coremltools not installed — run: pip install coremltools")
        return

    print("[NPU] Building Core ML model targeting Neural Engine...")

    # Build a heavy compute graph: conv → batchnorm → relu → conv, repeated
    # Core ML will route this to the ANE when using ALL compute units
    import coremltools.converters.mil as mil
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    batch, channels, h, w = 1, 64, 224, 224

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, channels, h, w))])
    def prog(x):
        # Stack conv layers to create a heavy workload the ANE will pick up
        for i in range(20):
            weight = np.random.randn(channels, channels, 3, 3).astype(np.float16)
            bias = np.random.randn(channels).astype(np.float16)
            x = mb.conv(x=x, weight=weight, bias=bias, pad_type="same", name=f"conv_{i}")
            x = mb.relu(x=x, name=f"relu_{i}")
        return x

    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.ALL,  # lets Core ML use the ANE
        minimum_deployment_target=ct.target.macOS13,
    )

    # Prepare input
    import numpy as np
    inp = {"x": np.random.randn(batch, channels, h, w).astype(np.float16)}

    # Warmup
    for _ in range(3):
        model.predict(inp)

    print(f"[NPU] Running inference loop for {duration}s (20-layer conv net on ANE)...")
    start = time.time()
    ops = 0
    while time.time() - start < duration:
        model.predict(inp)
        ops += 1
        if ops % 10 == 0:
            elapsed = time.time() - start
            print(f"\r[NPU] {elapsed:.0f}s | {ops} inferences", end="", flush=True)

    elapsed = time.time() - start
    print(f"\n[NPU] Done: {ops} inferences in {elapsed:.1f}s ({ops/elapsed:.1f} inf/s)")


if __name__ == "__main__":
    duration = 60

    print("=== GPU + NPU Stress Test ===\n")

    # Run both in parallel threads
    gpu_thread = threading.Thread(target=gpu_stress, args=(8192, duration))
    npu_thread = threading.Thread(target=npu_stress, args=(duration,))

    gpu_thread.start()
    npu_thread.start()

    gpu_thread.join()
    npu_thread.join()

    print("\nAll done.")
