import torch

BYTES_PER_GIB = 1024**3


def vram_gb() -> float:
    """Measure the total VRAM across every visible GPU.

    Returns:
        float: A capacity in GiB, or 0.0 on a machine without a GPU.
    """
    if not torch.cuda.is_available():
        return 0.0

    total = sum(
        torch.cuda.get_device_properties(index).total_memory
        for index in range(torch.cuda.device_count())
    )

    return total / BYTES_PER_GIB


def device_summary() -> str:
    """Describe the hardware a run landed on.

    Returns:
        str: A description of the visible GPUs, or of the lack of them.
    """
    if not torch.cuda.is_available():
        return "cpu, no GPU visible"

    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    counts = {name: names.count(name) for name in dict.fromkeys(names)}
    listed = ", ".join(f"{count}x {name}" for name, count in counts.items())

    return f"{listed}, {vram_gb():.1f} GiB VRAM in total"
