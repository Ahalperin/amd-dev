#!/usr/bin/env python3
# example.py
import torch
import time
from torchcomms import new_comm, ReduceOp

def main():
    # Initialize TorchComm with RCCLX backend
    device = torch.device("cuda")
    torchcomm = new_comm("rcclx", device, name="main_comm")

    # Get rank and world size
    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()

    # Calculate device ID
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    # Create a tensor with rank-specific data
    tensor = torch.full(
        (67108864,),  # 256 MB (256 * 1024 * 1024 / 4 bytes per float32)
        float(rank + 1),
        dtype=torch.float32,
        device=target_device
    )

    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Measure AllReduce duration
    start_time = time.perf_counter()
    
    # Perform synchronous AllReduce (sum across all ranks)
    torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")
    print(f"Rank {rank}: AllReduce duration: {duration_ms:.2f} ms")

    # Cleanup
    torchcomm.finalize()

if __name__ == "__main__":
    main()