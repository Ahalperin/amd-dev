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

    # print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    # Create a tensor with rank-specific data
    tensor = torch.full(
        (1*1024*1024//4,),  # 1 MB data (/4 to calc tesor size with 4 bytes per float32)
        float(rank + 1),
        dtype=torch.float32,
        device=target_device
    )

    # print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Calculate data size
    data_size_bytes = tensor.numel() * tensor.element_size()
    data_size_gb = data_size_bytes / (1024 ** 3)

    # Warmup run
    torchcomm.all_reduce(tensor.clone(), ReduceOp.SUM, async_op=False)
    torch.cuda.current_stream().synchronize()

    # Measure AllReduce duration
    start_time = time.perf_counter()
    
    # Perform synchronous AllReduce (sum across all ranks)
    torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()
    
    end_time = time.perf_counter()
    duration_s = end_time - start_time
    duration_ms = duration_s * 1000

    # Calculate bandwidth
    # Algorithm bandwidth: data_size / time
    alg_bw_gbps = data_size_gb / duration_s

    # Bus bandwidth for AllReduce: algBW * 2 * (n-1) / n
    # This accounts for the ring algorithm where each GPU sends/receives
    # (n-1)/n of data in both reduce-scatter and all-gather phases
    bus_bw_factor = 2.0 * (world_size - 1) / world_size
    bus_bw_gbps = alg_bw_gbps * bus_bw_factor

    # print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")
    print(f"Rank {rank}: Size: {data_size_bytes / (1024**2):.0f} MB | "
          f"Time: {duration_ms:.2f} ms | "
          f"AlgBW: {alg_bw_gbps:.2f} GB/s | "
          f"BusBW: {bus_bw_gbps:.2f} GB/s")

    # Cleanup
    torchcomm.finalize()

if __name__ == "__main__":
    main()