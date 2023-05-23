import torch.multiprocessing as mp

import os
from typing import Callable

import torch
import torch.distributed as dist

def init_process(rank: int, size: int, fn: Callable[[int, int], None], backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def hello_world(rank: int, size: int):
    print(f"[{rank}] say hi!")

def do_reduce(rank: int, size: int):    
    
    # create a group with all processors
    group = dist.new_group(list(range(size)))
        
    tensor = torch.ones(1)
    # sending all tensors to rank 0 and sum them
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    # can be dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN
    # only rank 0 will have four
    print(f"[{rank}] data = {tensor[0]}")


def do_all_reduce(rank: int, size: int):
    # create a group with all processors
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    # can be dist.ReduceOp.PRODUCT, dist.ReduceOp.MAX, dist.ReduceOp.MIN
    # will output 4 for all ranks
    print(f"[{rank}] data = {tensor[0]}")

if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_all_reduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  # waiting each child process completed.