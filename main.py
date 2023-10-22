import numpy as np
from numba import cuda
import time
import matplotlib.pyplot as plt

device = cuda.get_current_device()

array_size_step = 500
iter_count = 20
pass_count = 100
block_count = min(device.MAX_BLOCK_DIM_X, 1024)
thread_count = min(device.MAX_THREADS_PER_BLOCK, 1024)

print(device)
print(f"block count: {block_count}")
print(f"thread count: {thread_count}")

def sum_on_cpu(a: np.ndarray, b: np.ndarray):
    c = np.zeros_like(a)

    for i in range(a.size):
        c[i] = a[i] + b[i]
        
    return c

@cuda.jit
def add_kernel(a, b, c):
    thread_id = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(thread_id, len(a), stride):
        c[i] = a[i] + b[i]

def sum_on_gpu(a: np.ndarray, b: np.ndarray):
    c = np.zeros_like(a)

    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dc = cuda.to_device(c)

    add_kernel[block_count, thread_count](da, db, dc)

    dc.copy_to_host(c)

    return c

results = {}

for p in range(pass_count):
    for i in range(iter_count):
        array_size = (i + 1) * array_size_step
        
        a = np.random.randint(0, 100, size=array_size)
        b = np.random.randint(0, 100, size=array_size)

        start_time = time.time()
        sum_on_cpu(a, b)
        cpu_elapsed_time = time.time() - start_time

        start_time = time.time()
        sum_on_gpu(a, b)
        gpu_elapsed_time = time.time() - start_time

        print(f"pass: {p}\tsize: {array_size}\tcpu: {(cpu_elapsed_time * 1000):.4f} ms\tgpu: {(gpu_elapsed_time * 1000):.4f} ms")

        if array_size not in results:
            results[array_size] = {}
        if "cpu" not in results[array_size]:
            results[array_size]["cpu"] = []
        if "gpu" not in results[array_size]:
            results[array_size]["gpu"] = []

        results[array_size]["cpu"].append(cpu_elapsed_time * 1000)
        results[array_size]["gpu"].append(gpu_elapsed_time * 1000)

del results[next(iter(results))] # first result on gpu not good

averages_cpu = [np.mean(v["cpu"]) for k, v in results.items()]
averages_gpu = [np.mean(v["gpu"]) for k, v in results.items()]

keys = list(results.keys())
plt.plot(keys, averages_cpu, label="CPU", marker="o")
plt.plot(keys, averages_gpu, label="GPU", marker="o")

plt.xlabel("Arrays' size")
plt.ylabel("Average time (ms)")
plt.title("Array sum on CPU vs GPU")
plt.legend()
plt.grid()

plt.savefig(f"results/iter_count_{iter_count}_array_size_step_{array_size_step}.png")

plt.show()