import torch 
import time
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

print(f"Device set: {device}")
    
x = torch.randn(500, 1000)
y = torch.randn(1000, 700)

num_calculations_cpu = 10000
num_calculations_gpu = 10000
cpu_times = []
gpu_times = []

x.cpu()
y.cpu()
for i in tqdm(range(num_calculations_cpu), desc="CPU calculations"):
    start = time.time() 
    x @ y
    end = time.time()
    cpu_times.append(end - start)


print("Finished CPU calculations.") 
avg_cpu_time = sum(cpu_times) / len(cpu_times)
print(f"Average CPU time: {avg_cpu_time}")
print(f"Total CPU time: {sum(cpu_times)}")

x = x.to(device)
y = y.to(device)
for i in tqdm(range(num_calculations_gpu), desc="GPU calculations"):
    start = time.time() 
    torch.matmul(x, y)
    end = time.time()
    gpu_times.append(end - start)
 
avg_gpu_time = sum(gpu_times) / len(gpu_times)

print(f"Average GPU time: {avg_gpu_time}")
print(f"Total GPU time: {sum(gpu_times)}")
