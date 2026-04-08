import torch 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")   
x = torch.tensor([2.0], device=device, requires_grad=True)
y = torch.tensor([3.0], device=device, requires_grad=True)

z = (x * y) ** 2
alpha = torch.ones_like(z, device=device)
result = z + alpha

result.backward()
print(x.grad)
print(y.grad)


