import torch

cuda_is_available = torch.cuda.is_available()
print("CUDA is availabe:", cuda_is_available)

if cuda_is_available: 

    print("Nb. devices:", torch.cuda.device_count())

    current_device = torch.cuda.current_device()
    print("Current device:", current_device)
    print("Current device name:", torch.cuda.get_device_name(current_device))

    my_tensor = torch.tensor([1.0,2.0,3.0,4.0])
    print("Tensor:", my_tensor )
    print("Tensor device:", my_tensor.device)

    my_tensor = my_tensor.to('cuda')
    print(my_tensor)
    print("Tensor device:", my_tensor.device)
