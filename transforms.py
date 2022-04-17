from torchvision.transforms import transforms

def get_tensor_transform():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform