from torchvision.transforms import transforms

def get_tensor_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(1.0258, 1.0862, 1.0990),
            std=(0.4997, 0.5005, 0.5017)
        )
    ])
    return transform