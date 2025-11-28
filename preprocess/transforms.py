import torchvision.transforms as T

def get_transforms(image_size, is_train=True):
    # image_size: (H, W)
    
    if is_train:
        transform = T.Compose([
            T.Resize(image_size, interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5)
        ])
    else:
        transform = T.Compose([
            T.Resize(image_size, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    return transform
