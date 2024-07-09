
# IMAGE_HEIGHT = 256
# IMAGE_WIDTH = 256

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

opt = Option()

transform0 = A.Compose([
    A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    # ToTensorV2()
])

transform = A.Compose([
    A.Resize(width=IMAGE_WIDTH, height=IMAGE_HEIGHT),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25),
    A.Rotate(p=0.25),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
])

class Dataset_Class(Dataset):
    def __init__(self, root_dir, transform=transform0):
        self.root_dir = root_dir
        self.transform = transform
        self.folder_list = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir + '/' + self.folder_list[index] + '/' + "image.png")).convert("RGB")
        mask1 = Image.open(os.path.join(self.root_dir + '/' + self.folder_list[index] + '/' + "mask.png")).convert('L')
        image = np.array(img, dtype=np.float32)
        mask = np.array(mask1, dtype=np.float32)
        mask = np.expand_dims(mask, axis=-1)
        # mask1 = np.asarray(mask1)
        # k = np.expand_dims(mask1, axis=-1)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        image = image.type(opt.dtype)
        mask = mask.type(opt.dtype)
        image = torch.reshape(image, (3, IMAGE_WIDTH, IMAGE_HEIGHT))
        mask = torch.reshape(mask, (1, IMAGE_WIDTH, IMAGE_HEIGHT))
        mask = mask/255.0
        return image, mask
    
    
def get_train_loader(root_training_dir='/kaggle/input/unet-project-dataset-isic-2017/Total Dataset/training_data'):
    data = Dataset_Class(root_dir=root_training_dir)
    
    BATCH_SIZE = opt.batch_size
    dataloader_train = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True)

    print("TRAIN DATALOADER DONE")
    return dataloader_train


def get_val_loader(root_val_dir='/kaggle/input/unet-project-dataset-isic-2017/Total Dataset/validation_data'):
    
    data = Dataset_Class(root_dir=root_val_dir)
    
    BATCH_SIZE = opt.batch_size
    dataloader_train = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True)

    print("VAL DATALOADER DONE")
    return dataloader_train


# def get_test_loader(root_test_dir='testing_data'):
    
#     data = Dataset_Class(root_dir=root_test_dir)
#     BATCH_SIZE = opt.batch_size
#     dataloader_train = DataLoader(
#         data, batch_size=BATCH_SIZE, shuffle=True)