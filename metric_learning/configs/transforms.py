import albumentations as A
import torchvision.transforms as transforms

train_transform = transforms.Compose([transforms.Resize((640, 640)),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          (0.4914, 0.4822, 0.4465),
                                          (0.2023, 0.1994, 0.2010))])

val_transform = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010))])

train_albumentation = A.Compose([A.Resize(height=640, width=640),
                                 A.RandomCrop(height=480, width=480, p=0.5),
                                 A.Blur(blur_limit=3, p=0.4),
                                 A.ShiftScaleRotate(p=0.5),
                                 A.Rotate(limit=90),
                                 A.RandomBrightnessContrast(p=0.5),
                                 A.HorizontalFlip(p=0.5)])

test_transform = transforms.Compose([transforms.Resize((640, 640)),
                                     transforms.ToTensor()])
