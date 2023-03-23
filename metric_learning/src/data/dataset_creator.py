import albumentations as A
from glob import glob
from metric_learning.configs.transforms import train_albumentation
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)


class FruitVegetablesDatasetTrain(CustomDataset):
    def __init__(self, data: pd.DataFrame, transform: transforms = None, albumentation: A = None):
        self.data = data
        self.transform = transform
        self.targets = np.array(self.data['int_label'])
        self.images = np.array(self.data['filename'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data.filename.iloc[index]
        image = Image.open(image).convert('RGB')
        y_label = self.data.iloc[index, 2]
        if self.transform:
            image = self.transform(image)
        return image, y_label


class FruitVegetablesDatasetTest(CustomDataset):
    def __init__(self, data: pd.DataFrame, transform: transforms = None):
        self.data = data
        self.transform = transform
        self.images = np.array(self.data['filename'])

    def __getitem__(self, idx: int):
        image = self.data.filename.iloc[idx]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def load_data(data_dir: str, le_encoder: preprocessing, albumentate: bool = False) -> pd.DataFrame:
    """
    Загрузка данных из директории и создание датафрейма
    """
    list_filepaths = glob(data_dir + '**/*.jpg')
    data = pd.DataFrame(list_filepaths, columns=['filename'])
    data['label'] = data['filename'].apply(lambda x: x.split("\\")[-2])
    if albumentate:
        data = albumentation_dataset(data, train_albumentation, data_dir)
    data['int_label'] = le_encoder.fit_transform(data['label'])
    return data


def albumentation_dataset(df: pd.DataFrame, albumentation: A, base_path: str) -> pd.DataFrame:
    """
    Применение альбументации к изображениям датасета и сохранение новых изображений в папке data/albumentation
    """
    for idx in range(len(df)):
        image = df.filename.iloc[idx]
        image = Image.open(image).convert('RGB')
        image = albumentation(image=np.array(image))
        image = Image.fromarray(image['image'])
        path_for_albumentate = '\\'.join(base_path.split('\\')[:-2])
        if not os.path.isdir(f"{path_for_albumentate}\\albumentation"):
            os.mkdir(f"{path_for_albumentate}\\albumentation")
        img_path_save = f'{path_for_albumentate}\\albumentation\\{df.label.iloc[idx]}'
        if not os.path.isdir(img_path_save):
            os.mkdir(img_path_save)
        image.save(f'{img_path_save}\\album_image{idx}.jpg')
        df = df.append({'filename': f'{img_path_save}\\album_image{idx}.jpg', 'label': df.label.iloc[idx]}, ignore_index=True)
    return df


def dataloader_creator(data: pd.DataFrame, batch_size: int, flag_train: bool, is_shuffle: bool,
                       is_droplast: bool, transform: transforms = None, albumentation: A = None) -> DataLoader:
    """
    Создание DataLoader из датасета
    """
    dataset = FruitVegetablesDatasetTrain(data, transform=transform, albumentation=albumentation) \
        if flag_train else FruitVegetablesDatasetTest(data, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=is_droplast,
    )
    return dataloader