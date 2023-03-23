import datetime
import json
from metric_learning.src.data.dataset_creator import FruitVegetablesDatasetTrain
from metric_learning.configs.transforms import train_transform, test_transform
import os
import pandas as pd
from PIL import Image
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pytorch_metric_learning.distances import CosineSimilarity
import torch
from torchvision import transforms


def load_data(data_path: str):
    """
    Загружает данные, на которых тренировалась модель, классы которых ей известны
    """
    data = pd.read_csv(data_path)
    train_dataset = FruitVegetablesDatasetTrain(data, transform=train_transform)
    return train_dataset


def img_preprocessing(img_pth: str):
    """
    Предобработка изображения перед предсказанием
    """
    img = Image.open(img_pth).convert('RGB')
    img = test_transform(img)
    img = img.unsqueeze(0)
    normalize_img = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    img = normalize_img(img)
    return img


def predict_neighbors():
    """
    Выводит наименование классов k ближайших соседей входного изображения
    """
    with open(f'..\\..\\metric_learning\\configs\\predict_config.json', 'r') as f:
        config = json.load(f)
    train_dataset = load_data(config['path_train_csv'])
    model = torch.load(config['model_pth'])
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.8)
    inference_model = InferenceModel(model, match_finder=match_finder)
    img = img_preprocessing(config['image_path'])
    distances, indices = inference_model.get_nearest_neighbors(img, k=config['k_nearest_neighbors'])
    neighbors_label = [train_dataset[i.item()][1] for i in indices.cpu()[0]]
    neighbors = []
    for i in neighbors_label:
        elem = train_dataset[train_dataset['int_label'] == i].iloc[0]['label']
        neighbors.append(elem)
    current_time = str(datetime.datetime.now())
    if not os.path.isdir(config['predict_pth']):
        os.mkdir(config['predict_pth'])
    with open(f'nearest_neighbors_{current_time}.json', 'w') as f:
        result = {"image": config["image_path"], "neighbors": neighbors}
        json.dump(result, f)


if __name__ == "__main__":
    predict_neighbors()
