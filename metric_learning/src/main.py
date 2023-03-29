import json
from metric_learning.src.data import dataset_creator as d_c
from metric_learning.src.models import model
from metric_learning.configs import transforms, loss
import os
from pytorch_metric_learning import regularizers
import seed_func
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import warnings
warnings.filterwarnings('ignore')


def main():
    with open('..\\..\\metric_learning\\configs\\config.json', 'r') as f:
        config = json.load(f)
    if not os.path.isdir(config["checkpoint_path"]):
        os.mkdir(config["checkpoint_path"])
    seed_func.seed_everything(config["seed"], config['device'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    le = preprocessing.LabelEncoder()
    data_train = d_c.load_data(config["train_path"], le_encoder=le, albumentate=bool(config['album_flag']))
    data_train.to_csv('..\\..\\hw_metric_learning\\data\\train.csv', index=False)  # запись .csv для облегчения работы predict.py
    data_val = d_c.load_data(config["val_path"], le_encoder=le)
    best_accuracy = float('-inf')
    best_model = ''
    for l, params in loss.losses.items():
        train_dataloader = d_c.dataloader_creator(data_train, batch_size=config['batch_size'], flag_train=True,
                                                  is_shuffle=True, is_droplast=True,
                                                  transform=transforms.train_transform,
                                                  albumentation=transforms.train_albumentation)
        val_dataloader = d_c.dataloader_creator(data_val, batch_size=config['batch_size'], flag_train=True,
                                                is_shuffle=False, is_droplast=False, transform=transforms.val_transform)

        model_resnet18 = torchvision.models.resnet18(pretrained=True)
        for param in model_resnet18.parameters():
            param.requires_grad = False
        model_resnet18.fc = nn.Linear(in_features=512, out_features=params['embedding_size'])
        for param in model_resnet18.layer4.parameters():
            param.requires_grad = True
        model_resnet18.to(device)

        optimizer = optim.AdamW(list(model_resnet18.layer4.parameters()) + list(model_resnet18.fc.parameters()),
                                lr=config['optim_lr'],
                                weight_decay=config['weight_decay'])
        R = regularizers.RegularFaceRegularizer()
        loss_func = l(**params, weight_regularizer=R).to(device)
        loss_optimizer = torch.optim.AdamW(loss_func.parameters(), lr=config['loss_optim_lr'])

        accuracy = model.fit_model(model_resnet18, optimizer, loss_func, str(l), loss_optimizer, train_dataloader,
                                   val_dataloader, config['epochs'], device=device,
                                   checkpoint_path=config['checkpoint_path'])
        if accuracy > best_accuracy:
            best_model = model_resnet18
            best_accuracy = accuracy
    torch.save(best_model, '..\\..\\metric_learning\\src\\models\\best_model.pth')


if __name__ == "__main__":
    main()
