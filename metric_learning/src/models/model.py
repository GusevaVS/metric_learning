import glob
import os
from pytorch_metric_learning import testers
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable
import wandb


def train_model(model: torch.nn.Module, loss_function: Callable, optimizer: torch.optim, loss_optimizer: torch.optim,
                dataloader: DataLoader, device: torch.device) -> float:
    """
    Этап обучения модели
    """
    train_loss = 0.0
    inference_steps = int(len(dataloader.dataset) / dataloader.batch_size)
    model.train()

    for batch in tqdm(dataloader,  total=inference_steps):
        images, target = batch
        target = target.type(torch.LongTensor).to(device)
        images = images.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        loss_optimizer.zero_grad()

        embeddings = model(images)
        loss = loss_function(embeddings, target)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()

        train_loss += loss.item()
    train_loss /= len(dataloader.dataset)

    return train_loss


def validate_model(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, epoch: int):
    """
    Этап валидации модели
    """
    model.eval()
    tester = testers.GlobalEmbeddingSpaceTester()
    dataset_dict = {"train": train_dataloader.dataset, "val": val_dataloader.dataset}
    tester.test(dataset_dict, epoch, model)
    return tester.all_accuracies['train']['precision_at_1_level0'], tester.all_accuracies['val']['precision_at_1_level0']


def fit_model(model: torch.nn.Module, optimizer: torch.optim, loss_function: Callable, loss_name: str, loss_optimizer: torch.optim,
              train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: list, device: torch.device,
              checkpoint_path: str, scheduler=None):
    """
    Вызов этапов обучения и валидации модели необходимое количество эпох, в том числе и логгирование
    """
    lst_of_checkpoints = os.listdir(checkpoint_path)
    if lst_of_checkpoints:
        files = [os.path.join(checkpoint_path, file) for file in lst_of_checkpoints]
        last_checkpoint = max(files, key=os.path.getctime)
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_optimizer.load_state_dict(checkpoint['loss_optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0
    loss_name = str(loss_name)
    result = re.search(r"losses.+", loss_name)
    loss_name = result.group(0)[:-2]
    best_accuracy = 0.0
    for epoch in range(current_epoch, epochs[-1]):
        train_loss = train_model(model, loss_function, optimizer, loss_optimizer, train_dataloader, device)
        train_accuracy, val_accuracy = validate_model(model, train_dataloader, val_dataloader, epoch)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

        if scheduler:
            scheduler.step()

        if epoch+1 in epochs:
            wandb.init(project='MetricLearning', name='First_run')
            wandb.log({
                "epoch": epoch+1,
                "loss": loss_name,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy
            })
            wandb.finish()
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_optimizer_state_dict': loss_optimizer.state_dict(),
            'scheduler': scheduler.state_dict()}
        torch.save(checkpoint, f'{checkpoint_path}/model{epoch}.ckpt')

    for file in glob.glob(f'.\\{checkpoint_path}\\*'):
        os.remove(file)
    return best_accuracy
