import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel
from math import exp

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples.
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')
    
    train_perplexities = [exp(loss) for loss in train_losses]
    val_perplexities = [exp(loss) for loss in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch.
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices.to(device)
        lengths = torch.tensor(lengths).to(device)

        input_seq = indices[:, :-1]
        target_seq = indices[:, 1:]
        adjusted_lengths = lengths - 1

        optimizer.zero_grad()

        logits = model(input_seq, adjusted_lengths)
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = target_seq[:, :T].contiguous().view(-1)
        loss = criterion(logits_flat, targets_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * indices.size(0)

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch.
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        indices = indices.to(device)
        lengths = torch.tensor(lengths).to(device)
        
        input_seq = indices[:, :-1]
        target_seq = indices[:, 1:]
        adjusted_lengths = lengths - 1
        
        logits = model(input_seq, adjusted_lengths)
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = target_seq[:, :T].contiguous().view(-1)
        loss = criterion(logits_flat, targets_flat)
        
        val_loss += loss.item() * indices.size(0)
        
    val_loss /= len(loader.dataset)
    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5,
          model_save_path: Optional[str] = None, early_stopping_patience: int = 10):
    """
    Train language model for several epochs with early stopping.
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        plot_losses(train_losses, val_losses)

        print(f"Epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if model_save_path is not None:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, model_save_path)
                print(f"Checkpoint saved to {model_save_path} with val_loss {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
        
        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Validation loss has not improved for {early_stopping_patience} consecutive epochs. Early stopping!")
            break