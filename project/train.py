import os
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, num_epochs=25, title=None):

    if title is None:
        title = datetime.now().strftime("%Y%m%d-%H%M%S")

    # TensorBoard-Writer initialisieren
    base_path = os.path.dirname(os.path.abspath(__file__))  # Absoluter Pfad zum Verzeichnis dieser Datei
    path = os.path.join(base_path, 'runs', title)  # Relativer Pfad zum 'summary'-Ordner
    writer = SummaryWriter(log_dir=path, comment=title)

    model.to(device)  # Modell auf das Gerät (CPU oder GPU) verschieben

    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            phase_start_time = time.time()
            num_batches = len(dataloaders[phase])
            progress_bar = tqdm(enumerate(dataloaders[phase]), total=num_batches, desc=f"{phase} phase")

            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_time = time.time() - phase_start_time
                phase_start_time = time.time()
                remaining_batches = num_batches - batch_idx - 1
                estimated_remaining_time = remaining_batches * batch_time
                progress_bar.set_postfix({
                    'loss': running_loss / ((batch_idx + 1) * inputs.size(0)),
                    'accuracy': running_corrects.double() / ((batch_idx + 1) * inputs.size(0)),
                    'remaining time': str(timedelta(seconds=estimated_remaining_time))
                })

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss, epoch)
                writer.add_scalar('training accuracy', epoch_acc, epoch)
            else:
                writer.add_scalar('validation loss', epoch_loss, epoch)
                writer.add_scalar('validation accuracy', epoch_acc, epoch)

        epoch_time = time.time() - start_time
        remaining_time = epoch_time * (num_epochs - epoch - 1)
        print(f"Time for epoch {epoch + 1}: {str(timedelta(seconds=epoch_time))}")
        print(f"Estimated remaining time: {str(timedelta(seconds=remaining_time))}")

        print()
    writer.close()
    return model

def default_train_model(model, dataloaders, dataset_sizes, device, num_epochs=25, title='default_model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, num_epochs, title)
