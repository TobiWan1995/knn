import torch
import torch.optim as optim
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

'''Training for torchlayer.py'''


# Trainingsfunktionen
def train_torch(model, train_loader, valid_loader=None, epochs=1, lr=0.001, warmup_steps=None, weight_decay=None,
                early_stopping_patience=None, title=None, optimizer_cls=optim.Adam):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = model.loss_fn

    if warmup_steps is not None and weight_decay is not None:
        optimizer, scheduler = get_optimizer_with_warmup(model, lr, warmup_steps, weight_decay, optimizer_cls)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        scheduler = None

    early_stopping = None
    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(patience=early_stopping_patience)

    if title is None:
        title = datetime.now().strftime("%Y%m%d-%H%M%S")

    path = f"C:/Users/walte/github/knn/summary/{title}"
    writer = SummaryWriter(log_dir=path, comment=title)

    for epoch in range(epochs):
        model.train()
        running_loss, running_accuracy = run_training_epoch(model, train_loader, optimizer, criterion, device)
        avg_train_loss, avg_train_accuracy = calculate_train_stats(running_loss, running_accuracy, len(train_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_train_accuracy, epoch)

        if valid_loader is not None:
            avg_valid_loss, avg_valid_accuracy = evaluate_model(model, valid_loader, criterion, device)
            print(f"Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")

            writer.add_scalar('Loss/valid', avg_valid_loss, epoch)
            writer.add_scalar('Accuracy/valid', avg_valid_accuracy, epoch)

            if early_stopping is not None:
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(torch.load('checkpoint.pt'))
                    break

        if scheduler is not None:
            scheduler.step()

    writer.close()


def run_training_epoch(model, train_loader, optimizer, criterion, device):
    running_loss = 0.0
    running_accuracy = 0.0
    train_accuracy = torchmetrics.Accuracy(task=model.task, num_classes=model.num_classes).to(device)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += train_accuracy(outputs, labels.int()).item()

    return running_loss, running_accuracy


# Warm Up / Weight Decay
def get_optimizer_with_warmup(model, lr, warmup_steps, weight_decay, optimizer_cls):
    optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))
    return optimizer, scheduler


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_loss = val_loss
        torch.save(model.state_dict(), 'checkpoint.pt')


''' Statistics '''


def calculate_train_stats(running_loss, running_accuracy, num_batches):
    avg_train_loss = running_loss / num_batches
    avg_train_accuracy = running_accuracy / num_batches
    return avg_train_loss, avg_train_accuracy


def evaluate_model(model, valid_loader, criterion, device):
    model.eval()
    valid_running_loss = 0.0
    valid_running_accuracy = 0.0
    valid_accuracy = torchmetrics.Accuracy(task=model.task, num_classes=model.num_classes).to(device)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            valid_running_accuracy += valid_accuracy(outputs, labels.int()).item()

    avg_valid_loss = valid_running_loss / len(valid_loader)
    avg_valid_accuracy = valid_running_accuracy / len(valid_loader)

    return avg_valid_loss, avg_valid_accuracy
