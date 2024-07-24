import torchmetrics

from utils.torch.progressbar import *
from utils.torch.earlystopping import *
from utils.torch.tensorboard import *


class Trainer:
    def __init__(self, model, criterion, train_loader, valid_loader=None,
                 num_classes=10, task="binary", optimizer=None, scheduler=None):
        self.model = model
        self.criterion = criterion

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.num_classes = num_classes
        self.task = task

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.best_model_wts = None
        self.best_acc = 0.0

        self.progress_bar = None  # ProgressBar wird später initialisiert
        self.tensorboard_logger = None  # TensorBoardLogger wird später initialisiert

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accuracy_metric = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)

        self.model.to(self.device)

    def train(self, epochs=1, title=None, early_stopping_patience=None):
        self.tensorboard_logger = TensorBoardLogger(title)
        self.early_stopping = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience else None

        for epoch in range(epochs):
            train_loss, train_acc = self._run_epoch(epoch, epochs, self.train_loader, train=True)
            val_loss, val_acc = self._evaluate_model(epoch, epochs) if self.valid_loader else (None, None)
            self.tensorboard_logger.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

            if self.early_stopping and self.early_stopping.early_stop:
                print("Early stopping")
                self.model.load_state_dict(torch.load('checkpoint.pt'))
                break

            if val_acc and val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_model_wts = self.model.state_dict()

        if self.best_model_wts:
            self.model.load_state_dict(self.best_model_wts)
        self.tensorboard_logger.close()
        return self.model

    def _run_epoch(self, epoch, epochs, loader, train=True):
        phase = 'train' if train else 'eval'
        self.model.train() if train else self.model.eval()

        running_loss = 0.0
        running_accuracy = 0.0

        num_batches = len(loader)
        start_time = time.time()
        self.progress_bar = ProgressBar(num_batches)  # Initialisieren der ProgressBar

        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)
            loss, accuracy = self._process_batch(inputs, labels, train)
            running_loss += loss
            running_accuracy += accuracy
            self.progress_bar.update(epoch, epochs, batch_idx, num_batches, running_loss, running_accuracy, start_time, inputs, batch_size)

        self.progress_bar.close()

        if self.scheduler and train:
            self.scheduler.step()

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_accuracy / len(loader.dataset)
        return epoch_loss, epoch_acc

    def _process_batch(self, inputs, labels, train):
        if self.optimizer:
            self.optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            if train:
                loss.backward()
                if self.optimizer:
                    self.optimizer.step()

        return loss.item() * inputs.size(0), self.accuracy_metric(outputs, labels.int()).item() * inputs.size(0)

    def _evaluate_model(self, epoch, epochs):
        return self._run_epoch(epoch, epochs, self.valid_loader, train=False)
