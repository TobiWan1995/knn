import os
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    def __init__(self, title=None):
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = self._get_tensorboard_writer(title)

    def _get_tensorboard_writer(self, title):
        base_path = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if title:
            path = os.path.join(base_path, '..', '..', 'runs', f"{title} - {timestamp}")
        else:
            path = os.path.join(base_path, '..', '..', 'runs', f"untitled - {timestamp}")
        return SummaryWriter(log_dir=path)

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None):
        self.log_scalar('Loss/train', train_loss, epoch)
        self.log_scalar('Accuracy/train', train_acc, epoch)
        if val_loss is not None and val_acc is not None:
            self.log_scalar('Loss/val', val_loss, epoch)
            self.log_scalar('Accuracy/val', val_acc, epoch)

    def close(self):
        if self.writer:
            self.writer.close()
