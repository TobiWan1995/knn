from tqdm import tqdm
import time
from datetime import timedelta


class ProgressBar:
    def __init__(self, total_batches):
        self.progress_bar = tqdm(total=total_batches)

    def update(self, epoch, epochs, batch_idx, num_batches, running_loss, running_accuracy, start_time, inputs,
               batch_size):
        elapsed_time = time.time() - start_time
        estimated_epoch_time = elapsed_time * num_batches / (batch_idx + 1)
        total_remaining_time = estimated_epoch_time * (epochs - epoch - 1)

        self.progress_bar.set_postfix({
            'epoch': f'{epoch + 1}/{epochs}',
            'loss': running_loss / ((batch_idx + 1) * batch_size),
            'accuracy': running_accuracy / ((batch_idx + 1) * batch_size),
            'time/epoch': str(timedelta(seconds=estimated_epoch_time)),
            'remaining': str(timedelta(seconds=total_remaining_time))
        }, refresh=True)

        self.progress_bar.update(1)

    def close(self):
        self.progress_bar.close()
