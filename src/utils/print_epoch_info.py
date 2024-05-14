import time
import pandas as pd
import numpy as np
from .format_time import format_time

def print_epoch_info(epoch, epochs, start_time, last_epoch_start_time, accumulated_batch_loading_time, losses, lr, accuracy = "---"):

    estimated_finished_time = ((time.time() - start_time) / (epoch + 1)) * (epochs - epoch)
    current_time_str = time.strftime("%H:%M:%S", time.localtime())
    run_time = time.time() - start_time
    epoch_run_time = time.time() - last_epoch_start_time

    df = pd.DataFrame(
        {   f"": [  f"{epoch + 1} / {epochs} ({np.round((epoch + 1) / epochs * 100, 2)}%)",
                    f"{format(losses, '.6f')}",
                    f"{lr}",
                    f"{current_time_str}",
                    f"{format_time(estimated_finished_time)}",
                    f"{format_time(run_time)}",
                    f"{format_time(epoch_run_time)}",
                    f"{format_time(accumulated_batch_loading_time)}",
                    f"{accuracy}"
                    ]
            }, index=["Epoch", "Loss", "Learning rate", "Time", "Time remaining", "Total run time", "Epoch run time", "Accumulated batch loading time", "Accuracy"])

    stats_str = df.to_markdown()
    print(stats_str, flush=True)
