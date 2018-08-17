import os
import pandas as pd


def record_info(info, filename, mode):
    df = pd.DataFrame.from_dict(info)

    if mode == 'train':
        column_names = ['Train Accuracy']
    elif mode == 'test':
        column_names = ['Test Accuracy']
    else:
        return

    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else:  # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names)
