import os

import cv2
import numpy as np

FPS = 25
log_directory = 'data/log_file'
MetaDataFile = ['LeftMove', 'got', 'RightMove', 'Jump', 'Firekill']
offset = 20


def retrieve_data(root='data/'):
    """
    Main data retrieval function. Returns an X and y matrix
    :param root: the root data directory
    :return: preprocessed dataset
    """
    frames = [(root + "frames/" + frame_path) for frame_path in os.listdir(root + "frames/")]
    data = np.concatenate([cv2.resize(cv2.imread(z), dsize=(240, 360), interpolation=cv2.INTER_CUBIC).reshape((1, 3, 240, 360)) for z in frames], axis=0)
    target = get_video_log(len(frames))
    return data, target


def get_video_log(frame_num):
    current_file = open(log_directory)
    video_data = np.zeros([frame_num, 5])

    '''
    Player Firekill: EnemyType =  time = 76.96
    Player RightMove: StTime = 83.28 EdTime = 83.32
    Player Jump: StTime = 83.32 EdTime = 83.36
    Player RightMove: StTime = 83.4 EdTime = 83.44
    Player Jump: StTime = 83.44 EdTime = 83.52
    Player RightMove: StTime = 83.88 EdTime = 83.92
    '''
    for line in current_file:
        split_line = line.split(" ")
        label = split_line[1].replace(':', '')

        if label == "got":
            # can't really do this one since no timestamp
            # data[last_seen_time, md.index(label)] = 1
            continue

        elif label == 'Firekill':
            last_seen_time = float(split_line[-1].strip()) - offset
            video_data[int(last_seen_time * FPS), 4] = 1

        else:
            last_seen_time = float(split_line[-1].strip()) - offset
            start_time = float(split_line[4].strip()) - offset
            video_data[int(start_time * FPS):int(last_seen_time * FPS), MetaDataFile.index(label)] = 1

    return video_data
