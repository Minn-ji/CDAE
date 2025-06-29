import numpy as np
from tensorflow.keras.utils import get_file, to_categorical
from zipfile import ZipFile


def load_data():
    '''
    Load data from MovieLens 100K Dataset
    http://grouplens.org/datasets/movielens/

    Note: Uses ua.base and ua.test in the dataset.

    :return: train_users, train_x, test_users, test_x
    :rtype: list of int, np.array, list of int, np.array
    '''
    path = get_file('ml-100k.zip', origin='http://files.grouplens.org/datasets/movielens/ml-100k.zip')

    with ZipFile(path, 'r') as ml_zip:
        max_item_id = -1
        train_history = {}
        with ml_zip.open('ml-100k/ua.base', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                user_id, item_id = int(user_id), int(item_id)

                if user_id not in train_history:
                    train_history[user_id] = [item_id]
                else:
                    train_history[user_id].append(item_id)

                max_item_id = max(max_item_id, item_id)

        test_history = {}
        with ml_zip.open('ml-100k/ua.test', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                user_id, item_id = int(user_id), int(item_id)

                if user_id not in test_history:
                    test_history[user_id] = [item_id]
                else:
                    test_history[user_id].append(item_id)

    max_item_id += 1  # item_id starts from 1
    train_users = list(train_history.keys())
    train_x = np.zeros((len(train_users), max_item_id), dtype=np.int32)

    for i, hist in enumerate(train_history.values()):
        mat = to_categorical(hist, max_item_id)
        train_x[i] = np.sum(mat, axis=0)

    test_users = list(test_history.keys())
    test_x = np.zeros((len(test_users), max_item_id), dtype=np.int32)

    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = np.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x
