"""
Utility functions
"""

import logging
import torch
import datetime
import numpy as np


def save_model(save_path, model, valid_loss):

    if not save_path:
        return

    state_dict = {"model_state_dict": model.state_dict()}

    torch.save(state_dict, save_path)

    logging.info(f"Model saved to {save_path}")


def load_model(load_path, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not load_path:
        return

    state_dict = torch.load(load_path, map_location=device)
    logging.info(f"Model loaded from {load_path}")

    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]


def calc_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
