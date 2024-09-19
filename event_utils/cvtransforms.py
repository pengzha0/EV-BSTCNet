import random
import numpy as np

def CenterCrop(event_low, event_high, size):
    w, h = event_low.shape[-1], event_low.shape[-2]
    th, tw = size
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)
    event_low = event_low[..., y1: y1 + th, x1: x1 + tw]
    if event_high is not None:
        event_high = event_high[..., y1: y1 + th, x1: x1 + tw]
        return event_low, event_high
    else:
        return event_low, None

def RandomCrop(event_low, event_high, size):
    w, h = event_low.shape[-1], event_low.shape[-2]
    th, tw = size
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    event_low = event_low[..., y1: y1 + th, x1: x1 + tw]
    if event_high is not None:
        event_high = event_high[..., y1: y1 + th, x1: x1 + tw]
        return event_low, event_high
    else:
        return event_low, None

def HorizontalFlip(event_low, event_high):
    if random.random() > 0.5:
        event_low = np.ascontiguousarray(event_low[..., ::-1])
        if event_high is not None:
            event_high = np.ascontiguousarray(event_high[..., ::-1])
    return event_low, event_high
