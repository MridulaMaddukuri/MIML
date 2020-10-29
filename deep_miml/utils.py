from functools import wraps
from time import time
import numpy as np
from heapq import nlargest

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f s\n' % (f.__name__,
        #        args, kw,
        #       te-ts))
        print('func:%r took: %2.4f s\n' % (f.__name__, te-ts))
        return result
    return wrap



def precision_recall_helper(actual, predicted, k):
    active_actual_idxs = set([i for i, e in enumerate(actual) if e == 1])
    predicted_top_k_indices = set([i[0] for i in nlargest(k, enumerate(predicted), key=lambda x: x[1])])
    intersection = active_actual_idxs.intersection(predicted_top_k_indices)
    if len(active_actual_idxs) == 0:
        return 0, 1  # precision, recall
    precision = len(intersection) / k
    recall = len(intersection) / len(active_actual_idxs)
    return round(precision, 2), round(recall, 2)


def get_avg_batch_precision_recall_at_k(actual_lists, predicted_lists, k):
    assert len(actual_lists) == len(predicted_lists)
    batch_len = len(actual_lists)
    precision = [precision_recall_helper(actual_lists[i], predicted_lists[i], k)[0] for i in range(batch_len)]
    recall = [precision_recall_helper(actual_lists[i], predicted_lists[i], k)[1] for i in range(batch_len)]
    return np.mean(precision), np.mean(recall)