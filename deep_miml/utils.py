from functools import wraps
from heapq import nlargest
from time import time

import numpy as np
import torch
from tqdm import tqdm


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f s\n" % (f.__name__, te - ts))
        return result

    return wrap


def precision_recall_helper(actual, predicted, k):
    active_actual_idxs = set([i for i, e in enumerate(actual) if e == 1])
    predicted_top_k_indices = set(
        [i[0] for i in nlargest(k, enumerate(predicted), key=lambda x: x[1])]
    )
    intersection = active_actual_idxs.intersection(predicted_top_k_indices)
    if len(active_actual_idxs) == 0:
        return 0, 1  # precision, recall
    precision = len(intersection) / k
    recall = len(intersection) / len(active_actual_idxs)
    return round(precision, 2), round(recall, 2)


def get_avg_batch_precision_recall_at_k(actual_lists, predicted_lists, k):
    assert len(actual_lists) == len(predicted_lists)
    batch_len = len(actual_lists)
    precision = [
        precision_recall_helper(actual_lists[i], predicted_lists[i], k)[0]
        for i in range(batch_len)
    ]
    recall = [
        precision_recall_helper(actual_lists[i], predicted_lists[i], k)[1]
        for i in range(batch_len)
    ]
    return np.mean(precision), np.mean(recall)


def test_multi_instance_model(model, device, dataloader):
    model.eval()
    batch_apk_list = []
    batch_ark_list = []
    with torch.no_grad():
        for inputs, sizes, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            sizes = sizes.to(device)
            labels = labels.to(device)

            if inputs.shape[0] == 0:
                continue
            category_type_logits = model(inputs, sizes)
            batch_apk = [
                get_avg_batch_precision_recall_at_k(
                    labels.detach().cpu().tolist(),
                    category_type_logits.detach().cpu().tolist(),
                    k,
                )[0]
                for k in range(1, 7)
            ]
            batch_ark = [
                get_avg_batch_precision_recall_at_k(
                    labels.detach().cpu().tolist(),
                    category_type_logits.detach().cpu().tolist(),
                    k,
                )[1]
                for k in range(1, 7)
            ]
            batch_apk_list.append(batch_apk)
            batch_ark_list.append(batch_ark)
        test_apk = np.around(np.mean(batch_apk_list, axis=0), 3)
        test_ark = np.around(np.mean(batch_ark_list, axis=0), 3)
    results = {}
    results["precision_at"] = {k + 1: v for k, v in dict(enumerate(test_apk)).items()}
    results["recall_at"] = {k + 1: v for k, v in dict(enumerate(test_ark)).items()}
    return results
