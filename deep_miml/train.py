# https://lernapparat.de/debug-device-assert/
import argparse
import json
import os

# import copy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from deep_miml.cifar_bags import collate_fn
from deep_miml.models import Attention, Average
from deep_miml.utils import get_avg_batch_precision_recall_at_k

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# @timing
def train_miml_model(
    model,
    model_name,
    model_type,
    use_pretrained,
    device,
    dataloaders,
    criterion,
    optimizer,
    save_folder,
    lr=0.001,
    num_epochs=25,
    early_stopping=True,
    patience=5,
):
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_avg_precision = 0.0
    val_apk_history = []
    val_ark_history = []
    stop_train = False
    model = model.to(device)
    if isinstance(model.fc, list):
        for l in model.fc:
            if device != "cpu":
                l.cuda()
        for l in model.attn_layers:
            if device != "cpu":
                l.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch = 0
    print(device)
    while not stop_train and epoch <= num_epochs:
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                batch_iter_count = 0
                loss_list = []
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            num_items = 0
            batch_apk_list = []
            batch_ark_list = []
            # Iterate over data.
            for inputs, sizes, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                sizes = sizes.to(device)
                labels = labels.to(device)

                if inputs.shape[0] == 0:
                    # Checking batch_size >1
                    continue

                # zero the parameter gradients
                optimizer.zero_grad()
                # TODO handle out of memory errors?
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    category_type_logits = model(inputs, sizes)

                    # print(category_type_logits.shape, labels.shape)

                    loss = criterion(category_type_logits, labels.float())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        loss_list.append(loss.cpu().item())
                        batch_iter_count += 1
                        if batch_iter_count % 1000 == 0:
                            print(
                                f"\nRunning avg Loss at batch iter \
                                {batch_iter_count} is \
                                {np.mean(loss_list[-1000::])}"
                            )

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # instead of accuracy, do hamming loss
                num_items += inputs.size(0)
                # Calculating precision at k = 1, 2, 3, 4, 5,6 for each batch.
                # batch apk is a list of len =6
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

            epoch_loss = running_loss / num_items

            epoch_apk = np.mean(batch_apk_list, axis=0)
            epoch_ark = np.mean(batch_ark_list, axis=0)

            print("{} Loss: {:.4f}".format(phase, epoch_loss))
            if phase == "train":
                print(
                    f"\nTrain -- epoch average precision \
                    at k = 1, 2, 3, 4, 5, 6: {epoch_apk}"
                )
                print(
                    f"Train -- epoch average recall \
                    at k = 1, 2, 3, 4, 5, 6: {epoch_ark}\n"
                )
            else:
                print(
                    f"\nValidation -- epoch average precision \
                    at k = 1, 2, 3, 4, 5, 6: {epoch_apk}\n"
                )
                print(
                    f"\nValidation -- epoch average recall \
                    at k = 1, 2, 3, 4, 5, 6: {epoch_ark}\n"
                )

            # deep copy the model
            if phase == "val" and np.mean(epoch_apk) > best_avg_precision:
                # best_loss = epoch_loss
                best_avg_precision = np.mean(epoch_apk)
                early_stop_count = 0
                best_model_path = Path.cwd().joinpath(
                    save_folder,
                    "intermediate_{}_{}_pretrained_{}.pt".format(
                        model_name, model_type, int(use_pretrained)
                    ),
                )
                torch.save(model, best_model_path)
            if phase == "val" and early_stopping:
                if early_stop_count >= patience:
                    stop_train = True
                early_stop_count += 1
            if phase == "val":
                val_apk_history.append(epoch_apk)
                val_ark_history.append(epoch_ark)

        epoch += 1

    # load best model weights
    model = torch.load(best_model_path)
    return model  # , val_acc_history


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
                # Checking batch_size >1
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


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="MIML experiments")

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of epochs to train (default: 50)",
    )

    parser.add_argument("--batch_size", type=int, default=4, metavar="N")

    parser.add_argument("--cuda", action="store_true")

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    parser.add_argument(
        "--patience", type=int, default=5, metavar="S", help="patience (default: 5)"
    )

    parser.add_argument(
        "--cpu_workers", type=int, default=4, metavar="S", help="Number of workers"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="options: resnet18, resnext50 and alexnet",
    )

    parser.add_argument(
        "--use_pretrained", type=bool, required=True, help="True or False"
    )

    parser.add_argument(
        "--model_type", type=str, required=True, help="avg or attention"
    )

    parser.add_argument("--save_folder", type=str, default="/tmp")

    parser.add_argument(
        "--data_file_path", type=str, required=True, help="data/miml_data.pt"
    )

    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # root = '../data'

    # TODO: load saved TensorDataset
    image_datasets = torch.load(args.data_file_path)

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=(x == "train"),
            num_workers=args.cpu_workers,
            collate_fn=collate_fn,
        )
        for x in ["train", "val", "test"]
    }

    print("DataLoaders ready ... \n")
    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    num_categories = 10

    if args.model_type == "avg":
        model_ft = Average(num_classes=num_categories, model_name=args.model_name)
    elif args.model_type == "attention":
        model_ft = Attention(num_classes=num_categories, model_name=args.model_name)

    else:
        print("Enter valid model type")
        exit(0)

        # Send the model to GPU
    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)

    # You should pass logits to nn.BCEwithLogitsLoss
    # and probabilities (using "sigmoid") to nn.BCELoss.
    # Using BCEWithLogitsLoss because it is more stable
    criterion = torch.nn.BCEWithLogitsLoss(weight=None, reduction="mean")

    # Train and evaluate
    model_ft = train_miml_model(
        model=model_ft,
        device=device,
        dataloaders=dataloaders_dict,
        criterion=criterion,
        optimizer=optimizer_ft,
        save_folder=args.save_folder,
        num_epochs=args.epochs,
        early_stopping=True,
        patience=args.patience,
    )

    torch.save(
        model_ft,
        Path.cwd().joinpath(
            args.save_folder,
            "cifar10_{}_{}_pretrained_{}.pt".format(
                args.model_name, args.model_type, args.use_pretrained
            ),
        ),
    )

    # TESTING
    result = test_multi_instance_model(model_ft, device, dataloaders_dict["test"])

    print(result)
    dataset_name = args.data_file_path.split("/")[-1].split(".")[0]

    results_file = Path.cwd().joinpath(
        args.save_folder,
        "results_{}_{}_{}.json".format(args.model_name, args.model_type, dataset_name),
    )

    with open(results_file, "w") as fp:
        json.dump(
            result,
            fp,
            indent=4,
            sort_keys=True,
            separators=(", ", ": "),
            ensure_ascii=False,
        )
