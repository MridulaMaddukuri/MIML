import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from deep_miml.cifar_bags import collate_fn
from deep_miml.utils import get_avg_batch_precision_recall_at_k


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


# Training settings
parser = argparse.ArgumentParser(description="MIML experiments evaluation")

# parser.add_argument('--model_name', type=str,
# required=True, help='options: resnet18, resnext50 and alexnet')

parser.add_argument("--use_pretrained", type=bool, default=True, help="True or False")

# parser.add_argument('--model_type', type=str, required=True, help='avg or attention')

parser.add_argument("--batch_size", type=int, default=4, metavar="N")

parser.add_argument("--cuda", action="store_true")

parser.add_argument(
    "--data_file_path",
    type=str,
    default="data/miml_test_data_5000.pt",
    help="data/miml_test_data_5000.pt",
)

parser.add_argument("--save_folder", type=str, default="results/")

parser.add_argument(
    "--cpu_workers", type=int, default=4, metavar="S", help="Number of workers"
)

args = parser.parse_args()


print(args.data_file_path)
num_categories = 10

image_datasets = torch.load(args.data_file_path)

# Create training and validation dataloaders
dataloader = torch.utils.data.DataLoader(
    image_datasets["test"],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.cpu_workers,
    collate_fn=collate_fn,
)

if args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

result_dfs = []

for m_n in ["resnet18", "resnet34", "resnet50"]:
    for m_t in ["avg", "attention"]:
        file_path = Path.cwd().joinpath(
            args.save_folder,
            "cifar10_{}_{}_pretrained_{}.pt".format(m_n, m_t, args.use_pretrained),
        )

        # if m_t == "avg":
        #   model = Average(num_classes = num_categories,model_name=m_n)
        #   model.load_state_dict(torch.load(file_path))
        # elif m_t == 'attention':
        #   model = Attention(num_classes = num_categories, model_name= m_n)
        #   model.load_state_dict(torch.load(file_path))
        # else:
        #   print("Enter valid model type")
        #   exit(0)

        model = torch.load(file_path)

        # model = torch.load(model_path) #, map_location=torch.device("cpu"))
        model.to(device)

        res = test_multi_instance_model(model, device, dataloader)

        res_df = pd.DataFrame(res)
        print(res_df)
        res_df["model_name"] = m_n
        res_df["model_type"] = m_t
        print(res_df)

        result_dfs.append(res_df)


result_dfs = pd.concat(result_dfs)
print(result_dfs.head())


sns.lineplot(data=result_dfs, x="year", y="passengers", hue="month", style="month")

# results_file = Path.cwd().joinpath(args.save_folder,
# 'results_{}_{}_pretrained_{}.json'.format(args.model_name,
# args.model_type, args.use_pretrained))


# with open(results_file, 'w') as fp:

#     json.dump(result, fp, indent=4, sort_keys=True,
#               separators=(', ', ': '), ensure_ascii=False)
