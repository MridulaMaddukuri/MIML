import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch

from deep_miml.cifar_bags import collate_fn
from deep_miml.utils import test_multi_instance_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIML experiments evaluation")

    parser.add_argument(
        "--use_pretrained", action="store_true", help="use pretrained weights"
    )
    parser.add_argument("--batch_size", type=int, default=4, metavar="N")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--data_file_path",
        type=str,
        default="data/miml_test_data_5000.pt",
    )
    parser.add_argument("--save_folder", type=str, default="results/")
    parser.add_argument(
        "--cpu_workers", type=int, default=4, metavar="S", help="Number of workers"
    )

    args = parser.parse_args()

    print(args.data_file_path)

    image_datasets = torch.load(args.data_file_path)

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

            if not file_path.exists():
                print(f"Skipping {m_n} {m_t}: model file not found at {file_path}")
                continue

            model = torch.load(file_path)
            model.to(device)

            res = test_multi_instance_model(model, device, dataloader)

            res_df = pd.DataFrame(res)
            res_df["model_name"] = m_n
            res_df["model_type"] = m_t
            print(res_df)

            result_dfs.append(res_df)

    if not result_dfs:
        print("No model results found. Run training first.")
    else:
        result_dfs = pd.concat(result_dfs)
        result_dfs = result_dfs.reset_index().rename(columns={"index": "k"})
        result_dfs["k"] = result_dfs["k"] + 1

        sns.lineplot(
            data=result_dfs,
            x="k",
            y="precision_at",
            hue="model_name",
            style="model_type",
        )
        print(result_dfs)
