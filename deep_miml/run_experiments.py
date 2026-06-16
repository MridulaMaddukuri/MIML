import subprocess

MODEL_NAMES = ["resnet18", "resnet34", "resnet50"]
MODEL_TYPES = ["attention", "avg"]
DATA_PATH = "data/miml_data20000.pt"
SAVE_FOLDER = "results/"
BATCH_SIZE = 32

if __name__ == "__main__":
    for m_n in MODEL_NAMES:
        for m_t in MODEL_TYPES:
            print(f"Training {m_n} {m_t}")

            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "deep_miml.train",
                    "--model_name",
                    m_n,
                    "--model_type",
                    m_t,
                    "--use_pretrained",
                    "--data_file_path",
                    DATA_PATH,
                    "--save_folder",
                    SAVE_FOLDER,
                    "--batch_size",
                    str(BATCH_SIZE),
                ]
            )
