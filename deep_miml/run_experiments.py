import subprocess
import argparse

parser = argparse.ArgumentParser(description='Create and Save Multi-Instance Multi-Label Dataset')


model_names = [ "resnet18", "resnet34", "resnet50"] #"alexnet",

model_types = ["attention", "avg"]

data_path  =  "data/miml_data20000.pt"

save_folder = "results/"

batch_size = 32



result = {}


for m_n in model_names: 

	for m_t in model_types:

		print(f"Training {m_n} {m_t}")

		subprocess.check_call(["python",
								"-m", "deep_miml.train",
								"--model_name", m_n,
								"--model_type", m_t,
								"--use_pretrained", "True",
								"--data_file_path", data_path, 
								"--save_folder", save_folder,
								"--batch_size" , str(batch_size)])



