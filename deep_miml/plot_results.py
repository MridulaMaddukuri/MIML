import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# matplotlib.use("nbagg")  # matplotlib.use('TkAgg')


dfs = []
for m_n in ["resnet18", "resnet34"]:
    for m_t in ["avg", "attention"]:
        file_path = "results/results_" + m_n + "_" + m_t + "_miml_data20000.json"
        with open(file_path) as f:
            data = json.load(f)

            data_df = pd.DataFrame(data)

            print(data_df)
            data_df["model"] = m_n + "_" + m_t
            dfs.append(data_df)


dfs = pd.concat(dfs)
print(dfs.columns, "\n", dfs)

sns.lineplot(data=dfs, x=dfs.index, y="precision_at", hue="model", style="model")
plt.savefig("results/example_plot.pdf")
plt.show()
