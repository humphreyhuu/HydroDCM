import matplotlib.pyplot as plt
import numpy as np

models = ["DANN", "MLDG", "CondAdv", "IRM", "HydroDCM"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

data = {
    "MCR": {
        "Day 3": [(80.30,1.5), (80.68,1.4), (80.96,1.4), (80.20,1.5), (83.12,1.7)],
        "Day 4": [(76.60,2.0), (77.04,2.2), (77.33,2.0), (76.45,2.0), (79.87,2.3)],
        "Day 5": [(73.20,2.1), (74.23,1.5), (74.42,2.0), (73.05,2.0), (76.24,2.1)],
        "Day 6": [(68.90,2.0), (71.97,2.3), (72.14,1.8), (68.70,1.8), (73.86,1.9)],
        "Day 7": [(66.30,1.9), (69.82,1.9), (70.08,2.1), (66.05,2.0), (72.71,2.4)]
    },
    "JVR": {
        "Day 3": [(84.11,1.2), (84.03,1.5), (84.26,1.9), (83.75,1.9), (87.90,1.9)],
        "Day 4": [(80.04,2.0), (79.93,2.4), (77.33,2.3), (79.25,2.3), (82.17,2.2)],
        "Day 5": [(74.61,1.6), (74.23,1.7), (74.42,1.7), (73.10,2.3), (78.75,1.8)],
        "Day 6": [(71.50,2.0), (71.98,2.1), (72.14,1.8), (70.30,2.0), (74.14,2.4)],
        "Day 7": [(66.97,2.1), (69.47,2.3), (69.62,2.4), (65.75,2.4), (70.9,2.5)]
    },
    "MCP": {
        "Day 3": [(82.60,1.7), (86.41,1.9), (85.63,1.9), (81.70,2.1), (87.06,1.5)],
        "Day 4": [(81.85,2.2), (84.63,2.5), (84.74,2.0), (82.40,1.8), (84.75,2.3)],
        "Day 5": [(79.55,2.0), (82.27,2.0), (82.36,2.1), (79.35,1.8), (82.79,2.0)],
        "Day 6": [(75.35,1.8), (78.24,1.8), (78.33,1.9), (75.15,1.8), (80.73,1.8)],
        "Day 7": [(73.56,2.3), (74.98,2.2), (75.19,2.3), (72.80,2.3), (78.24,2.4)]
    }
}

fig, axes = plt.subplots(3, 5, figsize=(15, 6), sharex=True)

for i, (reservoir, res_data) in enumerate(data.items()):
    for j, (day, values) in enumerate(res_data.items()):
        ax = axes[i, j]
        means = [v[0] for v in values]
        stds = [v[1] for v in values]

        ax.bar(models, means, yerr=stds, color=colors, capsize=3)
        ax.set_ylim(min(means) - 5, max(means) + 5)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45)

        if i == 0:
            ax.set_title(day, fontsize=11)
        if j == 0:
            ax.set_ylabel(reservoir, fontsize=11)

plt.tight_layout()
plt.savefig("Reservoir_NSE_Day3-7.pdf", bbox_inches="tight")
plt.show()