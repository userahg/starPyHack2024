import common.star_api.design_manager as dm
import common.visualization as viz
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

version = dm.STARCCMInstall(r"E:\Siemens\STARCCM\starpy\STAR-CCM+19.04.007-2-ga404231\star\bin\starccm+.bat")
work_dir = Path(r"E:\OneDrive - Siemens AG\mdx\hackathon\2024\starPy\svd")
name = "svd_proj_test"


def plot_statistical_histories(proj: dm.DesignManagerProject):
    only_opt = [s for s in proj if "Sherpa_" in s.name]
    max_designs = -1
    for s in only_opt:
        current_number = len(s.get_all_designs().data)
        if current_number > max_designs:
            max_designs = current_number
    d = {"Design#": np.arange(1, max_designs + 1)}
    for study in only_opt:
        data = study.get_design_set("All Best").generate_history(col_name="L_over_D", max_designs=max_designs)
        d[study.name] = data[:, 1]

    data = pd.DataFrame(d)
    data2 = data[data.columns[1:]]
    data["Min"] = data2.min(axis=1)
    data["Average"] = data2.mean(axis=1)
    data["Median"] = data2.median(axis=1)
    data["Max"] = data2.max(axis=1)
    data["Q1"] = data2.quantile(0.25, axis=1)
    data["Q3"] = data2.quantile(0.75, axis=1)
    fig, ax = plt.subplots()
    x = data["Design#"].to_numpy()
    med = data["Median"].to_numpy()
    q1 = data["Q1"].to_numpy()
    q3 = data["Q3"].to_numpy()
    ax.plot(x, med, color="k", linestyle="-", linewidth=1.5, label="Median")
    ax.fill_between(x, q1, q3, alpha=0.25, color="k", label="IQR")
    ax.set_title("L/D Statistical Performance - SHERPA")
    ax.set_ylabel("L/D")
    ax.set_xlabel("Design Number")
    ax.set_xlim([0, max_designs])
    ax.set_ylim([75, 200])
    ax.set_yticks([v for v in range(75, 225, 25)])
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(work_dir.joinpath("svd_statistical.png"), dpi=300)


if __name__ == "__main__":
    dmprj = dm.DesignManagerProject.get_proj(work_dir=work_dir, dmprj=name, version=version)
    plot_statistical_histories(dmprj)
