import numpy as np
import common.star_api.design_manager as dm
from SALib.analyze import sobol
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def get_param_names(study: dm.Study):
    continuous = [p for p in study.parameters if p.type == dm.Parameter.Type.CONTINUOUS]
    param_names = [p.name for p in continuous]
    successful = study.get_design_set("Successful").data_frame()

    param_names_adjusted = []
    for p in param_names:
        found = False
        for c in successful.columns:
            regex = f"^{p}( \\(.*\\))?$"
            if re.match(regex, c):
                param_names_adjusted.append(c)
                found = True
                break
        if not found:
            print(f"Parameter name '{p}' not found in successful design columns. Using original name.")
            param_names_adjusted.append(p)
    return param_names_adjusted


def get_param_values(study: dm.Study):
    successful = study.get_design_set("Successful").data_frame()
    param_names = get_param_names(study)
    param_values = successful[param_names]
    return param_values


def get_responses(study: dm.Study, response_column: str):
    successful = study.get_design_set("Successful").data_frame()
    responses = successful[response_column].values
    return responses


def plot_histograms(param_values, param_names, responses, response_column):
    num_params = len(param_names)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(5, 2, wspace=0)
    axes = gs.subplots(sharey="row")
    # fig, axes = plt.subplots(5, 2, figsize=(14, 8), sharey=True)
    for i, param_name in enumerate(param_names):
        ax = axes.flatten()[i]
        sns.histplot(param_values[:, i], bins=30, kde=True, ax=ax)
        ax.set_title(f'Histogram of {param_name}')
    ax = axes.flatten()[num_params]
    sns.histplot(responses, bins=30, kde=True, ax=ax)
    ax.set_title(f'Histogram of {response_column}')
    plt.tight_layout()
    plt.show()


def plot_scatter_plots(param_values, param_names, responses, axes):
    x_label_prefixes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(param_values[:, i], responses, alpha=0.5)
        ax.set_xlabel(f"{x_label_prefixes[i]} = {param_name}")


def plot_sobol_indices(Si, param_values, param_names, responses):
    param_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    fig = plt.figure(layout="constrained", figsize=(15, 8))
    gs0 = gridspec.GridSpec(2, 2, figure=fig)
    gs00 = gs0[1, 0].subgridspec(4, 2, wspace=0)
    axes = gs00.subplots(sharey="row").flatten()
    ax1 = fig.add_subplot(gs0[0, 0])
    ax0 = fig.add_subplot(gs0[0, 1])
    ax3 = fig.add_subplot(gs0[1, 1])

    # First-order indices
    ax0.bar(param_names, Si['S1'])
    ax0.set_title('First-Order Sobol Indices')
    ax0.set_xlabel("Parameter")
    ax0.set_xticks([i for i in range(8)], param_labels)

    # Total indices
    ax1.bar(param_names, Si['ST'])
    ax1.set_title('Total Sobol Indices')
    ax1.set_xlabel("Parameter")
    ax1.set_xticks([i for i in range(8)], param_labels)

    sns.heatmap(Si['S2'], xticklabels=param_labels, yticklabels=param_labels, annot=True, fmt=".2f", cmap='viridis',
                ax=ax3)
    ax3.set_title('Second-order Sobol indices')
    ax3.set_xlabel("Parameter")
    ax3.set_ylabel("Parameter")

    plot_scatter_plots(param_values, param_names, responses, axes)
    plt.show()


def run_sobol_analysis(study: dm.Study, response_column: str):
    param_names = get_param_names(study)
    param_values = get_param_values(study).values
    responses = get_responses(study, response_column)

    D = len(param_names)  # Number of parameters
    N = len(responses)  # Number of samples
    calc_second_order = True  # Using first-order Sobol analysis

    if calc_second_order and N % (2 * D + 2) != 0:
        raise RuntimeError("Incorrect number of samples in model output file for second-order Sobol analysis.")
    elif not calc_second_order and N % (D + 2) != 0:
        raise RuntimeError("Incorrect number of samples in model output file for first-order Sobol analysis.")

    problem = {
        'num_vars': D,
        'names': param_names,
        'bounds': [[min(param_values[:, i]), max(param_values[:, i])] for i in range(D)]
    }

    assert not np.any(np.isnan(param_values)), "NaN values found in param_values"
    assert not np.any(np.isinf(param_values)), "Infinite values found in param_values"
    assert not np.any(np.isnan(responses)), "NaN values found in responses"
    assert not np.any(np.isinf(responses)), "Infinite values found in responses"

    Si = sobol.analyze(problem, responses, calc_second_order=calc_second_order, print_to_console=True)

    plot_sobol_indices(Si, param_values, param_names, responses)


if __name__ == "__main__":
    work_dir = Path(r"E:\OneDrive - Siemens AG\mdx\hackathon\2024\starPy\exhaust")
    name = "exhaust_test"
    version = dm.STARCCMInstall(r"E:\Siemens\STARCCM\starpy\STAR-CCM+19.04.007-2-ga404231\star\bin\starccm+.bat")
    study_name = "Single Opt - Min 1 Const"
    response_column = "Pressure Drop MA (Pa)"

    project = dm.DesignManagerProject.get_proj(work_dir=work_dir, dmprj=name, version=version)

    study = project.get_study(study_name)

    run_sobol_analysis(study, response_column)
