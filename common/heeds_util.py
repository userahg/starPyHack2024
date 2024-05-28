import re
import shutil
import sys
import pandas as pd
import common.util as util
from pathlib import Path
from typing import Union
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas.plotting import parallel_coordinates


def combine_study_res_files(study_dirs: [str], success_only: bool = False) -> Path:
    df_master = pd.DataFrame()
    for study_dir in study_dirs:
        p = _validate_heeds_study_dir(study_dir)
        r = get_res_as_df(p, success_only)
        df_master = pd.concat([df_master, r], ignore_index=True)
    output_path = Path.joinpath(study_dirs[0], "combined.csv")
    df_master.to_csv(output_path)
    return output_path


def combine_post_0(src_study_dir: Union[str, Path],
                   dest_study_dir: Union[str, Path],
                   src_number_column: str = "Design Name",
                   dest_number_column: str = "Evaluation  #",
                   post_from_src_column: str = "Design Source",
                   post_from_src_re: str = r"^.*\(UD\(.*\)\)$",
                   clean: bool = False,
                   extensions=None):
    if extensions is None:
        extensions = [".png", ".csv"]
    if isinstance(src_study_dir, str):
        src_study_dir = Path(src_study_dir)
    if isinstance(dest_study_dir, str):
        dest_study_dir = Path(dest_study_dir)
    if not src_study_dir.exists():
        raise FileNotFoundError(f"{src_study_dir} does not exist. Parameter src_study_dir must exist.")
    if not dest_study_dir.exists():
        raise FileNotFoundError(f"{dest_study_dir} does not exist. Parameter dest_study_dir must exist.")
    if not src_study_dir.is_dir():
        raise NotADirectoryError(f"{src_study_dir} is not a directory. Parameter src_study_dir must be a directory.")
    if not dest_study_dir.is_dir():
        raise NotADirectoryError(f"{dest_study_dir} is not a directory. Parameter dest_study_dir must be a directory")

    df = get_res_as_df(dest_study_dir)
    df2 = df[[post_from_src_column, src_number_column, dest_number_column]]
    mask = [True if re.match(post_from_src_re, val) else False for val in df2[post_from_src_column]]
    df3 = df2[mask]

    def copy_files(src_num, dest_num):
        src_design_dir = src_study_dir.joinpath("POST_0").joinpath(f"Design{src_num}")
        dest_design_dir = dest_study_dir.joinpath("POST_0").joinpath(f"Design{dest_num}")
        if dest_design_dir.exists():
            print(f"Design dir {dest_design_dir} already exists.")
            if clean:
                print(f"to delete: {dest_design_dir}")
                shutil.rmtree(dest_design_dir)
                dest_design_dir.mkdir()
        else:
            dest_design_dir.mkdir()
        analyses = [f for f in src_design_dir.iterdir() if f.is_dir()]
        for analysis in analyses:
            dest_analysis_dir = dest_design_dir.joinpath(analysis.name)
            if not dest_analysis_dir.exists():
                dest_analysis_dir.mkdir()
            to_copy = [f for f in analysis.iterdir() if f.suffix in extensions]
            for src_file in to_copy:
                dest_file = dest_analysis_dir.joinpath(src_file.name)
                shutil.copy(src_file, dest_file)

    for src_num_str, dest_num_str in zip(df3[src_number_column], df3[dest_number_column]):
        num_re = re.compile(r"\d+")
        src_num_str = num_re.search(src_num_str).group()
        dest_num_str = num_re.search(str(dest_num_str)).group()
        copy_files(src_num_str, dest_num_str)


def plot_history(study_dir: str, y_col: str, scatter_data: bool = False, axes: plt.Axes = None, **kwargs) -> plt.Axes:
    x_col = "Evaluation  #"
    if axes is None:
        fig, axes = plt.subplots()
    study_path = _validate_heeds_study_dir(study_dir)
    study_results = get_res_as_df(study_path, success_only=True)
    study_gph = get_gph_as_df(study_path, best_design_hist=True)
    if scatter_data:
        axes.plot(study_results[x_col].to_numpy(), study_results[y_col].to_numpy(), "o")
    axes.step(study_gph[x_col], study_gph[y_col], where="post", **kwargs)
    return axes


def plot_parallel(study_dir: str, col_re: str, axes: plt.Axes = None, **kwargs) -> plt.Axes:
    pattern = re.compile(col_re)
    if axes is None:
        fig, axes = plt.subplots()
    study_path = _validate_heeds_study_dir(study_dir)
    df = util.clean_csv_for_df(study_path.parent.joinpath("parallel.csv"), sort_col="")
    col_names = []
    for col in df.columns:
        if pattern.match(col):
            col_names += [col]
    axes = parallel_coordinates(df, "Study", cols=col_names, **kwargs)
    return axes


def get_res_as_df(study_dir: Path, success_only: bool = False) -> DataFrame:
    res_path = study_dir.joinpath("HEEDS0.res")
    df = util.clean_csv_for_df(res_path)
    if success_only:
        flag_col = 5
        include = ["FEASIBLE", "INFEASIBLE"]
        flag_col_name = df.columns[flag_col]
        success_mask = [True if val in include else False for val in df[flag_col_name]]
        df = df[success_mask]

    return df


def get_gph_as_df(study_dir: Path, best_design_hist: bool = False) -> DataFrame:
    res_path = study_dir.joinpath("HEEDS0.gph")
    df = util.clean_csv_for_df(res_path)
    if best_design_hist:
        perf = -sys.float_info.max
        mask = []
        for val in df["performance"]:
            if val > perf:
                perf = val
                mask += [True]
            else:
                mask += [False]
        df = df[mask]

    return df


def _validate_heeds_study_dir(study_dir: str) -> Path:
    res = "HEEDS0.res"
    p = Path(study_dir)
    if not p.exists():
        raise FileExistsError(f"{study_dir} does not exist.")
    if not p.is_dir():
        raise NotADirectoryError(f"{study_dir} is not a directory.")

    files = [x for x in p.iterdir() if x.is_file()]

    res_exists = False

    for f in files:
        if f.name == res:
            res_exists = True
            break

    if not res_exists:
        raise Exception("Study dir must contain HEEDS0.res file")

    return p


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPA", y_col="L_over_D", axes=ax,
                      label="SHERPA 1")
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretellComp_SHERPA2", y_col="L_over_D", axes=ax,
                      label="SHERPA 2")
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPAplus", y_col="L_over_D", axes=ax,
                      linestyle=":", label="SHERPA+ 1")
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPAplus2", y_col="L_over_D", axes=ax,
                      linestyle=":", label="SHERPA+ 2")
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPAplus3", y_col="L_over_D", axes=ax,
                      linestyle=":", label="SHERPA+ 3")
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPAplus4", y_col="L_over_D", axes=ax,
                      linestyle=":", label="SHERPA+ 4")
    ax = plot_history(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPAplus5", y_col="L_over_D", axes=ax,
                      linestyle=":", label="SHERPA+ 5")
    ax.set_xlabel("Design Number")
    ax.set_ylabel("L/D")
    ax.legend(loc="lower right")
    ax.set_ylim(75.0, 200.0)
    ax.set_yticks([75.0, 100.0, 125.0, 150.0, 175.0, 200.0])
    plt.tight_layout()
    plt.show()

    ax = plot_parallel(r"C:\Workdir\tests\heedsforetell\SVDAirfoil\foretell_SHERPA", r"^P_TE_|P_W_\d{1,2}$", [494])
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
