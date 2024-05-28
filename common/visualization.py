import os
import shutil
import subprocess
import re
from subprocess import CompletedProcess
from typing import Union
from pandas import DataFrame
from pathlib import Path
import common.util as util
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import common.star_api.design_manager as dm


def parallel_plot(
        n_x: int,
        y: npt.ArrayLike) -> None:
    x = np.array([i for i in range(n_x)])

    fig, ax = plt.subplots(1, n_x - 1, sharey='none')
    for i in range(n_x - 1):
        for y_data in y:
            ax[i].plot(x, y_data[0], 'k-')
        ax[i].set_ylim([-2, 10])
        ax[i].set_xlim([x[i], x[i + 1]])

    plt.subplots_adjust(wspace=0)
    plt.show()


def plot_history(df: DataFrame,
                 y_col: str,
                 x_col: str = "Design#",
                 perf: str = "Performance",
                 best_design_hist: bool = True,
                 best_design: bool = True,
                 scatter: bool = True,
                 ax: plt.Axes = None,
                 min_best: bool = False,
                 infeasible_col: str = None,
                 infeasible_color: str = "r",
                 **kwargs) -> plt.Axes:
    if x_col not in df.columns:
        raise AttributeError(f"{x_col} not found in Dataframe. Dataframe supplied must contain a {x_col} column.")

    plot_infeasible = infeasible_col is not None
    if plot_infeasible:
        if infeasible_col not in df.columns:
            raise AttributeError(f"{infeasible_col} not found in Dataframe. Dataframe supplied must contain"
                                 f" {infeasible_col} column.")

    if perf not in df.columns:
        perf = y_col

    if perf == y_col:
        perf_cols = [x_col, y_col]
    else:
        perf_cols = [x_col, y_col, perf]

    if not ax:
        fig, ax = plt.subplots()

    if plot_infeasible:
        infeasible = [True if "INFEASIBLE" in val else False for val in df[infeasible_col]]
        feasible = [not val for val in infeasible]
        df_feasible = df[feasible]
        df_infeasible = df[infeasible]
        x_data = df_feasible[x_col].to_numpy()
        y_data = df_feasible[y_col].to_numpy()
        x_data_2 = df_infeasible[x_col].to_numpy()
        y_data_2 = df_infeasible[y_col].to_numpy()
    else:
        x_data = df[x_col].to_numpy()
        y_data = df[y_col].to_numpy()
        x_data_2 = None
        y_data_2 = None

    hist_src = df[perf_cols].to_numpy()
    hist_data = hist_src[0, :].reshape((1, len(perf_cols)))

    best = hist_src[0, -1]

    for row in hist_src[1:, ]:
        current = row[-1]
        if min_best:
            if current < best:
                best = current
                hist_data = np.append(hist_data, row.reshape((1, len(perf_cols))), axis=0)
        else:
            if current > best:
                best = current
                hist_data = np.append(hist_data, row.reshape((1, len(perf_cols))), axis=0)

    if scatter:
        if x_data_2 is not None and y_data_2 is not None:
            ax.plot(x_data, y_data, "o", label="Feasible", **kwargs)
            kwargs2 = kwargs.copy()
            kwargs2["color"] = infeasible_color
            ax.plot(x_data_2, y_data_2, "o", label="Infeasible", **kwargs2)
            ax.legend(loc="upper right")
        else:
            ax.plot(x_data, y_data, "o", **kwargs)
    if best_design_hist:
        xx = hist_data[:, 0]
        yy = hist_data[:, 1]
        ax.step(xx, yy, where="post", **kwargs)
        if best_design:
            if "markersize" in kwargs:
                old = kwargs["markersize"]
                kwargs["markersize"] = old + 6
            else:
                kwargs["markersize"] = 10
            ax.plot(hist_data[-1, 0], hist_data[-1, 1], "*", **kwargs)

    return ax


def animate_history(output_dir: Union[str, Path],
                    df: DataFrame,
                    y_col: str,
                    x_col: str = "Design#",
                    perf: str = "Performance",
                    best_design_hist: bool = True,
                    scatter: bool = True,
                    min_best: bool = False,
                    highlight: bool = False,
                    render: bool = False,
                    sort_by_y_col: bool = False,
                    infeasible_col: str = None,
                    infeasible_color: str = "r",
                    template: plt.Axes = None,
                    **kwargs):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir()
    else:
        output_dir.mkdir()

    df = util.filter_nan(df, y_col)

    if sort_by_y_col:
        df = df.sort_values(y_col)

    n_designs = len(df)
    n_digits = len(str(n_designs))
    fmt = f"0{n_digits}"

    fs = (8.0, 6.0)
    if template is not None:
        fs = template.figure.get_size_inches()
    fig, ax = plt.subplots(figsize=fs)

    kwargs2 = kwargs.copy()
    if "markersize" in kwargs:
        ms = kwargs["markersize"]
    else:
        ms = 6
    kwargs2["markersize"] = ms
    kwargs2["marker"] = "o"

    for i in range(n_designs):
        img_name = f"img_{i:0{fmt}d}.png"
        img_path = Path.joinpath(output_dir, img_name)
        temp = df.head(i + 1)
        if highlight:
            ax = plot_history(df,
                              y_col=y_col,
                              x_col=x_col,
                              perf=perf,
                              best_design_hist=best_design_hist,
                              scatter=scatter,
                              min_best=min_best,
                              infeasible_col=infeasible_col,
                              infeasible_color=infeasible_color,
                              ax=ax, **kwargs)
            ax = plot_history(temp.tail(1),
                              y_col=y_col,
                              x_col=x_col,
                              perf=perf,
                              best_design_hist=False,
                              scatter=True,
                              min_best=min_best,
                              infeasible_col=infeasible_col,
                              infeasible_color=infeasible_color,
                              ax=ax, **kwargs2)
        else:
            ax = plot_history(temp,
                              y_col=y_col,
                              x_col=x_col,
                              perf=perf,
                              best_design_hist=best_design_hist,
                              scatter=scatter,
                              min_best=min_best,
                              infeasible_col=infeasible_col,
                              infeasible_color=infeasible_color,
                              ax=ax, **kwargs)
        if template:
            x_label = template.get_xlabel()
            y_label = template.get_ylabel()
            title = template.get_title()
            y_l, y_h = template.get_ylim()
            x_l, x_h = template.get_xlim()
            x_ticks = template.get_xticks()
            y_ticks = template.get_yticks()
            ax.set_ylim(y_l, y_h)
            ax.set_xlim(x_l, x_h)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
        plt.savefig(img_path, dpi=300, format="png")
        ax.clear()

    if render:
        process = animate(output_dir, image_base_name="img_", int_num_chars=n_digits, frame_rate=24)
        print(process.stdout)
        print(process.stderr)


def rename_images_for_animation(
        parent: Union[str, Path],
        img_tag: str = 'image',
        int_num_characters: int = 5,
        dm_study: bool = True) -> None:
    """
    Rename all images in the parent directory so that they include a formatted string that indicates the order in which
    they should be animated. The method is designed to work with images that were saved out by a STAR-CCM+ simulation
    with a format <scene_name>_<Base Filename>_<counter>.<ext>.  The img_tag is everything between <scene_name> and
    <counter>.  If more than one image was saved to the same directory this method will apply names appropriately as
    long as all images were saved with the same <Base Filename> which can be set in the Attributes -> Update -> Base
    Filename property of any Scene in a simulation.
    :param parent: str, required
        Path to the directory containing the images.
    :param img_tag: str, optional
        The tag used by STAR-CCM+ when naming an image. (Default is 'image')
    :param int_num_characters: int, optional
        Specifies the number of characters in the formatted integer for the ordered image names e.g. for 5 1 -> 00001.
        (Default is 5)
    :param dm_study: bool, optional
        The images were prepared from a design manager study and not directly from a simulation. (Default is True)
    :return None:
    """
    if isinstance(parent, str):
        parent = Path(parent)
    if not parent.exists():
        print(f'The directory {parent} cannot be found. Unable to continue.')
        return

    fmt = f'0{int_num_characters}d'
    file_types = {'.png', '.jpg', '.tif', '.pnm', '.bmp', '.ps'}
    img_files = [f for f in parent.iterdir() if f.is_file() and f.suffix in file_types]
    if dm_study:
        base_names = {img_tag}
    else:
        base_names = set([f.name.split(f'_{img_tag}_')[0] for f in img_files])

    def extract_tag(x):
        return x.stem.replace(base_name, '').replace('_', '').replace(img_tag, '')

    def sort_key(x):
        return float(extract_tag(x))

    for base_name in base_names:
        test = base_name if base_name != '' else img_tag
        images_to_rename = [f for f in img_files if f.name.startswith(test)]
        sorted_files = sorted(images_to_rename, key=sort_key)
        for i in range(len(sorted_files)):
            f = sorted_files[i]
            src = parent.joinpath(parent, f)
            s = extract_tag(f)
            t = f'{i + 1:{fmt}}'
            dst_name = f.name.replace(s, t)
            dest = parent.joinpath(dst_name)
            shutil.move(src, dest)


def prepare_dirs_for_animation(
        parent: Union[str, Path],
        scene_suffix: str,
        clean_first: bool = False,
        int_num_chars: int = 5) -> None:
    """
    Move and rename images in a parent directory in preparation for creating an animation with ffmpeg. The parent
    directory can contain images for multiple different animations. Each set of images will be placed in their own
    directory and renamed with an integer ordering formatted with the specified number of characters e.g. for 5
    characters 1 -> 00001.
    :param parent:
        directory containing images that are to be animated
    :type parent: str, required
    :param scene_suffix:
        suffix added by STAR-CCM+ prior to export of each image
    :type scene_suffix: str, required
    :param clean_first:
        if directory for image animation exists then delete any files prior to writing images from parent directory.
        (Default is False)
    :type clean_first: bool, optional
    :param int_num_chars:
        Specifies the number of characters in the formatted integer for the ordered image names e.g. for 5 1 -> 00001.
        (Default is 5)
    :type int_num_chars: int, optional
    :return: None
    """

    if isinstance(parent, str):
        parent = Path(parent)
    if not parent.is_absolute():
        parent = Path.joinpath(Path(os.getcwd()), parent)
    if not parent.exists():
        print(f'The directory {parent} cannot be found. Unable to continue.')
        return

    file_types = ['.png', '.jpg', '.tif', '.pnm', '.bmp', '.ps']
    img_files = [f for f in parent.iterdir() if f.is_file() and f.suffix in file_types]

    base_names = set([f.name.split(f'_{scene_suffix}_')[0] for f in img_files])

    for base_name in base_names:
        base_dir = parent.joinpath(base_name)
        if base_dir.exists():
            if not base_dir.is_dir():
                print(f'Path {base_dir} exists and references a file. Unable to proceed.')
                return
            else:
                if clean_first:
                    files = [f for f in base_dir.iterdir()]
                    if len(files) > 0:
                        for f in files:
                            os.remove(f)
        else:
            base_dir.mkdir()

    for base_name in base_names:
        dest_dir = parent.joinpath(base_name)
        files_to_copy = [f for f in img_files if f.name.startswith(base_name)]
        for f in files_to_copy:
            shutil.move(f, dest_dir)
        rename_images_for_animation(dest_dir, scene_suffix, int_num_characters=int_num_chars, dm_study=False)


def prepare_dirs_for_animation_dm(
        study_dir: Union[str, Path],
        clean_first: bool = True,
        image_names: [str] = None) -> None:
    """
    Collect image files from a design manager study into a new folder "images" inside of the specified study directory.
    For each image detected a subdirectory is created and a copy of each image from the design directories is copied to
    this directory named design_<image number>.<ext>.  The image number is an integer formatted with 5 characters (1 ->
    00001).

    :param study_dir: str, required
        Path to the study archive directory.
    :param clean_first: bool, optional
        If the images directory exists delete any of its contents prior to copying images from the study directory.
        (Default is True)
    :param image_names: list(str), optional
        Specify a list of str prefixes to limit which design images are copied to the images directory.  (Default is
        None which means all images found in the directory will be copied)
    :return: None
    """
    if isinstance(study_dir, str):
        study_dir = Path(study_dir)
    if not study_dir.is_absolute():
        cwd = os.getcwd()
        print(f'{study_dir} is relative, prepending current working director: {cwd}')
        study_dir = Path.joinpath(Path(cwd), study_dir)
    if not study_dir.exists():
        print(f'Study directory: {study_dir} does not exist.')
        raise FileNotFoundError(f'Study directory: {study_dir} does not exist.')
    if not study_dir.is_dir():
        print(f'Study directory {study_dir} is not a directory.')
        raise NotADirectoryError(f'Study directory {study_dir} is not a directory.')
    file_prefix = 'img_'
    if image_names is None:
        image_names = []

    img_dir = study_dir.joinpath('images')
    if img_dir.exists():
        if clean_first:
            for f in img_dir.iterdir():
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    os.remove(f)
    else:
        img_dir.mkdir()

    filetypes = {'.png', '.jpg', '.tif', '.pnm', '.bmp', '.ps'}
    design_dirs = [f for f in study_dir.iterdir() if f.is_dir()]
    design_dirs = [d for d in design_dirs if d.name.startswith('Design_')]
    design_numbers = set()

    def get_design_number(directory_name: Path) -> int:
        num = re.search(r'\d+', directory_name.name).group()
        return int(num)

    for design in design_dirs:
        design_number = get_design_number(design)
        if design_number in design_numbers:
            print(f'Duplicate directories found for Design {design_number}.  If this is not desired have design manager'
                  f' clean Project Artifacts.')
        design_numbers.add(design_number)

    for design in design_dirs:
        design_number = get_design_number(design)
        design_number = f'{design_number:05d}'
        design_dir = study_dir.joinpath(design.name)
        files = []
        if len(image_names) > 0:
            for image in image_names:
                if not design_dir.joinpath(image).exists():
                    temp = [f for f in design_dir.iterdir() if f.name.startswith(image)]
                    files.extend([f for f in temp if f.suffix in filetypes])
                else:
                    files.append(design_dir.joinpath(image))
        else:
            files = [f for f in design_dir.iterdir() if f.suffix in filetypes]
        for f in files:
            src = f
            dest_raw_dir_name = f.stem
            dest_processed_dir_name = dest_raw_dir_name.replace(' ', '_')
            dest_dir = Path.joinpath(img_dir, dest_processed_dir_name)
            if not dest_dir.exists():
                dest_dir.mkdir()
            f = file_prefix + '_' + design_number + f.suffix
            dest = dest_dir.joinpath(f)
            shutil.copy(src, dest)

    images = [f for f in img_dir.iterdir() if f.is_dir()]
    for image in images:
        idx = 1
        directory = img_dir.joinpath(image)
        for image_i in directory.iterdir():
            ordered_num = f'{idx:05d}'
            src = directory.joinpath(image_i)
            dest = directory.joinpath(f'{file_prefix}{ordered_num}{image_i.suffix}')
            os.rename(src, dest)
            idx += 1


def prep_dirs_for_animation_HEEDS(study_dir: Path,
                                  clean_first: bool = True,
                                  analysis_dirs: list[str] = None,
                                  image_names: list[str] = None) -> None:
    if isinstance(study_dir, str):
        study_dir = Path(study_dir)
    if not study_dir.is_absolute():
        cwd = os.getcwd()
        print(f'{study_dir} is relative, prepending current working director: {cwd}')
        study_dir = Path.joinpath(Path(cwd), study_dir)
    if not study_dir.exists():
        print(f'Study directory: {study_dir} does not exist.')
        raise FileNotFoundError(f'Study directory: {study_dir} does not exist.')
    if not study_dir.is_dir():
        print(f'Study directory {study_dir} is not a directory.')
        raise NotADirectoryError(f'Study directory {study_dir} is not a directory.')
    if not study_dir.name == "POST_0":
        print(f"Study directory {study_dir} is not a HEEDS POST_0 directory")
        raise Exception(f"Study directory must be a HEEDS POST_0 directory")

    img_dir = study_dir.joinpath('images')
    if img_dir.exists():
        if clean_first:
            for f in img_dir.iterdir():
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    os.remove(f)
    else:
        img_dir.mkdir()

    if analysis_dirs is None:
        analysis_dirs = []
    if image_names is None:
        image_names = []

    file_prefix = 'img_'
    filetypes = {'.png', '.jpg', '.tif', '.pnm', '.bmp', '.ps'}
    design_dirs = [f for f in study_dir.iterdir() if f.is_dir()]
    design_dirs = [d for d in design_dirs if d.name.startswith('Design')]

    def get_design_number(directory_name: Path) -> int:
        num = re.search(r'\d+', directory_name.name).group()
        return int(num)

    def get_design_images_from_analysis_dir(path: Path) -> list[Path]:
        if len(image_names) > 0:
            images = [f for f in path.iterdir() if f.stem in image_names and f.suffix in filetypes]
        else:
            images = [f for f in path.iterdir() if f.suffix in filetypes]
        return images

    def get_design_analysis_dirs_from_design_dir(path: Path) -> list[Path]:
        if len(analysis_dirs) > 0:
            dirs = [f for f in path.iterdir() if f.is_dir() and f.name in analysis_dirs]
        else:
            dirs = [f for f in path.iterdir() if f.is_dir()]
        return dirs

    for design in design_dirs:
        design_number = get_design_number(design)
        design_number = f'{design_number:05d}'
        design_dir = study_dir.joinpath(design.name)
        design_analysis_dirs = get_design_analysis_dirs_from_design_dir(design_dir)
        for analysis_dir_path in design_analysis_dirs:
            analysis_tag = analysis_dir_path.stem
            files = get_design_images_from_analysis_dir(analysis_dir_path)
            for f in files:
                src = f
                dest_raw_dir_name = f"{analysis_tag}_{f.stem}"
                dest_processed_dir_name = dest_raw_dir_name.replace(' ', '_')
                dest_dir = Path.joinpath(img_dir, dest_processed_dir_name)
                if not dest_dir.exists():
                    dest_dir.mkdir()
                f = file_prefix + '_' + design_number + f.suffix
                dest = dest_dir.joinpath(f)
                shutil.copy(src, dest)


def animate(img_dir: Union[str, Path],
            image_base_name: str,
            int_num_chars: int = 5,
            movie_name: str = 'animation.mp4',
            frame_rate: int = 24,
            ffmpeg_exe: str = r'E:\Program Files\ffmpeg\bin\ffmpeg.exe',
            rename_files: bool = True) -> Union[CompletedProcess, None]:
    """
    Create an mp4 from a directory of images using ffmpeg, which is required to be installed for this function to work.
    The images should be named with a common prefix and a formatted string that indicate the order in which they are to
    be animated. The number of characters in the formatted string can be set using the int_num_chars parameter.
    If the images are not already name appropriately then this can be accomplished with the
    common.visualization.name_images_for_animation function.

    :param img_dir: str, required
        Path to a folder containing images to be used for the animation.
    :param image_base_name: str, required
        Image prefix e.g. scalar_mach-00001.png -> scalar_mach-
    :param int_num_chars: int, optional
        number of characters used to format the integer values for the ordered images e.g. for 5 1 -> 00001. (Default is
        5)
    :param movie_name: str, optional
        Name of the movie created by this function.  (Default is animation.mp4)
    :param frame_rate: int, optional
        Frame rate for the animation. (Default is 24)
    :param ffmpeg_exe: str, optional
        Path to ffmpeg.exe. (Default = r'E:\Program Files\ffmpeg\bin\ffmpeg.exe')
    :param rename_files: bool, optional
        Call rename_files_for_animation function prior to running ffmpeg. (Default = True)
    :return:
        CompletedProcess object that results from the external call to ffmpeg or None if one of the checks fails in
        which case an error is printed to stout.
    """
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f'{img_dir} does not exist. Unable to continue')
    if not img_dir.is_dir():
        raise NotADirectoryError(f'{img_dir} is not a directory. Unable to continue')

    output_movie = img_dir.joinpath(movie_name)
    if output_movie.exists():
        os.remove(output_movie)

    if rename_files:
        movie_name = img_dir.joinpath(movie_name)
        img_dir = img_dir.joinpath("temp")
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir()
        for f in img_dir.parent.iterdir():
            if not f.is_dir():
                src = f
                dest = img_dir.joinpath(f.name)
                shutil.copy(src, dest)
        rename_images_for_animation(img_dir, img_tag=image_base_name, int_num_characters=int_num_chars, dm_study=True)

    fmt = f"%0{int_num_chars}d"
    images = [f for f in img_dir.iterdir() if f.name.startswith(image_base_name)]
    if len(images) < 2:
        print(f"No images named {image_base_name} found in {img_dir}.")
        return None
    ext = images[0].suffix

    args = [ffmpeg_exe,
            "-r",
            f"{frame_rate}",
            "-f",
            "image2",
            "-i",
            f"{image_base_name}{fmt}{ext}",
            "-vcodec",
            "libx264",
            "-crf",
            f"{frame_rate}",
            movie_name]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=img_dir
    )

    if rename_files:
        shutil.rmtree(img_dir)

    return result


if __name__ == "__main__":
    file = r"D:\Workdir\projects\2023\transonic_airfoil\svd\svd_proj.json"
    p = dm.DesignManagerProject.from_json(file)
    study = p.get_study("Sherpa1")
    data = study.get_all_designs().data
    axes = plot_history(data, "L_over_D", color="g", markersize=4)
    axes.set_ylim(0, 115)
    axes.set_xlabel("Design Number")
    axes.set_ylabel("L/D")
    axes.set_title("Sherpa1 L/D History")
    plt.tight_layout()
    image_dir = Path(file).parent.joinpath("img")
    if image_dir.exists():
        for f_i in image_dir.iterdir():
            os.remove(f_i)
        os.removedirs(image_dir)
    animate_history(image_dir, data.head(100), sort_by_y_col=True, y_col="L_over_D", best_design_hist=False,
                    template=axes, color="g", markersize=4)
