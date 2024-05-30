import os
import re
import csv
import math
import subprocess
import numpy as np
import numpy.typing as npt
from typing import Union, List
from shutil import copy2
from pathlib import Path
from subprocess import CompletedProcess
import pandas as pd
from common.star_versions import get_star_install
from common.star_versions import STARCCMInstall
from common.local_settings import default_star_ccm_plus
from common.local_settings import star_ccm_plus_install_dir
from common.local_settings import star_ccm_plus_bkup_install_dir
from common.local_settings import zip_exe


def launch_server(file: Union[str, Path], starccm: STARCCMInstall, print_output: bool = False) -> int:
    if not starccm.starccm().exists():
        raise FileNotFoundError(f"STARCCM+ {starccm} not found on disk.")
    elif starccm is None:
        starccm = STARCCMInstall(Path(default_star_ccm_plus).joinpath("starccm+.bat"))
    if isinstance(file, str):
        file = Path(file)
    _validate_starccm_file(file)

    sim = True
    if "dmprj" in file.suffix:
        sim = False
        raise NotImplementedError(f"launch_server does not currently support design manager projects")

    if sim:
        comm = [f"{starccm.starccm()}", "-server", file]
    else:
        comm = [f"{starccm.starlaunch()}",
                "--command", f"\"{starccm.starccm()} -server -dmproject {file}\"",
                "--slots", "0"]

    process = subprocess.Popen(comm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)

    regex = r"Server::start.*\d{5}"
    while True:
        port = None
        stdout_line = process.stdout.readline().decode("utf-8")
        stderr_line = process.stderr.readline().decode("utf-8")

        if print_output:
            print(f"Output: {stdout_line}")
            print(f"Errors: {stderr_line}")

        if stdout_line == "" and stderr_line == "":
            if port is not None:
                port = -1
                break

        match1 = re.search(regex, stdout_line)
        match2 = re.search(regex, stderr_line)
        if match1:
            port = match1.group().split(":")[-1]
            port = int(port)
            break
        if match2:
            port = match2.group().split(":")[-1]
            port = int(port)
            break

    if port is None:
        port = -1

    return port


def play_macro(macro: Union[str, Path],
               file: Union[str, Path] = None,
               starccm: STARCCMInstall = None,
               port: int = None,
               delete_macro: bool = False,
               is_mdx: bool = False) -> CompletedProcess:
    if isinstance(file, str):
        file = Path(file)
    if isinstance(macro, str):
        macro = Path(macro)

    live_server = False
    if file is None and port is None:
        raise ValueError("arguments file and port cannot both be none.")
    if port is not None:
        live_server = True
        if starccm is None:
            raise ValueError("port argument specified without a valid version.")
        elif not starccm.exists():
            raise ValueError(f"port argument specified without a valid version.\n{starccm} does not exist.")
        if file is not None:
            if isinstance(file, str):
                file = Path(file)
            if not file.exists():
                file = None
            else:
                print("A valid file and port were supplied. Priority given to running server.")
    elif starccm is None:
        starccm = get_starccm_version(file)
    if not live_server:
        _validate_starccm_file(file)

    def compile_args(running_server: bool) -> (str, Path):
        if running_server:
            a = [starccm.starccm(), "-batch", macro, "-port", f"{port}"]
            w = macro.parent
        else:
            a = [starccm.starccm(), "-batch", macro, file]
            w = file.parent
        if is_mdx:
            a.append("-dmproject")
        return a, w

    arguments, work_dir = compile_args(live_server)
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        cwd=work_dir
    )

    if delete_macro:
        os.remove(macro)

    return result


def get_starccm_version(file: Union[str, Path]) -> STARCCMInstall:
    if isinstance(file, str):
        file = Path(file)
    _validate_starccm_file(file)
    parent = file.parent.absolute()

    starccm = Path.joinpath(Path(default_star_ccm_plus), "starccm+.bat")
    args = [starccm, '-info', file]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=parent
    )
    print(result.stdout)
    print(result.stderr)
    pattern = r"\d{2}\.\d{2}\.\d{3}"
    version = re.search(pattern, result.stdout)
    version = version.group()
    pattern = r"-r8"
    dp = True if re.search(pattern, result.stdout) else False
    if dp:
        version += '-R8'

    star_version = get_star_install(version=version)

    if star_version is None:
        raise ValueError(f"No valid installation of STAR-CCM+ {version} found.\n"
                         f"Default install directory: {star_ccm_plus_install_dir}\n"
                         f"Backup install directory:  {star_ccm_plus_bkup_install_dir}")

    return star_version


def get_files_by_type(path: str, ext: str) -> [str]:
    """
    List all files with a specified extension in a parent directory.
    :param path:    Path of directory to search for files
    :param ext:     Find all files with this extension.
    :return:        A list of all files with the specified extension
    """
    if not str(ext).startswith('.'):
        ext = '.' + ext

    contents = os.listdir(path)
    files = list(file for file in contents if file.endswith(ext))
    return files


def remote_directory_exists(test_dir: str, host: str) -> bool:
    """
    Check if a remote directory exists on a specified host.

    :param test_dir:    The remote path of the directory that is being searched for.
    :param host:        The hostname or IP address to search for the test directory.
    :return:            True if the remote directory exists, False otherwise.
    """
    dir_name = test_dir.split("/")[-1]
    test_dir = test_dir.replace(dir_name, '')
    dir_test_result = ssh_comm(host, f'ls {test_dir}')
    test_dirs = dir_test_result.stdout.split('\n')
    return dir_name in test_dirs


def remote_file_exists(test_file: str, host: str) -> bool:
    """
    Check if a remote file exists on a specified host.

    :param test_file:   The remote path of the file that is being searched for.
    :param host:        The hostname or IP address to search for the test file.
    :return:            True if the remote file exists, False otherwise.
    """
    file_name = test_file.split("/")[-1]
    test_dir = test_file.replace(file_name, '')
    file_test_result = ssh_comm(host, f'ls {test_dir}')
    test_dir_contents = file_test_result.stdout.split('\n')
    return file_name in test_dir_contents


def create_remote_directory(dir_path: str, host: str) -> CompletedProcess:
    """
    Create a new directory on a remote host.

    :param dir_path:    Remote path of the new directory.
    :param host:        The hostname or IP address on which to create the new directory.
    :return:            The CompletedProcess object from the remote command to create the directory.
    """
    print(f'Creating directory {dir_path} on {host}')
    result = ssh_comm(host, f'mkdir {dir_path}')
    return result


def copy_file_to_remote_host(remote_host: str, src: str, trg: str, overwrite: bool = False) \
        -> Union[CompletedProcess, None]:
    """
    Copy a file to a remote host.

    :param remote_host:     The hostname or IP address of the remote machine on which to copy the file.
    :param src:             The local file path to copy to the remote host.
    :param trg:             The remote file path for the copied file.
    :param overwrite:       Overwrites the remote file with the local if it already exists.
    :return:                The CompletedProcess object from the copy command or None if the remote file already exists
                            and overwrite=False.
    """
    if overwrite:
        return _scp_comm(remote_host, src, trg)
    else:
        remote_file_name = Path(trg + '/' + Path(src).as_posix().split('/')[-1]).as_posix()
        if not remote_file_exists(remote_file_name, remote_host):
            return _scp_comm(remote_host, src, trg)
        else:
            return None


def copy_file_from_remote_host(remote_host: str, src: str, trg: str, overwrite=False) \
        -> Union[CompletedProcess, None]:
    """
    Copy a file from a remote host.

    :param remote_host:     The hostname or IP address of the remote machine from which file is copied.
    :param src:             The remote file path for the file to copy.
    :param trg:             The local file path to copy the file to.
    :param overwrite:       If True overwrites the local file with the remote if it already exists.
    :return:                The CompletedProcess object from the copy command or None if the local file already exists
                            and overwrite=False.
    """
    if overwrite:
        return _scp_comm(remote_host, src, trg, remote_is_src=True)
    else:
        local_file_name = Path(trg + '/' + Path(src).as_posix().split('/')[-1])
        if not os.path.exists(local_file_name):
            return _scp_comm(remote_host, src, trg, remote_is_src=True)
        else:
            return None


def copy_files(files_2_copy: [str], src: str, dest: str, overwrite: bool = False) -> None:
    """
    Copies a list of files from a source directory to a destination directory.  Overwriting those files if overwrite
    = True.

    :param files_2_copy:    List of file names that exist in the source directory to be copied to the destination
                            directory.
    :param src:             Path to the directory containing the files to copy.
    :param dest:            Path to the destination directory.
    :param overwrite:       If True overwrites the files in the destination directory if they exist.
    :return:                None
    """
    for file_i in files_2_copy:
        src_path = os.path.join(src, file_i)
        dest_path = os.path.join(dest, file_i)
        if not os.path.exists(dest_path):
            print(f'copying from {src_path} to {dest_path}')
            copy2(src_path, dest_path)
        else:
            print(f'{file_i} found in {dest}.')
            if overwrite:
                print(f'Overwriting {dest_path} with file from {src_path}')
                copy2(src_path, dest_path)


def delete_files_by_type(parent_dir: str, types: tuple[str] = ('.sim~', '.dmprj~'), test: bool = False) -> None:
    total_size = 0
    count = 0
    for root, sub_dirs, files in os.walk(parent_dir):
        for f in files:
            match = any(f.endswith(t) for t in types)
            if match:
                count += 1
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                total_size += size
                if test:
                    print(f'{path} ({size/1000000000} GB) is marked to be deleted')
                else:
                    try:
                        os.remove(path)
                    except PermissionError:
                        print(f'Permission error for file {path}')
    test_str = 'marked to be ' if test else ''
    print(f'{count} files totaling {total_size/1000000000} GB {test_str}deleted')


def compress_files_by_type(parent_dir: Union[str, Path],
                           types: tuple[str] = ".sim",
                           test: bool = False,
                           delete_original: bool = False) -> None:
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    total_size = 0
    total_compressed_size = 0
    count = 0
    for root, sub_dirs, files in os.walk(parent_dir):
        for f in files:
            match = any(f.endswith(t) for t in types)
            if match:
                count += 1
                path = Path.joinpath(Path(root), f)
                z_file = path.parent.joinpath(f"{path.stem}.7z")
                size = os.path.getsize(path)
                total_size += size
                if test:
                    print(f'{path} ({size / 1000000000} GB) is marked to be compressed')
                else:
                    args = [zip_exe, "a", f"{z_file}", f"{path}"]
                    result = subprocess.run(
                        args,
                        capture_output=True,
                        text=True
                    )
                    print(result.stderr)
                    print(result.stdout)
                    total_compressed_size += os.path.getsize(z_file)
                    if delete_original:
                        try:
                            os.remove(path)
                        except PermissionError:
                            print(f'Permission error for file {path}')
    test_str = 'marked to be ' if test else ''
    print(f'{count} files totaling {total_size / 1000000000} GB {test_str}compressed')
    print(f"Total compressed size {total_compressed_size/ 1000000000} GB. "
          f"Saving {(total_size - total_compressed_size) / 1000000000} GB")


def ssh_comm(host: str, comm: str) -> CompletedProcess:
    """
    Execute a connection on a remote host using ssh.

    :param host:    The hostname or ip address of the remote host on which to execute the command.
    :type host:     str
    :param comm:    The command to be executed.
    :type comm:     str
    :return:        The CompletedProcess result from the remote command.
    :rtype:         CompletedProcess
    """
    args = [r'C:\Windows\System32\OpenSSH\ssh.exe',
            f'cd8unu@{host}',
            comm]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True
    )
    return result


def _scp_comm(host, src, trg, remote_is_src=False):
    if remote_is_src:
        args = [r'C:\Windows\System32\OpenSSH\scp.exe',
                '-r',
                f'cd8unu@{host}:{src}',
                trg]
    else:
        args = [r'C:\Windows\System32\OpenSSH\scp.exe',
                '-r',
                src,
                f'cd8unu@{host}:{trg}']
    result = subprocess.run(
        args,
        capture_output=True,
        text=True
    )
    return result


def update_macro_parameters(val_dict: {str: str}, itc_macro: str) -> None:
    """
    Using the supplied {str: str} dictionary searches a supplied java macro for each instance of a dictionary entry's
    key and replaces the assigned value in the java macro with they associated dictionary entry's value.  Only lines in
    the macro that satisfy the regular expression \"^ *String p_[A-Za-z\\d_ ]*=.*;$\" are considered.

    Some examples:

            String p_docName = "staticMixer.CATAnalysis";
            String p_proxyNetwork = ".net.plm.eds.com";

    For the above examples the assigned value of the Java variable p_docName or p_ProxyNetwork can be replaced by a
    different value if either of those variable names is included in the dictionary.

    :param val_dict:    Dictionary used to update the java variables. Keys must satisfy the regular expression following
    the pattern p_[A-Za-z\\d_ ].
    :type val_dict:     {str: str}
    :param itc_macro:   Path to the Java macro that is to be updated.
    :type itc_macro:    str
    :return:            None
    """
    with open(itc_macro, 'r') as macro:
        pattern = re.compile(r"^ *String p_[A-Za-z\d_ ]*=.*;$")
        lines = macro.readlines()
        replacement = ''
        for line in lines:
            result = pattern.search(line)
            if result is None:
                replacement += line
            else:
                temp = line.split('=')
                key = temp[0].replace(' ', '')
                key = key.replace('String', '')
                new_line = temp[0] + "= \"" + val_dict.get(key) + '\";\n'
                replacement += new_line
    with open(itc_macro, 'w') as macro:
        macro.write(replacement)


def recursive_play_star_macro(parent_dir: Union[str, Path],
                              macro_path: Union[str, Path],
                              star_version: STARCCMInstall,
                              log_tag: str = '') -> None:
    """
    Recursively play a STAR-CCM+ java macro on every sim file found in a specified directory and its sub-directories
    using a specified STAR-CCM+ executable (starccm+.exe).  Optionally extract the content of each line of a
    simulation's stout that begins with a user specified tag and append these to a file extract.csv in the root
    directory of the search.

    :param parent_dir:      An absolute path that will be the starting point for the crawl looking for any .sim files
    :type parent_dir:       str, required
    :param macro_path:      An absolute path to the STAR-CCM+ java macro that is to be played on each sim found
    :type macro_path:       str, required
    :param star_version:    An absolute path to the starccm+.exe executable of the version to be used when running the
                            macro
    :type star_version:     str, required
    :param log_tag:         String that servers as a tag to extract information from the STAR-CCM+ stout stream for each
                            simulation. Any line that starts with this String will be appended to extract.csv in the
                            parent_dir. (Default is '' which disables the extract).
    :type: log_tag:         str
    :return:                None
    """
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)
    if isinstance(macro_path, str):
        macro_path = Path(macro_path)
    for root, sub_dirs, files in os.walk(parent_dir):
        for f in files:
            if f.endswith('.sim'):
                sim_name = f.replace('.sim', '')
                if sim_name.startswith('Design_'):
                    sim_name = sim_name.replace('Design_', '')
                sim = Path(root).joinpath(f)
                args = [star_version.starccm(), '-batch', macro_path, sim]
                print(args)
                result = subprocess.run(args, capture_output=True, text=True)
                if log_tag:
                    for line in result.stdout.split('\n'):
                        if line.startswith(log_tag):
                            value = line.replace(log_tag, '')
                            with open(os.path.join(parent_dir, 'extract.csv'), 'a') as cr_file:
                                cr_file.write(f'{sim_name},{value}\n')


def subsample(population: [], proportion: float = 0.5) -> tuple[[], []]:
    """
    Randomly selects a specified proportion of items from a provided list of items (each item must be hashable).

    :param population:  List from which to sample
    :type population:   List of hashable items, required
    :param proportion:  Value between 0.0 and 1.0.  The number of subsampled items is equal to the floor of the
                        proportion * len(population) (default is 0.5).
    :type proportion:   float, optional

    :return:            The randomly sampled list and the remaining items which were not sampled.
    """
    total = len(population)
    n = math.floor(proportion * total)
    idx = np.random.choice(np.arange(len(population)), size=n, replace=False)
    sub = [population[i] for i in range(len(population)) if i in idx]
    remain = [population[i] for i in range(len(population)) if i not in idx]
    return sub, remain


def subsample_n(population: [], n_subsamples: int) -> tuple[[], []]:
    """
    Randomly selects a specified proportion of items from a provided list of items (each item must be hashable).

    :param population:      List from which to sample
    :type population:       List of hashable items, required
    :param n_subsamples:    The number of subsampled items to be randomly selected from population.
    :type n_subsamples:     int, required

    :return:            The randomly sampled list and the remaining items which were not sampled.
    """
    idx = np.random.choice(np.arange(len(population)), size=n_subsamples, replace=False)
    sub = [population[i] for i in range(len(population)) if i in idx]
    remain = [population[i] for i in range(len(population)) if i not in idx]
    return sub, remain


def read_str_to_ndarray(string: str) -> npt.ArrayLike:
    """
    Reads in a string and first replaces all commas with a space ' ' then replaces all characters  except numbers, +, -,
    ' ', and . with an empty string ''. Then converts the remaining string to an np.ndarray using eval.  Though it does
    return an ndarray all arrays will be a single row.  Multi dimensional arrays should be sent one line at a time as
    all \n characters will be replaced with an empty string thus compressing down to a single row.

    :param string:  The string to convert back to an ndarray.
    :type string:   str
    :return         An numpy.ndarray of the string.  This will be a single row array.
    """
    replacement = re.sub(r'[^0-9 \-+\.]', '', string)
    replacement = " ".join(replacement.split())
    replacement = replacement.split(' ')
    ndarray = np.array([eval(val) for val in replacement if val != '.' and val != ''])

    return ndarray


def bin_index(bins: npt.ArrayLike, value: float) -> int:
    for i in range(1, len(bins)):
        if bins[i] >= value:
            return i - 1
    return len(bins) - 1


def clean_csv_for_df(csv_file: Union[str, Path], sort_col: str = "Evaluation  #") -> pd.DataFrame:
    if isinstance(csv_file, str):
        csv_file = Path(csv_file)

    csv_path = Path.joinpath(csv_file.parent, csv_file.name + ".tmp")
    with open(csv_file, "r", newline="") as input_file, open(csv_path, "w", newline="") as csv_file:
        r = csv.reader(input_file, delimiter=",")
        w = csv.writer(csv_file, delimiter=",")
        for line in r:
            pattern = re.compile(r"^\s*$")
            cleaned = []
            for field in line:
                if pattern.match(field):
                    cleaned += [""]
                else:
                    cleaned += [field]
            w.writerow(cleaned)
    study_results = pd.read_csv(csv_path)
    os.remove(csv_path)
    study_results.columns = study_results.columns.str.strip()
    names = {old: old.replace("\"", "") for old in study_results.columns}
    study_results.rename(columns=names, inplace=True)

    for column in study_results:
        column_df = study_results[column]
        if column_df.dtype == object:
            study_results[column] = column_df.str.strip()

    if sort_col:
        study_results = study_results.sort_values(by=[sort_col])

    return study_results


def filter_nan(df: pd.DataFrame, col: Union[str, List[str]], reset_idx: bool = True) -> pd.DataFrame:
    if isinstance(col, str):
        col = [col]

    mask = [False for _ in range(len(df))]

    for c in col:
        m1 = df[c].isnull()
        m2 = df[c].isna()
        mask = [False if any(z) else True for z in zip(mask, m1, m2)]

    df = df[mask]
    if reset_idx:
        df = df.reset_index(drop=True)

    return df


def _validate_starccm_file(file: Path):
    files = ('.sim', '.dmprj', '.sim~', '.dmprj~')
    if file is None:
        raise ValueError("argument file cannot be None type")
    if not file.exists():
        raise FileNotFoundError(f"File {file} must exist.")
    if file.suffix not in files:
        message = f'{file} must be a Simcenter STAR-CCM+ sim file or dmprj file.'
        raise ValueError(f"{message}")
