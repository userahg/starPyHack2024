import re
from pathlib import Path
from typing import Union
from common.local_settings import default_star_ccm_version
from common.local_settings import star_ccm_plus_install_dir
from common.local_settings import star_ccm_plus_bkup_install_dir

star_version_re = r"^\d{1,2}.0[1-6].\d{3}(-R8)?$"


class STARCCMInstall:

    def __init__(self, install_dir: Union[str, Path]):
        if isinstance(install_dir, str):
            install_dir = Path(install_dir)
        self._bin_dir = install_dir.parent
        self._root_dir = install_dir.parent.parent.parent.parent

    def version(self) -> str:
        return self._root_dir.name

    def starccm(self) -> Path:
        starccm = self._bin_dir.joinpath("starccm+.bat")
        return starccm

    def starlaunch(self) -> Path:
        starlaunch = self._bin_dir.joinpath("starlaunch.bat")
        return starlaunch

    def cad_client_installed(self) -> bool:
        version = self.version().replace("-R8", "")
        cad_dir = self._root_dir.joinpath(f"STAR-CAD{version}")
        return cad_dir.exists()

    def exists(self):
        return self.starccm().exists()

    def __str__(self):
        return str(self.starccm())


def validate_version(version: str) -> bool:
    match = re.search(star_version_re, version)
    is_valid = True if match else False
    return is_valid


def validate_install_dir(p: Path) -> bool:
    exp = star_version_re
    if not p.exists():
        return False
    if not p.is_dir():
        return False
    if not p.exists():
        return False
    if not p.is_dir():
        return False
    one_valid_dir = False
    for f in p.iterdir():
        m = re.search(exp, f.name)
        if m:
            one_valid_dir = True
    return one_valid_dir


def list_installed_versions(p: Path) -> list[STARCCMInstall]:
    exp = star_version_re
    version_paths = []
    if validate_install_dir(p):
        for file in p.iterdir():
            match = re.search(exp, file.name)
            if match:
                installed_version_path = file.joinpath(f"STAR-CCM+{file.name}")
                installed_version_path = installed_version_path.joinpath("star")
                installed_version_path = installed_version_path.joinpath("bin")
                installed_version_path = installed_version_path.joinpath("starccm+.bat")
                if installed_version_path.exists():
                    installed_version = STARCCMInstall(installed_version_path)
                    version_paths.append(installed_version)
    return version_paths


def all_installed_versions(install_dir: Union[str, Path] = None) -> list[STARCCMInstall]:
    install_paths = []

    if install_dir is not None:
        if isinstance(install_dir, str):
            install_dir = Path(install_dir)
        install_paths.append(install_dir)
    install_paths.append(Path(star_ccm_plus_install_dir))
    install_paths.append(Path(star_ccm_plus_bkup_install_dir))

    starccm_paths = []

    for install_path in install_paths:
        starccm_paths.extend(list_installed_versions(install_path))

    return starccm_paths


def get_star_install(version: str = default_star_ccm_version,
                     install_dir: Union[str, Path] = None) -> Union[STARCCMInstall, None]:
    exp = star_version_re
    match = re.search(exp, version)
    if not match:
        raise ValueError(f"Version {version} does not identify a STAR-CCM+ version number.")

    installed_version = None

    for p in all_installed_versions(install_dir=install_dir):
        if version == p.version():
            installed_version = p

    return installed_version
