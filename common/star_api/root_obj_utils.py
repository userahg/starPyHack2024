from __future__ import annotations
import re
import os
import common.starccm_macros as sm
import common.util as util
from typing import Union
from enum import Enum
from pathlib import Path
from subprocess import CompletedProcess
from common.star_versions import STARCCMInstall
from common.star_versions import get_star_install
from common.star_versions import get_star_install_local


class CommandRecorder:

    def __init__(self, name: str, is_mdx: bool = True):
        self.record_all = False
        self.name = name
        self.is_mdx = is_mdx
        self.star_install = get_star_install_local()
        self.macro = self.new_macro()

    def clear_commands(self):
        self.macro = self.new_macro()

    def new_macro(self, name: str = None) -> sm.MacroWriter:
        if name is None:
            name = f"RecordedActionsOn{self.name}"
        macro = sm.MacroWriter(name, is_mdx=self.is_mdx)
        macro.save_root_obj = True
        return macro

    def update_version(self, version: Union[str, STARCCMInstall]):
        if version is None:
            return
        if isinstance(version, str):
            star_install = get_star_install(version)
            if star_install is not None:
                self.star_install = star_install
        elif isinstance(version, STARCCMInstall):
            self.star_install = version

    def play_macro(self,
                   star_file: Union[Path, int],
                   macro: Path,
                   delete_macro: bool = False) -> CompletedProcess:
        is_server_address = False
        if isinstance(star_file, int):
            port_re = r"^\d{5}$"
            if not re.match(port_re, str(star_file)):
                raise ValueError(f"star_file of type int must specify a valid STAR-CCM+ server port.\n"
                                 f"{star_file} is not a valid port number.")
            is_server_address = True
        else:
            valid_extensions = [".sim", ".dmprj"]
            if isinstance(star_file, Path):
                if not star_file.exists():
                    raise ValueError(f"star_file of type Path must specify an existing STAR-CCM+ sim or dmprj file.\n"
                                     f"{star_file} does not exist.")
                if star_file.suffix not in valid_extensions:
                    raise ValueError(f"star_file of type Path must specify an existing STAR-CCM+ sim or dmprj file.\n"
                                     f"star_file extension {star_file.suffix} is not recognized.")

        if is_server_address:
            result = util.play_macro(macro=macro, starccm=self.star_install, port=star_file, delete_macro=False,
                                     is_mdx=self.is_mdx)
        else:
            result = util.play_macro(macro=macro, file=star_file, starccm=self.star_install, delete_macro=False,
                                     is_mdx=self.is_mdx)
        if delete_macro:
            os.remove(macro)
        return result

    def n_recorded_actions(self) -> int:
        count = 0
        for _ in self.macro.iter_methods():
            count += 1
        return count


class STARObject:

    def __init__(self, recorder: CommandRecorder, obj_id: STARObjectID):
        self.recorder = recorder
        self.id = obj_id

    def __setattr__(self, key, value):
        api_key = f"_{key}"
        if api_key in self.__dict__:
            api_obj = self.__dict__[api_key]
            if isinstance(api_obj, StarAPIProperty):
                if self.recorder.record_all:
                    api_obj.set_property(value, record=True, macro=self.recorder.macro)
                    super(STARObject, self).__setattr__(key, value)
                else:
                    api_obj.starapi_value = value
                    super(STARObject, self).__setattr__(key, value)
            else:
                super(STARObject, self).__setattr__(key, value)
        else:
            super(STARObject, self).__setattr__(key, value)


class JSONTag:

    def __init__(self, tag: str):
        self.tag = tag
        self.r = self.tag
        self.w = f"\"{self.tag}\""


class STARObjectID:

    def __init__(self, obj_id: int, java_class: str, is_mdx: bool = True):
        self.id = obj_id
        self.java_class_s_name = java_class.split(".")[-1]
        self.java_class_f_name = java_class
        self.is_mdx = is_mdx
        self.aux_java_classes = []

    def java(self) -> sm.MacroMethod:
        method = sm.MacroMethod(f"get{self.id}Obj", arguments=dict(), return_type=self.java_class_s_name)
        method.packages.add(self.java_class_f_name)
        proj_var_name = sm.MacroWriter.get_root_obj_var_name(self.is_mdx)
        var_name = "object"
        method.add_line(f"{self.java_class_s_name} {var_name} = ({self.java_class_s_name}) {proj_var_name}"
                        f".getObjectRegistry().getObjectById({self.id}l)")
        method.add_line(f"return {var_name}")
        return method

    def packages(self) -> list[str]:
        packages = [item for item in self.aux_java_classes]
        packages.append(self.java_class_f_name)
        return packages


class STARAPIEnum(Enum):

    def java_class_path(self) -> str:
        pass


class StarAPIProperty:

    def __init__(self, value, name: str, setter_comm: str, getter_comm: str, obj_id: STARObjectID):
        self.name = name
        self.starccm_value = value
        self.starapi_value = value
        self.setter_comm = setter_comm
        self.getter_comm = getter_comm
        self.id = obj_id

    def set_property(self, value, **kwargs):
        if "record" in kwargs and kwargs["record"]:
            if "macro" not in kwargs:
                raise ValueError(f"set_property called with record kwarg but no macro kwarg.\n"
                                 f"To record a MacroWriter must be supplied with kwarg macro.")
            if not isinstance(kwargs["macro"], sm.MacroWriter):
                raise ValueError
            macro = kwargs["macro"]
            method = self._get_setter_method(value, macro)
            macro.add_method(method)
        self.starapi_value = value

    def _get_setter_method(self,
                           value,
                           macro: sm.MacroWriter) -> sm.MacroMethod:
        name = macro.generate_method_name(f"set{self.id.id}{self.name}")
        method = sm.MacroMethod(name, dict())
        method.packages.update(self.id.packages())
        getter = self.id.java()
        getter.call_in_execute = False
        macro.add_method(getter)
        body = sm.BodyComponent()
        body.new_lines = False
        param_var_name = f"obj{self.id.id}"
        body.add_line(f"{getter.return_type} {param_var_name} = {getter.name}()")
        if isinstance(value, str):
            value = f"\"{value}\""
        elif issubclass(type(value), STARAPIEnum):
            value = f"{value.java_class_path()}.{value.name}"
        body.add_line(f"{param_var_name}.{self.setter_comm}({value})")
        method.add_component(body)
        return method

    def __str__(self):
        return self.starapi_value
