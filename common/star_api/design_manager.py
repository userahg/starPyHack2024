from __future__ import annotations
import json
import shutil
import numpy as np
import common.starccm_macros as sm
import common.star_versions as sv
from numpy.typing import ArrayLike
from common.starccm_macros import LineType as LT
from common.star_api.root_obj_utils import *
import pandas as pd
from pathlib import Path
from typing import Union, List
from matplotlib import pyplot as plt
import datetime
import common.util as util
import common.math_util as m_util
import common.visualization as viz
from common.local_settings import default_star_ccm_plus


t_name = JSONTag("name")
t_log = JSONTag("log")
t_log_m = JSONTag("log_message")
t_messages = JSONTag("messages")
t_id = JSONTag("object_ID")
t_studies = JSONTag("studies")
t_design_sets = JSONTag("design sets")
t_parameters = JSONTag("parameters")
t_responses = JSONTag("responses")
t_list_vals = JSONTag("list_values")
t_designs = JSONTag("designs")
t_val = JSONTag("value")
t_baseline = JSONTag("baseline")
t_min = JSONTag("min")
t_max = JSONTag("max")
t_res = JSONTag("resolution")
t_type = JSONTag("type")
t_dir = JSONTag("dir")
t_vis_files = JSONTag("vis_files")
t_path = JSONTag("path")
_api_props = []


class DesignManagerProject:

    @staticmethod
    def get_java_packages() -> set[str]:
        return {"star.mdx.MdxDesignStudy", "star.mdx.MdxDesignSet"}

    @staticmethod
    def empty_macro(name: str) -> sm.MacroWriter:
        macro = sm.MacroWriter(class_name=name, java_imports=DesignManagerProject.get_java_packages(), is_mdx=True)
        return macro

    @staticmethod
    def get_loop_all_studies(proj_var_name="_proj", iterate_name: str = "study_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesignStudy"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{proj_var_name}.getDesignStudyManager().getDesignStudies()"
        return loop

    @classmethod
    def from_json(cls, json_file: Union[str, Path]) -> DesignManagerProject:
        if isinstance(json_file, str):
            json_file = Path(json_file)

        with open(json_file) as f:
            file_contents = json.load(f)

        dmprj_path = Path.joinpath(json_file.parent, f'{file_contents[t_name.r]}.dmprj')
        if not dmprj_path.exists():
            raise FileNotFoundError(f"project {dmprj_path} not found")

        proj = DesignManagerProject(dmprj_path)
        proj._from_json(json_file)
        return proj

    @classmethod
    def get_proj(cls,
                 work_dir: Union[str, Path],
                 dmprj: str,
                 sync: bool = False,
                 version: str = "19.02.009-R8") -> DesignManagerProject:
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        if dmprj.endswith(".dmprj"):
            split_name = dmprj.split(".")
            dmprj = ""
            for s in split_name[:-1]:
                dmprj += s

        dmprj_path = work_dir.joinpath(f"{dmprj}.dmprj")
        if not dmprj_path.exists():
            raise ValueError(f"No project {dmprj}.dmprj found in workdir: {work_dir}")
        star_install = sv.get_star_install(version=version)
        if star_install is None:
            raise ValueError(f"Version {version} not found.")

        if sync:
            prj = DesignManagerProject(dmprj_path)
            prj.recorder.update_version(star_install)
            prj.sync()
        else:
            json_path = work_dir.joinpath(f"{dmprj}.json")
            if json_path.exists():
                prj = DesignManagerProject.from_json(json_path)
                prj.recorder.update_version(star_install)
            else:
                prj = DesignManagerProject(dmprj_path)
                prj.recorder.update_version(star_install)
                prj.sync()
        return prj

    @classmethod
    def get_proj_distrib(cls,
                 work_dir: Union[str, Path],
                 dmprj: str,
                 sync: bool = False,
                 distrib: str = r"/install/STAR-CCMP/lin64-r8/19.02.009_01/STAR-CCM+19.02.009-R8/star/bin/starccm+") -> DesignManagerProject:
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        if dmprj.endswith(".dmprj"):
            split_name = dmprj.split(".")
            dmprj = ""
            for s in split_name[:-1]:
                dmprj += s

        dmprj_path = work_dir.joinpath(f"{dmprj}.dmprj")
        if not dmprj_path.exists():
            raise ValueError(f"No project {dmprj}.dmprj found in workdir: {work_dir}")
        star_install = sv.get_star_install_local(distrib=distrib)
        if star_install is None:
            raise ValueError(f"Version {version} not found.")

        if sync:
            prj = DesignManagerProject(dmprj_path)
            prj.recorder.update_version(star_install)
            prj.sync()
        else:
            json_path = work_dir.joinpath(f"{dmprj}.json")
            if json_path.exists():
                prj = DesignManagerProject.from_json(json_path)
                prj.recorder.update_version(star_install)
            else:
                prj = DesignManagerProject(dmprj_path)
                prj.recorder.update_version(star_install)
                prj.sync()
        return prj

    @classmethod
    def get_live_proj(cls,
                      dmprj_path: Union[str, Path],
                      port: int,
                      version: str = "19.02.009-R8") -> DesignManagerProject:
        if isinstance(dmprj_path, str):
            dmprj_path = Path(dmprj_path)
        if not dmprj_path.suffix == ".dmprj":
            raise ValueError(f"{dmprj_path} is not a STAR-CCM+ Design Manager dmprj file.")
        if not dmprj_path.exists():
            raise ValueError(f"No project {dmprj_path.name} exists in directory {dmprj_path.parent}.")
        port_re = r"^\d{5}$"
        if not re.match(port_re, str(port)):
            raise ValueError(f"star_file of type int must specify a valid STAR-CCM+ server port.\n"
                             f"{port} is not a valid port number.")

        dmprj = DesignManagerProject(path=dmprj_path)
        dmprj.port = port

        if version is not None:
            star_install = get_star_install(version=version)
            if star_install is None:
                raise ValueError(f"Version {version} not found.")
            else:
                dmprj.recorder.update_version(star_install)

        json_file = dmprj._to_json(run=True)
        dmprj._from_json(json_file=json_file)
        return dmprj

    def __init__(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        self.name = path.stem
        self.recorder = CommandRecorder(self.name, is_mdx=True)
        self.path = path
        self.studies = list()
        self.port = None

    def __iter__(self):
        return iter(self.studies)

    def get_study(self, name: str) -> Study:
        for s in self.studies:
            if s.name == name:
                return s
        raise AttributeError(f"No study named {name} found.")

    def sync(self,
             push_to_star: bool = False,
             recorded_only: bool = True,
             delete_json: bool = False,
             delete_macro: bool = True,
             version: Union[str, STARCCMInstall] = None):
        if version is not None:
            self.recorder.update_version(version)

        if push_to_star:
            macro = None
            if recorded_only:
                if self.recorder.n_recorded_actions() > 0:
                    macro = self.recorder.macro
            else:
                macro = self._prep_macro_writer()
            if macro is not None:
                if self.port is not None:
                    macro.save_root_obj = False
                    sf = self.port
                else:
                    sf = self.path
                macro_path = macro.build(self.path.parent)
                result = self.recorder.play_macro(star_file=sf, macro=macro_path, delete_macro=delete_macro)
                print(result.stdout)
                print(result.stderr)

        json_file = self._to_json(run=True)
        self._from_json(json_file, delete_json=delete_json)

    def clear_commands(self):
        self.recorder.clear_commands()

    def start(self, version: Union[str, Path] = None):
        raise NotImplementedError(f"method start not implemented for STAR-CCM+ Design Manager Projects.\n"
                                  f"Launch the server and then manually set the port attribute.")

    def set_port(self, port: int):
        port_re = r"^\d{5}$"
        if not re.match(port_re, str(port)):
            raise ValueError(f"star_file of type int must specify a valid STAR-CCM+ server port.\n"
                             f"{port} is not a valid port number.")
        self.port = port

    def _prep_macro_writer(self) -> Union[sm.MacroWriter, None]:
        global _api_props
        macro_name = f"ApplySTARApiValuesTo{self.name}"
        macro = self.recorder.new_macro(macro_name)
        n_actions = 0
        for sapi_prop in _api_props:
            v1 = sapi_prop.starapi_value
            v2 = sapi_prop.starccm_value
            if v1 != v2:
                n_actions += 1
                sapi_prop.set_property(v1, record=True, macro=macro)
        if n_actions > 0:
            return macro
        else:
            return None

    def _to_json(self, run: bool = False) -> Path:
        p = {"star.mdx.MdxDesignStudy",
             "star.mdx.MdxDesignSet",
             "star.mdx.MdxStudyParameter",
             "java.util.Map",
             "java.util.HashMap",
             "javax.json.*",
             "javax.json.stream.JsonGenerator"
             }
        macro_name = f"ExportProjectToJSON"
        macro = sm.MacroWriter(macro_name, p, is_mdx=True)
        workdir = self.path.parent
        macro_output = Path.joinpath(workdir, f"{self.name}.json")
        method = sm.MacroMethod("convertProjectToJSON", dict())
        except_method = sm.get_print_ex_method(True)
        proj_var_name = macro.get_root_obj_var_name(True)
        method.add_line("JsonArrayBuilder studyArrayBuilder = Json.createArrayBuilder()")
        study_loop = DesignManagerProject.get_loop_all_studies()
        study_loop.new_lines = True
        loop_var = study_loop.for_all_loop_iterate_name
        study_json_method = Study.to_json(macro)
        study_loop.add_line(f"{study_json_method.return_type} studyJSON = {study_json_method.name}({loop_var})")
        study_loop.add_line(f"studyArrayBuilder.add(studyJSON)")
        method.add_component(study_loop)

        write_body = sm.BodyComponent()
        declare_proj_json = f"JsonObject projJSON = Json.createObjectBuilder()"
        write_body.add_line(f"{declare_proj_json}", line_type=LT.START)
        write_body.add_line(f".add({t_name.w}, {proj_var_name}.getPresentationName())", line_type=LT.MIDDLE)
        write_body.add_line(f".add({t_studies.w}, studyArrayBuilder)", line_type=LT.MIDDLE)
        write_body.add_line(f".build()", line_type=LT.END)
        write_body.add_line("Map<String, Boolean> config = new HashMap<>()")
        write_body.add_line("config.put(JsonGenerator.PRETTY_PRINTING, true)")
        write_body.add_line("JsonWriterFactory writerFactory = Json.createWriterFactory(config)")

        method.add_component(write_body)

        catch_comp = sm.CatchComponent()
        catch_comp.add_line(f"{except_method.name}({catch_comp.except_var_name})")
        try_resource = sm.TryFileWriteResource(str(macro_output))
        try_comp = sm.TryComponent(catch_comp, resource=try_resource)
        try_comp.add_line("writerFactory.createWriter(writer).write(projJSON)")
        try_comp.new_lines = False
        method.add_component(try_comp)
        macro.add_method(method)
        macro.add_method(except_method)
        macro.build(workdir)

        if run:
            macro = macro.get_macro_path(workdir)
            sf = self.path if self.port is None else self.port
            output = self.recorder.play_macro(star_file=sf, macro=macro, delete_macro=True)
            print(output.stdout)
            print(output.stderr)

        return macro_output

    def _from_json(self, json_file: Path, delete_json: bool = False):
        with open(json_file) as f:
            file_contents = json.load(f)

        self.studies.clear()

        for study_dict in file_contents[t_studies.r]:
            self.studies.append(Study.from_json(study_dict, self))

        if delete_json:
            os.remove(json_file)


class Study:

    @staticmethod
    def loop_all_design_sets(study_var_name: str, iterate_name: str = "set_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesignSet"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{study_var_name}.getDesignSets().getDesignSets()"
        return loop

    @staticmethod
    def loop_all_parameters(study_var_name: str, iterate_name: str = "param_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxStudyParameter"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{study_var_name}.getStudyParameters().getObjectsOf(MdxStudyParameter.class)"
        return loop

    @staticmethod
    def loop_all_log_messages(study_var_name: str, iterate_name: str = "pair_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.packages = {"star.common.Pair",
                         "star.mdx.MdxLogMessagesSeverityEnum"}
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "Pair<MdxLogMessagesSeverityEnum, String>"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{study_var_name}.getLogMessages()" \
                                     f".getMessages({study_var_name}.getDesignStudies().getLogContexts(), false)"
        return loop

    @staticmethod
    def to_json(macro: sm.MacroWriter) -> sm.MacroMethod:
        java_class = "MdxDesignStudy"
        study_var = "study"
        args = {java_class: study_var}
        method = sm.MacroMethod("convertStudyToJSON", args, "JsonObject")
        method.add_line("JsonArrayBuilder parameterArrayBuilder = Json.createArrayBuilder()")
        method.add_line("JsonArrayBuilder designSetArrayBuilder = Json.createArrayBuilder()")
        method.add_line("JsonArrayBuilder logMessageArrayBuilder = Json.createArrayBuilder()")
        method.add_newline()

        param_loop = Study.loop_all_parameters(study_var)
        design_set_loop = Study.loop_all_design_sets(study_var)
        messages_loop = Study.loop_all_log_messages(study_var)
        param_loop.new_lines = False
        design_set_loop.new_lines = False
        param_loop_var = param_loop.for_all_loop_iterate_name
        design_set_loop_var = design_set_loop.for_all_loop_iterate_name
        message_loop_var = messages_loop.for_all_loop_iterate_name
        json_param_method = Parameter.to_json(macro)
        json_design_set_method = DesignSet.to_json(macro)

        param_loop.add_line(f"{json_param_method.return_type} paramJSON = {json_param_method.name}({param_loop_var})")
        param_loop.add_line(f"parameterArrayBuilder.add(paramJSON)")

        design_set_loop.add_line(f"{json_design_set_method.return_type} designSetJSON = "
                                 f"{json_design_set_method.name}({design_set_loop_var})")
        design_set_loop.add_line(f"designSetArrayBuilder.add(designSetJSON)")

        message_body = sm.BodyComponent()
        message_body.add_line(f"JsonObject messageJSON = Json.createObjectBuilder()", line_type=LT.START)
        message_body.add_line(f".add({t_log_m.w}, {message_loop_var}.getRight().toString())", line_type=LT.MIDDLE)
        message_body.add_line(f".build()", line_type=LT.END)
        message_body.add_line(f"logMessageArrayBuilder.add(messageJSON)")
        message_body.new_lines = False
        messages_loop.add_component(message_body)

        method.add_component(param_loop)
        method.add_component(design_set_loop)
        method.add_component(messages_loop)

        body = sm.BodyComponent()
        declare_study_json = f"JsonObject studyJSON = Json.createObjectBuilder()"
        body.add_line(f"{declare_study_json}", line_type=LT.START)
        body.add_line(f".add({t_name.w}, {study_var}.getPresentationName())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_id.w}, {study_var}.getObjectId())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_parameters.w}, parameterArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(f".add({t_design_sets.w}, designSetArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(f".add({t_messages.w}, logMessageArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(f".build()", line_type=LT.END)
        body.add_line("return studyJSON")
        body.new_lines = False
        method.add_component(body)
        method.call_in_execute = False
        macro.add_method(method)
        return method

    @staticmethod
    def from_json(study_dict: dict, project: DesignManagerProject) -> Study:
        oid = STARObjectID(study_dict[t_id.r], "star.mdx.MdxDesignStudy")
        study = Study(oid, study_dict[t_name.r], project)
        for design_set_dict in study_dict[t_design_sets.r]:
            study.design_sets.append(DesignSet.from_json(design_set_dict, study))
        for parameter_dict in study_dict[t_parameters.r]:
            study.parameters.append(Parameter.from_json(parameter_dict, study))
        messages = [m[t_log_m.r] for m in study_dict[t_messages.r]]
        study.log_messages = messages
        return study

    def __init__(self,
                 obj: STARObjectID,
                 name: str,
                 project: DesignManagerProject):
        self.project = project
        self.name = name
        self.id = obj
        self.design_sets = list()
        self.parameters = list()
        self.log_messages = list()

    def add_design_to_mdx_table(self, values: dict[str, float]):
        method_name = f"addDesignTo{self.name}Study"
        method_name = self.project.recorder.macro.generate_method_name(method_name)
        add_table_method = sm.MacroMethod(method_name, dict())
        add_table_method.packages.add("star.mdx.MdxDesignTable")
        add_table_method.packages.add("star.mdx.MdxStudyParameterColumn")
        add_table_method.packages.add("star.common.Units")
        add_table_method.packages.add("star.common.DoubleWithUnits")
        getter_methods = []
        get_study_method = self.id.java()
        get_study_method.call_in_execute = False
        getter_methods.append(get_study_method)
        body = sm.BodyComponent()
        study_var_name = "study"
        table_var_name = "mdxTable"
        n_rows_var_name = "rowIndex"
        param_name = "studyParam"
        units_name = "units"
        param_column_name = "studyParamColumn"
        body.add_line(f"{get_study_method.return_type} {study_var_name} = {get_study_method.name}()")
        body.add_line(f"MdxDesignTable {table_var_name} = {study_var_name}.getDesignTable()")
        body.add_line(f"{table_var_name}.addNewRow()")
        body.add_line(f"Long nRows = {table_var_name}.getNumRows()")
        body.add_line(f"int {n_rows_var_name} = nRows.intValue() - 1")
        body.add_line(f"MdxStudyParameter {param_name}")
        body.add_line(f"Units {units_name}")
        body.add_line(f"MdxStudyParameterColumn {param_column_name}")

        for k, v in values.items():
            param = self.get_parameter(k)
            param_getter_method = param.id.java()
            param_getter_method.call_in_execute = False
            getter_methods.append(param_getter_method)
            column_name = k.replace("\\", "\\\\")
            body.add_line(f"{param_name} = {param_getter_method.name}()")
            body.add_line(f"{units_name} = {param_name}.getBaselineQuantity().getUnits()")
            body.add_line(f"{param_column_name} = (MdxStudyParameterColumn) {table_var_name}.getTableColumns()."
                          f"getDesignTableColumn(\"{column_name}\")")
            body.add_line(f"{table_var_name}.setValue({param_column_name}, {n_rows_var_name},"
                          f" new DoubleWithUnits({v}, {units_name}))")
        add_table_method.add_component(body)
        self.project.recorder.macro.add_method(add_table_method)
        for method in getter_methods:
            self.project.recorder.macro.add_method(method)

    def get_design_set(self, name: str) -> DesignSet:
        for d in self.design_sets:
            if d.name == name:
                return d
        raise AttributeError(f"No Design Set {name} found.")

    def get_all_designs(self) -> AllDesignsSet:
        for ds in self.design_sets:
            if isinstance(ds, AllDesignsSet):
                return ds
        raise AttributeError("No AllDesignsSet found.")

    def df(self) -> pd.DataFrame:
        return self.get_all_designs().data

    def get_parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise AttributeError(f"No parameter {name} found.")

    def update_parameter_baseline_vals(self, macro: sm.MacroWriter, vals: list) -> sm.MacroMethod:
        method = sm.MacroMethod(f"set{self.name}BaselineValues", dict())
        method.packages = {"star.mdx.MdxStudyParameter"}
        for val in vals:
            for param_i in self.parameters:
                if val["name"] == param_i.name:
                    getter = param_i.id.java()
                    getter.call_in_execute = False
                    param_var_name = f"p{param_i.id.id}"
                    new = val["baseline"]
                    macro.add_method(getter)
                    body = sm.BodyComponent()
                    body.add_line(f"{getter.return_type} {param_var_name} = {getter.name}()")
                    body.add_line(f"{param_var_name}.getBaselineQuantity().setValue({new})")
                    method.add_component(body)
                    continue
            print(f"Unable to find parameter {val['name']} in study {self.name}.")
        return method

    def runtime(self):
        fmt = "%a %b  %d %H:%M:%S %Y"
        start_messages = []
        end_messages = []
        for message in self.log_messages:
            if message.startswith("Begins at"):
                start_messages.append(message)
            if message.startswith("Ends at"):
                end_messages.append(message)

        if len(start_messages) != len(end_messages):
            offset = 1
        else:
            offset = 0

        dur = datetime.timedelta()
        for i in range(len(start_messages) - offset):
            start_str = start_messages[i].replace("Begins at ", "")
            end_str = end_messages[i].replace("Ends at ", "")
            start = datetime.datetime.strptime(start_str, fmt)
            end = datetime.datetime.strptime(end_str, fmt)
            dur += end - start

        return dur


class Parameter(STARObject):

    class Type(STARAPIEnum):
        CONSTANT = 0
        CONTINUOUS = 1
        DISCRETE = 2

        def java_class_path(self) -> str:
            return "MdxStudyParameterBase.ParameterType"

        @staticmethod
        def names():
            return ["constant", "continuous", "discrete"]

        @staticmethod
        def from_str(s: str) -> Parameter.Type:
            s_lower = s.lower().strip()
            if s_lower not in Parameter.Type.names():
                raise ValueError(f"{s} is not a valid type. Must be one of {Parameter.Type.names}")
            if s_lower == Parameter.Type.names()[0]:
                return Parameter.Type.CONSTANT
            elif s_lower == Parameter.Type.names()[1]:
                return Parameter.Type.CONTINUOUS
            else:
                return Parameter.Type.DISCRETE

    ref_definitions = ["baseline", "boundary"]
    definition_modes = ["rel", "abs"]

    @staticmethod
    def get_loop_all_list_values(param_var_name: str, iterate_name: str = "val_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "Double"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{param_var_name}.getDiscreteParameterValue().getQuantity().getArray()"
        return loop

    @staticmethod
    def to_json(macro: sm.MacroWriter) -> sm.MacroMethod:
        param_class = "MdxStudyParameter"
        param_var = "param"
        args = {param_class: param_var}
        method = sm.MacroMethod(f"convertParamToJSON", args, "JsonObject")
        method.add_line("JsonArrayBuilder listArrayBuilder = Json.createArrayBuilder()")
        list_vals_loop = Parameter.get_loop_all_list_values(param_var)
        list_vals_loop.new_lines = False
        list_var = list_vals_loop.for_all_loop_iterate_name
        declare_val_json = f"JsonObject listValJSON = Json.createObjectBuilder()"
        list_vals_loop.add_line(f"{declare_val_json}", line_type=LT.START)
        list_vals_loop.add_line(f".add({t_val.w}, {list_var})", line_type=LT.MIDDLE)
        list_vals_loop.add_line(f".build()", line_type=sm.LineType.END)
        list_vals_loop.add_line(f"listArrayBuilder.add(listValJSON)")
        method.add_component(list_vals_loop)
        declare_param_json = f"JsonObject paramJSON = Json.createObjectBuilder()"
        param_body = sm.BodyComponent()
        param_body.add_line(f"{declare_param_json}", line_type=LT.START)
        param_body.add_line(f".add({t_name.w}, param.getPresentationName())", line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_id.w}, param.getObjectId())", line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_list_vals.w}, listArrayBuilder)", line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_baseline.w}, param.getBaselineQuantity().getRawValue())", line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_min.w}, param.getMinimumQuantity().getRawValue())", line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_max.w}, param.getMaximumQuantity().getRawValue())", line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_res.w}, param.getContinuousParameterValue().getResolution())",
                            line_type=LT.MIDDLE)
        param_body.add_line(f".add({t_type.w}, param.getParameterType().toString())", line_type=LT.MIDDLE)
        param_body.add_line(f".build()", line_type=LT.END)
        param_body.add_line(f"return paramJSON")
        param_body.new_lines = False
        method.add_component(param_body)
        method.call_in_execute = False
        macro.add_method(method)
        return method

    @staticmethod
    def from_json(parameter_dict: dict, study: Study) -> Parameter:
        oid = STARObjectID(parameter_dict[t_id.r], "star.mdx.MdxStudyParameter")
        oid.aux_java_classes.append("star.mdx.MdxStudyParameterBase")
        vals = list()
        for val in parameter_dict[t_list_vals.r]:
            vals.append(val["value"])
        t = Parameter.Type.from_str(parameter_dict[t_type.r])
        param = Parameter(oid, study, parameter_dict[t_name.r], parameter_dict[t_baseline.r], parameter_dict[t_min.r],
                          parameter_dict[t_max.r], parameter_dict[t_res.r], t, vals)
        return param

    def __init__(self,
                 obj_id: STARObjectID,
                 study: Study,
                 name: str,
                 baseline: float,
                 min_val: float,
                 max_val: float,
                 resolution: int,
                 param_type: Parameter.Type,
                 list_values=None):
        recorder = study.project.recorder
        super().__init__(recorder=recorder, obj_id=obj_id)
        if list_values is None:
            list_values = list()
        self.study = study
        self.name = name
        self.baseline = baseline
        self.min = min_val
        self.max = max_val
        self.resolution = resolution
        self.type = param_type
        self.list_values = list_values
        self._populate_api_hooks()

    def __iter__(self):
        return iter(self.list_values)

    def set_baseline(self, val: float, record: bool = True):
        is_recording = self.recorder.record_all
        self.recorder.record_all = record
        self.baseline = val
        self.recorder.record_all = is_recording

    def set_resolution(self, val: float, record: bool = True):
        is_recording = self.recorder.record_all
        self.recorder.record_all = record
        self.resolution = val
        self.recorder.record_all = is_recording

    def set_type(self, val: Parameter.Type, record: bool = True):
        is_recording = self.recorder.record_all
        self.recorder.record_all = record
        self.type = val
        self.recorder.record_all = is_recording

    def set_min(self, val: float, record: bool = True):
        is_recording = self.recorder.record_all
        self.recorder.record_all = record
        self.min = val
        self.recorder.record_all = is_recording

    def set_max(self, val: float, record: bool = True):
        is_recording = self.recorder.record_all
        self.recorder.record_all = record
        self.max = val
        self.recorder.record_all = is_recording

    def set_list_values(self, vals: list[float], record: bool = True):
        is_recording = self.recorder.record_all
        self.recorder.record_all = record
        self.list_values = vals
        self.recorder.record_all = is_recording

    def set_ranges(self,
                   ref: str = "baseline",
                   mode: str = "rel",
                   u_bnd: float = 5.0,
                   l_bnd: float = 5.0,
                   sig_fig: int = None,
                   record: bool = True):
        if ref not in self.ref_definitions:
            raise ValueError(f"ref value {ref} is not recognized. ref must be one of {self.ref_definitions}")
        if mode not in self.definition_modes:
            raise ValueError(f"mode value {mode} is not recognized. mode must be one of {self.definition_modes}")

        u_bnd, l_bnd = self._get_new_bounds(ref, mode, u_bnd, l_bnd, sig_fig)

        recording = self.recorder.record_all
        self.recorder.record_all = record
        self.max = u_bnd
        self.min = l_bnd
        self.recorder.record_all = recording

    def _get_new_bounds(self, ref: str, mode: str, u_bnd: float, l_bnd: float, sig_fig: int):
        if ref == "baseline":
            if mode == "rel":
                u_bnd, l_bnd = self._baseline_rel_bounds(u_bnd, l_bnd)
            else:
                u_bnd, l_bnd = self._baseline_abs_bounds(u_bnd, l_bnd)
        else:
            if mode == "rel":
                u_bnd, l_bnd = self._boundary_rel_bounds(u_bnd, l_bnd)
            else:
                u_bnd, l_bnd = self._boundary_abs_bounds(u_bnd, l_bnd)

        if sig_fig is not None:
            u_bnd = m_util.ceiling(u_bnd, sig_fig)
            l_bnd = m_util.floor(l_bnd, sig_fig)

        return u_bnd, l_bnd

    def _baseline_rel_bounds(self, u_bnd: float, l_bnd: float) -> (float, float):
        if u_bnd < 0.0:
            raise ValueError(f"Value {u_bnd} is less than 0.0. u_bnd must be greater than 0.0.")
        if l_bnd > 0.0:
            raise ValueError(f"Value {l_bnd} is greater than 0.0. l_bnd must be less than 0.0.")
        u_bnd /= 100.0
        l_bnd /= 100.0
        u_bnd = abs(self.baseline * u_bnd)
        u_bnd = self.baseline + u_bnd
        l_bnd = abs(self.baseline * l_bnd)
        l_bnd = self.baseline - l_bnd
        return u_bnd, l_bnd

    def _baseline_abs_bounds(self, u_bnd: float, l_bnd: float) -> (float, float):
        if u_bnd < 0.0:
            raise ValueError(f"Value {u_bnd} is less than 0.0. u_bnd must be greater than 0.0.")
        if l_bnd > 0.0:
            raise ValueError(f"Value {l_bnd} is less than 0.0. l_bnd must be greater than 0.0.")
        u_bnd = self.baseline + u_bnd
        l_bnd = self.baseline + l_bnd
        return u_bnd, l_bnd

    def _boundary_rel_bounds(self, u_bnd: float, l_bnd: float) -> (float, float):
        u_bnd /= 100.0
        l_bnd /= 100.0
        u_bnd = self.max + abs(self.max) * u_bnd
        l_bnd = self.min + abs(self.min) * l_bnd
        if self.baseline > u_bnd:
            raise ValueError(f"Computed upper bound {u_bnd} is less than baseline {self.baseline}.")
        if self.baseline < l_bnd:
            raise ValueError(f"Computed lower bound {l_bnd} is greater than baseline {self.baseline}.")
        return u_bnd, l_bnd

    def _boundary_abs_bounds(self, u_bnd: float, l_bnd: float) -> (float, float):
        u_bnd = self.max + u_bnd
        l_bnd = self.min + l_bnd
        if self.baseline > u_bnd:
            raise ValueError(f"Computed upper bound {u_bnd} is less than baseline {self.baseline}.")
        if self.baseline < l_bnd:
            raise ValueError(f"Computed lower bound {l_bnd} is greater than baseline {self.baseline}.")
        return u_bnd, l_bnd

    def _populate_api_hooks(self):
        global _api_props
        self._name = StarAPIProperty(value=self.name,
                                     name="PresentationName",
                                     setter_comm="setPresentationName",
                                     getter_comm="getPresentationName",
                                     obj_id=self.id)
        self._baseline = StarAPIProperty(value=self.baseline,
                                         name="Baseline",
                                         setter_comm="getBaselineQuantity().setValue",
                                         getter_comm="getBaselineQuantity().getRawValue",
                                         obj_id=self.id)
        self._max = StarAPIProperty(value=self.max,
                                    name="Maximum",
                                    setter_comm="getMaximumQuantity().setValue",
                                    getter_comm="getMaximumQuantity().getRawValue",
                                    obj_id=self.id)
        self._min = StarAPIProperty(value=self.min,
                                    name="Minimum",
                                    setter_comm="getMinimumQuantity().setValue",
                                    getter_comm="getMinimumQuantity().getRawValue",
                                    obj_id=self.id)
        self._resolution = StarAPIProperty(value=self.resolution,
                                           name="Resolution",
                                           setter_comm="getContinuousParameterValue().setResolution",
                                           getter_comm="getContinuousParameterValue().getResolution",
                                           obj_id=self.id)
        self._type = StarAPIProperty(value=self.type,
                                     name="ParameterType",
                                     setter_comm="setParameterType",
                                     getter_comm="getParameterType",
                                     obj_id=self.id)
        self._list_values = StarAPIProperty(value=self.list_values,
                                            name="DiscreteValues",
                                            setter_comm="getDiscreteParameterValue().getQuantity().setArray",
                                            getter_comm="getDiscreteParameterValue().getQuantity().getArray",
                                            obj_id=self.id)
        _api_props.extend([self._name, self._baseline, self._min, self._max, self._resolution, self._type,
                           self._list_values])

    def __repr__(self):
        keys = ["name", "baseline", "min", "max", "list_values"]
        s = ""
        for k in keys:
            s += f"{k}: {self.__dict__[k]}\n"
        return s


class DesignSet:

    @staticmethod
    def loop_all_designs(design_set_var_name: str, iterate_name: str = "design_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesign"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{design_set_var_name}.getDesigns()"
        return loop

    @staticmethod
    def to_json(macro: sm.MacroWriter) -> sm.MacroMethod:
        ds_class = "MdxDesignSet"
        ds_var = "designSet"
        args = {ds_class: ds_var}
        proj_var = sm.MacroWriter.get_root_obj_var_name(True)
        method = sm.MacroMethod(f"convertDesignSetToJSON", args, "JsonObject")
        method.packages = {"java.io.IOException", "java.nio.file.Files", "java.nio.file.Paths"}
        method.call_in_execute = False
        method.add_line("JsonArrayBuilder designArrayBuilder = Json.createArrayBuilder()")

        design_loop = DesignSet.loop_all_designs(ds_var)
        it_name = design_loop.for_all_loop_iterate_name
        design_method = Design.to_json(macro)
        design_loop.add_line(f"JsonObject designJSON = {design_method.name}({it_name})")
        design_loop.add_line("designArrayBuilder.add(designJSON)")
        method.add_component(design_loop)

        body = sm.BodyComponent()
        body.add_line("String fs = File.separator")
        body.add_line(f"String projName = {proj_var}.getPresentationName()")
        body.add_line(f"String studyName = {ds_var}.getDesignStudy().getPresentationName()")
        body.add_line(f"String hidden = {proj_var}.getSessionDir() + fs + \"._ds_exp\"")
        body.add_line(f"String csvFile = \"\"")
        method.add_component(body)
        catch_comp = sm.CatchComponent(except_cls="IOException")
        catch_comp.add_line("print(ex)")
        try_comp = sm.TryComponent(catch_comp)
        try_comp.add_line(f"Files.createDirectories(Paths.get(hidden))")
        try_comp.add_line(f"csvFile = hidden + fs + projName + \"_\" + studyName + \"_AllDesigns.csv\"")
        method.add_component(try_comp)

        if_all = sm.IfComponent(f"{ds_var}.getPresentationName().equals(\"All\")")
        export_body = sm.BodyComponent()
        export_body.add_line(f"{ds_var}.exportCsvFile(csvFile)")
        export_body.new_lines = False
        if_all.add_component(export_body)
        method.add_component(if_all)

        body = sm.BodyComponent()
        declare_ds_json = f"JsonObject designSetJSON = Json.createObjectBuilder()"
        body.add_line(f"{declare_ds_json}", line_type=LT.START)
        body.add_line(f".add({t_name.w}, {ds_var}.getPresentationName())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_id.w}, {ds_var}.getObjectId())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_path.w}, csvFile)", line_type=LT.MIDDLE)
        body.add_line(f".add({t_designs.w}, designArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(f".build()", line_type=LT.END)
        body.add_line(f"return designSetJSON")
        method.add_component(body)

        macro.add_method(method)

        return method

    @staticmethod
    def from_json(design_set_dict: dict, study: Study) -> DesignSet:
        oid = STARObjectID(design_set_dict[t_id.r], "star.mdx.MdxDesignSet")
        if design_set_dict[t_name.r] == "All":
            design_set = AllDesignsSet(oid, study, design_set_dict[t_name.r], design_set_dict[t_path.r])
        else:
            design_set = DesignSet(oid, study, design_set_dict[t_name.r])
        for design_dict in design_set_dict[t_designs.r]:
            design_set.designs.append(Design.from_json(design_dict, design_set))
        return design_set

    def __init__(self,
                 obj_id: STARObjectID,
                 study: Study,
                 name: str):
        self.study = study
        self.name = name
        self.id = obj_id
        self.designs = list()

    def __iter__(self):
        return iter(self.designs)

    def get_design(self, name: str) -> Design:
        for design in self.designs:
            if design.name == name:
                return design
        raise AttributeError(f"No design {name} found.")

    def data_frame(self) -> pd.DataFrame:
        all_ds = self.study.get_all_designs().data
        keep = self.included_designs()
        name_series = all_ds["Name"]
        in_mask = [True if val in keep else False for val in name_series]
        return all_ds[in_mask]

    def included_designs(self) -> List[str]:
        names = []
        for d in self.designs:
            names.append(f'\"{d.name}\"')
        return names

    def generate_history(self, col_name: str = "Performance", max_designs: int = -1) -> ArrayLike:
        total_designs = len(self.study.get_all_designs().data)
        columns = ["Design#", col_name]
        ds_data = self.data_frame()[columns].to_numpy()
        series_data = np.zeros((total_designs, 2))
        series_data[:, 0] = np.arange(1, total_designs + 1)
        for i in range(1, len(ds_data[:, 0])):
            lo = int(ds_data[i-1, 0]) - 1
            hi = int(ds_data[i, 0] - 1)
            series_data[lo:hi, 1] = ds_data[i - 1, 1]
        lo = int(ds_data[-1, 0]) - 1
        series_data[lo:total_designs, 1] = ds_data[-1, 1]
        counter = len(series_data[:, 0]) + 1
        while counter <= max_designs:
            repeat = np.array([[counter, series_data[-1, 1]]])
            series_data = np.append(series_data, repeat, axis=0)
            counter += 1

        return series_data

    def animate_design_history(self,
                               clean_first: bool = True,
                               y_col: str = "Performance",
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
                               **kwargs) -> Union[None, Path]:
        if len(self.designs) < 1:
            return
        output_dir = self.designs[0].path.parent
        directory_name = f"{self.name}_" + y_col.replace(" ", "_") + "_history"
        output_dir = output_dir.joinpath(directory_name)
        if not output_dir.exists():
            output_dir.mkdir()
        elif clean_first:
            shutil.rmtree(output_dir)
            output_dir.mkdir()

        viz.animate_history(output_dir=output_dir, df=self.data_frame(), y_col=y_col, x_col=x_col, perf=perf,
                            best_design_hist=best_design_hist, scatter=scatter, min_best=min_best,
                            highlight=highlight, render=render, sort_by_y_col=sort_by_y_col,
                            infeasible_col=infeasible_col, infeasible_color=infeasible_color, template=template,
                            **kwargs)
        return output_dir

    def prep_viz_files_for_animation(self,
                                     file_names: [str] = None,
                                     int_num_chars: int = 5,
                                     clean_first: bool = False) -> Union[None, Path]:
        if len(self.designs) == 0:
            return Path(self.study.project.path.parent)
        ds_anim_path = self.designs[0].path.parent.joinpath(f"{self.name}Animation")
        if not ds_anim_path.exists():
            ds_anim_path.mkdir()
        elif clean_first:
            shutil.rmtree(ds_anim_path)
            ds_anim_path.mkdir()

        prep_all = True if file_names is None else False
        for design in self:
            for v_file in design:
                v_f_name = v_file.stem
                v_f_p_name = v_f_name.replace(" ", "_")
                if prep_all or v_file.name in file_names:
                    v_file_anim_dir = ds_anim_path.joinpath(v_f_p_name)
                    if not v_file_anim_dir.exists():
                        v_file_anim_dir.mkdir()
                    design_num = design.get_design_number()
                    fmt = f"0{int_num_chars}d"
                    design_num_str = f"{design_num:{fmt}}"
                    target_path = v_file_anim_dir.joinpath(f"img_{design_num_str}{v_file.suffix}")
                    shutil.copy2(v_file, target_path)

        return ds_anim_path


class AllDesignsSet(DesignSet):

    def __init__(self, obj_id: STARObjectID, study: Study, name: str, csv_data: Union[str, Path] = None):
        DesignSet.__init__(self, obj_id, study, name)
        if isinstance(csv_data, str):
            csv_data = Path(csv_data)
        if csv_data is not None:
            df = util.clean_csv_for_df(csv_data, sort_col="")
            self.data = df
        else:
            self.data = pd.DataFrame()

    def data_frame(self) -> pd.DataFrame:
        return self.data


class Design:

    @staticmethod
    def loop_all_scene_files(design_var_name: str, iterate_name: str = "scene_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesignScene"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{design_var_name}.getDesignScenes().getObjects()"
        return loop

    @staticmethod
    def loop_all_plot_files(design_var_name: str, iterate_name: str = "plot_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesignPlot"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{design_var_name}.getDesignPlots().getObjects()"
        return loop

    @staticmethod
    def loop_all_params(design_var_name: str, iterate_name: str = "param_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesignParameter"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{design_var_name}.getDesignParameters().getObjectsOf(MdxDesignParameter.class)"
        return loop

    @staticmethod
    def loop_all_reports(design_var_name: str, iterate_name: str = "report_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "MdxDesignReport"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{design_var_name}.getDesignReports().getObjects()"
        return loop

    @staticmethod
    def loop_all_log_messages(design_var_name: str, iterate_name: str = "pair_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.packages = {"star.common.Pair",
                         "star.mdx.MdxLogMessagesSeverityEnum"}
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "Pair<MdxLogMessagesSeverityEnum, String>"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{design_var_name}.getLogMessages()" \
                                     f".getMessages({design_var_name}.getDesignManager()" \
                                     f".getDesignStudy().getDesignStudies().getLogContexts(), false)"
        return loop

    @staticmethod
    def to_json(macro: sm.MacroWriter) -> sm.MacroMethod:
        d_class = "MdxDesign"
        d_var = "design"
        args = {d_class: d_var}
        method = sm.MacroMethod(f"convertDesignToJSON", args, "JsonObject")
        method.call_in_execute = False
        method.packages = {"star.mdx.MdxDesign",
                           "star.mdx.MdxVisFormatEnum",
                           "java.io.File",
                           "star.mdx.MdxDesignScene",
                           "star.mdx.MdxDesignPlot",
                           "star.mdx.MdxDesignParameter",
                           "star.mdx.MdxDesignReport"
                           }
        method.add_line("JsonArrayBuilder designVisArrayBuilder = Json.createArrayBuilder()")
        method.add_line("JsonArrayBuilder designParamArrayBuilder = Json.createArrayBuilder()")
        method.add_line("JsonArrayBuilder designReportArrayBuilder = Json.createArrayBuilder()")
        method.add_line("JsonArrayBuilder logMessageArrayBuilder = Json.createArrayBuilder()")
        method.add_newline()

        scene_loop = Design.loop_all_scene_files(d_var)
        it_name = scene_loop.for_all_loop_iterate_name
        if_png = sm.IfComponent(f"{it_name}.getSceneFormat() == MdxVisFormatEnum.HARDCOPY")
        if_body = sm.BodyComponent()
        if_body.add_line(f"File f = new File({it_name}.getFileName())")
        if_body.add_line(f"JsonObject designSceneJSON = Json.createObjectBuilder()", line_type=LT.START)
        if_body.add_line(f".add({t_name.w}, f.getName())", line_type=LT.MIDDLE)
        if_body.add_line(f".build()", line_type=LT.END)
        if_body.add_line(f"designVisArrayBuilder.add(designSceneJSON)")
        if_png.add_component(if_body)
        scene_loop.add_component(if_png)
        scene_loop.new_lines = False
        method.add_component(scene_loop)

        plot_loop = Design.loop_all_plot_files(d_var)
        it_name = plot_loop.for_all_loop_iterate_name
        if_png = sm.IfComponent(f"{it_name}.getPlotFormat() == MdxVisFormatEnum.HARDCOPY")
        if_body = sm.BodyComponent()
        if_body.add_line(f"File f = new File({it_name}.getFileName())")
        if_body.add_line(f"JsonObject designSceneJSON = Json.createObjectBuilder()", line_type=LT.START)
        if_body.add_line(f".add({t_name.w}, f.getName())", line_type=LT.MIDDLE)
        if_body.add_line(f".build()", line_type=LT.END)
        if_body.add_line(f"designVisArrayBuilder.add(designSceneJSON)")
        if_png.add_component(if_body)
        plot_loop.add_component(if_png)
        plot_loop.new_lines = False
        method.add_component(plot_loop)

        messages_loop = Design.loop_all_log_messages(d_var)
        it_name = messages_loop.for_all_loop_iterate_name
        message_body = sm.BodyComponent()
        message_body.add_line(f"JsonObject messageJSON = Json.createObjectBuilder()", line_type=LT.START)
        message_body.add_line(f".add({t_log_m.w}, {it_name}.getRight().toString())", line_type=LT.MIDDLE)
        message_body.add_line(f".build()", line_type=LT.END)
        message_body.add_line(f"logMessageArrayBuilder.add(messageJSON)")
        message_body.new_lines = False
        messages_loop.add_component(message_body)
        method.add_component(messages_loop)

        body = sm.BodyComponent()
        body.add_line("JsonObject designJSON = Json.createObjectBuilder()", line_type=LT.START)
        body.add_line(f".add({t_name.w}, {d_var}.getPresentationName())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_dir.w}, {d_var}.getDirPath())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_log.w}, {d_var}.getLogFileName())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_vis_files.w}, designVisArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(f".add({t_messages.w}, logMessageArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(".build()", line_type=LT.END)
        body.add_line("return designJSON")
        body.new_lines = False
        method.add_component(body)

        macro.add_method(method)

        return method

    @staticmethod
    def from_json(design_dict: dict, design_set: DesignSet) -> Design:
        design = Design(design_dict[t_name.r], design_set, design_dict[t_dir.r], design_dict[t_log.r])
        v_files = [v for v in design_dict[t_vis_files.r]]
        design.viz_files = v_files
        messages = [m[t_log_m.r] for m in design_dict[t_messages.r]]
        design.log_messages = messages
        return design

    def __init__(self, name: str, design_set: DesignSet, path: Union[str, Path], log_file: str = None):
        if isinstance(path, str):
            path = Path(path)
        self.name = name
        self.design_set = design_set
        self.path = path
        if log_file is None:
            log_file = ""
        self.log_file = log_file
        self.viz_files = list()
        self.log_messages = list()

    def __iter__(self):
        paths = [self.path.joinpath(f["name"]) for f in self.viz_files]
        return iter(paths)

    def data_frame(self) -> pd.DataFrame:
        all_data = self.design_set.study.get_all_designs().data_frame()
        names = all_data["Name"]
        names = [name for name in names]
        name_filter = [True if name == f'"{self.name}"' else False for name in names]
        df = all_data[name_filter]
        return df

    def get_design_number(self) -> int:
        return self.data_frame()["Design#"].iloc[0]

    def runtime(self):
        fmt = "%a %b  %d %H:%M:%S %Y"
        start_message = None
        end_message = None
        for message in self.log_messages:
            if message.startswith("Begins at"):
                start_message = message
            if message.startswith("Ends at"):
                end_message = message

        if start_message is not None and end_message is not None:
            start_str = start_message.replace("Begins at ", "")
            end_str = end_message.replace("Ends at ", "")
            start = datetime.datetime.strptime(start_str, fmt)
            end = datetime.datetime.strptime(end_str, fmt)
            dur = end - start
        else:
            dur = datetime.timedelta()

        return dur


def example_set_study_baseline_values():
    workdir = Path(r"D:\Workdir\projects\2023\transonic_airfoil\svd")
    proj = DesignManagerProject.from_json(Path.joinpath(workdir, "svd_proj.json"))
    study = proj.get_study("Sherpa_B")
    macro = proj.empty_macro("Test")
    baseline_df = pd.read_csv(Path.joinpath(workdir, "baseline_designs.csv"))
    header = list(baseline_df.columns)
    data = []
    for idx, row in baseline_df.iterrows():
        if row[header[0]] == "CRM":
            for param in header[1:]:
                name = param
                baseline = row[param]
                data.append({"name": name, "baseline": baseline})
    m = study.update_parameter_baseline_vals(macro, data)
    macro.add_method(m, order=0)
    macro.build(workdir)


if __name__ == "__main__":
    test = Path(r"C:\Users\cd8unu\Desktop\Demo\python_api\quickMan.dmprj")
    dmp = DesignManagerProject(test)
    dmp.sync(push_to_star=False, recorded_only=True, delete_json=False, delete_macro=True)
