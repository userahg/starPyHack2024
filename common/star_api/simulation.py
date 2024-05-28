from __future__ import annotations
import common.util as util
import os
from subprocess import CompletedProcess
from common.star_versions import STARCCMInstall
from common.star_api.root_obj_utils import *
from common.starccm_macros import LineType as LT
from typing import Union
from pathlib import Path
import json


t_name = JSONTag("name")
t_class = JSONTag("class")
t_def = JSONTag("definition")
t_dim = JSONTag("dimension")
t_id = JSONTag("object_ID")
t_params = JSONTag("parameters")
_api_props = []


class Simulation:

    @staticmethod
    def get_loop_all_parameters(sim_var_name="_sim", iterate_name: str = "param_i") -> sm.LoopComponent:
        loop = sm.LoopComponent()
        loop.loop_type = sm.LoopType.FOR_ALL
        loop.for_all_loop_iterate_class = "GlobalParameterBase"
        loop.for_all_loop_iterate_name = iterate_name
        loop.for_all_loop_iterable = f"{sim_var_name}.get(GlobalParameterManager.class)." \
                                     f"getObjectsOf({loop.for_all_loop_iterate_class}.class)"
        return loop

    @classmethod
    def from_json(cls, json_file: Union[str, Path]):
        if isinstance(json_file, str):
            json_file = Path(json_file)

        with open(json_file) as f:
            file_contents = json.load(f)

        sim_path = Path.joinpath(json_file.parent, f'{file_contents[t_name.r]}.sim')
        if not sim_path.exists():
            raise FileNotFoundError(f"project {sim_path} not found")

        sim = Simulation(sim_path)
        sim._from_json(json_file)
        return sim

    def __init__(self, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        self.name = path.stem
        self.recorder = CommandRecorder(self.name, is_mdx=False)
        self.path = path
        self.parameters = list()
        self.port = None

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
                macro = macro.build(self.path.parent)
                result = self.recorder.play_macro(star_file=self.path, macro=macro, delete_macro=delete_macro)
                print(result.stdout)
                print(result.stderr)

        json_file = self._to_json(run=True)
        self._from_json(json_file, delete_json=delete_json)

    def clear_commands(self):
        self.recorder.clear_commands()

    def start(self, version: Union[str, Path] = None):
        starccm = get_star_install(version)
        port = util.launch_server(self.path, starccm)
        if port < 0:
            raise Exception(f"STAR-CCM+ process did not return a valid port number. port = {port}")
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
        p = {"star.common.GlobalParameterBase",
             "star.common.GlobalParameterManager",
             "java.io.BufferedWriter",
             "java.io.FileWriter",
             "java.util.Map",
             "java.util.HashMap",
             "javax.json.*",
             "javax.json.stream.JsonGenerator"
             }
        macro_name = f"ExportSimulationToJSON"
        macro = sm.MacroWriter(macro_name, p)
        workdir = self.path.parent
        macro_output = Path.joinpath(workdir, f"{self.name}.json")
        method = sm.MacroMethod("convertSimulationToJSON", dict())
        except_method = sm.get_print_ex_method()
        sim_var_name = macro.get_root_obj_var_name(False)
        method.add_line("JsonArrayBuilder parameterArrayBuilder = Json.createArrayBuilder()")
        param_loop = Simulation.get_loop_all_parameters()
        param_loop.new_lines = True
        loop_var = param_loop.for_all_loop_iterate_name
        param_json_method = Parameter.to_json(macro)
        param_loop.add_line(f"{param_json_method.return_type} paramJSON = {param_json_method.name}({loop_var})")
        param_loop.add_line(f"parameterArrayBuilder.add(paramJSON)")
        method.add_component(param_loop)

        write_body = sm.BodyComponent()
        declare_sim_json = f"JsonObject simJSON = Json.createObjectBuilder()"
        write_body.add_line(f"{declare_sim_json}", line_type=LT.START)
        write_body.add_line(f".add({t_name.w}, {sim_var_name}.getPresentationName())", line_type=LT.MIDDLE)
        write_body.add_line(f".add({t_params.w}, parameterArrayBuilder)", line_type=LT.MIDDLE)
        write_body.add_line(f".build()", line_type=LT.END)
        write_body.add_line("Map<String, Boolean> config = new HashMap<>()")
        write_body.add_line("config.put(JsonGenerator.PRETTY_PRINTING, true)")
        write_body.add_line("JsonWriterFactory writerFactory = Json.createWriterFactory(config)")
        method.add_component(write_body)

        catch_comp = sm.CatchComponent()
        catch_comp.add_line(f"{except_method.name}({catch_comp.except_var_name})")
        try_resource = sm.TryFileWriteResource(str(macro_output))
        try_comp = sm.TryComponent(catch_comp, resource=try_resource)
        try_comp.add_line("writerFactory.createWriter(writer).write(simJSON)")
        try_comp.new_lines = False
        method.add_component(try_comp)
        macro.add_method(method)
        macro.add_method(except_method)
        macro.build(workdir)

        if run:
            macro = macro.get_macro_path(workdir)
            self.recorder.play_macro(star_file=self.path, macro=macro, delete_macro=True)

        return macro_output

    def _from_json(self, json_file: Union[str, Path], delete_json: bool = False):
        if isinstance(json_file, str):
            json_file = Path(json_file)

        with open(json_file) as f:
            file_contents = json.load(f)

        self.parameters.clear()

        for param_dict in file_contents[t_params.r]:
            self.parameters.append(Parameter.from_json(param_dict, self))

        if delete_json:
            os.remove(json_file)

    def __setattr__(self, key, value):
        if key == "star_install":
            if isinstance(value, STARCCMInstall):
                super.__setattr__(self, key, value)
            else:
                print(f"Unable to assign value of type {type(value)} to star_install.\n"
                      f"Must be a STARInstall object.")
        else:
            super.__setattr__(self, key, value)


class Parameter(STARObject):

    @staticmethod
    def to_json(macro: sm.MacroWriter) -> sm.MacroMethod:
        java_class = "GlobalParameterBase"
        param_var = "param"
        args = {java_class: param_var}
        method = sm.MacroMethod("convertParamToJSON", args, "JsonObject")
        method.add_line("JsonArrayBuilder dimArrayBuilder = Json.createArrayBuilder()")
        dim_loop = sm.LoopComponent()
        dim_loop.loop_type = sm.LoopType.FOR_ALL
        dim_loop.for_all_loop_iterate_class = "Integer"
        dim_loop.for_all_loop_iterate_name = "i"
        dim_loop.for_all_loop_iterable = f"{param_var}.getDimensionsVector()"
        dim_loop.add_line("dimArrayBuilder.add(i)")
        method.add_component(dim_loop)
        body = sm.BodyComponent()
        body.add_line(f"JsonObject paramJSON = Json.createObjectBuilder()", line_type=LT.START)
        body.add_line(f".add({t_name.w}, {param_var}.getPresentationName())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_id.w}, {param_var}.getObjectId())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_class.w}, {param_var}.getClass().getName())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_def.w}, {param_var}.getQuantity().getDefinition())", line_type=LT.MIDDLE)
        body.add_line(f".add({t_dim.w}, dimArrayBuilder)", line_type=LT.MIDDLE)
        body.add_line(f".build()", line_type=LT.END)
        body.add_line(f"return paramJSON")
        method.add_component(body)
        method.call_in_execute = False
        macro.add_method(method)
        return method

    @staticmethod
    def from_json(param_dict: dict, sim: Simulation) -> Parameter:
        class_name = param_dict[t_class.r]
        oid = STARObjectID(param_dict[t_id.r], class_name, is_mdx=False)
        name = param_dict[t_name.r]
        definition = param_dict[t_def.r]
        dim = param_dict[t_dim.r]
        param = Parameter(oid, sim, name, definition, dim)
        return param

    def __init__(self, obj_id: STARObjectID, sim: Simulation, name: str, definition: str, dim: list[int]):
        super().__init__(sim.recorder, obj_id)
        self.sim = sim
        self.name = name
        self.definition = definition
        self.dim = dim
        self._populate_api_hooks()

    def set_definition(self, definition: str):
        self._definition.set_property(definition, record=True, macro=self.sim.recorder.macro, obj_id=self.id)
        self.definition = definition

    def set_dim(self, dimensions: list[int]):
        self._dim.set_property(dimensions, record=True, macro=self.sim.recorder.macro, obj_id=self.id)
        self.dim = dimensions

    def set_name(self, name: str):
        self._name.set_property(name, record=True, macro=self.sim.recorder.macro, obj_id=self.id)
        self.name = name

    def _populate_api_hooks(self):
        global _api_props
        self._name = StarAPIProperty(value=self.name,
                                     name="PresentationName",
                                     setter_comm="setPresentationName",
                                     getter_comm="getPresentationName",
                                     obj_id=self.id)
        self._definition = StarAPIProperty(value=self.definition,
                                           name="Definition",
                                           setter_comm="getQuantity().setDefinition",
                                           getter_comm="getQuantity().getDefinition",
                                           obj_id=self.id)
        self._dim = StarAPIProperty(value=self.dim,
                                    name="Dimension",
                                    setter_comm="setDimensionsVector",
                                    getter_comm="getDimensionsVector",
                                    obj_id=self.id)
        _api_props.extend([self._name, self._definition, self._dim])
