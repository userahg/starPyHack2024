from __future__ import annotations
from enum import Enum
from typing import Set, Dict, Union
from pathlib import Path


def tab(i) -> str:
    s = ''
    for _ in range(i):
        s += '    '
    return s


def format_line_ending(line: str) -> str:
    if not line.endswith('\n'):
        if not line.endswith(';'):
            line += ';\n'
        else:
            line += '\n'
    else:
        if not line.endswith(';\n'):
            line = line[:len(line) - 1] + ';\n'
    return line


class LoopType(Enum):
    FOR_I = 0
    FOR_ALL = 1
    WHILE = 2


class LineType(Enum):
    FULL = 0
    START = 1
    MIDDLE = 2
    END = 3


class TryFileWriteResource:

    def __init__(self, file: str):
        self.file = file
        self.res_class = "BufferedWriter"
        self.res_var_name = "writer"
        self.res_construct_start = "new BufferedWriter(new FileWriter(\""
        self.res_construct_end = "\"))"
        self.packages = {"java.io.BufferedWriter", "java.io.FileWriter"}

    def get_resource_text(self):
        s = self.file.replace("\\", "\\\\")
        return f"{self.res_class} {self.res_var_name} = {self.res_construct_start}{s}{self.res_construct_end}"


class MacroComponent:

    def __init__(self):
        self.body = []
        self.packages = set()
        self._components = dict()
        self.new_lines = True

    def add_line(self, line: str, line_type: LineType = LineType.FULL):
        if line_type == LineType.FULL or line_type == LineType.END:
            line = format_line_ending(line)
        if line_type == LineType.MIDDLE or line_type == LineType.END:
            line = tab(2) + line
        if line_type == LineType.START or line_type == LineType.MIDDLE:
            if not line.endswith("\n"):
                line += "\n"
        self.body.append(line)

    def add_newline(self):
        self.body.append('\n')

    def add_component(self, component: MacroComponent, order: int = -1):
        n_methods = len(self._components.keys())
        if order < 0:
            order = n_methods
            self._components[order] = component
        elif order > n_methods + 1:
            order = n_methods
            self._components[order] = component
        else:
            new_methods = {order: component}
            for o, m in self._components.items():
                if o >= order:
                    new_methods[o + 1] = m
                else:
                    new_methods[o] = m
            sorted_keys = [new_methods.keys()]
            sorted_keys.sort()
            self._components = {i: new_methods[i] for i in sorted_keys}
        self.packages.update(component.packages)

    def generate_body_text(self, ind: int = 1) -> str:
        body = ""
        for line in self.body:
            body += f"{tab(ind)}{line}"
        return body

    def generate_comp_text(self, ind: int = 1) -> str:
        comp_text = ""
        for k, v in sorted(self._components.items()):
            comp_text += v.build(ind)
            if v.new_lines:
                comp_text += "\n"
        return comp_text

    def build(self) -> str:
        pass

    def __str__(self):
        return self.build()

    def __repr__(self):
        return self.build()


class BodyComponent(MacroComponent):

    def __init__(self):
        MacroComponent.__init__(self)

    def build(self, ind: int = 1) -> str:
        text = self.generate_body_text(ind)
        text += self.generate_comp_text(ind)
        return text


class LoopComponent(MacroComponent):

    def __init__(self, loop_type: LoopType = LoopType.FOR_I, *,
                 i_name: str = 'i', i_start: int = 0, i_comparator: str = '<', i_end: int = 1, i_inc: int = 1,
                 iterate_class: str = 'int', iterate_name: str = 'i', iterable: str = 'new int[] {1, 2, 3}',
                 while_cond: str = 'i < 10'):
        MacroComponent.__init__(self)
        self.loop_type = loop_type
        self.i_loop_i_name = i_name
        self.i_loop_start = i_start
        self.i_loop_comparator = i_comparator
        self.i_loop_end = i_end
        self.i_loop_inc = i_inc
        self.for_all_loop_iterate_class = iterate_class
        self.for_all_loop_iterate_name = iterate_name
        self.for_all_loop_iterable = iterable
        self.while_loop_cond = while_cond

    def build(self, ind: int = 1) -> str:
        if self.loop_type == LoopType.FOR_I:
            return self._build_for_i(ind)
        if self.loop_type == LoopType.FOR_ALL:
            return self._build_for_for_all(ind)
        return self._build_for_while(ind)

    def _build_for_i(self, ind: int = 1):
        loop_text = f'{tab(ind)}for (int {self.i_loop_i_name} = {self.i_loop_start}; {self.i_loop_i_name} ' \
                    f'{self.i_loop_comparator} {self.i_loop_end}; {self.i_loop_i_name} += {self.i_loop_inc}) '
        loop_text += '{\n'

        ind += 1
        loop_text += self.generate_body_text(ind)
        loop_text += self.generate_comp_text(ind)
        ind -= 1

        loop_text += f'{tab(ind)}' + '}\n'

        return loop_text

    def _build_for_for_all(self, ind: int = 1):
        loop_text = f'{tab(ind)}for ({self.for_all_loop_iterate_class} {self.for_all_loop_iterate_name} ' \
                    f': {self.for_all_loop_iterable}) ' + '{\n'

        ind += 1
        loop_text += self.generate_body_text(ind)
        loop_text += self.generate_comp_text(ind)
        ind -= 1

        loop_text += f'{tab(ind)}' + '}\n'

        return loop_text

    def _build_for_while(self, ind: int = 1):
        loop_text = f'{tab(ind)}while ({self.while_loop_cond}) ' + '{\n'

        ind += 1
        loop_text += self.generate_body_text(ind)
        loop_text += self.generate_comp_text(ind)
        ind -= 1

        loop_text += f'{tab(ind)}' + '}\n'

        return loop_text


class IfComponent(MacroComponent):

    def __init__(self, predicate: str):
        MacroComponent.__init__(self)
        self.predicate = predicate

    def build(self, ind: int = 1) -> str:
        text = f"{tab(ind)}if ({self.predicate})" + " {\n"
        ind += 1
        text += self.generate_body_text(ind)
        text += self.generate_comp_text(ind)
        ind -= 1
        text += f"{tab(ind)}" + "}\n"
        return text


class CatchComponent(MacroComponent):

    def __init__(self, except_cls: str = "Exception", except_var_name: str = "ex"):
        MacroComponent.__init__(self)
        self.except_cls = except_cls
        self.except_var_name = except_var_name

    def build(self, ind: int = 1) -> str:
        text = tab(ind) + "} " + f"catch ({self.except_cls} {self.except_var_name}) " + "{\n"
        ind += 1
        text += self.generate_body_text(ind)
        text += self.generate_comp_text(ind)
        ind -= 1
        text += tab(ind) + "}\n"
        return text


class TryComponent(MacroComponent):

    def __init__(self, catch: CatchComponent, resource: TryFileWriteResource = None):
        MacroComponent.__init__(self)
        self.catch = catch
        self.resource = resource
        if resource:
            self.packages.update(resource.packages)

    def build(self, ind: int = 1) -> str:
        text = f"{tab(ind)}try"
        if self.resource:
            text += f" ({self.resource.get_resource_text()}) " + "{\n"
        else:
            text += " {\n"
        ind += 1
        text += self.generate_body_text(ind)
        text += self.generate_comp_text(ind)
        ind -= 1
        text += self.catch.build(ind)
        return text


class MacroMethod(MacroComponent):

    def __init__(self, name: str, arguments: Dict[str, str], return_type: str = 'void'):
        MacroComponent.__init__(self)
        self.name = name.replace('()', '')
        self.return_type = return_type
        self.arguments = arguments
        self.call_in_execute = True

    def build(self, ind: int = 1) -> str:
        method_text = ''
        method_text += f'{tab(ind)}private {self.return_type} {self.name}('
        n_args = len(self.arguments.keys())
        arg_idx = 1
        for arg_type, name in self.arguments.items():
            if arg_idx == n_args:
                method_text += f'{arg_type} {name}'
            else:
                method_text += f'{arg_type} {name},'
            arg_idx += 1

        method_text += ') {\n'

        ind += 1
        method_text += self.generate_body_text(ind)
        method_text += self.generate_comp_text(ind)
        ind -= 1
        method_text += tab(ind) + '}\n'

        return method_text


class MacroWriter:

    @classmethod
    def get_root_obj_var_name(cls, is_mdx: bool):
        return '_proj' if is_mdx else '_sim'

    def __init__(self, class_name: str, java_imports: Set[str] = None, is_mdx: bool = False):
        if java_imports is None:
            java_imports = set()
        self.class_name = class_name
        self.packages = java_imports
        self.class_vars = set()
        self.is_mdx = is_mdx
        self.save_root_obj = False
        self.macro_text = ''
        self._methods = dict()
        self._add_default_packages()

    def get_macro_path(self, dir_path: Union[str, Path]) -> Path:
        if isinstance(dir_path, str):
            macro_path = Path(dir_path)
        else:
            macro_path = dir_path

        if not macro_path.exists():
            raise FileNotFoundError(f'{dir_path} does not exist. path variable must reference an existing directory')

        if not macro_path.is_dir():
            parent = macro_path.parent.absolute()
            macro_path = parent.joinpath(f'{self.class_name}.java')
        else:
            macro_path = macro_path.joinpath(f'{self.class_name}.java')

        return macro_path

    def add_method(self, method: MacroMethod, order: int = -1):
        duplicate = False
        for m in self._methods.values():
            if method.name == m.name:
                duplicate = True
        if duplicate:
            return

        n_methods = len(self._methods.keys())
        if order < 0:
            order = n_methods
            self._methods[order] = method
        elif order > n_methods + 1:
            order = n_methods
            self._methods[order] = method
        else:
            new_methods = {order: method}
            for o, m in self._methods.items():
                if o >= order:
                    new_methods[o + 1] = m
                else:
                    new_methods[o] = m
            self._methods = new_methods
        self.packages.update(method.packages)

    def iter_methods(self):
        return iter(self._methods.values())

    def generate_method_name(self, name: str):
        search = name if "__" not in name else name.split("__")[0]
        count = 0
        for method in self._methods.values():
            if search in method.name:
                count += 1

        if count == 0:
            return name

        name = f"{search}__{count}"
        return name

    def build(self, directory: Union[Path, str]) -> Path:
        if isinstance(directory, str):
            directory = Path(directory)

        if self.save_root_obj:
            self.add_method(get_save_root_obj_method(is_mdx=self.is_mdx))
        macro_text = self._build()
        macro_path = self.get_macro_path(directory)

        with open(macro_path, 'w', newline='') as macro_file:
            macro_file.write(macro_text)

        return macro_path

    def _add_default_packages(self):
        if self.is_mdx:
            self.packages.add('star.mdx.MdxProject')
            self.packages.add('star.mdx.MdxMacro')
        else:
            self.packages.add('star.common.Simulation')
            self.packages.add('star.common.StarMacro')

    def _get_superclass(self):
        return 'MdxMacro' if self.is_mdx else 'StarMacro'

    def _get_root_obj_type(self):
        return 'MdxProject' if self.is_mdx else 'Simulation'

    def _get_root_obj_comm(self):
        return 'getActiveMdxProject()' if self.is_mdx else 'getActiveSimulation()'

    def _build(self) -> str:
        ct = 1
        macro_text = ''
        for package in self.packages:
            macro_text += f'import {package};\n'

        macro_text += '\n'
        macro_text += f'public class {self.class_name} extends {self._get_superclass()} ' + '{\n\n'
        macro_text += f'{tab(ct)}{self._get_root_obj_type()} {self.get_root_obj_var_name(is_mdx=self.is_mdx)};\n'
        for class_var in self.class_vars:
            macro_text += f'{tab(ct)}{class_var}\n'
        macro_text += f'\n'
        macro_text += f'{tab(ct)}@Override\n'
        macro_text += f'{tab(ct)}public void execute() ' + '{\n'
        ct += 1
        macro_text += f'{tab(ct)}{self.get_root_obj_var_name(is_mdx=self.is_mdx)} = {self._get_root_obj_comm()};\n'

        for method in sorted(self._methods):
            method = self._methods[method]
            if method.call_in_execute:
                if method.return_type != 'void':
                    macro_text += f'{tab(ct)}{method.return_type} {method.name}{method.return_type} =' \
                                  f' {method.name}('
                else:
                    macro_text += f'{tab(ct)}{method.name}('
                for arg in method.arguments.values():
                    macro_text += f'{arg},'
                if macro_text.endswith(','):
                    macro_text = macro_text[:len(macro_text) - 1]
                macro_text += f');\n'
        ct -= 1
        macro_text += f'{tab(ct)}' + '}\n\n'

        for method in sorted(self._methods):
            method = self._methods[method]
            macro_text += method.build(ct)
            macro_text += "\n"
        macro_text += '}'

        return macro_text

    def __str__(self):
        return self._build()

    def __repr__(self):
        return self._build()


def get_save_root_obj_method(path: Path = None, is_mdx: bool = False) -> MacroMethod:
    method_name = "saveProject" if is_mdx else "saveSimulation"
    file_ext = ".dmprj" if is_mdx else ".sim"
    root_obj_name = MacroWriter.get_root_obj_var_name(is_mdx)
    root_obj_tag = root_obj_name.replace("_", "")
    root_obj_path_name = f"{root_obj_tag}Path"
    if path is None:
        root_obj_path = f"{root_obj_name}.getSessionDir() + File.separator + " \
                        f"{root_obj_name}.getPresentationName() + \"{file_ext}\""
    else:
        root_obj_path = str(path)
    method = MacroMethod(method_name, dict())
    method.packages.add("java.io.File")
    method.add_line(f"String {root_obj_path_name} = {root_obj_path}")
    method.add_line(f"{root_obj_name}.saveState({root_obj_path_name})")
    return method


def get_print_ex_method(is_mdx: bool = False):
    args = {'Exception': 'ex'}
    method = MacroMethod('print', args)
    method.call_in_execute = False
    method.packages.add('java.io.PrintWriter')
    method.packages.add('java.io.StringWriter')
    method.add_line('StringWriter sw = new StringWriter()')
    method.add_line('PrintWriter pw = new PrintWriter(sw)')
    method.add_line('ex.printStackTrace(pw)')
    method.add_line(f'{MacroWriter.get_root_obj_var_name(is_mdx)}.println(sw)')
    return method


def get_run_studies_macro(studies: []) -> MacroWriter:
    macro = MacroWriter('RunStudies', {'star.mdx.MdxDesignStudy'}, is_mdx=True)
    run_studies = MacroMethod('runStudies', dict())
    count = 0
    for study in studies:
        if count == 0:
            run_studies.add_line(f'MdxDesignStudy study = _proj.getDesignStudyManager().getDesignStudy("{study}")')
            run_studies.add_line('study.runDesignStudy()')
        else:
            run_studies.add_line(f'study = _proj.getDesignStudyManager().getDesignStudy("{study}")')
            run_studies.add_line('study.runDesignStudy()')
        count += 1
    macro.add_method(run_studies)
    return macro


def hello_world_macro():
    packages = {'star.common.Simulation', 'star.common.StarMacro'}
    macro = MacroWriter('SimulationHelloWorld', packages)
    hello_world = MacroMethod('helloWorld', dict())
    hello_world.add_line(f'{macro.get_root_obj_var_name(is_mdx=False)}' + r'.println("Hello World!!!");')
    macro.add_method(hello_world)
    macro.build(r'D:\PycharmProjects\siemens\common')


def hello_world_macro_loop():
    packages = {'star.common.Simulation', 'star.common.StarMacro'}
    macro = MacroWriter('SimulationHelloWorldLoop', packages)
    hello_world = MacroMethod('helloWorld', dict())
    hello_world.add_line('int i = 0')
    loop = LoopComponent()
    loop.loop_type = LoopType.WHILE
    loop.while_loop_cond = 'i < 3'
    loop.add_line(f'{macro.get_root_obj_var_name(is_mdx=False)}.println("Hello World " + '
                  f'Integer.toString(i) + "!")')
    loop.add_line(f'i++')
    hello_world.add_component(loop)
    macro.add_method(hello_world)
    macro.build(r'D:\PycharmProjects\siemens\common')


def hello_world_with_args():
    packages = {'star.common.Simulation', 'star.common.StarMacro'}
    macro = MacroWriter('SimulationHelloWorldWithArgs', packages)
    args = {'Simulation': macro.get_root_obj_var_name(is_mdx=False)}
    hello_world = MacroMethod('helloWorld', args)
    hello_world.add_line(f'{macro.get_root_obj_var_name(is_mdx=False)}' + r'.println("Hello World!!!");')
    macro.add_method(hello_world)
    macro.build(r'D:\PycharmProjects\siemens\common')


def macro_with_method_return():
    packages = {'star.common.Simulation', 'star.common.StarMacro'}
    macro = MacroWriter('PrintSimulationPresentationName', packages)
    args = {'Simulation': macro.get_root_obj_var_name(is_mdx=False)}
    presentation_name = MacroMethod('helloWorld', args, return_type='String')
    presentation_name.add_line(f'String name = {macro.get_root_obj_var_name(is_mdx=False)}.getPresentationName()')
    presentation_name.add_line(f'return name')
    macro.add_method(presentation_name)
    macro.build(r'D:\PycharmProjects\siemens\common')


if __name__ == '__main__':
    hello_world_macro_loop()
