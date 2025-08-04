import os
import argparse

import dill as pickle
from component import CustomComponent


def is_float(expr: str):
    try:
        float(expr)
        return True
    except ValueError:
        return False


def is_bool(expr):
    return expr.lower() in ["true", "false"]


class Generator:
    def __init__(self, formula):
        self.formula = formula

    def extract_functions(self, formula):
        function = ""
        args = []
        current_arg = ""
        count = 0
        for c in formula:
            if c == "(":
                count += 1
                if count == 1:
                    continue
            elif c == ")":
                count -= 1

            if count == 0 and c == ")" or count == 1 and c == ",":
                current_arg = current_arg.replace(" ", "")
                args.append(current_arg)
                current_arg = ""
                continue

            if count == 0:
                if c != ")":
                    function += c
            else:
                current_arg += c

        return function, args

    def get_function_code(self, function, args):
        operators = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "lt": "<",
            "gt": ">",
            "and_": "&&",
            "or_": "||",
        }
        if function in operators:
            return f"({args[0]}) {operators[function]} ({args[1]})"

        if function == "div":
            return f"({args[1]}) == 0 ? 1 : ({args[0]})/({args[1]})"

        if function == "if_then_else":
            return f"{args[0]} ? {args[1]} : {args[2]}"

        if function == "not":
            return f"!({args[0]})"

        if function == "neg":
            return f"-({args[0]})"

        if function == "round":
            return f"round({args[0]})"

        return f"std::{function}({args[0]}, {args[1]})"



    def parse(self, expr: str):

        if expr.startswith("ARG"):
            index = expr[3:]
            code = f"features[{index}]"
        elif is_float(expr):
            code = f"static_cast<double>({expr})"
        elif  is_bool(expr):
            code = expr.lower()
        else:
            function, args = self.extract_functions(expr)
            args = [self.parse(arg) for arg in args]
            code = self.get_function_code(function, args)

        return code

    def generate_file(self, name, path, template_path="observer_generation/template.cpp"):
        with open(template_path, "r") as f:
            content = f.read()

        content = content.replace("template_name", name)

        code = self.parse(self.formula)
        content = content.replace("#define FORMULA 0", f"#define FORMULA {code}")

        content = content.replace("#define FORMULA_STR \"\"", f"#define FORMULA_STR \"{self.formula}\"")

        p = os.path.join(path, f"{name}.cpp")
        with open(p, "w") as f:
            f.write(content)


if __name__ == "__main__":
    from utils import create_tool_box

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--individual", help="Individual file", type=str, required=True)
    parser.add_argument("-o", "--out", help="Output directory", type=str, required=True)
    args = parser.parse_args()

    observer = CustomComponent()
    toolbox, pset = create_tool_box(observer=observer)
    individial = pickle.load(open(args.individual, "rb"))
    print(individial)
    gen = Generator(str(individial))

    current_dir = os.getcwd()
    template_dir = os.path.join(os.path.abspath(os.path.join(current_dir, os.pardir)), "observer_generation", "template.cpp")


    _, name = os.path.split(args.individual)

    gen.generate_file(name, args.out, template_dir)