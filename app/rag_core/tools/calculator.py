import ast
import operator
import re


class Calculator:

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def calculate(self, expression: str):

        try:

            expression = self.extract_expression(
                expression
            )

            if not expression:
                return None

            tree = ast.parse(
                expression,
                mode="eval"
            )

            result = self._evaluate(
                tree.body
            )

            return result

        except Exception:

            return None

    # -----------------------------------------
    # Convert natural language to expression
    # -----------------------------------------

    def extract_expression(self, text: str):

        text = text.lower().strip()

        # -------------------------------------
        # Addition
        # -------------------------------------

        match = re.search(
            r"addition of\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)",
            text
        )

        if match:

            return (
                f"{match.group(1)}+{match.group(2)}"
            )

        # -------------------------------------
        # Subtraction
        # -------------------------------------

        match = re.search(
            r"subtraction of\s+(-?\d+(?:\.\d+)?)\s+from\s+(-?\d+(?:\.\d+)?)",
            text
        )

        if match:

            return (
                f"{match.group(2)}-{match.group(1)}"
            )

        # -------------------------------------
        # Multiplication
        # -------------------------------------

        match = re.search(
            r"(?:multiplication of|product of)\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)",
            text
        )

        if match:

            return (
                f"{match.group(1)}*{match.group(2)}"
            )

        # -------------------------------------
        # Division
        # -------------------------------------

        match = re.search(
            r"(?:division of|quotient of)\s+(-?\d+(?:\.\d+)?)\s+(?:by|and)\s+(-?\d+(?:\.\d+)?)",
            text
        )

        if match:

            return (
                f"{match.group(1)}/{match.group(2)}"
            )

        # -------------------------------------
        # General natural-language operators
        # -------------------------------------

        replacements = {

            "what is": "",

            "calculate": "",

            "compute": "",

            "solve": "",

            "multiplied by": "*",

            "multiply by": "*",

            "times": "*",

            "divided by": "/",

            "divide by": "/",

            "plus": "+",

            "add": "+",

            "minus": "-",

            "subtract": "-",

            "modulo": "%",

            "mod": "%",

            "to the power of": "**",

            "power of": "**",
        }

        for phrase, replacement in replacements.items():

            text = text.replace(
                phrase,
                replacement
            )

        # -------------------------------------
        # Remove question mark
        # -------------------------------------

        text = text.replace("?", "")

        # -------------------------------------
        # Remove spaces
        # -------------------------------------

        text = text.replace(" ", "")

        # -------------------------------------
        # Percentage
        # -------------------------------------

        percentage_match = re.fullmatch(
            r"(\d+(?:\.\d+)?)%of(\d+(?:\.\d+)?)",
            text
        )

        if percentage_match:

            percentage = percentage_match.group(1)

            number = percentage_match.group(2)

            return (
                f"({percentage}/100)*{number}"
            )

        # -------------------------------------
        # Safety check
        # -------------------------------------

        if not re.fullmatch(
            r"[0-9+\-*/%.()]+",
            text
        ):

            return None

        return text

    # -----------------------------------------
    # Safely evaluate expression
    # -----------------------------------------

    def _evaluate(self, node):

        # Numbers
        if isinstance(node, ast.Constant):

            if isinstance(
                node.value,
                (int, float)
            ):

                return node.value

            raise ValueError(
                "Invalid number"
            )

        # Binary operations
        if isinstance(node, ast.BinOp):

            left = self._evaluate(
                node.left
            )

            right = self._evaluate(
                node.right
            )

            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Unsupported operator"
                )

            return operation(
                left,
                right
            )

        # Unary operations
        if isinstance(node, ast.UnaryOp):

            operand = self._evaluate(
                node.operand
            )

            operation = self.OPERATORS.get(
                type(node.op)
            )

            if operation is None:

                raise ValueError(
                    "Unsupported operator"
                )

            return operation(
                operand
            )

        raise ValueError(
            "Invalid mathematical expression"
        )