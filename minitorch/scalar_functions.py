from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Compute the backward pass of the function.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_out (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the inputs.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Compute the forward pass of the function.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            *inps (float): The input values.

        Returns:
        -------
            float: The result of the function.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of ScalarLike values.

        Args:
        ----
            *vals (ScalarLike): The input values, which can be either Scalar or a numerical value.

        Returns:
        -------
            Scalar: The result of the function applied to the inputs.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for addition.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the addition.

        """
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for addition.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the logarithm.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the logarithm.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the logarithm.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiply function: f(x, y) = x*y"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the inputs.

        """
        (a, b) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inv function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for inversion.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the inversion.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for inversion.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Neg function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for negation.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the negation.

        """
        return -float(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for negation.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid function.

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        (a,) = ctx.saved_values
        sigm = operators.sigmoid(a)
        return d_output * sigm * (1 - sigm)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the ReLU function.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the ReLU function.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the exponential function.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: The gradient of the input.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    """Lt function: <"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for the less-than function.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the less-than comparison.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for the less-than function.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the inputs.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Eq function: =="""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for the equality function.

        Args:
        ----
            ctx (Context): The context for the forward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the equality comparison.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for the equality function.

        Args:
        ----
            ctx (Context): The context for the backward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: The gradients of the inputs.

        """
        return 0.0, 0.0
