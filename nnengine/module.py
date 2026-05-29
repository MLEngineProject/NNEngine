import nn_core

class Module(nn_core.Module):
    """Python base class for composing NNEngine layers.

    Subclasses typically register child layers with add_module and implement a
    forward method that accepts the native Tape and Tensor objects.

    Example:
        >>> class MyNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = self.add_module(nn_core.DenseLayer(16, 32))
        ...
        ...     def forward(self, tape, x):
        ...         return self.fc(tape, x)
    """

    def __init__(self):
        """Initialize the native module base class."""
        super().__init__()