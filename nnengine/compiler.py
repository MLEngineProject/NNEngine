import nn_core

class JITCompiler:
    """High-level Python wrapper around the native JIT training graph.

    The heavy training loop, validation, and early-stopping logic execute in
    C++ through the bound JITGraph object, so the GIL is released while the
    compiled engine is replaying batches.

    Example:
        >>> trainer = JITCompiler(model, optimizer, loss_fn)
        >>> trainer.fit(train_loader, epochs=20, val_dataloader=val_loader)
    """

    def __init__(self, model, optimizer, loss_fn, regularizer=None):
        """Initialize a compiled training wrapper.

        Args:
            model: Layer or Module instance to optimize.
            optimizer: Native optimizer implementation.
            loss_fn: Native loss implementation.
            regularizer: Optional regularization term.
        """
        self._cpp_engine = nn_core.JITGraph(model, optimizer, loss_fn, regularizer)
        self.is_compiled = False
        
    def fit(self, dataloader, epochs, val_dataloader=None, tol=1e-4, n_iter_no_change=10, verbose=True):
        """Train the compiled graph with optional validation and early stopping.

        Args:
            dataloader: Training DataLoader.
            epochs: Number of epochs to train.
            val_dataloader: Optional validation DataLoader.
            tol: Minimum improvement required to reset early stopping.
            n_iter_no_change: Number of epochs without improvement before stop.
            verbose: Whether to print progress from the native engine.

        Returns:
            None

        Notes:
            The first batch is traced in C++, and the remaining training,
            validation, and early-stopping work runs natively with the GIL
            released.
        """
        if not self.is_compiled:
            self._cpp_engine.trace_batch(dataloader)
            self.is_compiled = True
            self._cpp_engine.fast_loop(dataloader)
            epochs -= 1
        if epochs > 0:
            self._cpp_engine.fast_fit(dataloader, val_dataloader, epochs, tol, n_iter_no_change, verbose)