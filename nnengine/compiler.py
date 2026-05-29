import nn_core

class JITCompiler:
    def __init__(self, model, optimizer, loss_fn, regularizer=None):
        self._cpp_engine = nn_core.JITGraph(model, optimizer, loss_fn, regularizer)
        self.is_compiled = False
        
    def fit(self, dataloader, epochs, tol=1e-4, n_iter_no_change=10, verbose=True):
        if not self.is_compiled:
            self._cpp_engine.trace_batch(dataloader)
            self.is_compiled = True
            self._cpp_engine.fast_loop(dataloader)
            epochs -= 1
        if epochs > 0:
            self._cpp_engine.fast_fit(dataloader, epochs, tol, n_iter_no_change, verbose)