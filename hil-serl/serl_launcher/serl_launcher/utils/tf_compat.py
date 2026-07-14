import os


try:
    import tensorflow as tf
except ModuleNotFoundError:
    class _GFile:
        def __init__(self, path, mode):
            self.path = path
            self.mode = mode
            self._file = None

        def __enter__(self):
            self._file = open(self.path, self.mode)
            return self._file

        def __exit__(self, exc_type, exc_value, traceback):
            self._file.close()

    class _GFileModule:
        GFile = _GFile

        @staticmethod
        def exists(path):
            return os.path.exists(path)

        @staticmethod
        def makedirs(path):
            os.makedirs(path, exist_ok=True)

        @staticmethod
        def join(*parts):
            return os.path.join(*parts)

    class _IOModule:
        gfile = _GFileModule()

    class _Tensor:
        pass

    class _TensorFlowCompat:
        Tensor = _Tensor
        io = _IOModule()

    tf = _TensorFlowCompat()
