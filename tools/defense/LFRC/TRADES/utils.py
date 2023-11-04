import os
import sys

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a')
    def __del__(self):
        self.close()
    def __enter__(self):
        pass
    def __exit__(self, *args):
        self.close()
    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())
    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()