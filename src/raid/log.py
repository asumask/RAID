import sys
from datetime import datetime
from pathlib import Path


class _Tee:
    ## writes to the terminal and the log file; flushes every write so `tail -f` follows a running job
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_output(log_path):
    """Copy everything printed (stdout and stderr, incl. tracebacks) to log_path, appending.

    Each run starts with a header line holding the time and the full command, so reruns in the
    same output_dir stay apart.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}  python3 {' '.join(sys.argv)}")
