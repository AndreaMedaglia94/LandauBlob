"""Small dependency-free progress bar."""
import sys
import time


class ProgressBar:
    def __init__(self, total, width=30, label=""):
        self.total = max(int(total), 1)
        self.width = width
        self.label = label
        self.start = time.time()

    def update(self, step):
        frac = min(max(step / self.total, 0.0), 1.0)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = elapsed / step * (self.total - step) if step > 0 else 0.0
        msg = f"\r{self.label}[{bar}] {frac*100:5.1f}%  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if step >= self.total:
            sys.stdout.write("\n")
