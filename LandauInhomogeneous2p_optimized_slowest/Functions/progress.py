"""
A tiny dependency-free progress bar (percentage + elapsed/ETA), so you
don't need to pip install anything just to see how a long run is going.
If you already have tqdm and prefer it, swap print_progress(...) for
tqdm(range(...)) in main.py -- this is just the zero-dependency default.
"""
import sys
import time


class ProgressBar:
    def __init__(self, total, width=30, label=""):
        self.total = total
        self.width = width
        self.label = label
        self.start = time.time()

    def update(self, step):
        frac = step / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = elapsed / step * (self.total - step) if step > 0 else 0.0
        msg = f"\r{self.label}[{bar}] {frac*100:5.1f}%  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s"
        sys.stdout.write(msg)
        sys.stdout.flush()
        if step >= self.total:
            sys.stdout.write("\n")
