"""Measure the progress and real latency of Abort on a long AI leveling run (GPU).

usage: python tests/manual/abort_check.py <long_audio> [engine] [seconds_before_abort]
"""
import sys
import threading
import time

import torch

from parla_studio.core import audio_io, leveler as lv, models, smart


def main() -> None:
    path = sys.argv[1]
    engine = sys.argv[2] if len(sys.argv) > 2 else "inworld"
    wait_s = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    x, sr = audio_io.load(path)
    print(f"audio {x.shape[0] / sr / 60:.1f} min · engine {engine}", flush=True)
    prog = []
    ctx = lv.Ctx(progress=lambda v: prog.append((time.perf_counter(), v)),
                 log=lambda m, l: print("   ", m, flush=True))
    state = {}

    def run():
        try:
            smart.run_smart(x, sr, smart.SmartParams(engine=engine), ctx)
            state["end"] = "completed"
        except lv.Cancelled:
            state["end"] = "aborted"
        state["t_end"] = time.perf_counter()

    th = threading.Thread(target=run)
    t0 = time.perf_counter()
    th.start()
    time.sleep(wait_s)
    t_cancel = time.perf_counter()
    last = prog[-1][1] if prog else 0
    ctx.cancel.set()
    th.join()
    models.free_vram()
    alloc = torch.cuda.memory_allocated() / 2**30
    print(f"progress at abort: {last * 100:.1f}% after {t_cancel - t0:.0f} s · {len(prog)} progress bar updates", flush=True)
    print(f"result: {state['end']} · stopped {state['t_end'] - t_cancel:.2f} s after the abort request · VRAM in use after freeing: {alloc:.2f} GB (models loaded, ready to reuse)", flush=True)


if __name__ == "__main__":
    main()
