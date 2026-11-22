"""Real test of the three models on the GPU (downloads the weights into ./models the first time).

usage: python tests/manual/smoke_models.py <audio> <output_folder> [seconds] [language]
"""
import sys
import time
from pathlib import Path

import numpy as np

from parla_studio.core import audio_io, dataset, leveler as lv, models, smart


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    lang = sys.argv[4] if len(sys.argv) > 4 else "en"
    out.mkdir(parents=True, exist_ok=True)
    x, sr = audio_io.load(src)
    x = x[: int(secs * sr)]
    print(f"device: {models.device_description()} · audio {x.shape[0] / sr:.0f} s @ {sr} Hz, {x.shape[1]} channel(s)")
    log = lambda m, l: print("   ", m)
    ctx = lv.Ctx(log=log)

    t = time.perf_counter(); models.isolator(); print(f"voice-isolator load: {time.perf_counter() - t:.1f} s")
    t = time.perf_counter(); voice = models.isolate_voice(x, sr); dt = time.perf_counter() - t
    print(f"voice-isolator: {dt:.2f} s ({secs / dt:.0f}x real time)")
    t = time.perf_counter(); models.sava_model(); print(f"SAVA load: {time.perf_counter() - t:.1f} s")
    t = time.perf_counter(); sp = models.speech_probability(voice.mean(axis=1), sr); dt = time.perf_counter() - t
    print(f"SAVA: {dt:.2f} s · {sp.size} frames · voice {100 * (sp > 0.5).mean():.0f}%")

    t = time.perf_counter()
    r = smart.run_smart(x, sr, smart.SmartParams(final_db=-18.0), ctx, voice=voice, speech=sp)
    print(f"voice-aware leveling: {time.perf_counter() - t:.2f} s")
    audio_io.save_wav24(out / "nivelado_ia.wav", r.output, sr)
    audio_io.save_wav24(out / "voice.wav", voice, sr)
    audio_io.save_wav24(out / "background.wav", x - voice, sr)
    c = lv.run_pipeline(x, sr, lv.Params(final_db=-18.0))
    audio_io.save_wav24(out / "nivelado_clasico.wav", c.output, sr)

    # noise in pauses: background level where SAVA detects no voice, classic vs AI
    hops = x.shape[0] // lv.hop_frames(sr, lv.HOP_S)
    speech = models.speech_on_hops(sp, hops, lv.HOP_S) >= 0.5
    pause = ~smart._dilate(speech, 30)
    def pause_db(y):
        e = lv.hop_meansq(y, sr)[: pause.size]
        return 10 * np.log10(e[pause[: e.size]].mean() + 1e-20)
    def speech_spread(y):
        e = lv.loudness_db(lv.hop_meansq(y, sr), 0.4)[: speech.size]
        v = e[speech[: e.size]]
        return np.percentile(v, 90) - np.percentile(v, 10)
    for name, y in (("original", x), ("classic", c.output), ("IA", r.output)):
        print(f"  {name:9s} noise in pauses {pause_db(y):7.1f} dB · voice spread (p90-p10) {speech_spread(y):5.1f} dB")

    t = time.perf_counter(); models.crisper(); print(f"CrisperWhisper 2.0 large load: {time.perf_counter() - t:.1f} s")
    mono = audio_io.to_mono(voice)
    t = time.perf_counter(); text, words = models.transcribe(mono, sr, lang); dt = time.perf_counter() - t
    print(f"CrisperWhisper: {dt:.2f} s ({secs / dt:.0f}x real time) · {len(words)} words")
    print("  text:", text[:300])
    print("  first words:", [(w.text, round(w.start, 2), round(w.end, 2), round(w.prob, 2)) for w in words[:12]])
    groups = dataset.segment(words, dataset.DatasetParams())
    print(f"  segmentation: {len(groups)} clips:", [round(words[j - 1].end - words[i].start, 1) for i, j in groups])


if __name__ == "__main__":
    main()
