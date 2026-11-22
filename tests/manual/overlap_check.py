"""Real test of overlapping voices (TIGER) and enrollment with reference clips.

Builds a conversation with two real voices that take turns and overlap in 3 segments, and measures against
the original voices (known because the mix is built here).

usage: python tests/manual/overlap_check.py <podcast_es.wav> <interview_en.wav> <folder>
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from parla_studio.core import audio_io, models, speakers as spk

SR = 16000


def sisdr(est, ref):
    ref, est = ref - ref.mean(), est - est.mean()
    s = (est @ ref) / (ref @ ref + 1e-12) * ref
    return 10 * np.log10((s @ s) / ((est - s) @ (est - s) + 1e-12))


def main() -> int:
    es_p, en_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    es, s1 = audio_io.load(es_p)
    en, s2 = audio_io.load(en_p)
    A = audio_io.resample(audio_io.to_mono(es), s1, SR)           # host (Spanish)
    B = audio_io.resample(audio_io.to_mono(en), s2, SR)[: 10 * SR]  # interviewer (English), first turn only
    norm = lambda v: v / (np.sqrt(np.mean(v ** 2)) + 1e-9) * 0.08
    # 40 s conversation: A speaks, B cuts in over him, A answers over B, etc.
    plan = [("A", 0.0, A[int(29.9 * SR):int(38.0 * SR)]), ("B", 6.0, B[:int(8.0 * SR)]),
            ("A", 12.0, A[int(45.4 * SR):int(53.0 * SR)]), ("B", 17.5, B[int(1.0 * SR):int(9.5 * SR)]),
            ("A", 24.0, A[int(53.6 * SR):int(61.0 * SR)]), ("B", 29.0, B[int(0.5 * SR):int(9.0 * SR)])]
    total = int(40 * SR)
    ref = {"A": np.zeros(total, np.float32), "B": np.zeros(total, np.float32)}
    for who, t, seg in plan:
        a = int(t * SR)
        seg = norm(seg)[: total - a]
        ref[who][a:a + seg.size] += seg
    mix = ref["A"] + ref["B"]
    audio_io.save_wav(out / "conversacion.wav", mix, SR)
    both = (np.abs(ref["A"]) > 0) & (np.abs(ref["B"]) > 0)
    print(f"mix: {total / SR:.0f} s · both voices at once ~{both.mean() * total / SR:.1f} s (includes internal pauses)")

    diar = spk.diarize(mix, SR)
    people = spk.describe(diar, mix, SR, None, 0.8)
    print(f"diarization: {len(people)} speakers · overlapped {diar.overlap.sum() * diar.frame_s:.1f} s · segments to separate: {[(round(a, 1), round(b, 1)) for a, b in spk.overlap_regions(diar)]}")
    sep = spk.separate_overlaps(mix, SR, diar, people)
    # which diarizer speaker is A: the one that best matches A's activity
    actA = np.abs(ref["A"][: diar.active.shape[0] * 160].reshape(-1, 160)).max(axis=1) > 1e-4
    kA = max((p.idx for p in people), key=lambda k: (diar.active[: actA.size, k] & actA).sum())
    kB = next(p.idx for p in people if p.idx != kA)
    ok = True
    for name, k, r in (("A (Spanish)", kA, ref["A"]), ("B (English)", kB, ref["B"])):
        gated = spk.speaker_track(mix[:, None], SR, diar, k)[:, 0]
        withsep = spk.speaker_track(mix[:, None], SR, diar, k, separated=sep)[:, 0]
        audio_io.save_wav(out / f"track_{name[0]}_tiger.wav", withsep, SR)
        audio_io.save_wav(out / f"track_{name[0]}_no_tiger.wav", gated, SR)
        g, w = [], []
        for spn in sep:
            a, b = int(spn.start * SR), int(spn.end * SR)
            g.append(sisdr(gated[a:b], r[a:b]))
            w.append(sisdr(withsep[a:b], r[a:b]))
        print(f"  track {name}: in overlapped segments SI-SDR without TIGER {np.mean(g):5.1f} dB → with TIGER {np.mean(w):5.1f} dB")
        ok &= np.mean(w) > np.mean(g) + 6
    print("  assignment confidence per segment:", [round(s.confidence, 2) for s in sep])

    # enrollment with reference clips: 3 clips of the host + 1 of the interviewer (must be discarded)
    tmp = Path(tempfile.mkdtemp(dir=out))
    files = []
    for n, (a, b) in enumerate(((90, 100), (110, 121), (140, 150))):
        f = tmp / f"host_{n + 1}.wav"
        audio_io.save_wav(f, A[a * SR:b * SR], SR)
        files.append(f)
    intruder = tmp / "other.wav"
    audio_io.save_wav(intruder, B[: 8 * SR], SR)
    files.append(intruder)
    store = spk.ProfileStore(out / "perfiles.json")
    for p in list(store.profiles):
        store.remove(p)
    prof, reports = spk.enroll_from_clips(store, "Locutor", files, isolate=True)
    for r in reports:
        print(f"  clip {r.file:12s} {'✓' if r.accepted else '✗'} {r.seconds:4.1f} s · voices {r.speakers} · similarity {r.similarity:.2f} {r.note}")
    ok &= prof is not None and prof.samples == 3 and not reports[-1].accepted
    people2 = spk.describe(diar, mix, SR, store, 0.8)
    print("  recognition in the conversation:", [(p.name, round(p.score, 2)) for p in people2])
    ok &= any(p.idx == kA and p.identity == "Locutor" for p in people2) and \
        not any(p.idx == kB and p.identity for p in people2)
    print("RESULT:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
