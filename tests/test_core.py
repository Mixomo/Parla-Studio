"""Core tests: same reference values as the Python/Rust/C++ versions."""
from pathlib import Path

import numpy as np
import pytest

from parla_studio.core import audio_io, dataset, leveler as lv, models, smart
from parla_studio.i18n import tr

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "interview_real.wav"


@pytest.fixture(scope="module")
def audio():
    if not DATA.exists():  # reference audio from the parent repository (not published)
        pytest.skip(f"reference audio not found: {DATA}")
    return audio_io.load(DATA)


def test_rms_matches_reference(audio):
    x, sr = audio
    r = lv.run_pipeline(x, sr, lv.Params(final_db=-18.0))
    assert abs(r.level_db + 40.41) < 0.05
    assert abs(r.offset_db - 22.41) < 0.05
    assert r.limited_blocks == 227
    assert abs(r.max_reduction_db - 8.90) < 0.05


def test_lufs_matches_reference(audio, tmp_path):
    x, sr = audio
    r = lv.run_pipeline(x, sr, lv.Params(final_db=-16.0, lufs=True))
    assert abs(r.level_db + 40.60) < 0.05
    assert abs(r.offset_db - 24.60) < 0.05
    assert r.limited_blocks == 1264
    out = tmp_path / "o.wav"
    audio_io.save_wav24(out, r.output, sr)
    y, _ = audio_io.load(out)
    assert abs(lv.lufs_integrated(lv.k_subblocks(y, sr)) + 16.09) < 0.1


def test_peak_never_exceeds_ceiling(audio):
    x, sr = audio
    r = lv.run_pipeline(x, sr, lv.Params(final_db=-0.1))
    assert 20 * np.log10(np.abs(r.output).max()) <= -0.25


def test_cancel(audio):
    x, sr = audio
    ctx = lv.Ctx()
    ctx.cancel.set()
    with pytest.raises(lv.Cancelled):
        lv.run_pipeline(x, sr, lv.Params(), ctx)


def test_segmenter_cuts_at_pauses_and_sentence_ends():
    W = models.Word
    words, t = [], 0.0
    # 3 sentences of ~3 s separated by 0.6 s pauses, with a short internal pause
    for s in range(3):
        for k in range(6):
            txt = f"p{s}{k}" + ("." if k == 5 else "")
            words.append(W(txt, t, t + 0.4))
            t += 0.5 if k != 2 else 0.55
        t += 0.6
    groups = dataset.segment(words, dataset.DatasetParams(min_s=2.0, max_s=4.0, min_pause_s=0.2))
    assert groups == [(0, 6), (6, 12), (12, 18)]


def test_dilate():
    m = np.array([0, 0, 1, 0, 0, 0], bool)
    assert smart._dilate(m, 1).tolist() == [False, True, True, True, False, False]


# ---------------------------------------------------------------- censoring
from parla_studio.core import censor  # noqa: E402


def _words(text, step=0.5):
    return [models.Word(t, k * step, k * step + 0.4) for k, t in enumerate(text.split())]


def test_search_accents_phrases_wildcards():
    ws = _words("Qué MIERDA, dijo. La mierda está acá. Mierdoso y puta madre, ¡Putísima!")
    assert [m.text for m in censor.search(ws, "mierda")] == ["MIERDA,", "mierda"]
    assert [m.text for m in censor.search(ws, "mierd*")] == ["MIERDA,", "mierda", "Mierdoso"]
    assert [m.text for m in censor.search(ws, "puta madre")] == ["puta madre,"]
    assert [m.text for m in censor.search(ws, "putisima")] == ["¡Putísima!"]  # without accents
    assert [m.text for m in censor.search(ws, "acá, que")] == ["Qué", "acá."]  # several terms, in order
    assert [m.text for m in censor.search(ws, "put", whole_word=False)] == ["puta", "¡Putísima!"]
    assert censor.search(ws, "putisima", ignore_accents=False) == []


def test_beep_replaces_only_voice_and_cut_shortens():
    sr = 16000
    t = np.arange(sr * 3) / sr
    voice = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]
    bg = (0.05 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)[:, None]
    x = voice + bg
    m = [censor.Match(0, 1, 1.0, 2.0, "x", "x")]
    sil = censor.render(x, sr, m, "silence", voice=voice)
    mid = slice(int(1.2 * sr), int(1.8 * sr))
    assert np.allclose(sil[mid], bg[mid], atol=1e-5)          # the background continues, the voice doesn't
    assert np.allclose(sil[: sr // 2], x[: sr // 2])          # outside the segment, untouched
    beep = censor.render(x, sr, m, "beep", voice=voice, beep_hz=1000)
    spec = np.abs(np.fft.rfft((beep[mid] - bg[mid])[:, 0]))
    assert abs(np.argmax(spec) * sr / (mid.stop - mid.start) - 1000) < 5  # the replacement is the beep
    cut = censor.render(x, sr, m, "cut")
    assert abs(cut.shape[0] - 2 * sr) <= int(0.01 * sr)


def test_cut_time_mapping_matches_rendered_audio():
    sr = 8000
    x = (np.arange(sr * 10) / sr).astype(np.float32)[:, None]  # each sample is worth its time in s
    ms = [censor.Match(0, 1, 2.0, 2.5, "a", "a"), censor.Match(1, 2, 6.0, 7.0, "b", "b"),
          censor.Match(2, 3, 8.0, 8.2, "c", "c", enabled=False)]
    y = censor.render(x, sr, ms, "cut")
    for t in (1.0, 4.0, 5.9, 7.5, 9.0):
        u = censor.map_time_after_cut(t, ms)
        assert abs(float(y[int(round(u * sr)), 0]) - t) < 0.002, (t, u)
    # inside a cut segment: lands on the cut point
    assert abs(censor.map_time_after_cut(6.5, ms) - censor.map_time_after_cut(6.0, ms)) < 1e-9


# ---------------------------------------------------------------- speakers
from parla_studio.core import speakers as spk  # noqa: E402


def _diar(spans, total_s=10.0, k=3):
    """Synthetic diarization: spans = [(start, end, speaker)]."""
    T = int(total_s / 0.01)
    p = np.zeros((T, 8), np.float32)
    for a, b, s in spans:
        p[int(a / 0.01):int(b / 0.01), s] = 0.9
    return spk.Diarization(p)


def test_diarization_cleanup_segments_and_overlap():
    d = _diar([(0.0, 2.0, 0), (2.1, 4.0, 0), (3.5, 6.0, 1), (7.0, 7.1, 2)])
    assert d.speakers == [0, 1]                                 # 0.1 s island discarded, S2 doesn't count
    segs = [(round(s.start, 2), round(s.end, 2), s.speaker) for s in d.segments()]
    assert segs == [(0.0, 4.0, 0), (3.5, 6.0, 1)]               # 0.1 s gap closed
    assert abs(d.overlap.sum() * 0.01 - 0.5) < 0.02             # 3.5-4.0 overlapped
    assert d.who(0.5, 1.0) == (0, 0.0)
    who, ov = d.who(3.6, 3.9)
    assert ov == 1.0


def test_turns_srt_rttm_and_tracks():
    d = _diar([(0.0, 3.0, 0), (3.2, 6.0, 1)])
    W = models.Word
    words = [W("Hola", 0.1, 0.5), W("qué", 0.6, 0.8), W("tal.", 0.9, 1.2), W("Bien,", 3.3, 3.6), W("gracias.", 3.7, 4.2)]
    tr = spk.turns(words, d)
    assert [(t.speaker, t.text) for t in tr] == [(0, "Hola qué tal."), (1, "Bien, gracias.")]
    srt = spk.to_srt(tr, {0: "Ana", 1: "Luis"})
    assert "00:00:00,100 --> 00:00:01,200\nAna: Hola qué tal." in srt and "Luis: Bien, gracias." in srt
    rttm = spk.to_rttm(d, {0: "Ana López", 1: "Luis"}, "f")
    assert rttm.splitlines()[0] == "SPEAKER f 1 0.000 3.000 <NA> <NA> Ana_López <NA> <NA>"
    sr = 1000
    x = np.ones((6 * sr, 1), np.float32)
    t0 = spk.speaker_track(x, sr, d, 0)
    assert t0[int(1.5 * sr), 0] > 0.99 and t0[int(5 * sr), 0] < 1e-6   # sounds where they speak, silence where they don't


def test_profiles_enroll_identify(tmp_path):
    st = spk.ProfileStore(tmp_path / "p.json")
    a = np.array([1, 0, 0], np.float32)
    b = np.array([0, 1, 0], np.float32)
    st.enroll("Ana", a)
    st.enroll("Luis", b)
    assert st.identify(np.array([0.95, 0.31, 0], np.float32), 0.8)[0] == "Ana"
    assert st.identify(np.array([0.7, 0.71, 0], np.float32), 0.8)[0] is None   # doubtful: unknown
    st2 = spk.ProfileStore(tmp_path / "p.json")                                  # persiste en disco
    assert set(st2.profiles) == {"Ana", "Luis"}


def test_segment_never_mixes_speakers():
    W = models.Word
    words, t = [], 0.0
    for k in range(12):  # 12 words in a row without pauses; the speaker changes at word 5 and 9
        words.append(W(f"w{k}", t, t + 0.4))
        t += 0.45
    owners = [0] * 5 + [1] * 4 + [None] + [0] * 2
    groups = dataset.segment(words, dataset.DatasetParams(min_s=0.5, max_s=30), owners)
    assert groups == [(0, 5), (5, 10), (10, 12)]  # the word without a speaker follows the current turn


def test_turns_do_not_orphan_sentence_end():
    d = _diar([(0.0, 9.0, 0)])
    W = models.Word
    txt = "And this spills over into normative arguments made by [UH] other scholars.".split()
    words = [W(t, 0.3 * k, 0.3 * k + 0.25) for k, t in enumerate(txt)]
    tr = spk.turns(words, d, max_chars=66)  # "...other" just fits; "scholars." would not
    assert [t.text for t in tr] == [" ".join(txt)]  # "scholars." is not left alone


def test_words_in_separated_overlap_go_to_louder_voice():
    d = _diar([(0.0, 6.0, 0), (2.0, 4.0, 1)])  # 2-4 s: both active
    sr = 1000
    v0 = np.zeros(int(3.0 * sr), np.float32)
    v1 = np.zeros(int(3.0 * sr), np.float32)
    v0[: int(1.0 * sr)] = 0.5   # separated segment 1.5-4.5 s: S0 sounds in 1.5-2.5, S1 in 2.5-4.5
    v1[int(1.0 * sr):] = 0.5
    sep = [spk.Separated(1.5, 4.5, {0: v0, 1: v1}, 1.0, sr)]
    W = models.Word
    words = [W("uno", 1.0, 1.4), W("dos", 2.1, 2.4), W("tres", 3.0, 3.4), W("cuatro", 5.0, 5.4)]
    owners = [o for o, _ in spk.assign_words(words, d, sep)]
    assert owners == [0, 0, 1, 0]


def test_slots_mapping_auto_and_limit(tmp_path):
    st = spk.SlotStore(tmp_path / "s.json", initial=4)
    assert [s.name for s in st.slots] == [tr(f"Speaker {i}") for i in range(1, 5)]  # name according to the language
    st.slots[2].name = "Ana"
    sp = [spk.Speaker(k, f"Speaker {k + 1}", seconds=10 - k) for k in range(6)]
    sp[3].identity = "Ana"
    spk.apply_slots(sp, st, auto_add=True)
    assert sp[3].name == "Ana" and sp[3].slot == 2            # the person goes to their slot
    assert len(st.slots) == 6 and all(s.slot >= 0 for s in sp)  # automatic: slots are created
    assert len({s.color for s in st.slots}) == 6
    st2 = spk.SlotStore(tmp_path / "s2.json", initial=4)
    sp2 = [spk.Speaker(k, f"Speaker {k + 1}") for k in range(6)]
    spk.apply_slots(sp2, st2, auto_add=False)
    assert len(st2.slots) == 4 and sum(s.slot < 0 for s in sp2) == 2


def test_limit_speakers_merges_into_most_similar():
    d = _diar([(0, 3, 0), (3, 6, 1), (6, 7, 2)])
    e = lambda v: np.array(v, np.float32) / np.linalg.norm(v)
    sp = [spk.Speaker(0, "a", seconds=3, embedding=e([1, 0])), spk.Speaker(1, "b", seconds=3, embedding=e([0, 1])),
          spk.Speaker(2, "c", seconds=1, embedding=e([0.1, 1]))]
    kept = spk.limit_speakers(d, sp, 2)
    assert [s.idx for s in kept] == [0, 1] and d.speakers == [0, 1]
    assert d.who(6.2, 6.8)[0] == 1                             # c merged into b (most similar)


def test_separated_words_replace_main_in_overlap():
    W = models.Word
    main = [W("hola", 0.0, 0.5), W("xx", 2.0, 2.4), W("yy", 2.5, 2.9), W("chau", 5.0, 5.5)]
    sep = [spk.Separated(1.5, 3.5, {0: np.zeros(32000, np.float32), 1: np.zeros(32000, np.float32)}, 1.0, 16000)]
    fake = {0: [W("dime", 0.6, 0.9)], 1: [W("no", 0.7, 1.0), W("sé", 1.1, 1.4)]}
    calls = iter([fake[0], fake[1]])
    out = spk.merge_separated_words(main, sep, lambda v, sr: next(calls))
    assert [(w.text, w.speaker) for w in out] == [("hola", None), ("dime", 0), ("no", 1), ("sé", 1), ("chau", None)]


def test_turns_keep_overlapping_sentences_whole():
    d = _diar([(0.0, 6.0, 0), (2.0, 5.0, 1)])
    W = models.Word
    a = [W(t, 0.5 + 0.6 * k, 0.9 + 0.6 * k, 1.0, 0) for k, t in enumerate("uno dos tres cuatro cinco seis siete.".split())]
    b = [W(t, 2.1 + 0.6 * k, 2.5 + 0.6 * k, 1.0, 1) for k, t in enumerate("aa bb cc dd.".split())]
    words = sorted(a + b, key=lambda w: w.start)
    tr = spk.turns(words, d)
    assert [(t.speaker, t.text) for t in tr] == [(0, "uno dos tres cuatro cinco seis siete."), (1, "aa bb cc dd.")]


# ---------------------------------------------------------------- preventive censoring by lexicon
from parla_studio.core import profanity as pf  # noqa: E402


def _lex(lines, lang="es"):
    lex = pf.Lexicon(lang)
    for l in lines:
        lex.add_line(l, "custom")
    return lex


def test_fold_keeps_enye_and_neutralizes_obfuscation():
    assert pf.fold("¡Coño!") == "coño" and pf.fold("cono") == "cono"
    assert pf.fold("PÚTÍSIMA") == "putisima" and pf.fold("pu!ta") == "puta" and pf.fold("p.u.t.a") == "puta"
    assert pf.fold("m1erd4") == "mierda" and pf.fold("2024") == "2024"   # bare numbers are not touched


def test_lexicon_wildcards_phrases_exceptions_and_enye():
    lex = _lex(["coño", "put*", "*mierda*", "hijo de puta", "!computadora", "comput*", "hijo"])
    ws = _words("el cono de helado, coño, putísimas, comemierda, mieeeerda, hijo de puta, computadora, computin, su hijo")
    got = [(m.text, m.source) for m in lex.scan(ws)]
    assert got == [("coño,", "DB · custom"), ("putísimas,", "DB · custom"), ("comemierda,", "DB · custom"),
                   ("mieeeerda,", "DB · custom"), ("hijo de puta,", "DB · custom"), ("computin,", "DB · custom"),
                   ("hijo", "DB · custom")]
    # "cono" (ice-cream cone) is NOT censored anymore: ñ matters; "computadora" is an exception


def test_lexicon_cjk_substring():
    lex = _lex(["ばか"], "ja")
    ws = [models.Word("お前", 0, 0.3), models.Word("は", 0.3, 0.4), models.Word("ばか", 0.4, 0.8), models.Word("だ", 0.8, 1.0)]
    assert [m.text for m in lex.scan(ws)] == ["ばか"]


def test_sanitize_restores_enye_dedupes_and_compacts():
    old = ["cono", "conos", "mierda", "Mierda", "chingada", "chingadas", "chingado", "chingados", "puta"]
    out, rep = pf.sanitize(old, {"coño"})
    assert "coño" in out and "coños" in out and "cono" not in out
    assert rep["duplicates"] == 1 and "chingad*" in out and "chingada" not in out


def test_store_own_list_and_migration(tmp_path):
    st = pf.LexiconStore(tmp_path)
    legacy = tmp_path / "profanidad.txt"
    legacy.write_text("cono\nconos\nmierda\n", encoding="utf-8")
    rep = pf.migrate_legacy(st, legacy, "es")
    assert rep and (tmp_path / "profanidad_original.txt").exists() and not legacy.exists()
    lex = st.load("es")
    assert "coño" in lex.exact and "cono" not in lex.exact and lex.size == 3
    assert st.add_own("es", "Gilipollas") and not st.add_own("es", "gilipollas")


def test_merge_marks_lexicon_and_search():
    from parla_studio.core import censor
    ws = _words("vaya mierda de día")
    merged = pf.merge(_lex(["mierda"]).scan(ws), censor.search(ws, "mierda, día"))
    assert [(m.text, m.source) for m in merged] == [("mierda", "DB · custom + search"), ("día", "search")]


# ---------------------------------------------------------------- blocks, progress and abort
from parla_studio.core import blocks  # noqa: E402


def test_process_blocks_identity_progress_and_cancel():
    sr = 1000
    x = np.random.default_rng(1).standard_normal((95 * sr, 2)).astype(np.float32)
    seen = []
    y = blocks.process_blocks(x, sr, lambda b, inner: (inner(0.5), b)[1], 10.0, 2.0, seen.append)
    assert np.allclose(y, x, atol=1e-6)                       # the crossfade rebuilds the signal
    assert seen == sorted(seen) and seen[-1] == 1.0           # monotonic progress up to 100 %
    ctx = lv.Ctx()
    calls = []

    def fn(b, inner):
        calls.append(1)
        if len(calls) == 3:
            ctx.cancel.set()                                  # the user aborts during the 3rd block
        return b
    with pytest.raises(lv.Cancelled):
        blocks.process_blocks(x, sr, fn, 10.0, 2.0, None, ctx.check)
    assert len(calls) == 3                                    # no further block is processed


def test_ctx_sub_maps_progress():
    got = []
    sub = lv.Ctx(progress=got.append).sub(0.35, 0.45)
    sub.progress(0.5)
    assert abs(got[0] - 0.40) < 1e-9


def test_smart_without_separation_uses_sava_on_original(monkeypatch):
    sr = 16000
    t = np.arange(sr * 12) / sr
    talk = ((t % 4) < 2.5).astype(np.float32)                 # 2.5 s of "voice", 1.5 s of pause
    x = (0.2 * np.sin(2 * np.pi * 180 * t) * talk * (1 + (t > 6)) + 0.002 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)[:, None]
    called = {}
    monkeypatch.setattr(models, "isolate_voice", lambda *a, **k: called.setdefault("iso", True))
    monkeypatch.setattr(models, "speech_probability",
                        lambda mono, sr_, ctx=None, *a, **k: np.interp(np.arange(int(12 / 0.04)) * 0.04, t, talk).astype(np.float32))
    r = smart.run_smart(x, sr, smart.SmartParams(separate=False, final_db=-20.0))
    assert "iso" not in called                                # nothing was separated
    assert np.allclose(r.background, 0.0)                     # the whole audio was treated as voice
    e = lv.hop_meansq(r.output, sr)
    hops = np.arange(e.size) * 0.005
    a, b = e[(hops % 4 > 0.5) & (hops % 4 < 2.0) & (hops < 6)], e[(hops % 4 > 0.5) & (hops % 4 < 2.0) & (hops > 6.5)]
    assert abs(10 * np.log10(a.mean() / b.mean())) < 2.0      # both parts were leveled


def test_process_blocks_splits_when_out_of_vram():
    class OutOfMemoryError(RuntimeError):
        pass
    sr = 1000
    x = np.random.default_rng(2).standard_normal((60 * sr, 1)).astype(np.float32)
    sizes, logs = [], []

    def fn(b, inner):
        sizes.append(b.shape[0] / sr)
        if b.shape[0] > 20 * sr:          # simulates that blocks longer than 20 s don't fit in VRAM
            raise OutOfMemoryError("CUDA out of memory")
        return b
    y = blocks.process_blocks(x, sr, fn, 40.0, 2.0, None, None, lambda m, l: logs.append(m))
    assert np.allclose(y, x, atol=1e-6) and max(s for s in sizes if s <= 20) <= 20 and logs
    with pytest.raises(ValueError):       # other errors are not hidden
        blocks.process_blocks(x, sr, lambda b, i: (_ for _ in ()).throw(ValueError("x")), 40.0, 2.0)


# ---------------------------------------------------------------- audio from online videos
from parla_studio.core import fetch  # noqa: E402


def test_fetch_parses_times_and_links():
    assert fetch.parse_time("") is None and fetch.parse_time("90") == 90 and fetch.parse_time("1:02:03.5") == 3723.5
    with pytest.raises(ValueError):
        fetch.parse_time("1:xx")
    text = "see https://youtu.be/a\nhttps://youtu.be/a https://www.youtube.com/watch?v=b&t=3"
    assert fetch.split_urls(text) == ["https://youtu.be/a", "https://www.youtube.com/watch?v=b&t=3"]


def test_fetch_decodes_and_trims_to_wav(tmp_path):
    import av
    import soundfile as sf
    sr = 48000
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 4) / sr)).astype(np.float32)
    src = tmp_path / "clip.webm"
    with av.open(str(src), "w") as box:     # what YouTube serves: Opus in WebM
        st = box.add_stream("libopus", rate=sr)
        st.layout = "stereo"
        for i in range(0, len(x) - 959, 960):
            fr = av.AudioFrame.from_ndarray(np.stack([x[i:i + 960]] * 2), format="fltp", layout="stereo")
            fr.sample_rate, fr.pts = sr, i
            for pk in st.encode(fr):
                box.mux(pk)
        for pk in st.encode(None):
            box.mux(pk)
    full = sf.info(str(fetch.decode_to_wav(src, tmp_path / "full.wav")))
    cut = sf.info(str(fetch.decode_to_wav(src, tmp_path / "cut.wav", start=1.0, end=2.5)))
    assert (full.samplerate, full.channels, full.subtype) == (sr, 2, "PCM_24") and abs(full.duration - 4.0) < 0.05
    assert abs(cut.duration - 1.5) < 1e-3
    with pytest.raises(ValueError):
        fetch.decode_to_wav(src, tmp_path / "none.wav", start=10.0, end=11.0)
    assert not (tmp_path / "none.wav.part").exists()


# ---------------------------------------------------------------- dataset export and undo history
def _clips(n: int = 3) -> list:
    return [dataset.Clip(f"src{k % 2}.wav", k, k + 1.0, f"text {k}", 2, 0.9, 1.0,
                         audio=np.zeros((1600, 1), np.float32)) for k in range(n)]


def test_export_audio_only_and_folder(tmp_path):
    import zipfile
    prm = dataset.DatasetParams(out_sr=16000)
    n = dataset.export(_clips(), tmp_path / "a", prm, layout="audio")
    names = zipfile.ZipFile(tmp_path / "a.zip").namelist()
    assert n == 3 and len(names) == 3 and all(x.endswith(".wav") for x in names)  # no .txt nor metadata
    out = tmp_path / "sub" / "dataset"
    assert dataset.export(_clips(), out, prm, layout="pairs", package="folder") == 3
    files = sorted(p.name for p in out.iterdir())
    assert sum(f.endswith(".wav") for f in files) == 3 and sum(f.endswith(".txt") for f in files) == 3
    assert "metadata.jsonl" in files and not (out.parent / "dataset.part").exists()
    with pytest.raises(FileExistsError):          # never writes over an existing folder
        dataset.export(_clips(), out, prm, package="folder")


def test_history_undo_redo_and_merge():
    from parla_studio.ui.common import History
    state = {"v": 0}
    h = History(lambda: state["v"], lambda v: state.update(v=v))
    for v in (1, 2):
        h.record()
        state["v"] = v
    for v in (3, 4, 5):                           # typing: one single step
        h.record(merge="text")
        state["v"] = v
    h.undo()
    assert state["v"] == 2
    h.undo()
    assert state["v"] == 1
    h.redo()
    assert state["v"] == 2 and h.can_redo()
    h.record()                                    # a new change drops the redo branch
    state["v"] = 9
    assert not h.can_redo()
    h.undo()
    assert state["v"] == 2


# ---------------------------------------------------------------- projects (.parla)
from parla_studio.core import project as prj  # noqa: E402


def test_project_roundtrip_keeps_everything(tmp_path):
    from pathlib import Path as P
    wav = tmp_path / "in" / "talk.wav"
    wav.parent.mkdir()
    audio_io.save_wav(wav, np.zeros((4800, 1), np.float32), 48000)
    rng = np.random.default_rng(0)
    long = (rng.standard_normal((600_000, 2)) * 0.1).astype(np.float32)      # stored as FLAC
    diar = spk.Diarization(np.clip(rng.random((300, 8)), 0, 1).astype(np.float32))
    state = {
        "path": str(wav), "audio": long, "words": [models.Word("hola", 0.1, 0.4, 0.9, 1)],
        "diar": diar, "names": {0: "Ana", 3: "Luis"}, "span": (1.5, 2.5),
        "speakers": [spk.Speaker(0, "Ana", embedding=np.ones(4, np.float32))],
        "separated": [spk.Separated(1.0, 2.0, {0: np.zeros(160, np.float32)})],
        "matches": [censor.Match(0, 1, 0.1, 0.4, "hola", "hola", False, "DB")],
        "clips": [dataset.Clip(str(wav), 0.0, 1.0, "hola", 1, 0.9, 1.0)], "src": P(wav), "none": None,
    }
    f = prj.save(tmp_path / "p", "censor", state, [str(wav)])
    assert f.suffix == ".parla" and prj.read_header(f)["kind"] == "censor"
    kind, st = prj.load(f)
    assert kind == "censor" and st["path"] == str(wav) and st["names"] == {0: "Ana", 3: "Luis"}
    assert st["span"] == (1.5, 2.5) and st["src"] == P(wav) and st["none"] is None
    assert np.abs(st["audio"] - long).max() < 1e-6 and st["audio"].dtype == np.float32
    assert np.array_equal(st["diar"].prob, diar.prob) and np.array_equal(st["diar"].active, diar.active)
    assert st["words"][0].speaker == 1 and st["speakers"][0].embedding.tolist() == [1, 1, 1, 1]
    assert st["separated"][0].voices[0].shape == (160,) and st["matches"][0].enabled is False
    assert st["clips"][0].text == "hola"


def test_project_relinks_moved_files(tmp_path):
    a = tmp_path / "old" / "a.wav"
    a.parent.mkdir()
    audio_io.save_wav(a, np.zeros((480, 1), np.float32), 48000)
    f = prj.save(tmp_path / "proj" / "x.parla", "dataset", {"files": [str(a)], "pre": {str(a): 1}}, [str(a)])
    new = tmp_path / "moved" / "deep" / "a.wav"
    new.parent.mkdir(parents=True)
    a.rename(new)
    found, missing = prj.resolve_files(prj.read_header(f)["files"], f)
    assert not found and [m["name"] for m in missing] == ["a.wav"]
    hits = prj.search_folder(tmp_path / "moved", {"a.wav"})
    assert hits == {"a.wav": str(new)}
    _kind, st = prj.load(f, {str(a): str(new)})
    assert st["files"] == [str(new)] and st["pre"] == {str(new): 1}          # every reference follows the file
    prj.relink(f, {str(a): str(new)})                                     # or rewrite the project without opening it
    found, missing = prj.resolve_files(prj.read_header(f)["files"], f)
    assert not missing and prj.load(f)[1]["files"] == [str(new)]


def test_project_rejects_unknown_types(tmp_path):
    import json
    import zipfile
    with pytest.raises(TypeError):
        prj.save(tmp_path / "bad", "leveler", {"x": object()}, [])
    f = tmp_path / "evil.parla"
    doc = {"app": "Parla Studio", "format": 1, "kind": "leveler", "files": [],
           "state": {"x": {"__dc__": "os:system", "f": {}}}}
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("project.json", json.dumps(doc))
    with pytest.raises(prj.ProjectError):
        prj.load(f)
    assert not (tmp_path / "bad.parla").exists() and not (tmp_path / "bad.parla.part").exists()


def test_transcription_pieces_cover_and_cut_at_pauses():
    sr = 16000
    assert models.transcription_pieces(sr * 60, sr) == [(0, sr * 60)]          # short audio: one call
    x = np.full(sr * 1000, 0.3, np.float32)
    quiet = [int(sr * 290.0), int(sr * 588.0)]                               # pauses inside the cut search windows
    for q in quiet:
        x[q:q + sr // 5] = 0.0
    pieces = models.transcription_pieces(x.size, sr, x)
    assert pieces[0][0] == 0 and pieces[-1][1] == x.size
    assert all(a == pb for (_, pb), (a, _) in zip(pieces, pieces[1:]))     # contiguous, no gaps or overlaps
    assert all(abs(b - q) < sr * 0.2 for (_, b), q in zip(pieces, quiet))   # each cut lands in the pause
    assert max(b - a for a, b in pieces) <= sr * models.CW_PIECE_S * 1.25


def test_settings_roundtrip(tmp_path, monkeypatch):
    from parla_studio import paths, settings
    monkeypatch.setattr(paths, "SETTINGS", tmp_path / "settings.json")
    assert settings.get("chime", True) is True
    settings.set("chime", False)
    settings.set("language", "es")
    assert settings.get("chime") is False and settings.get("language") == "es"


def test_worker_unloads_models_when_the_last_process_ends(monkeypatch):
    from PySide6.QtCore import QCoreApplication
    from parla_studio.ui import common
    QCoreApplication.instance() or QCoreApplication([])
    calls = []
    monkeypatch.setattr(models, "_cache", {"crisper": object()})
    monkeypatch.setattr(models, "unload", lambda name=None: calls.append(name) or models._cache.clear())
    monkeypatch.setattr(common, "chime", lambda kind="done": calls.append(kind))
    for fn in (lambda ctx: 1, lambda ctx: 1 / 0):
        models._cache["crisper"] = object()
        w = common.Worker(fn)
        w.run()                      # synchronous: same code path as the thread
    assert calls == [None, None] and not models._cache and common.Worker._running == 0


def test_transcribe_survives_an_empty_piece(monkeypatch):
    """CrisperWhisper returns words=None when every chunk of a piece is empty: no crash, one retry, logged."""
    from types import SimpleNamespace as NS
    sr = 16000
    calls, logs = [], []

    class Fake:
        def transcribe(self, x, **kw):
            calls.append(kw["hallucination_mitigation"])
            if len(calls) <= 2:          # first piece: empty twice (first try and retry)
                return NS(text=None, words=None)
            return NS(text="hola", words=[NS(word=" hola", start=1.0, end=1.5, probability=0.9)])

    monkeypatch.setattr(models, "crisper", lambda: Fake())
    monkeypatch.setattr(models, "CW_PIECE_S", 300.0)
    x = (0.1 * np.sin(np.arange(sr * 700) / 10)).astype(np.float32)
    ctx = lv.Ctx(log=lambda m, lvl: logs.append(lvl))
    text, words = models.transcribe(x, sr, "es", ctx=ctx)
    assert calls[:2] == [True, False] and text == "hola hola"
    assert [w.text for w in words] == ["hola", "hola"] and words[0].start > 280   # offset of the second piece (cut within the last 20 s)
    assert logs.count(1) == 2   # "came back empty, retrying" + "no transcribable speech"


def test_project_is_light_and_derived_audio_goes_to_the_media_folder(tmp_path):
    rng = np.random.default_rng(1)
    long = (rng.standard_normal((1_200_000, 2)) * 0.1).astype(np.float32)     # ~10 MB of float audio
    f = prj.save(tmp_path / "p.parla", "censor", {"voice": long, "small": np.ones(10, np.float32)}, [])
    media = prj.media_dir(f)
    assert f.stat().st_size < 50_000                                          # only the state
    assert len(list(media.glob("*.flac"))) == 1
    stamp = next(media.glob("*.flac")).stat().st_mtime_ns
    prj.save(f, "censor", {"voice": long, "small": np.ones(10, np.float32)}, [])
    assert next(media.glob("*.flac")).stat().st_mtime_ns == stamp             # unchanged audio is not rewritten
    _k, st = prj.load(f)
    assert np.abs(st["voice"] - long).max() < 1e-6 and "__missing_media__" not in st
    prj.save(f, "censor", {"voice": None}, [])                                # no longer used: removed
    assert not media.exists()
    prj.save(f, "censor", {"voice": long}, [])
    import shutil
    shutil.rmtree(media)                                                      # media folder deleted by the user
    _k, st = prj.load(f)
    assert st["voice"] is None and len(st["__missing_media__"]) == 1


def test_aligner_rejects_collapsed_words(monkeypatch):
    from parla_studio.core import aligner

    class Eng:
        def __init__(self, *a):
            pass

        def tokens(self, text):
            return [1]

        def chunk(self, x, texts):           # 2nd word squeezed to zero length
            return [(0.5, 0.8), (1.2, 1.2), (1.3, 1.6)]

    monkeypatch.setattr(aligner, "_Qwen", Eng)
    words = [models.Word("hola", 0.5, 0.8), models.Word("antes", 0.9, 1.25), models.Word("que", 1.3, 1.6)]
    out, rows = aligner.align(np.zeros(16000 * 3, np.float32), 16000, words, engine="qwen", language="es")
    assert [r.status for r in rows] == ["aligned", "collapsed", "aligned"]
    assert (out[1].start, out[1].end) == (0.9, 1.25)                         # CrisperWhisper's times kept


def test_censor_span_covers_both_estimates():
    cw = [models.Word("puta", 1.00, 1.30), models.Word("mierda", 2.00, 2.40)]
    al = [models.Word("puta", 1.05, 1.42), models.Word("mierda", 1.50, 2.30)]  # 2nd: aligner 500 ms early
    out = censor.union_spans(cw, al)
    assert (out[0].start, round(out[0].end, 2)) == (1.00, 1.42)               # aligner's later end kept
    assert round(out[1].start, 2) == 1.85 and out[1].end == 2.40              # extension capped at 150 ms


def _talk(n=40, word=0.3, pause_every=8, pause=0.5):
    out, t = [], 0.0
    for k in range(n):
        txt = f"w{k}" + ("." if (k + 1) % pause_every == 0 else "")
        out.append(models.Word(txt, t, t + word))
        t += word + (pause if (k + 1) % pause_every == 0 else 0.0)
    return out


def test_segment_never_makes_one_word_clips_when_the_mask_finds_no_pause():
    words = _talk()
    prm = dataset.DatasetParams(language="es")
    groups = dataset.segment(words, prm, None, np.zeros(len(words)))      # silence mask failed: every pause is 0
    durs = [words[j - 1].end - words[i].start for i, j in groups]
    assert all(j - i > 1 for i, j in groups) and min(durs) >= prm.min_s
    assert all(words[j - 1].text.endswith(".") for i, j in groups[:-1])  # cuts at the timestamps' pauses


def test_merge_clips_joins_times_and_transcript(tmp_path):
    sr = 16000
    words = _talk(n=24)
    dur = words[-1].end + 0.5
    src = dataset.Source(path=tmp_path / "a.wav", mono=np.zeros(int(dur * sr), np.float32), sr=sr, words=words,
                         speech=np.ones(int(dur / models.SAVA_FRAME_S) + 1, np.float32),
                         edb=np.zeros(int(dur / 0.01) + 1, np.float32), word_db=np.zeros(len(words), np.float32))
    prm = dataset.DatasetParams(language="es")
    a = dataset.make_clip(src, words[0].start, words[7].end, prm)
    b = dataset.make_clip(src, words[8].start, words[15].end, prm)
    m = dataset.merge_clips(src, [b, a], prm)
    assert (m.start, m.end) == (a.start, b.end) and m.text.split() == [w.text for w in words[:16]]
    b.text, b.text_edited = "corregido", True
    m = dataset.merge_clips(src, [a, b], prm)
    assert m.text_edited and m.text == a.text + " corregido"
    with pytest.raises(ValueError):
        dataset.merge_clips(src, [a], prm)


def test_dataset_without_sava_cuts_at_the_timestamps_pauses(tmp_path, monkeypatch):
    sr = 16000
    words = _talk(n=96)                                                              # ~33 s: needs several clips
    dur = words[-1].end + 0.5
    x = (0.1 * np.sin(np.arange(int(dur * sr)) / 7)).astype(np.float32)[:, None]   # tone under everything
    called = []
    monkeypatch.setattr(models, "speech_probability", lambda *a, **k: called.append(1))
    monkeypatch.setattr(models, "transcribe", lambda *a, **k: ("", [models.Word(w.text, w.start, w.end) for w in words]))
    prm = dataset.DatasetParams(language="es", isolate=False, align=False, use_sava=False, write_log=False)
    clips, src = dataset.build_clips(tmp_path / "a.wav", prm, lv.Ctx(), audio=(x, sr))
    assert not called and src.quiet is None
    assert len(clips) >= 3 and all(c.text.endswith(".") for c in clips)            # cut only at sentence pauses
    assert all(c.keep for c in clips)


def test_dataset_pre_cut_clips_are_only_transcribed(tmp_path, monkeypatch):
    sr = 16000
    words = _talk(n=96)                                                              # ~33 s: would be several clips
    dur = words[-1].end + 0.5
    x = (0.1 * np.sin(np.arange(int(dur * sr)) / 7)).astype(np.float32)[:, None]
    def boom(*a, **k):
        raise AssertionError("pre-cut clips use no aligner or SAVA")
    monkeypatch.setattr(models, "speech_probability", boom)
    from parla_studio.core import aligner
    monkeypatch.setattr(aligner, "align", boom)
    monkeypatch.setattr(models, "transcribe", lambda *a, **k: ("", [models.Word(w.text, w.start, w.end) for w in words]))
    prm = dataset.DatasetParams(language="es", isolate=False, align=True, use_sava=True, write_log=False,
                                pre_cut=True, max_s=12.0)
    clips, src = dataset.build_clips(tmp_path / "a.wav", prm, lv.Ctx(), audio=(x, sr))
    assert len(clips) == 1 and (clips[0].start, clips[0].end) == (0.0, pytest.approx(dur))
    assert clips[0].keep and clips[0].words == len(words)                          # not "too long"
    monkeypatch.setattr(models, "transcribe", lambda *a, **k: ("", []))
    clips, _ = dataset.build_clips(tmp_path / "a.wav", prm, lv.Ctx(), audio=(x, sr))
    assert len(clips) == 1 and not clips[0].keep and "no text" in clips[0].reason
    assert dataset.DatasetParams().pre_cut is False


def test_split_clip_at_a_point(tmp_path):
    sr = 16000
    words = _talk(n=16)
    dur = words[-1].end + 0.5
    src = dataset.Source(path=tmp_path / "a.wav", mono=np.zeros(int(dur * sr), np.float32), sr=sr, words=words,
                         speech=np.ones(int(dur / models.SAVA_FRAME_S) + 1, np.float32),
                         edb=np.zeros(int(dur / 0.01) + 1, np.float32), word_db=np.zeros(len(words), np.float32))
    prm = dataset.DatasetParams(language="es")
    c = dataset.make_clip(src, words[0].start, words[15].end, prm)
    t = (words[7].end + words[8].start) / 2                                  # in the pause after the sentence
    a, b = dataset.split_clip(src, c, t, prm)
    assert (a.start, a.end, b.start, b.end) == (c.start, t, t, c.end)
    assert a.text.split() == [w.text for w in words[:8]] and b.text.split() == [w.text for w in words[8:16]]
    c.text, c.text_edited = " ".join(f"x{k}" for k in range(16)), True    # corrected text: split at the same word
    a, b = dataset.split_clip(src, c, t, prm)
    assert a.text_edited and a.text.split()[-1] == "x7" and b.text.split()[0] == "x8"
    with pytest.raises(ValueError):
        dataset.split_clip(src, c, c.start + 0.01, prm)


def test_energy_speech_marks_talk_and_pauses():
    sr = 16000
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.001, sr * 6).astype(np.float32)            # quiet noise floor
    x[sr:2 * sr] += rng.normal(0, 0.2, sr).astype(np.float32)        # "speech" 1-2 s
    x[4 * sr:5 * sr] += rng.normal(0, 0.2, sr).astype(np.float32)    # "speech" 4-5 s
    p = smart.energy_speech(x, sr)
    t = (np.arange(p.size) + 0.5) * models.SAVA_FRAME_S
    assert p[(t > 1.2) & (t < 1.8)].min() > 0.9 and p[(t > 4.2) & (t < 4.8)].min() > 0.9
    assert p[(t > 2.5) & (t < 3.5)].max() < 0.1


def test_smart_without_sava_or_separation_needs_no_models(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no model should be used")
    monkeypatch.setattr(models, "speech_probability", boom)
    monkeypatch.setattr(models, "isolate_voice", boom)
    sr = 16000
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.001, (sr * 8, 1)).astype(np.float32)
    for a, g in ((1, 0.05), (4, 0.3)):                              # a quiet and a loud talker
        x[a * sr:(a + 2) * sr, 0] += rng.normal(0, g, 2 * sr).astype(np.float32)
    r = smart.run_smart(x, sr, smart.SmartParams(separate=False, use_sava=False))
    assert r.output.shape == x.shape and 0.2 < r.speech_ratio < 0.8


def test_dataset_sava_off_by_default_and_karaoke_engine():
    assert dataset.DatasetParams().use_sava is False
    assert "voc_fv7+gabox_denoise+becruily_karaoke" in models.ENGINES
    m = models.MSS_MODELS["becruily_karaoke"]
    assert m["yaml"].exists() and m.get("model_type") == "bs_roformer" and m["voice"] == "Vocals"
    for k in models.ENGINES:
        assert k == "inworld" or all(p in models.MSS_MODELS or p == "inworld" for p in k.split("+"))
        assert k in models.ENGINE_HINTS


def test_words_with_voice_drops_only_voiceless_words():
    speech = np.zeros(100, np.float32)          # 4 s of SAVA frames
    speech[10:40] = 0.9                          # voice 0.4-1.6 s
    ws = [models.Word("hola", 0.5, 0.9), models.Word("borde", 1.62, 1.8), models.Word("fantasma", 2.8, 3.2)]
    kept, dropped = models.words_with_voice(ws, speech)
    assert [w.text for w in kept] == ["hola", "borde"] and [w.text for w in dropped] == ["fantasma"]


def test_gate_with_speech_removes_speaker_outside_voice():
    from parla_studio.core import speakers as spk
    T = 600                                       # 6 s at 10 ms
    prob = np.zeros((T, 8), np.float32)
    prob[:, 0] = 0.9                              # "speaker" active the whole time (music included)
    d = spk.Diarization(prob, 0.01)
    speech = np.zeros(150, np.float32)
    speech[25:75] = 0.9                           # real voice 1-3 s
    g = spk.gate_with_speech(d, speech)
    segs = g.segments()
    assert len(segs) == 1 and abs(segs[0].start - 0.8) < 0.05 and abs(segs[0].end - 3.2) < 0.05
    assert "becruily_karaoke+gabox_denoise" in models.ENGINES


def test_projects_folder_lists_only_the_projects_of_each_tab(tmp_path):
    from parla_studio.core import project
    lib = tmp_path / "projects"
    (lib / "presets").mkdir(parents=True)
    project.save(lib / "voz.parla", "dataset", {"controls": {"a": 1}}, [])
    project.save(lib / "presets" / "suave.parla", "leveler", {"controls": {"b": 2}}, [])
    project.save(lib / "tacos.parla", "censor", {"controls": {}}, [])
    (lib / "roto.parla").write_bytes(b"not a zip")                     # damaged file: ignored
    (lib / "x.parla-media").mkdir()
    rec = project.RecentProjects(tmp_path / "recent.json", folder=lib)
    assert [it["name"] for it in rec.library("dataset")] == ["voz"]
    assert [it["name"] for it in rec.library("leveler")] == ["presets/suave"]
    assert [it["name"] for it in rec.library("censor")] == ["tacos"]
    assert rec.library("speakers") == []
    assert {it["kind"] for it in rec.library()} == {"dataset", "leveler", "censor"}
    # a recent project from elsewhere plus the folder: each one once, still filtered by tab
    other = tmp_path / "otro.parla"
    project.save(other, "dataset", {"controls": {}}, [])
    rec.touch(other, "dataset")
    rec.touch(lib / "voz.parla", "dataset")
    names = [it["name"] for it in rec.everything("dataset")]
    assert sorted(names) == ["otro", "voz"] and len(names) == 2
    assert all(it["kind"] == "dataset" for it in rec.everything("dataset"))
    assert project.load(lib / "presets" / "suave.parla")[1]["controls"] == {"b": 2}   # works as a preset


def test_default_engine_is_lead_voice_only_everywhere():
    assert models.DEFAULT_ENGINE == "voc_fv7+gabox_denoise+becruily_karaoke" and models.DEFAULT_ENGINE in models.ENGINES
    assert dataset.DatasetParams().engine == models.DEFAULT_ENGINE
    assert smart.SmartParams().engine == models.DEFAULT_ENGINE


def test_project_saves_aligner_rows(tmp_path):
    from parla_studio.core import aligner, project
    rows = [aligner.AlignRow("hola", 0.1, 0.4, 0.9, 0.12, 0.38, "aligned")]
    p = project.save(tmp_path / "a.parla", "dataset", {"sources": {"x": {"rows": rows}}}, [])
    back = project.load(p)[1]["sources"]["x"]["rows"]
    assert back[0].text == "hola" and back[0].status == "aligned" and isinstance(back[0], aligner.AlignRow)


def test_memory_budget_scales_with_the_hardware():
    from parla_studio.core import memory
    y = models.MSS_MODELS["voc_fv7"]["yaml"]
    # separation block: grows with free RAM, clamped to 30..600 s
    assert memory.separation_block_s("voc_fv7", 2, ram_gb=64) == 600
    assert memory.separation_block_s("voc_fv7", 2, ram_gb=3.0) == 30   # nothing above the reserve: minimum
    mid = memory.separation_block_s("voc_fv7", 2, ram_gb=4.5)
    assert 30 < mid < 600
    # voice-isolator is also limited by VRAM
    assert memory.separation_block_s("inworld", 2, ram_gb=64, vram_gb=4) < memory.separation_block_s("inworld", 2, ram_gb=64, vram_gb=22)
    # GPU batch: 1 on a small GPU or CPU, more on a big one, never above the cap
    assert memory.mss_batch(y, vram_gb=0) == 1 and memory.mss_batch(y, vram_gb=3) == 1
    assert 1 < memory.mss_batch(y, vram_gb=22) <= memory.MSS_BATCH_MAX
    k = models.MSS_MODELS["becruily_karaoke"]["yaml"]   # longer chunks (20 s) → smaller batch for the same VRAM
    assert memory.mss_batch(k, vram_gb=12) <= memory.mss_batch(y, vram_gb=12)
    # SAVA window: whole minutes, 1..60, more with more VRAM
    assert memory.sava_window_s(vram_gb=2) < memory.sava_window_s(vram_gb=20) <= 3600
    assert memory.sava_window_s(vram_gb=20) % 60 == 0
    assert memory.ram_free_gb() > 0


def test_auto_is_the_default_block_and_window():
    assert dataset.DatasetParams().sep_block_s is None
    p = smart.SmartParams()
    assert p.sep_block_s is None and p.sava_window_s is None


def test_wrapped_label_gets_the_height_of_all_its_lines():
    # in its own process: a QApplication inside the test run would clash with the other tests
    import os
    import subprocess
    import sys
    code = """
import sys
sys.path.insert(0, %r)
from PySide6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
app = QApplication([])
from qfluentwidgets import CaptionLabel
from parla_studio.ui.level_page import fit
w = QWidget()
lay = QVBoxLayout(w)
lbl = fit(CaptionLabel("short"))
row = QHBoxLayout()
row.addWidget(lbl, 1)
lay.addLayout(row)
lay.addStretch(1)
w.resize(220, 400)
w.show()
lbl.setText("Spanish: 1828 active terms (custom 565 · textdetox 1195 · ldnoobw 68)")
for _ in range(5):
    app.processEvents()
assert lbl.height() >= lbl.heightForWidth(lbl.width()) > lbl.fontMetrics().lineSpacing(), (lbl.height(), lbl.heightForWidth(lbl.width()))
print("ok")
""" % str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_english_is_the_default_transcription_language():
    src = Path(__file__).resolve().parents[1] / "src" / "parla_studio" / "ui"
    for page in ("dataset_page.py", "censor_page.py", "speakers_page.py"):
        assert 'self.lang.setCurrentText("English")' in (src / page).read_text(encoding="utf-8")


def test_projects_go_to_their_tab_folder(tmp_path):
    from parla_studio.core import project
    root = tmp_path / "projects"
    (root / "presets").mkdir(parents=True)
    (root / "censor").mkdir()
    a = project.save(root / "voz.parla", "dataset", {"x": np.ones(2_000_000, np.float32) * 0.1}, [])   # with media
    b = project.save(root / "presets" / "suave.parla", "leveler", {"controls": {}}, [])
    c = project.save(root / "censor" / "mal.parla", "dataset", {"controls": {}}, [])   # in another tab's folder
    d = project.save(root / "dataset" / "ok.parla", "dataset", {"controls": {}}, [])    # already in place
    taken = project.save(root / "dataset" / "voz.parla", "dataset", {"controls": {}}, [])
    assert project.media_dir(a).is_dir()
    rec = project.RecentProjects(tmp_path / "recent.json", folder=root)
    rec.touch(a, "dataset")
    moved = dict(project.organize(root))
    assert moved[a] == root / "dataset" / "voz (2).parla"            # never overwrites
    assert project.media_dir(moved[a]).is_dir() and not project.media_dir(a).exists()
    assert moved[b] == root / "leveler" / "presets" / "suave.parla"
    assert moved[c] == root / "dataset" / "mal.parla"
    assert d not in moved and taken not in moved and d.exists()
    assert project.load(moved[a])[1]["x"].shape == (2_000_000,)      # the media folder still matches
    for old, new in moved.items():
        rec.rename(old, new)
    assert rec.of("dataset")[0]["path"] == str(moved[a].resolve())
    assert sorted(it["name"] for it in rec.library("dataset")) == ["mal", "ok", "voz", "voz (2)"]
    assert [it["name"] for it in rec.library("leveler")] == ["presets/suave"]
    assert project.organize(root) == []                               # a second pass moves nothing
