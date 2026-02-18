#!/usr/bin/env python3
# batch_transcribe_fw.py
#
# Whisper-only batch transcription with:
#  - audio preprocessing to 16k mono wav (ffmpeg)
#  - SolidWorks low-volume fallback: auto-retry with VAD OFF + "skip less"
#  - temperature schedule: try multiple temps and choose best output
#  - optional de-looping on BIN text to reduce repetitive hallucinations
#
# Example:
#   python -u batch_transcribe_fw.py --root . --bins 10 20 30 --outfolder whisper_large_results --nlp-clean --clean-outfolder --sw-no-vad-fallback

from faster_whisper import WhisperModel
import srt
import datetime as dt
from pathlib import Path
import argparse
import csv
import math
import re
import subprocess
import time
import inspect
import shutil
import hashlib

# -----------------------------
# helpers
# -----------------------------

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def wipe_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def clean_text_for_nlp(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    # keep letters/numbers/spaces/apostrophes/hyphens
    text = re.sub(r"[^A-Za-z0-9\s'\-]", "", text)
    return text.strip()

def write_srt(subs, out_path: Path):
    safe_mkdir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

def write_phrases_csv(phrase_rows, out_path: Path):
    safe_mkdir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_index", "start_s", "end_s", "text"])
        for r in phrase_rows:
            w.writerow([r["i"], f"{r['start']:.3f}", f"{r['end']:.3f}", r["text"]])

def write_words_csv(words, out_path: Path):
    safe_mkdir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word_index", "start_s", "end_s", "word"])
        for i, item in enumerate(words, start=1):
            w.writerow([i, f"{item['start']:.3f}", f"{item['end']:.3f}", item["text"]])

def write_bins_csv(records, out_path: Path):
    safe_mkdir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_index", "start_s", "end_s", "n_words", "text"])
        for r in records:
            w.writerow([r["bin_index"], f"{r['start_s']:.3f}", f"{r['end_s']:.3f}", r["n_words"], r["text"]])

def records_to_srt(records):
    subs = []
    for i, r in enumerate(records, start=1):
        subs.append(
            srt.Subtitle(
                index=i,
                start=dt.timedelta(seconds=float(r["start_s"])),
                end=dt.timedelta(seconds=float(r["end_s"])),
                content=(r["text"] or "").strip(),
            )
        )
    return subs

def words_from_segments(segs):
    words = []
    for seg in segs:
        for w in (getattr(seg, "words", None) or []):
            raw = w.word if w.word is not None else ""
            txt = raw.strip()
            if txt == "":
                continue
            words.append({"start": float(w.start), "end": float(w.end), "raw": raw, "text": txt})
    words.sort(key=lambda x: x["start"])
    return words

def make_binned_records(words, bin_size_s: float, include_empty: bool = False):
    if not words:
        return []

    max_t = max(w["end"] for w in words)
    n_bins = int(math.ceil(max_t / bin_size_s))

    records = []
    wi = 0

    for b in range(n_bins):
        t0 = b * bin_size_s
        t1 = min((b + 1) * bin_size_s, max_t)

        while wi < len(words) and words[wi]["start"] < t0:
            wi += 1

        bucket = []
        wj = wi
        while wj < len(words) and words[wj]["start"] < t1:
            bucket.append(words[wj])
            wj += 1

        raw_text = "".join(w["raw"] for w in bucket).strip()

        if raw_text or include_empty:
            records.append({
                "bin_index": b + 1,
                "start_s": t0,
                "end_s": t1,
                "n_words": len(bucket),
                "text": raw_text if raw_text else "",
            })

    return records

def ffprobe_duration_seconds(media_path: Path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(media_path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except Exception:
        return None

def pick_device_and_compute(force_device: str | None, force_compute: str | None):
    if force_device and force_compute:
        return force_device, force_compute

    device = "cpu"
    compute = "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device = "cuda"
            compute = "float16"
        else:
            device = "cpu"
            compute = "int8"
    except Exception:
        device = "cpu"
        compute = "int8"

    if force_device:
        device = force_device
    if force_compute:
        compute = force_compute

    return device, compute

def make_prefix(video_path: Path) -> str:
    stem = video_path.stem
    stem = stem.replace(" Video", "")
    stem = re.sub(r"\s+", "_", stem.strip())
    return stem

def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    except Exception:
        return kwargs

def _hash_str(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()[:8]

def ensure_wav_16k_mono(src: Path, cache_dir: Path, audio_filter: str = "") -> Path:
    safe_mkdir(cache_dir)
    h = _hash_str(audio_filter.strip())
    out_wav = cache_dir / (src.stem.replace(" ", "_") + f"__{h}__16k_mono.wav")

    try:
        if out_wav.exists() and out_wav.stat().st_mtime >= src.stat().st_mtime:
            return out_wav
    except Exception:
        pass

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg or run with --no-preprocess-audio.")

    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    if audio_filter.strip():
        cmd += ["-af", audio_filter.strip()]
    cmd += [str(out_wav)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_wav

def default_domain_prompt() -> str:
    return (
        "Think-aloud technical CAD modeling session. "
        "Terms: AR-CAD, SolidWorks, STL, CAD, sketch, extrude, revolve, constraints, feature tree."
    )

def video_kind(video: Path, arcad_suffix: str, sw_suffix: str):
    low = video.name.lower()
    if low.endswith(arcad_suffix.lower()):
        return "arcad"
    if low.endswith(sw_suffix.lower()):
        return "solidworks"
    return "other"

# -----------------------------
# de-looping (light postprocess)
# -----------------------------

_word_re = re.compile(r"\b[\w']+\b")

def _dedupe_runs(tokens, max_run=3):
    out = []
    run = 0
    prev = None
    for t in tokens:
        if prev is not None and t.lower() == prev.lower():
            run += 1
        else:
            run = 1
        if run <= max_run:
            out.append(t)
        prev = t
    return out

def reduce_repetition_text(text: str) -> str:
    """
    Safe-ish: removes extreme immediate repeats (word runs).
    Doesn't try to rewrite sentences.
    """
    toks = _word_re.findall(text or "")
    if len(toks) < 10:
        return text
    toks2 = _dedupe_runs(toks, max_run=3)
    # keep original spacing roughly by joining with space
    return " ".join(toks2)

def repetition_score(text: str) -> float:
    toks = [t.lower() for t in _word_re.findall(text or "")]
    if len(toks) < 10:
        return 0.0
    uniq = len(set(toks))
    return 1.0 - (uniq / max(1, len(toks)))  # higher = more repetitive

# -----------------------------
# decode selection / retries
# -----------------------------

def score_decode(segs) -> float:
    # Prefer more words, penalize repetition.
    full = " ".join([(getattr(s, "text", "") or "").strip() for s in segs]).strip()
    toks = _word_re.findall(full)
    n = len(toks)
    rep = repetition_score(full)
    # If it's super repetitive, punish hard
    return float(n) - 80.0 * float(rep)

def transcribe_best_of_temps(model: WhisperModel, audio_path: Path, base_kwargs: dict, temps: list[float]):
    best = None
    best_info = None
    best_score = -1e18
    best_temp = None

    for t in temps:
        kwargs = dict(base_kwargs)
        kwargs["temperature"] = float(t)
        kwargs = _filter_kwargs_for_callable(model.transcribe, kwargs)

        segments, info = model.transcribe(str(audio_path), **kwargs)
        segs = list(segments)

        sc = score_decode(segs)
        if sc > best_score:
            best_score = sc
            best = segs
            best_info = info
            best_temp = t

    return best or [], best_info, best_temp, best_score

# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch transcribe participant folders (Whisper-only, SW fallback).")
    ap.add_argument("--root", default=".", help="Root folder containing participant subfolders")
    ap.add_argument("--outfolder", default="whisper_results", help="Output folder created inside each participant folder")

    ap.add_argument("--model", default="large-v3", help="Whisper model")
    ap.add_argument("--beam", type=int, default=5, help="Beam size")
    ap.add_argument("--best-of", type=int, default=5, help="best_of (if supported)")
    ap.add_argument("--language", default="en", help="Force language (en recommended). Use 'auto' to disable forcing.")

    ap.add_argument("--no-vad", action="store_true", help="Disable VAD for ALL videos")
    ap.add_argument("--sw-no-vad-fallback", action="store_true",
                    help="If SolidWorks output is too small, auto-retry with VAD OFF")

    ap.add_argument("--vad-min-silence-ms", type=int, default=800)
    ap.add_argument("--vad-speech-pad-ms", type=int, default=200)
    ap.add_argument("--vad-max-speech-s", type=float, default=30.0)

    ap.add_argument("--prompt", default="", help="Optional domain prompt. If empty uses built-in.")
    ap.add_argument("--no-prompt", action="store_true", help="Disable prompt")

    ap.add_argument("--bins", nargs="*", type=float, default=[10, 20, 30])
    ap.add_argument("--include-empty", action="store_true")
    ap.add_argument("--nlp-clean", action="store_true")

    ap.add_argument("--no-preprocess-audio", action="store_true")
    # Good defaults:
    ap.add_argument("--audio-filter", default="highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
                    help="ffmpeg -af filter for ALL videos")
    ap.add_argument("--audio-filter-sw", default="highpass=f=80,lowpass=f=8000,compand=attacks=0.05:decays=0.2:points=-80/-80|-30/-12|0/-3,loudnorm=I=-16:TP=-1.5:LRA=11",
                    help="ffmpeg -af filter for SolidWorks videos (low volume). If empty, uses --audio-filter.")

    ap.add_argument("--clean-outfolder", action="store_true",
                    help="Delete the existing output folder contents before writing new results")

    # decode “skip less” knobs (passed only if supported by your faster-whisper version)
    ap.add_argument("--compression-ratio-threshold", type=float, default=2.2,
                    help="Lower catches repetitive loops sooner (if supported)")
    ap.add_argument("--logprob-threshold", type=float, default=-2.0,
                    help="More negative = skip less (if supported)")
    ap.add_argument("--no-speech-threshold", type=float, default=0.95,
                    help="Higher = skip less quiet speech (if supported)")
    ap.add_argument("--condition-on-previous-text", action="store_true",
                    help="Enable context carry-over (can increase loops; default OFF)")

    ap.add_argument("--temps", nargs="*", type=float, default=[0.0, 0.2, 0.4, 0.6],
                    help="Temperature schedule to try; best output is selected")

    ap.add_argument("--bin-dedupe-loops", action="store_true",
                    help="Apply light de-looping on bin text (reduces repetitive hallucinations)")

    # performance knobs
    ap.add_argument("--device", default=None)
    ap.add_argument("--compute", default=None)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)

    # file matching
    ap.add_argument("--sw-suffix", default="SW Video.mp4")
    ap.add_argument("--arcad-suffix", default="ARCAD Video.mp4")

    # compatibility flags (ignored)
    ap.add_argument("--use-txt-for-bins", action="store_true", help=argparse.SUPPRESS)

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    device, compute = pick_device_and_compute(args.device, args.compute)

    prompt_text = ""
    if not args.no_prompt:
        prompt_text = args.prompt.strip() if args.prompt.strip() else default_domain_prompt()

    print("=" * 72, flush=True)
    print("Batch Faster-Whisper Transcription (Whisper-only)", flush=True)
    print(f"Python:     {shutil.which('python') or ''}", flush=True)
    print(f"Root:       {root}", flush=True)
    print(f"Model:      {args.model}", flush=True)
    print(f"Device:     {device}", flush=True)
    print(f"Compute:    {compute}", flush=True)
    print(f"Beam/BestOf:{args.beam}/{args.best_of}", flush=True)
    print(f"Temps:      {args.temps}", flush=True)
    print(f"VAD:        {'OFF' if args.no_vad else 'ON'} (max_speech={args.vad_max_speech_s}s)", flush=True)
    print(f"SW fallback:{args.sw_no_vad_fallback}", flush=True)
    print(f"Clean out:  {args.clean_outfolder}", flush=True)
    print(f"Bins:       {args.bins}", flush=True)
    print(f"Outfolder:  {args.outfolder}", flush=True)
    print("=" * 72, flush=True)

    model_kwargs = dict(device=device, compute_type=compute)
    if args.cpu_threads and args.cpu_threads > 0:
        model_kwargs["cpu_threads"] = args.cpu_threads
    if args.num_workers and args.num_workers > 0:
        model_kwargs["num_workers"] = args.num_workers

    print("Loading Whisper model...", flush=True)
    t_load = time.time()
    model = WhisperModel(args.model, **model_kwargs)
    print(f"Model loaded in {time.time() - t_load:.1f}s\n", flush=True)

    participant_dirs = [p for p in root.iterdir() if p.is_dir()]
    participant_dirs.sort(key=lambda p: p.name.lower())

    def match_videos(pdir: Path):
        vids = []
        for f in pdir.iterdir():
            if not f.is_file():
                continue
            low = f.name.lower()
            if low.endswith(args.sw_suffix.lower()) or low.endswith(args.arcad_suffix.lower()):
                vids.append(f)
        vids.sort(key=lambda p: p.name.lower())
        return vids

    total_videos = sum(len(match_videos(pd)) for pd in participant_dirs)
    done_videos = 0
    print(f"Found {len(participant_dirs)} participant folders, {total_videos} target videos.\n", flush=True)

    for pd in participant_dirs:
        vids = match_videos(pd)
        if not vids:
            continue

        outdir = pd / args.outfolder

        if args.clean_outfolder:
            if outdir.parent == pd and outdir.name == args.outfolder:
                wipe_dir(outdir)
            else:
                raise RuntimeError(f"Refusing to clean unexpected outdir path: {outdir}")
        else:
            safe_mkdir(outdir)

        audio_cache_dir = outdir / "_audio_cache"
        safe_mkdir(audio_cache_dir)

        print("-" * 72, flush=True)
        print(f"Participant: {pd.name}", flush=True)
        print(f"Output dir:  {outdir}", flush=True)
        print(f"Videos:      {len(vids)}", flush=True)
        print("-" * 72, flush=True)

        for video in vids:
            done_videos += 1
            prefix = make_prefix(video)
            duration = ffprobe_duration_seconds(video)
            kind = video_kind(video, args.arcad_suffix, args.sw_suffix)

            print(f"\n[{done_videos}/{total_videos}] Transcribing: {video.name}", flush=True)
            if duration:
                print(f"Duration: {duration/60:.1f} min", flush=True)

            # Choose audio filter
            audio_filter = args.audio_filter
            if kind == "solidworks" and (args.audio_filter_sw or "").strip():
                audio_filter = args.audio_filter_sw

            audio_path = video
            if not args.no_preprocess_audio:
                try:
                    audio_path = ensure_wav_16k_mono(video, audio_cache_dir, audio_filter)
                except Exception as e:
                    print(f"WARNING: preprocess failed ({e}); using original media.", flush=True)
                    audio_path = video

            vad_params = dict(
                min_silence_duration_ms=int(args.vad_min_silence_ms),
                speech_pad_ms=int(args.vad_speech_pad_ms),
                max_speech_duration_s=float(args.vad_max_speech_s),
            )

            base_kwargs = dict(
                word_timestamps=True,
                beam_size=int(args.beam),
                best_of=int(args.best_of),
                task="transcribe",
                language=None if args.language.strip().lower() == "auto" else args.language.strip(),
                initial_prompt=prompt_text if prompt_text else None,
                prompt=prompt_text if prompt_text else None,
                condition_on_previous_text=bool(args.condition_on_previous_text),

                # These are only used if your faster-whisper build supports them:
                compression_ratio_threshold=float(args.compression_ratio_threshold),
                log_prob_threshold=float(args.logprob_threshold),
                no_speech_threshold=float(args.no_speech_threshold),
            )

            # Pass 1: normal VAD unless globally disabled
            use_vad_1 = (not args.no_vad)
            base_kwargs_1 = dict(base_kwargs)
            base_kwargs_1["vad_filter"] = bool(use_vad_1)
            base_kwargs_1["vad_parameters"] = vad_params if use_vad_1 else None

            segs1, info1, temp1, sc1 = transcribe_best_of_temps(model, audio_path, base_kwargs_1, list(args.temps))

            # If SolidWorks and output is tiny, retry with VAD OFF (quiet speech often gets cut by VAD)
            segs = segs1
            info = info1
            chosen = f"pass1(vad={'ON' if use_vad_1 else 'OFF'}, temp={temp1})"

            def _n_words(segs_):
                return len(_word_re.findall(" ".join([(getattr(s, 'text', '') or '') for s in segs_])))

            if kind == "solidworks" and args.sw_no_vad_fallback:
                nw1 = _n_words(segs1)
                # heuristic: if we got almost nothing, retry
                if nw1 < 80 and (duration or 0) > 120:
                    base_kwargs_2 = dict(base_kwargs)
                    base_kwargs_2["vad_filter"] = False
                    base_kwargs_2["vad_parameters"] = None

                    segs2, info2, temp2, sc2 = transcribe_best_of_temps(model, audio_path, base_kwargs_2, list(args.temps))
                    nw2 = _n_words(segs2)

                    # choose better score; prefer more words if scores close
                    if (sc2 > sc1) or (nw2 > nw1 * 1.15):
                        segs = segs2
                        info = info2
                        chosen = f"pass2(vad=OFF, temp={temp2})"
                        print(f"SW fallback used: pass1_words={nw1}, pass2_words={nw2}", flush=True)

            lang = getattr(info, "language", "unknown") if info is not None else "unknown"
            lp = getattr(info, "language_probability", 0.0) if info is not None else 0.0
            print(f"Detected language: {lang} ({lp:.1%}) | chosen={chosen}", flush=True)
            print(f"Finished decoding. Total segments: {len(segs)}", flush=True)

            # 1) Phrase-level outputs
            phrase_subs = []
            phrase_rows = []
            for i, seg in enumerate(segs, start=1):
                t0 = float(getattr(seg, "start", 0.0))
                t1 = float(getattr(seg, "end", 0.0))
                txt = (getattr(seg, "text", "") or "").strip()
                phrase_subs.append(
                    srt.Subtitle(index=i, start=dt.timedelta(seconds=t0), end=dt.timedelta(seconds=t1), content=txt)
                )
                phrase_rows.append({"i": i, "start": t0, "end": t1, "text": txt})

            write_srt(phrase_subs, outdir / f"{prefix}_phrases.srt")
            write_phrases_csv(phrase_rows, outdir / f"{prefix}_phrases.csv")

            # 2) Word-level outputs
            words = words_from_segments(segs)

            word_subs = []
            for j, w in enumerate(words, start=1):
                word_subs.append(
                    srt.Subtitle(
                        index=j,
                        start=dt.timedelta(seconds=float(w["start"])),
                        end=dt.timedelta(seconds=float(w["end"])),
                        content=w["text"],
                    )
                )
            write_srt(word_subs, outdir / f"{prefix}_words.srt")
            write_words_csv(words, outdir / f"{prefix}_words.csv")

            # 3) Fixed-time bins
            for bs in args.bins:
                if bs <= 0:
                    continue
                tag = (str(bs).replace(".", "p")) + "s"
                records = make_binned_records(words, float(bs), include_empty=args.include_empty)

                if args.bin_dedupe_loops:
                    for r in records:
                        r["text"] = reduce_repetition_text(r.get("text", ""))

                if args.nlp_clean:
                    for r in records:
                        r["text"] = clean_text_for_nlp(r.get("text", ""))

                write_srt(records_to_srt(records), outdir / f"{prefix}_bin{tag}.srt")
                write_bins_csv(records, outdir / f"{prefix}_bin{tag}.csv")

            print(f"✓ Saved: {prefix}_phrases.(srt/csv), {prefix}_words.(srt/csv), bins: {args.bins}", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()

