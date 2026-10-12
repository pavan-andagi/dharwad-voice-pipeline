"""
English speech (mic) -> English text -> Kannada -> Dharwad Kannada (your model) -> speaker

Stages
  1. ASR            : faster-whisper (English)
  2. EN -> KN       : ai4bharat/indictrans2-en-indic-dist-200M (eng_Latn -> kan_Knda) [gated: accept terms on HF]
  3. KN -> Dharwad  : YOUR fine-tuned IndicTrans2 indic-indic-1B checkpoint
  4. TTS            : ai4bharat/indic-parler-tts (Kannada)

LOW-RAM MODE: this version loads ONE model into memory at a time, uses it, then
releases it before loading the next. This is necessary on machines with ~8 GB RAM
or less, since all four models loaded simultaneously would exceed available memory.
It is slower per turn (every stage reloads its model each time) but it fits.
If you later run this on a machine with more RAM (16 GB+) or a GPU, switch to the
"load everything once" version instead for much faster turns.

Install (Python 3.12 venv, WSL2/Linux recommended on Windows):
  pip install torch transformers sentencepiece sacremoses sounddevice soundfile numpy faster-whisper
  pip install indictranstoolkit indic-nlp-library
  pip install git+https://github.com/huggingface/parler-tts.git --no-deps
  python -c "from huggingface_hub import login; login()"   # needed for the gated en-indic model
"""

import ctypes
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from IndicTransToolkit.processor import IndicProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def report_memory(label: str):
    """Print this process's actual resident memory — proof of what's really loaded."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    print(f"  [mem] {label}: {line.split()[1]} {line.split()[2]}")
                    return
    except FileNotFoundError:
        pass  # not on Linux


def free_memory(*objs):
    """Drop references, force garbage collection, and — critically on Linux — force
    glibc to actually hand freed memory back to the OS. del + gc.collect() alone
    often leaves memory held by the process's allocator for reuse, which `free -h`
    still counts as used and can accumulate across stages until the system runs
    out of real memory. malloc_trim(0) fixes that."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _libc is not None:
        _libc.malloc_trim(0)

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # will resolve to "cpu" — no GPU on this machine
# bfloat16 on CPU: roughly half the memory of float32, and — unlike float16 —
# has solid CPU operator support in PyTorch, so it won't hit "not implemented
# for CPU" errors the way float16 sometimes does. This is what makes
# preloading all four models feasible on a low-RAM machine.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.bfloat16

MIC_SR = 16_000                      # Whisper expects 16 kHz mono
WHISPER_SIZE = "tiny.en"             # smallest English model — more headroom on this machine; "base.en"/"small.en" for better accuracy if RAM allows

EN_KN_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"  # smaller distilled model — far less RAM than the 1B version
DIALECT_MODEL = "pavan007123/dharwad-kannada-indictrans2"  # your fine-tuned, merged checkpoint on HF Hub

# IMPORTANT: use exactly the language tags / preprocessing you used while fine-tuning.
# If you trained kan_Knda -> kan_Knda, keep these as-is. If you introduced a custom
# target tag for the dialect, put it here.
DIALECT_SRC_TAG = "kan_Knda"
DIALECT_TGT_TAG = "kan_Knda"

TTS_MODEL = "ai4bharat/indic-parler-tts"
# Check the model card for the recommended Kannada speaker names and use one here.
TTS_DESCRIPTION = (
    "A female speaker delivers a clear, natural Kannada speech with a moderate "
    "speed and pitch. The recording is of very high quality, with no background noise."
)

# Every turn's English/Kannada/Dharwad text gets appended here, UTF-8, so the
# Kannada script renders correctly in any real text editor regardless of how
# the terminal itself displays it. ONE file for every run and every turn —
# always opened in append mode, so nothing already in the file is ever
# overwritten, no matter how many times the script is restarted.
OUTPUT_LOG = Path("translations_log.txt")


def log_turn(english: str, kannada: str, dharwad: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        f.write(f"English : {english}\n")
        f.write(f"Kannada : {kannada}\n")
        f.write(f"Dharwad : {dharwad}\n")
        f.write("-" * 60 + "\n")


# --------------------------------------------------------------------------- #
# STAGE 1: microphone + English ASR
# --------------------------------------------------------------------------- #
def record_until_enter() -> np.ndarray:
    """Push-to-talk: press Enter to start, Enter again to stop. Returns float32 mono @16 kHz."""
    frames = []

    def cb(indata, n_frames, time_info, status):
        frames.append(indata.copy())

    input("\n[Enter] to start recording...")
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", callback=cb):
        input("Recording... [Enter] to stop.")
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames, axis=0).squeeze()


class ASR:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_SIZE,
            device=DEVICE,
            compute_type="float16" if DEVICE == "cuda" else "int8",
        )

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()


# --------------------------------------------------------------------------- #
# STAGES 2 & 3: IndicTrans2 wrapper (same code serves stock and fine-tuned models)
# --------------------------------------------------------------------------- #
class IndicTrans2:
    def __init__(self, name_or_path: str):
        self.tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                name_or_path, trust_remote_code=True, torch_dtype=DTYPE
            )
            .to(DEVICE)
            .eval()
        )
        self.ip = IndicProcessor(inference=True)

    @torch.inference_mode()
    def __call__(self, sentences, src_lang: str, tgt_lang: str, num_beams: int = 5):
        batch = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        self.tok.src_lang = src_lang  # matches training-notebook inference call
        inputs = self.tok(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)
        out = self.model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=num_beams,
            num_return_sequences=1,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        decoded = self.tok.batch_decode(
            out.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,  # matches training-notebook inference settings
        )
        return self.ip.postprocess_batch(decoded, lang=tgt_lang)


# --------------------------------------------------------------------------- #
# STAGE 4: Kannada TTS + playback
# --------------------------------------------------------------------------- #
class TTS:
    def __init__(self):
        # bfloat16 causes "Got unsupported ScalarType BFloat16" inside parler-tts's
        # audio codec layers on CPU — those ops have no bf16 CPU kernel. Load in
        # float32, then apply PyTorch's own dynamic quantization to the Linear
        # layers (most of this 0.9B-param model's weights) — converts them to
        # int8 in memory, roughly a 4x reduction vs float32 for those layers,
        # without needing a different checkpoint or any extra download.
        model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=torch.float32).to(DEVICE)
        self.model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.prompt_tok = AutoTokenizer.from_pretrained(TTS_MODEL)
        # Parler uses a separate tokenizer for the voice description
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sr = self.model.config.sampling_rate

    @torch.inference_mode()
    def speak(self, text: str, description: str = TTS_DESCRIPTION):
        desc = self.desc_tok(description, return_tensors="pt").to(DEVICE)
        prompt = self.prompt_tok(text, return_tensors="pt").to(DEVICE)
        audio = self.model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        wav = audio.cpu().numpy().squeeze().astype(np.float32)
        sd.play(wav, self.sr)
        sd.wait()
        return wav


# --------------------------------------------------------------------------- #
# MAIN LOOP — fully sequential: only ONE model resident at a time, always.
#
# Both "keep everything loaded" and "keep the light two loaded" were tested
# and overshot this machine's ~6.8 GB RAM (swap engaged, available dropped
# to double-digit MB). This machine's real budget only supports one model
# in memory at once, even in bfloat16. Every stage loads fresh, runs, and
# is released before the next stage starts — slower per turn, but stable.
# --------------------------------------------------------------------------- #
def main():
    print(f"Device: {DEVICE}, dtype: {DTYPE}. Fully sequential mode — one model at a time.")
    print("Ready. Ctrl+C to quit.")

    while True:
        audio = record_until_enter()
        if audio.size < MIC_SR // 2:          # <0.5 s -> ignore
            print("Too short, try again.")
            continue

        report_memory("before ASR")
        print("Loading ASR...")
        asr = ASR()
        report_memory("ASR loaded")
        english = asr(audio)
        free_memory(asr)
        report_memory("ASR released")
        if not english:
            print("Nothing recognised, try again.")
            continue
        print(f"[EN ] {english}")

        print("Loading EN->KN model...")
        en_kn = IndicTrans2(EN_KN_MODEL)
        report_memory("EN->KN loaded")
        kannada = en_kn([english], src_lang="eng_Latn", tgt_lang="kan_Knda")[0]
        free_memory(en_kn)
        report_memory("EN->KN released")
        print(f"[KN ] {kannada}")

        print("Loading Dharwad model...")
        kn_dharwad = IndicTrans2(DIALECT_MODEL)
        report_memory("Dharwad loaded")
        dharwad = kn_dharwad(
            [kannada], src_lang=DIALECT_SRC_TAG, tgt_lang=DIALECT_TGT_TAG
        )[0]
        free_memory(kn_dharwad)
        report_memory("Dharwad released")
        print(f"[DWD] {dharwad}")

        log_turn(english, kannada, dharwad)
        print(f"(saved to {OUTPUT_LOG.resolve()})")

        print("Loading TTS...")
        try:
            tts = TTS()
            report_memory("TTS loaded")
            tts.speak(dharwad)
            free_memory(tts)
            report_memory("TTS released")
        except Exception as e:
            print(f"TTS failed ({e}) — translation was still saved to {OUTPUT_LOG.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
