"""
English speech (mic) -> English text -> Kannada -> Dharwad Kannada (your model) -> speaker

Stages
  1. ASR            : faster-whisper (English)
  2. EN -> KN       : ai4bharat/indictrans2-en-indic-1B   (eng_Latn -> kan_Knda)   [gated: accept terms on HF]
  3. KN -> Dharwad  : YOUR fine-tuned IndicTrans2 indic-indic-1B checkpoint
  4. TTS            : ai4bharat/indic-parler-tts (Kannada)

Install (Python 3.10/3.11, CUDA GPU recommended):
  pip install torch transformers sentencepiece sacremoses sounddevice soundfile numpy faster-whisper
  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git      # provides IndicTransToolkit.processor
  pip install git+https://github.com/huggingface/parler-tts.git
  huggingface-cli login       # needed for the gated en-indic-1B model
"""

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from IndicTransToolkit.processor import IndicProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MIC_SR = 16_000                      # Whisper expects 16 kHz mono
WHISPER_SIZE = "small.en"            # "base.en" (faster) / "medium.en" (more accurate)

EN_KN_MODEL = "ai4bharat/indictrans2-en-indic-1B"   # or ...-en-indic-dist-200M for less VRAM
DIALECT_MODEL = "pavan007123/dharwad-kannada-indictrans2"  # HF Hub repo id (see README for upload steps)

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
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL).to(DEVICE)
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
# MAIN LOOP
# --------------------------------------------------------------------------- #
def main():
    print(f"Device: {DEVICE}. Loading models (one-time)...")
    asr = ASR()
    en_kn = IndicTrans2(EN_KN_MODEL)
    kn_dharwad = IndicTrans2(DIALECT_MODEL)
    tts = TTS()
    print("Ready. Ctrl+C to quit.")

    while True:
        audio = record_until_enter()
        if audio.size < MIC_SR // 2:          # <0.5 s -> ignore
            print("Too short, try again.")
            continue

        english = asr(audio)
        if not english:
            print("Nothing recognised, try again.")
            continue
        print(f"[EN ] {english}")

        kannada = en_kn([english], src_lang="eng_Latn", tgt_lang="kan_Knda")[0]
        print(f"[KN ] {kannada}")

        dharwad = kn_dharwad(
            [kannada], src_lang=DIALECT_SRC_TAG, tgt_lang=DIALECT_TGT_TAG
        )[0]
        print(f"[DWD] {dharwad}")

        tts.speak(dharwad)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
