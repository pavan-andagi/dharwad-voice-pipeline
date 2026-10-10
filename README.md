# Dharwad Kannada Voice Pipeline

English speech (mic) → English text → Kannada → Dharwad Kannada (fine-tuned) → speaker.

Model weights live on **Hugging Face Hub**. Code lives in **this GitHub repo**. Everything runs **locally**.

## 1. Upload your model to Hugging Face Hub (one-time)

```bash
pip install huggingface_hub
huggingface-cli login          # paste a token with "write" access, from hf.co/settings/tokens
```

Create the repo and upload your merged checkpoint folder (the one with `config.json`,
`model.safetensors`, `dict.SRC`/`dict.TGT`, `model.SRC`/`model.TGT`, etc.):

```bash
huggingface-cli repo create dharwad-kannada-indictrans2 --private   # drop --private to make it public
huggingface-cli upload <your-username>/dharwad-kannada-indictrans2 \
    /path/to/dharwad-lora-model/final_merged .
```

Verify it uploaded correctly:

```bash
python -c "from huggingface_hub import list_repo_files; print(list_repo_files('<your-username>/dharwad-kannada-indictrans2'))"
```

You should see `config.json`, `model.safetensors`, `dict.SRC.json`, `dict.TGT.json`,
`model.SRC`, `model.TGT`, `special_tokens_map.json`, `tokenizer_config.json`,
`generation_config.json`.

## 2. Set up the GitHub repo (code only, no weights)

```bash
mkdir dharwad-voice-pipeline && cd dharwad-voice-pipeline
git init
# copy in: dharwad_voice_pipeline.py, requirements.txt, .gitignore, README.md
git add .
git commit -m "Initial pipeline"
git remote add origin git@github.com:<your-username>/dharwad-voice-pipeline.git
git push -u origin main
```

The `.gitignore` already excludes model weight files and virtual environments, so you
won't accidentally try to push a multi-GB `.safetensors` file to GitHub.

## 3. Point the script at your Hub model

In `dharwad_voice_pipeline.py`:

```python
DIALECT_MODEL = "<your-username>/dharwad-kannada-indictrans2"
```

If the repo is private, the script needs your HF token available at runtime. Easiest way:

```bash
huggingface-cli login        # once per machine; caches the token locally
```

`from_pretrained()` will then pick it up automatically — no code change needed. If you'd
rather not log in globally, set `HF_TOKEN` as an environment variable instead and the same
call will pick it up.

## 4. Clone and run locally (on any machine, going forward)

```bash
git clone git@github.com:<your-username>/dharwad-voice-pipeline.git
cd dharwad-voice-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# parler-tts pins transformers==4.46.1 exactly, which conflicts with the
# transformers==4.48.0 this project needs for IndicTransToolkit. Install it
# separately with --no-deps so it doesn't downgrade transformers:
pip install git+https://github.com/huggingface/parler-tts.git --no-deps

python dharwad_voice_pipeline.py
```

The first run downloads and caches `en-indic-1B`, your `dharwad-kannada-indictrans2`,
Whisper, and Parler-TTS from their respective hubs (a few GB total) — subsequent runs
load from the local cache.

## Notes

- `ai4bharat/indictrans2-en-indic-1B` is **gated**: accept its terms on its Hugging Face
  model page with the same account you ran `huggingface-cli login` with, or loading it
  will fail with a 403.
- Keep `DIALECT_SRC_TAG` / `DIALECT_TGT_TAG` as `kan_Knda` / `kan_Knda` — confirmed from
  the training notebook (`dharwad_kannada_training_128_256_4000.ipynb`).
- Updating the model later: re-run the `huggingface-cli upload` command with a new
  `final_merged` folder — it versions automatically on the Hub. No change needed on the
  GitHub side unless the loading code itself changes.
