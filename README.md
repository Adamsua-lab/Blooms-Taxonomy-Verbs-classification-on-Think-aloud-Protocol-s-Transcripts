# Bloom’s Taxonomy Verb Classification on Think-Aloud Protocol Transcripts

End-to-end, reproducible pipeline to classify **Bloom’s Taxonomy (cognitive domain)** verbs in **think-aloud protocol transcripts** (e.g., AR-CAD vs SolidWorks).  
The workflow covers **Whisper transcription → transcript cleaning → time-bin segmentation → embeddings → multi-label Bloom labeling → analysis + visualization exports**.

---

## What’s in this repo

### 1) Batch transcription (Whisper / Faster-Whisper)
A root-level batch script is provided to automatically traverse participant folders and transcribe recorded session videos.  
It generates **phrase-level** and **word-level** outputs, plus **fixed-time bins** (e.g., 10s/20s/30s) for downstream analysis.

- Script: `batch_transcribe_fw.py` :contentReference[oaicite:0]{index=0}  
- Key features:
  - ffmpeg audio preprocessing to 16k mono wav
  - SolidWorks low-volume fallback (optional retry with VAD off)
  - temperature sweep + best-output selection
  - optional light “de-looping” to reduce repetitive hallucinations
  - exports SRT/CSV (easy to convert to plain TXT if needed)

### 2) Analysis notebook (Bloom coding + plots)
The Jupyter notebook contains the downstream pipeline to load transcripts/bins, apply the Bloom verb classification model, and generate analyses/figures for replication.

> Notebook: `BT5 User Study Applied using Bloom Taxonomy Classification Model.ipynb`

---

## Repository origin / credit
The Bloom verb classification model and the original pipeline foundation were adapted from:

```text
https://github.com/Adamsua-lab/Blooms-Taxonomy-Extension.git
