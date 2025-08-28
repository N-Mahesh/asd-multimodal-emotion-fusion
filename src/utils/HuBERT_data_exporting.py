"""
Create embeddings for the ReCANVO dataset using HuBERT.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

HuBERT = torchaudio.pipelines.HUBERT_XLARGE

ReCANVO_csv = pd.read_csv("data/ReCANVO/directory_w_train_test.csv")
labels = ReCANVO_csv.Label.value_counts()

# From: https://github.com/julianrosen/erdos_dl_recanvo_project
ReCANVO_files = ReCANVO_csv.loc[
    ReCANVO_csv.Label.isin(labels[labels >= 30].index)
    & (ReCANVO_csv.is_test == 0)
].copy()

ReCANVO_files["session"] = ReCANVO_files.Filename.apply(
    lambda name: name.split("-")[0][:-3]
)

dir_to_wav = "data/ReCANVO"
output_dir = "data/ReCANVO_HuBERT"
output_dir.mkdir(parents=True, exist_ok=True)

with torch.no_grade():
    for i in tqdm(range(len(ReCANVO_files))):
        fname = ReCANVO_files.Filename.iloc[i]
        wave, sample = torchaudio.load(dir_to_wav / fname)
        wave = torchaudio.functional.resample(wave, sample, 16000)
        features, _ = HuBert.extract_features(wave)
        out = features[0].mean((0, 1))
        torch.save(out, output_dir / f"{fname.removesuffix('.wav')}.pt")