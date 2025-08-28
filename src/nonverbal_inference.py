from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class HeadClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.dropout0 = nn.Dropout(0.3)
        self.norm = nn.BatchNorm1d(1280)
        self.layer1 = nn.Linear(1280, 1280)
        self.activation1 = nn.GELU()
        self.dropout1=nn.Dropout(0.2)
        self.layer2 = nn.Linear(1280, 256)
        self.activation2 = nn.GELU()
        self.dropout2=nn.Dropout(0.2)
        self.output = nn.Linear(256, num_labels)

    def forward(self, x):
        x = self.dropout0(x)
        x = self.norm(x)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x


class NonverbalAudioEmotionClassifier:
    """
    Importable classifier for nonverbal emotion inference from .wav audio.
    """

    def __init__(
        self,
        model_path = None,
        labels_path = None,
        device = None,
    ) -> None:
        ROOT = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else ROOT / "models/nonverbal_classifier/model.pt"
        self.labels_path = Path(labels_path) if labels_path else ROOT / "models/nonverbal_classifier/labels.json"

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load labels
        with open(self.labels_path) as f:
            label_to_idx = json.load(f)
        self.idx_to_label = {v: k for k, v in label_to_idx.items()}
        num_labels = len(self.idx_to_label)

        # Model head
        self.model = HeadClassifier(num_labels=num_labels).to(self.device)
        state = torch.load(self.model_path, map_location=self.device)
        # Support loading checkpoints that may include extra keys (e.g., wrapped in dict)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            # As a fallback, try to load under a potential nested key
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                self.model.load_state_dict(state["model"], strict=False)
            else:
                raise
        self.model.eval()

        # HuBERT feature extractor
        self.bundle = torchaudio.pipelines.HUBERT_XLARGE
        self.hubert = self.bundle.get_model().to(self.device).eval()

    @torch.inference_mode()
    def _extract_features(self, wav_path):
        wav_path = Path(wav_path)
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.bundle.sample_rate)
        waveform = waveform.to(self.device)
        feats, _ = self.hubert.extract_features(waveform)
        x = feats[0].mean((0, 1)).detach()
        return x

    @torch.inference_mode()
    def predict(self, wav_path, topk=5):
        """Return top-k (label, probability) pairs for a given wav file."""
        x = self._extract_features(wav_path).unsqueeze(0)
        x = x.to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        topk = max(1, min(topk, len(self.idx_to_label)))
        vals, inds = torch.topk(probs, k=topk)
        results = [(self.idx_to_label[i.item()], v.item()) for v, i in zip(vals, inds)]
        return results

    @torch.inference_mode()
    def predict_top1(self, wav_path):
        results = self.predict(wav_path, topk=1)
        return results[0]

    def __call__(self, wav_path, topk=5):
        return self.predict(wav_path, topk=topk)