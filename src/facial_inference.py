import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm


# Model Checkpoint only
class InferenceBundle:
    def __init__(self, model: nn.Module, class_names: List[str], img_size: int, device: str):
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.img_size = img_size
        self.device = device

        # Validation/inference transforms (just matches what we did in training)
        self.tfms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    @torch.no_grad()
    def predict_image(self, path: str, topk: int = 3, return_probs: bool = True):
        """Predict class probabilities"""
        img = Image.open(path).convert("RGB")
        x = self.tfms(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        topk_idx = probs.argsort()[::-1][:topk]
        result = [(self.class_names[i], float(probs[i])) for i in topk_idx]
        return (result, probs) if return_probs else result

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        """Recreate the ViT + MLP head architecture used during training."""
        backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        emb_dim = backbone.num_features
        head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )
        return nn.Sequential(backbone, head)

    @staticmethod
    def _safe_load_json(p: Path) -> Optional[dict]:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path, device: Optional[str] = None) -> "InferenceBundle":
        artifacts_path = Path(artifacts_dir).resolve()
        if device not in ("cuda", "cpu"):
            device = "cuda" if torch.cuda.is_available() else "cpu"

        best_state_path = artifacts_path / "best_state.pth"
        weights_path = artifacts_path / "best_model_weights.pth"
        label_map_path = artifacts_path / "label_map.json"

        class_names: List[str]
        img_size = 224 

        state_dict = None
        if best_state_path.exists():
            blob = torch.load(best_state_path, map_location="cpu")
            state_dict = blob.get("model")
            class_names = list(blob.get("class_names", []))
            cfg = blob.get("cfg", {}) or {}
            img_size = int(cfg.get("img_size", img_size))
            if not class_names and label_map_path.exists():
                lm = cls._safe_load_json(label_map_path) or {}
                class_names = [lm[str(i)] if isinstance(i, int) else lm[i] for i in sorted(map(int, lm.keys()))]
        else:
            if not weights_path.exists() or not label_map_path.exists():
                raise FileNotFoundError(
                    f"Could not find required artifacts. Looked for: {best_state_path} or ({weights_path} and {label_map_path})."
                )
            state_dict = torch.load(weights_path, map_location="cpu")
            lm = cls._safe_load_json(label_map_path) or {}
            if not lm:
                raise RuntimeError("label_map.json is empty or invalid; cannot get class names.")
            class_names = [lm[str(i)] if isinstance(i, int) else lm[i] for i in sorted(map(int, lm.keys()))]

        if not state_dict:
            raise RuntimeError("No model model state found.")

        model = cls._build_model(num_classes=len(class_names))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return cls(model=model, class_names=class_names, img_size=img_size, device=device)
