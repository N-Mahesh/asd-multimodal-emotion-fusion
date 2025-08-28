import torch
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


class VAD:
    """
    Voice Activity Detection using Silero VAD, with a backend-free WAV reader.
    """

    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
        )
        # utils = (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        (self.get_speech_timestamps, _, _read_audio_unused, _, self.collect_chunks) = self.utils

    @staticmethod
    def _read_wav_mono_16k(path: str) -> torch.Tensor:
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        else:
            audio = data.astype(np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        target_sr = 16000
        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr).astype(np.float32)

        return torch.from_numpy(audio)

    def inference(self, wav_path: str) -> bool:
        wav_tensor = self._read_wav_mono_16k(wav_path)
        speech_timestamps = self.get_speech_timestamps(wav_tensor, self.model, sampling_rate=16000)

        total_len = wav_tensor.shape[0]
        speech_len = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        speech_ratio = (speech_len / total_len) if total_len > 0 else 0.0

        return speech_ratio >= 0.4


if __name__ == "__main__":
    VAD_model = VAD()
    print(VAD_model.inference("output.wav"))