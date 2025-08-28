from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class emotion2vec_bundle:
    def __init__(self):
        self.inference_pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model="iic/emotion2vec_plus_large"
        )

    def inference(self, wav_path):
        rec_result = self.inference_pipeline(wav_path, granularity="utterance", extract_embedding=False)

        return rec_result
