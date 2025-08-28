# A POST PROCESSING PIPELINE, REALTIME USE IS SIMPLE, MAKE SURE ALL INFERENCES ARE NON-BLOCKING TO THE MAIN THREAD AND ENSURE FFMPEG DOESN'T OVERWRITE
# AND YOU CAN NOW FEED LIVE VIDEO STEAMS
import base64
import ffmpeg

from vision_utils import VLM_inference

from facial_inference import InferenceBundle as Facial_Inference

inference_util_vision = VLM_inference(api_key="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")

facial_inference = Facial_Inference.from_artifacts("../models/ViTClassification", device=None)  # auto-selects cuda/cpu

from retinaface_utils import draw_and_crop_single_face_b64

from verbal_inference import emotion2vec_bundle

from nonverbal_inference import NonverbalAudioEmotionClassifier


non_verbal_inference = NonverbalAudioEmotionClassifier()

verbal_inference = emotion2vec_bundle()

# with open(r"C:\Users\nmahesh\OneDrive - Eastside Preparatory School\Documents\code\asd-multimodal-emotion-fusion\output.png", "rb") as f:
#     img_b64 = base64.b64encode(f.read()).decode("utf-8")

# print(inference_util_vision.visual_background_inference(base64_image=img_b64))

import cv2

vid_cap = cv2.VideoCapture("current.mp4")

fps = vid_cap.get(cv2.CAP_PROP_FPS)

current_frame = 0

current_visual_inference_image = None

context = {
    "interaction_branch": None,
    "visual_branch": None,
    "verbal_branch": None,
    "nonverbal_branch": None,
    "facial_branch": None,
}

if not vid_cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = vid_cap.read()
    if not ret:
        break

    # First frame of second
    if (current_frame % fps == 0):
        _, buffer = cv2.imencode('.png', frame)
        current_visual_inference_image = base64.b64encode(buffer).decode('utf-8')
        current_visual_inference_image, crop = draw_and_crop_single_face_b64(current_visual_inference_image)
        if crop is not None:
            cv2.imwrite("temp_facial.jpg", crop)
            predictions, _ = facial_inference.predict_image("temp_facial.jpg", topk=3, return_probs=True)
            context["facial_branch"] = predictions
            
    if (current_frame % (fps*1.5) == 0):
        # Every 1.5 seconds, do audio inference
        (
            ffmpeg
            .input("current.mp4")
            .output("output.wav", vn=None, acodec="pcm_s16le")
            .run()
        )

        context["nonverbal_branch"] = non_verbal_inference.predict("output.wav")
        context["verbal_branch"] = verbal_inference.inference("output.wav")

    current_frame += 1

    if (current_frame % (fps*2) == 0):
        # Every 2 seconds, run all VLM inference
        context["visual_branch"] = inference_util_vision.visual_background_inference(base64_image=current_visual_inference_image)
        context["interaction_branch"] = inference_util_vision.human_interaction_inference(base64_image=current_visual_inference_image)

    if (current_frame % (fps*3) == 0):
        # Run fusion
        print("=============Input Fusion:==============")
        print(context)
        print("=============Output Fusion:==============")
        print(VLM_inference.fusion_inference(context))


vid_cap.release()