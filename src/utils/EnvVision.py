from pydantic import BaseModel
from openai import OpenAI
import cv2
import tempfile
import base64

# Output from model
class envComprehension(BaseModel):
    entities: list[dict[str, str]]  # Each dict contains object_name, object_description, object_interaction, object_impact
    general_explanation: str # Explanation of how the individual may feel

class Environment_Vision():

    def __init__(self):
        self.client = OpenAI(api_key=process.env.OPENAI_API_KEY)

    def get_frame(self, video_base64):
        vid_bytes = base64.b64decode(video_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.avi') as tmp_file:
            tmp_file.write(vid_bytes)
            tmp_file_path = tmp_file.name

        # Open with OpenCV
        capture = cv2.VideoCapture(tmp_file_path)
        if not capture.isOpened():
            raise ValueError("Could not open the decoded video")
        
        # Get the frame
        count = 0
        frame_30 = None

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Get the 30th frame for vision
            count += 1
            if count == 30:
                frame_30 = frame
                break

        capture.release()

        if frame_30 is None:
            raise ValueError("Could not retrieve the 30th frame from the video for Env Vision")
        
        # Encode as a base64
        _, buffer = cv2.imencode('.jpg', frame_30)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        return frame_base64

    def comprehend_world(self, image_base64):
        response = self.client.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "user",
                    "content": f"""
    The following image contains a child with ASD and their environment. Your job is to analyze the image and provide a detailed description of the child's surroundings, including any objects or entities present in the following format
    A list of entities:
    object_name: Name of object
    object_description: Describe the object breifly, and how it could contribute to the situation
    object_interaction: How the object is interacting with the autistic child
    object_impact: How the object could impact how they feel (emotion)

    Then provide a general explanation of the situation and how it may contribute to how the child with ASD feels.
    : {image_base64}
                    """
                }
            ],
            reasoning="minimal",
            verbosity="medium",
            response_format=envComprehension
        )
        return response.choices[0].message.content
