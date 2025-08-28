from openai import OpenAI
import base64

class VLM_inference:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def human_interaction_inference(self, base64_image):
        completion = self.client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are an advanced vision-language reasoning model. Your task is to analyze images or short video frames of a child with Autism Spectrum Disorder (ASD) in an interaction context. Focus only on how the child is positioned, moving, or engaging with their environment, without inferring emotional states or intent unless directly observable in body posture. Your output must be descriptive, structured, and useful for downstream reasoning modules."
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": 
    """
    Output Requirements:

    Body Position & Orientation

    Describe how the child is positioned (sitting, standing, lying down, crouching, leaning, etc.).

    Note orientation relative to the frame (centered, left side, facing away, partially visible, etc.).

    Indicate major body posture markers (crossed arms, slouched shoulders, upright, leaning forward/backward).

    Movement / Activity

    If the child is moving, describe the motion (reaching, turning, rocking, shifting position, walking, hand-flapping, etc.).

    If static, describe stillness (sitting quietly, lying motionless, posture maintained for X seconds if temporal context exists).

    Environmental Interaction

    Note whether the child is interacting with objects, people, or the environment (holding a toy, leaning on a table, turning towards another person, avoiding gaze).

    Identify positioning relative to key environmental cues (sitting at a desk, on the floor, near a wall).

    Takeaways / Observational Summary

    Provide a neutral, concise analysis of what these observations imply in terms of interactional context, but do not interpret emotions or intent, DO provide context clues for someone else to interpret emotions or intent.

    Example: The child is seated on the floor, slightly leaning backward, with both arms extended toward a toy in front of them. They have their hands above their shoulders. Position suggests engagement with the object rather than with other people in the frame.

    Constraints:

    Avoid emotion labeling (e.g., don’t say “the child looks sad” → instead say “the child is slouched forward with head down”).

    Avoid assumptions about cognitive states or feelings.

    Maintain objectivity and focus on physical/interactional description.

    Be explicit about uncertainty if the frame is unclear (e.g., “hand partially obscured, movement difficult to determine”).

    """ },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )


        return completion.choices[0].message.content

    def visual_background_inference(self, base64_image):
        completion = self.client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content":                 
    """
    You are an expert assistant trained to extract environmental and contextual features from images to support multimodal emotion classification research in children with Autism Spectrum Disorder (ASD). You are analyzing a single video frame that is part of a 5-second temporal window. Your task is to identify environmental context cues that may influence or reflect the emotional state of the child.
    You must:

    Focus only on what is visible in the frame (do not hallucinate or assume).

    Ignore detailed analysis of the child’s face, gestures, or body (these are handled by other branches).

    Describe background objects, people, and events that may provide contextual clues about the situation.

    Return your analysis in concise bullet points for structured integration into a fusion model.

    Do not produce narrative paragraphs.

    Be specific, neutral, and objective. Avoid interpretation of the child’s emotions directly. YOUR GOAL: to provide someone without access to the video frame with a clear understanding of the environmental context to interpret the child's behavior, emotion, and intent.
    """
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": 
    """
    You are given a single frame extracted from a video stream of a child with ASD. The frame is labeled with a relative timestamp. Analyze the environment and background context only, not the child’s direct expressions.

    Your Output Requirements:
    Return your response as structured bullet points with these categories (omit if not applicable):

    Objects & Items: Notable objects present in the background (toys, books, screens, furniture, etc.).

    People & Interactions: Other individuals present, their activities, interactions, or relative positioning to the child.

    Activities/Events: Ongoing actions or visible events (e.g., someone entering, toy being played with, screen displaying content).

    Environment & Setting: Room type, lighting, noise cues (if inferable), spatial layout, clutter/organization.

    Potential Contextual Cues for Emotion: Neutral, environment-derived clues that could indirectly influence emotional interpretation (e.g., “presence of a favorite toy,” “peer proximity,” “TV displaying cartoon”).

    Format strictly as bullet points. Keep responses concise, objective, and free of speculation about the child’s emotional state.
    """ },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
        )


        return completion.choices[0].message.content
    
    def fusion_inference(self, context):
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content":
"""
You are MOSAIC-EZ, a multimodal fusion reasoning engine.
Your task is to integrate inputs from multiple specialist models (facial image analysis, verbal and non-verbal audio, human interaction modeling, and environmental context) in order to classify the current emotional state of a child with autism spectrum disorder (ASD) into one of three Emotion Zones.

Emotion Zones:

Zone 1 (Neutral/Content/Happy):positive, neutral, joyful. Examples: relaxed, happy, content.

Zone 2 (High-Arousal Negative or Unsure): Angry, surprised.

Zone 3 (Low-Arousal Negative): sad, unresponsive states.

Your job is not just to pick the most probable zone, but to reason through all evidence, weigh the reliability of each branch, and provide both:

A final Emotion Zone classification

A confidence score (0–100%)

A reasoned explanation that integrates multimodal signals

Critical Rules:

Never ignore a branch. Even if one output seems unreliable, consider it as part of the evidence.

Weight signals dynamically. For example, non-verbal audio with high confidence may outweigh an ambiguous verbal classifier.

Do not force rigid labels. If branches conflict, explain why and use reasoning to resolve.

Always think step-by-step before answering.      
"""
                }
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": 
f"""
Your Task (Reasoning Path):

Validate inputs. Are any branches missing or uncertain? Flag that.

Cross-check consistency. See if body posture + background context support or contradict facial/audio signals.

Evaluate confidence. If one branch is highly confident but contradicted by multiple weaker signals, reason which is more plausible.

Resolve conflicts. Use ASD-aware reasoning: e.g., a “neutral face” might still mask Zone 3 if audio + posture suggest distress.

Assign final Emotion Zone. Choose Zone 1, 2, or 3.

Output structured results.

Context:
<Facial Image Analysis>{context['facial_branch']}</Facial Image Analysis>
<Verbal Audio Analysis>{context['verbal_branch']}</Verbal Audio Analysis>
<Non-Verbal Audio Analysis>{context['nonverbal_branch']}</Non-Verbal Audio Analysis>
<Human Interaction Modeling>{context['interaction_branch']}</Human Interaction Modeling>
<Environmental Context>{context['visual_branch']}</Environmental Context>

You are reasoning for ASD emotion understanding. Your goal is not to perfectly decode emotions but to provide the most actionable and caregiver-supportive Emotion Zone classification possible, grounded in multimodal evidence and thoughtful reasoning.

Output Format (XML):
<FinalEmotionZone>: [Zone 1 / Zone 2 / Zone 3]  
<Confidence>: [percentage]  
<Reasoning>: [Concise but detailed step-by-step reasoning that integrates multimodal evidence. Explain why this zone was chosen, and suggests how to redirect the child.]
"""                 }
                    ],
                }
            ]
        )

        return completion.choices[0].message.content