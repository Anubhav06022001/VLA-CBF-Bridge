# policies/vla_policy.py
import torch
import json
import re
from tranfromers import Qwen2_5_VLForConditionalGeneration, AutoProcessor ,BitsAndBytesConfig 
from PIL import Image

class VLAPolicy:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
        print("Loading VLA in 4-bit...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # --- NEW: Use BitsAndBytesConfig for 4-bit loading ---
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config  # Pass the config here
        )
        self.prompt = "Observe the wrist camera view. Provide the 3D translation delta (dx, dy, dz) in meters to move the robot hand closer to the green cube. Output strictly in JSON format like: {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 0.0}"

    def predict_action(self, image_array):
        image = Image.fromarray(image_array)
        inputs = self.processor(text=[self.prompt], images=[image], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=40)
            
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Regex to extract JSON from the LLM's chatty response
        match = re.search(r'\{.*?\}', output_text)
        if match:
            try:
                data = json.loads(match.group(0))
                return [data.get('dx', 0.0), data.get('dy', 0.0), data.get('dz', 0.0)]
            except:
                pass
        return [0.0, 0.0, 0.0]































# class VLAPolicy:
#     def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct"):
#         print("Loading VLA in 4-bit...")
#         self.processor = AutoProcessor.from_pretrained(model_id)
#         # self.model = AutoModelForVision2Seq.from_pretrained(
#         #     model_id,
#         #     device_map="auto",
#         #     load_in_4bit=True  # Crucial for RTX 5060
#         # )
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             model_id,
#             device_map="auto",
#             load_in_4bit=True  
#         )
#         self.prompt = "Observe the wrist camera view. Provide the 3D translation delta (dx, dy, dz) in meters to move the robot hand closer to the green cube. Output strictly in JSON format like: {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 0.0}"

#     def predict_action(self, image_array):
#         image = Image.fromarray(image_array)
#         inputs = self.processor(text=[self.prompt], images=[image], return_tensors="pt").to("cuda")
        
#         with torch.no_grad():
#             generated_ids = self.model.generate(**inputs, max_new_tokens=40)
            
#         output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         # Regex to extract JSON from the LLM's chatty response
#         match = re.search(r'\{.*?\}', output_text)
#         if match:
#             try:
#                 data = json.loads(match.group(0))
#                 return [data.get('dx', 0.0), data.get('dy', 0.0), data.get('dz', 0.0)]
#             except:
#                 pass
#         return [0.0, 0.0, 0.0] 
