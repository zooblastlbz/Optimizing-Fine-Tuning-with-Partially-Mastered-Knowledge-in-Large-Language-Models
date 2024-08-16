from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("path to base model")
peft_model_id ="path to peft model"
model = PeftModel.from_pretrained(base_model, peft_model_id)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("path to save merged model")