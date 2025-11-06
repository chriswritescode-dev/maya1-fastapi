import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf

# Load the best open source voice AI model
model = AutoModelForCausalLM.from_pretrained(
    "maya-research/maya1", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1")

# Load SNAC audio decoder (24kHz)
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")

# Design your voice with natural language
description = "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing."
text = "Hello! This is Maya1 <laugh> the best open source voice AI model with emotions."

# Create prompt with voice design
prompt = f'<description="{description}"> {text}'

# Generate emotional speech
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=500, 
        temperature=0.4, 
        top_p=0.9, 
        do_sample=True
    )

# Extract SNAC audio tokens
generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
snac_tokens = [t.item() for t in generated_ids if 128266 <= t <= 156937]

# Decode SNAC tokens to audio frames
frames = len(snac_tokens) // 7
codes = [[], [], []]
for i in range(frames):
    s = snac_tokens[i*7:(i+1)*7]
    codes[0].append((s[0]-128266) % 4096)
    codes[1].extend([(s[1]-128266) % 4096, (s[4]-128266) % 4096])
    codes[2].extend([(s[2]-128266) % 4096, (s[3]-128266) % 4096, (s[5]-128266) % 4096, (s[6]-128266) % 4096])

# Generate final audio with SNAC decoder
codes_tensor = [torch.tensor(c, dtype=torch.long, device="cuda").unsqueeze(0) for c in codes]
with torch.inference_mode():
    audio = snac_model.decoder(snac_model.quantizer.from_codes(codes_tensor))[0, 0].cpu().numpy()

# Save your emotional voice output
sf.write("output.wav", audio, 24000)
print("Voice generated successfully! Play output.wav")
