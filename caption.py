import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Image captioning with Transformers.")
parser.add_argument("--image-path", required=True, help="Path to the input image.")
args = parser.parse_args()


IMAGE_PATH = args.image_path
#PROMPT = "Write a long descriptive caption for this image in a formal tone."
#PROMPT = "Write a descriptive caption for this image in a casual tone."
PROMPT = "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt."
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
)

# Load JoyCaption
# bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
# device_map=0 loads the model into the first GPU
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
)
llava_model.eval()

with torch.no_grad():
    # Load image
    image = Image.open(IMAGE_PATH)

    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": PROMPT,
        },
    ]

    # Format the conversation
    # WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
    # but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
    # if not careful, which can make the model perform poorly.
    convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
    assert isinstance(convo_string, str)

    # Process the inputs
    inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    # Generate the captions
    generate_ids = llava_model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        suppress_tokens=None,
        use_cache=True,
        temperature=0.6,
        top_k=None,
        top_p=0.9,
    )[0]

    # Trim off the prompt
    generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

    # Decode the caption
    caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    caption = caption.strip()
    print(caption)

