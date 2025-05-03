import re
import json
import torch
import logging
from PIL import Image
from symspellpy import SymSpell, Verbosity
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
IMAGE_PATH = "/content/Maggie.jpg"

MAX_OCR_CHARS = 3000
MAX_INGREDIENTS = 100
SYM_SPELL_EDIT_DISTANCE = 1

# --- Setup ---
try:
    sym_spell = SymSpell(max_dictionary_edit_distance=SYM_SPELL_EDIT_DISTANCE, prefix_length=7)
    sym_spell.load_dictionary('frequency_dict.txt', term_index=0, count_index=1)
    logging.info("SymSpell dictionary loaded.")
except Exception as e:
    logging.error(f"Failed to load SymSpell dictionary: {e}")
    sym_spell = None

# --- OCR Function (Keep as is) ---
def ocr_to_text(image_path):
    """Performs OCR on the image and returns the extracted text."""
    try:
        img = Image.open(image_path)
        img_gray = img.convert('L')
        # Try a different PSM mode if default fails often (e.g., 4 or 11)
        text = pytesseract.image_to_string(img_gray, config='--psm 6').strip()
        logging.info(f"OCR successful for {image_path}. Text length: {len(text)}")
        return text[:MAX_OCR_CHARS]
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error during OCR processing: {e}")
        return None

# --- Ingredient Cleaning Function  ---
def clean_ingredients(text):
    """Cleans the OCR text to extract a list of ingredients."""
    if not text:
        return []
    text = ' '.join(text.split()).lower()
    ingredient_markers = ["ingredients:", "contains:", "ingredients :", "contains :"]
    start_index = -1
    ingredient_text = text # Default to full text if no marker found
    for marker in ingredient_markers:
        try:
            idx = text.index(marker)
            # Check if this marker is preceded by nutrition info (less likely to be the real start)
            preceding_text = text[:idx]
            if "nutrition facts" not in preceding_text and "serving size" not in preceding_text:
                 start_index = idx + len(marker)
                 logging.info(f"Found potential ingredient marker '{marker}'")
                 ingredient_text = text[start_index:]
                 break # Use the first valid marker found
        except ValueError:
            continue 

        if start_index == -1:
            logging.warning("Could not find a clear ingredient start marker. Attempting cleanup on full text.")
        
        end_markers = ["nutrition facts", "serving size", "% daily value", "manufactured by", "distributed by", "produced by"]
    for marker in end_markers:
        marker_index = ingredient_text.find(marker)
        if marker_index != -1:
            logging.info(f"Removing text after '{marker}'")
            ingredient_text = ingredient_text[:marker_index].strip()

    # Improved splitting: handle commas/semicolons, periods followed by space, 'and', but respect parentheses
    # Remove common clutter like 'less than 2% of:' before splitting
    ingredient_text = re.sub(r'less than \d+% of\s*[:]*\s*', '', ingredient_text, flags=re.IGNORECASE)
    ingredient_text = re.sub(r'contains \d+% or less of\s*[:]*\s*', '', ingredient_text, flags=re.IGNORECASE)

    potential_ingredients = re.split(r'[;,]\s*(?![^()]*\))|\.\s+(?![^()]*\))|\s+and\s+(?![^()]*\))', ingredient_text)


    cleaned = []
    for ing in potential_ingredients:
        cleaned_ing = re.sub(r"^[^\w(]+|[^\w)]+$", "", ing.strip()).strip() # Allow starting '(' and ending ')'
        cleaned_ing = cleaned_ing.replace(' :', '') # Remove stray colons

        if cleaned_ing and len(cleaned_ing) > 1 and not cleaned_ing.isdigit():
            corrected_ing = cleaned_ing
            if sym_spell:
                suggestions = sym_spell.lookup(cleaned_ing, Verbosity.CLOSEST, max_edit_distance=SYM_SPELL_EDIT_DISTANCE, include_unknown=True)
                if suggestions and suggestions[0].distance < SYM_SPELL_EDIT_DISTANCE + 1: # Only correct if close enough
                    best_suggestion = suggestions[0].term
                    # Avoid correcting ingredient names containing numbers like 'red 40' into words
                    if not (any(char.isdigit() for char in cleaned_ing) and not any(char.isdigit() for char in best_suggestion)):
                        if best_suggestion != cleaned_ing:
                            logging.info(f"SymSpell corrected '{cleaned_ing}' to '{best_suggestion}' (distance {suggestions[0].distance})")
                        corrected_ing = best_suggestion
                    else:
                        logging.info(f"SymSpell skipped correction for '{cleaned_ing}' due to number mismatch.")

            if corrected_ing and corrected_ing not in cleaned: # Check non-empty after potential correction
                cleaned.append(corrected_ing)

    logging.info(f"Found {len(cleaned)} potential ingredients.")
    return cleaned[:MAX_INGREDIENTS]

# --- LLM Analysis Function ---
model = None
tokenizer = None
pipe = None

def load_model():
    global model, tokenizer, pipe
    if pipe is not None and hasattr(pipe, 'model_name') and pipe.model_name == MODEL_NAME: # Check if correct model loaded
        logging.info(f"Model {MODEL_NAME} already loaded.")
        return True

    logging.info(f"Loading model: {MODEL_NAME}")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
             # Important: Set pad_token_id correctly based on model specifics if eos isn't right
             logging.info("Tokenizer missing pad token, setting to EOS token.")
             tokenizer.pad_token = tokenizer.eos_token
             # Some models might need pad_token_id = 0 or another specific ID

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2" # Optional: If flash-attn is installed and supported, can speed up
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # Store model name for checking later
        pipe.model_name = MODEL_NAME
        logging.info("Model loaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Error loading model {MODEL_NAME}: {e}", exc_info=True) # Log traceback
        model = None
        tokenizer = None
        pipe = None
        return False