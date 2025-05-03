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

