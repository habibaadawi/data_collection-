import json
import os
from pathlib import Path
from tqdm import tqdm
import requests

# ---- Config ----
QA_FILE = Path("/Users/habibaadawi/Documents/projects/graduation_project/data_for_LLM_model/ALL_Q&A_with_fixed_instruction.jsonl")
OUTPUT_FILE = Path("/Users/habibaadawi/Documents/projects/graduation_project/data_for_LLM_model/QA_with_anim_cues.jsonl")
EMPTY_CUES_FILE = Path("/Users/habibaadawi/Documents/projects/graduation_project/data_for_LLM_model/empty_anim_cues.jsonl")
CHECKPOINT_FILE = Path("/Users/habibaadawi/Documents/projects/graduation_project/data_for_LLM_model/qa_checkpoint.txt")
BATCH_SIZE = 10  # process 10 QA items at a time

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"  # Change this to your model name (e.g., "llama2:7b", "llama3:8b", etc.)

# ---- Test Ollama connection ----
def test_ollama_connection():
    """Test if Ollama is running and the model is available."""
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "test",
                "stream": False
            },
            timeout=10
        )
        if response.status_code == 200:
            print(f"✓ Successfully connected to Ollama with model: {MODEL_NAME}")
            return True
        else:
            print(f"✗ Ollama returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama. Make sure Ollama is running (run 'ollama serve')")
        return False
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        return False

print("Testing Ollama connection...")
if not test_ollama_connection():
    print("\nPlease ensure:")
    print("1. Ollama is installed (https://ollama.ai)")
    print("2. Ollama is running (run 'ollama serve' in terminal)")
    print(f"3. Model '{MODEL_NAME}' is pulled (run 'ollama pull {MODEL_NAME}')")
    exit(1)

# ---- Load checkpoint ----
start_idx = 0
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE, "r") as f:
        start_idx = int(f.read().strip())
print(f"Resuming from index {start_idx}...")

# ---- Load QA file as JSON Lines ----
qa_items = []
with open(QA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            qa_items.append(json.loads(line))
print(f"Total QA items: {len(qa_items)}")

# ---- Function to check if anim_cues are empty ----
def has_empty_cues(anim_cues):
    """Check if any animation cue field is empty."""
    required_fields = ["emotion", "facial_expression", "gesture", "head_movement", "tts_style"]
    for field in required_fields:
        if field not in anim_cues or anim_cues[field] == "":
            return True
    return False

# ---- Function to generate anim_cues using Ollama ----
def generate_anim_cues(input_text, response_text):
    """
    Uses Ollama with LLaMA 7B to generate animation cues.
    Returns a dictionary of animation cues.
    """
    prompt = f"""Based on the following conversation, generate animation cues for an avatar.

Input: {input_text}
Response: {response_text}

Generate the following animation cues in JSON format:
- emotion (e.g., neutral, happy, sad, concerned, excited, calm, proud, anxious, angry, hopeful, amused, somber, curious, fearful)
- facial_expression (e.g., serious, smiling, laughing, frowning, thinking, neutral, raised_eyebrows, narrowed_eyes, wide_eyes, soft_smile, grimace, smirk)
- gesture (e.g., hands_together, open_palms, pointing, waving, crossed_arms, hand_on_chest, shrug, fist_clench, slow_reach, palm_up, palm_down)
- head_movement (e.g., nod_slow, nod_fast, shake, tilt, still, lean_forward, lean_back, glance_side, bow_slight, look_down, look_up)
- tts_style (e.g., low, medium, high, enthusiastic, calm, whisper, authoritative, warm, hesitant, dramatic, soft, firm)


Output ONLY a valid JSON object with these fields, nothing else:"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 200
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API returned status code {response.status_code}")
        
        result = response.json()
        generated_text = result.get("response", "")
        
        # Extract JSON from the generated text
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = generated_text[json_start:json_end]
            anim_cues = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["emotion", "facial_expression", "gesture", "head_movement", "tts_style"]
            if all(field in anim_cues for field in required_fields):
                return anim_cues
            else:
                raise ValueError("Missing required fields in JSON output")
        else:
            raise ValueError("No valid JSON found in output")
            
    except (json.JSONDecodeError, ValueError, requests.exceptions.RequestException) as e:
        print(f"  Warning: Failed to generate valid cues ({e}), using defaults")
        # Fallback to defaults
        return {
            "emotion": "",
            "facial_expression": "",
            "gesture": "",
            "head_movement": "",
            "tts_style": ""
        }

# ---- Open output files for appending ----
with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f, \
     open(EMPTY_CUES_FILE, "a", encoding="utf-8") as empty_f:
    
    for idx in tqdm(range(start_idx, len(qa_items)), desc="Processing QA items"):
        item = qa_items[idx]
        
        # Skip if anim_cues already exist
        if "anim_cues" in item.get("output", {}):
            continue
        
        try:
            # Generate anim_cues based on input + response
            anim_cues = generate_anim_cues(item["input"], item["output"]["response"])
            item["output"]["anim_cues"] = anim_cues
            
            # Check if any cue is empty
            if has_empty_cues(anim_cues):
                # Write to empty cues file
                empty_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                empty_f.flush()
            
            # Write to output file as JSON line (write all items)
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()  # ensure it writes to disk
            
            # Update checkpoint every BATCH_SIZE
            if (idx + 1) % BATCH_SIZE == 0:
                with open(CHECKPOINT_FILE, "w") as f:
                    f.write(str(idx + 1))
                    
        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            # Save checkpoint and continue
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(idx))
            continue

print("\n✓ Processing complete!")
# Remove checkpoint after finishing
if CHECKPOINT_FILE.exists():
    os.remove(CHECKPOINT_FILE)