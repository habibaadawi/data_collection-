import os
import re
import json
import requests
import hashlib
from collections import defaultdict
from typing import List, Set, Tuple
import numpy as np

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 📋 Configuration
# ==========================================

SEMANTIC_THRESHOLD = 0.55
MIN_CHUNK_SIZE = 3
REDUNDANCY_SEMANTIC_THRESHOLD = 0.90
REDUNDANCY_QUICK_THRESHOLD = 0.99

# ==========================================
# 🟢 Enhanced Sentence Segmenter
# ==========================================

historical_abbrevs = [
    "ca.", "c.", "cir.", "approx.",
    "B.C.", "BC.", "B.C.E.", "BCE.",
    "A.D.", "AD.", "A.C.E.", "ACE.",
    "fig.", "Fig.", "ref.", "Ref.",
    "vol.", "Vol.", "no.", "No.",
    "pp.", "Pg.", "pg.",
    "ch.", "Chap.", "chap.",
    "vs.", "Vs.",
    "e.g.", "i.e.", "etc.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "St.",
    "Prof.", "Rev.", "Hon.", "Sr.", "Jr."
]

def smart_sentence_split(text: str) -> List[str]:
    """
    Robust sentence splitting without pysbd dependency.
    Uses regex-based approach with abbreviation handling.
    """
    protected_text = text
    replacements = {}
    
    for i, abbrev in enumerate(historical_abbrevs):
        placeholder = f"__ABBREV_{i}__"
        protected_text = protected_text.replace(abbrev, placeholder)
        replacements[placeholder] = abbrev
    
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    raw_sentences = re.split(sentence_pattern, protected_text)
    
    sentences = []
    for sent in raw_sentences:
        for placeholder, abbrev in replacements.items():
            sent = sent.replace(placeholder, abbrev)
        sent = sent.strip()
        if sent:
            sentences.append(sent)
    
    return sentences

# ==========================================
# 🧮 Quality Content Filtering
# ==========================================

def is_quality_content(text: str) -> bool:
    """
    Filter out low-quality content like coordinates, references, etc.
    """
    if len(text.strip()) < 25:
        return False
    
    # Filter out map coordinates and measurements
    if re.search(r'\d+°\s*[NS]\.?\s*\d+°\s*[EW]', text):
        return False
    if re.search(r'lat\.\s*\d+|long\.\s*\d+', text.lower()):
        return False
    
    # Filter out page references and publication info
    if re.search(r'\b(pg?\.|page)\s*\d+', text.lower()):
        return False
    if re.search(r'\b(vol|fig|chap?)\.?\s*\d+', text.lower()):
        return False
    
    # Filter out publication metadata
    if re.search(r'gutenberg|ebook|project gutenberg', text.lower()):
        return False
    
    # Filter out index-like entries
    words = text.split()
    if len(words) <= 3 and text.isupper():
        return False
    
    # Filter out table of contents patterns
    if re.search(r'\.{3,}\s*\d+$', text):
        return False
    
    # Keep only narrative content with proper structure
    if not re.search(r'[.!?]$', text.strip()):
        return False
    
    # Reasonable word count
    if len(words) < 5 or len(words) > 100:
        return False
    
    return True

def detect_content_type(text: str) -> str:
    """
    Classify content type to prioritize narrative content.
    """
    text_lower = text.lower()
    
    narrative_indicators = [
        'said', 'stated', 'explained', 'described', 'noted',
        'however', 'therefore', 'furthermore', 'consequently',
        'because', 'although', 'while', 'during', 'after'
    ]
    
    reference_indicators = [
        'see page', 'see fig', 'as shown in', 'according to',
        'map', 'chart', 'diagram', 'illustration', 'reference'
    ]
    
    narrative_score = sum(1 for indicator in narrative_indicators if indicator in text_lower)
    reference_score = sum(1 for indicator in reference_indicators if indicator in text_lower)
    
    if narrative_score > reference_score:
        return "narrative"
    elif reference_score > narrative_score:
        return "reference"
    else:
        return "neutral"

# ==========================================
# 📄 Local File Processing Functions
# ==========================================

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a .txt file with encoding detection.
    """
    print(f"📄 Reading TXT file: {file_path}")
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                print(f"✅ Successfully read TXT file with {encoding} encoding")
                return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ Error reading TXT file with {encoding}: {e}")
            continue
    
    print("❌ Failed to read TXT file with any encoding")
    return ""

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2 or pdfplumber.
    Falls back to basic extraction if libraries are not available.
    """
    print(f"📄 Reading PDF file: {file_path}")
    
    try:
        # Try PyPDF2 first
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            print("✅ Successfully extracted text using PyPDF2")
            return text
        except ImportError:
            print("⚠️  PyPDF2 not available, trying pdfplumber...")
        
        # Try pdfplumber second
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            print("✅ Successfully extracted text using pdfplumber")
            return text
        except ImportError:
            print("⚠️  pdfplumber not available, using fallback method...")
        
        # Fallback: use basic command-line tools if available
        try:
            import subprocess
            # Try pdftotext (common Linux/Unix tool)
            result = subprocess.run(['pdftotext', file_path, '-'], 
                                 capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("✅ Successfully extracted text using pdftotext")
                return result.stdout
        except:
            pass
        
        print("❌ No PDF extraction libraries available. Please install PyPDF2 or pdfplumber.")
        return ""
        
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        return ""

def process_local_file(file_path: str) -> str:
    """
    Process local .txt or .pdf files and return cleaned text.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return ""
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.txt':
        raw_text = extract_text_from_txt(file_path)
    elif file_ext == '.pdf':
        raw_text = extract_text_from_pdf(file_path)
    else:
        print(f"❌ Unsupported file format: {file_ext}")
        return ""
    
    if not raw_text:
        return ""
    
    # Apply the same cleaning as HTML processing
    # Remove common noise patterns
    text = re.sub(r"\[Illustration[^\]]*\]", "", raw_text)
    text = re.sub(r"\[Footnote[^\]]*\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    
    # Fix broken patterns
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[_]{3,}", "", text)
    text = re.sub(r"[\*]{3,}", "", text)
    
    # Remove common artifacts
    text = re.sub(r"[|]{2,}", "", text)
    text = re.sub(r"[=]{3,}", "", text)
    text = re.sub(r"[-]{4,}", "", text)
    
    print(f"✅ Processed local file: {len(text)} characters")
    return text.strip()

# ==========================================
# 🧹 STEP 3: Enhanced Structural Cleaner
# ==========================================

def advanced_structural_cleaner(sentence: str) -> str:
    """
    Remove book-specific noise patterns with enhanced filtering.
    """
    # Remove page numbers and references
    sentence = re.sub(r'\b[Pp]age\s+\d+\b', '', sentence)
    sentence = re.sub(r'\b\d+\s*$', '', sentence)
    sentence = re.sub(r'^\s*\d+\s*$', '', sentence)
    
    # Remove chapter headers
    sentence = re.sub(r'^CHAPTER\s+[IVXLCDM\d]+\.?\s*$', '', sentence, flags=re.I)
    sentence = re.sub(r'^Chapter\s+\d+\.?\s*$', '', sentence)
    
    # Remove running headers
    if sentence.isupper() and len(sentence.split()) <= 6:
        return ""
    
    # Remove table of contents patterns
    sentence = re.sub(r'^[IVXLCDM]+\.\s+[A-Z]', '', sentence)
    sentence = re.sub(r'\.{3,}', '', sentence)
    
    # Remove "continued" patterns
    sentence = re.sub(r'\(continued\)', '', sentence, flags=re.I)
    sentence = re.sub(r'\bcontinued\b', '', sentence, flags=re.I)
    
    # Remove section markers
    sentence = re.sub(r'^\*\s+\*\s+\*$', '', sentence)
    sentence = re.sub(r'^§\s*\d+', '', sentence)
    
    # Remove transcriber notes
    sentence = re.sub(r'\[Transcriber.*?\]', '', sentence, flags=re.I)
    sentence = re.sub(r'\[Editor.*?\]', '', sentence, flags=re.I)
    
    # Enhanced filtering for historical texts
    sentence = re.sub(r'\b\d+\s*B\.C\.?|\b\d+\s*A\.D\.?', '', sentence)
    sentence = re.sub(r'^[IVXLCDM]+\.\s*$', '', sentence)
    sentence = re.sub(r'\b\d+\s*degrees?\s*[NSWE]\b', '', sentence, flags=re.I)
    
    sentence = sentence.strip()
    
    # Apply quality filter
    if not is_quality_content(sentence):
        return ""
    
    return sentence

# ==========================================
# 🔗 STEP 4: Enhanced Semantic Chunking
# ==========================================

def group_semantically(sentences: List[str], 
                       threshold: float = SEMANTIC_THRESHOLD, 
                       min_chunk_size: int = MIN_CHUNK_SIZE) -> List[str]:
    """
    Group sentences into semantically coherent paragraphs.
    """
    if not sentences:
        return []
    
    if len(sentences) < 2:
        return sentences
    
    try:
        print("\n🔗 Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("🧠 Encoding sentences...")
        batch_size = 100
        embeddings_list = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)
        
        import torch
        embeddings = torch.cat(embeddings_list, dim=0)
        
        paragraphs = []
        current_chunk = [sentences[0]]
        
        print("🔗 Grouping sentences...")
        for i in range(1, len(sentences)):
            try:
                similarity = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
                
                if similarity >= threshold or len(current_chunk) < min_chunk_size:
                    current_chunk.append(sentences[i])
                else:
                    paragraphs.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
            except Exception as e:
                print(f"⚠️  Error processing sentence {i}: {e}")
                current_chunk.append(sentences[i])
        
        if current_chunk:
            paragraphs.append(" ".join(current_chunk))
        
        print(f"✅ Created {len(paragraphs)} semantic paragraphs")
        return paragraphs
    
    except Exception as e:
        print(f"⚠️  Semantic grouping failed: {e}")
        print("🔄 Falling back to fixed-size chunking...")
        paragraphs = []
        for i in range(0, len(sentences), min_chunk_size):
            chunk = sentences[i:i+min_chunk_size]
            paragraphs.append(" ".join(chunk))
        return paragraphs

def group_semantically_enhanced(sentences: List[str], 
                               threshold: float = SEMANTIC_THRESHOLD, 
                               min_chunk_size: int = MIN_CHUNK_SIZE) -> List[str]:
    """
    Enhanced semantic grouping that prioritizes narrative content.
    """
    if not sentences:
        return []
    
    # Filter and classify sentences
    narrative_sentences = []
    other_sentences = []
    
    for sent in sentences:
        content_type = detect_content_type(sent)
        if content_type == "narrative":
            narrative_sentences.append(sent)
        elif content_type == "neutral":
            other_sentences.append(sent)
    
    # Use semantic grouping on narrative content
    narrative_paragraphs = group_semantically(narrative_sentences, threshold, min_chunk_size)
    
    # Add neutral content only if it fits well
    final_paragraphs = narrative_paragraphs
    
    print(f"📊 Content breakdown: {len(narrative_sentences)} narrative, {len(other_sentences)} other")
    return final_paragraphs

# ==========================================
# 🔍 STEP 5: Efficient Redundancy Filter
# ==========================================

def efficient_redundancy_filter(sentences: List[str],
                               semantic_threshold: float = REDUNDANCY_SEMANTIC_THRESHOLD,
                               quick_threshold: float = REDUNDANCY_QUICK_THRESHOLD) -> List[str]:
    """
    Remove duplicate and near-duplicate sentences efficiently.
    """
    print("\n🔍 Starting redundancy filtering...")
    
    # Quick Check: Exact and near-exact duplicates
    seen_hashes = set()
    seen_normalized = set()
    unique_sentences = []
    
    for sent in sentences:
        normalized = re.sub(r'\s+', ' ', sent.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        sent_hash = hashlib.md5(sent.encode()).hexdigest()
        
        if sent_hash in seen_hashes or normalized in seen_normalized:
            continue
        
        seen_hashes.add(sent_hash)
        seen_normalized.add(normalized)
        unique_sentences.append(sent)
    
    print(f"✅ Quick filter: {len(sentences)} → {len(unique_sentences)} sentences")
    
    # Semantic Check: Use embeddings for semantic similarity
    if len(unique_sentences) < 2:
        return unique_sentences
    
    try:
        print("🧠 Loading embedding model for semantic filtering...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("🧠 Encoding sentences...")
        batch_size = 50
        embeddings_list = []
        
        for i in range(0, len(unique_sentences), batch_size):
            batch = unique_sentences[i:i+batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)
        
        embeddings_np = np.vstack(embeddings_list)
        
        final_sentences = []
        final_embeddings = []
        
        for i, sent in enumerate(unique_sentences):
            if not final_embeddings:
                final_sentences.append(sent)
                final_embeddings.append(embeddings_np[i])
                continue
            
            similarities = cosine_similarity([embeddings_np[i]], final_embeddings)[0]
            
            if np.max(similarities) < semantic_threshold:
                final_sentences.append(sent)
                final_embeddings.append(embeddings_np[i])
        
        print(f"✅ Semantic filter: {len(unique_sentences)} → {len(final_sentences)} sentences")
        return final_sentences
    
    except Exception as e:
        print(f"⚠️  Semantic filtering failed: {e}")
        print("🔄 Returning quick-filtered results only")
        return unique_sentences

# ==========================================
# 💾 Save Results
# ==========================================

def save_results(book_data: dict, output_dir: str = "output"):
    """Save processed book data to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', book_data['title'])
    filepath = os.path.join(output_dir, f"{safe_title}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(book_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved: {filepath}")

# ==========================================
# 🚀 MAIN PIPELINE - ENHANCED FOR FOLDER PROCESSING
# ==========================================

def process_local_file_pipeline(file_path: str, book_title: str = None) -> dict:
    """
    Process a local .txt or .pdf file through the pipeline WITHOUT Ollama correction.
    """
    print(f"\n{'='*60}")
    print(f"📚 Processing Local File: {file_path}")
    print(f"{'='*60}")
    
    try:
        # STEP 1: Extract text from local file
        raw_text = process_local_file(file_path)
        if not raw_text:
            return None
        
        # STEP 2: Sentence splitting WITHOUT Ollama correction
        print("\n✂️  Splitting into sentences...")
        try:
            sentences = smart_sentence_split(raw_text)
        except Exception as e:
            print(f"⚠️  Sentence splitting failed: {e}")
            sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if s.strip()]
        
        print(f"📊 Total sentences: {len(sentences)}")
        
        # STEP 3: Enhanced structural cleaning with quality filtering
        print("\n🧹 Applying enhanced structural cleaning...")
        cleaned_sentences = []
        for s in sentences:
            cleaned = advanced_structural_cleaner(s)
            if cleaned and is_quality_content(cleaned):
                cleaned_sentences.append(cleaned)
        
        print(f"✅ After enhanced cleaning: {len(cleaned_sentences)} quality sentences")
        
        # STEP 4: Redundancy filtering
        unique_sentences = efficient_redundancy_filter(cleaned_sentences)
        
        # STEP 5: Enhanced semantic chunking
        paragraphs = group_semantically_enhanced(unique_sentences)
        
        # Final quality check on paragraphs
        final_paragraphs = [p for p in paragraphs if len(p.split()) >= 15 and len(p.split()) <= 300]
        
        # Prepare output
        result = {
            "source": file_path,
            "title": book_title or os.path.basename(file_path),
            "total_quality_sentences": len(unique_sentences),
            "total_paragraphs": len(final_paragraphs),
            "paragraphs": final_paragraphs,
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Critical error processing local file: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_supported_files(root_directory: str) -> List[str]:
    """
    Recursively find all .txt and .pdf files in the directory tree.
    """
    supported_files = []
    
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith(('.txt', '.pdf')):
                full_path = os.path.join(root, file)
                supported_files.append(full_path)
    
    return supported_files

def process_directory_structure(root_directory: str, output_base_dir: str = "processed_books") -> dict:
    """
    Process all .txt and .pdf files in a directory structure.
    Organizes output by maintaining folder structure.
    """
    print(f"\n{'='*80}")
    print(f"📁 PROCESSING DIRECTORY STRUCTURE: {root_directory}")
    print(f"{'='*80}")
    
    if not os.path.exists(root_directory):
        print(f"❌ Root directory not found: {root_directory}")
        return {}
    
    # Find all supported files
    all_files = find_supported_files(root_directory)
    print(f"📊 Found {len(all_files)} supported files (.txt, .pdf)")
    
    if not all_files:
        print("❌ No supported files found in the directory structure")
        return {}
    
    # Process files and organize by original folder structure
    results = {}
    processed_count = 0
    
    for file_path in all_files:
        # Get relative path to maintain folder structure
        relative_path = os.path.relpath(file_path, root_directory)
        folder_name = os.path.dirname(relative_path)
        
        # Create output directory that mirrors folder structure
        if folder_name:
            output_dir = os.path.join(output_base_dir, folder_name)
        else:
            output_dir = output_base_dir
        
        print(f"\n\n{'#'*80}")
        print(f"📄 Processing: {relative_path}")
        print(f"📁 Output folder: {output_dir}")
        print(f"{'#'*80}")
        
        try:
            # Use folder name as part of book title for better organization
            if folder_name:
                book_title = f"{folder_name}_{os.path.splitext(os.path.basename(file_path))[0]}"
            else:
                book_title = os.path.splitext(os.path.basename(file_path))[0]
            
            result = process_local_file_pipeline(file_path, book_title)
            
            if result:
                save_results(result, output_dir)
                processed_count += 1
                
                # Track results by folder
                if folder_name not in results:
                    results[folder_name] = []
                results[folder_name].append({
                    'file': relative_path,
                    'paragraphs': len(result['paragraphs']),
                    'sentences': result['total_quality_sentences']
                })
                
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
            continue
    
    # Print comprehensive summary
    print(f"\n\n{'='*80}")
    print(f"✅ DIRECTORY PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Overall Statistics:")
    print(f"   • Total files found: {len(all_files)}")
    print(f"   • Successfully processed: {processed_count}")
    print(f"   • Success rate: {(processed_count/len(all_files))*100:.1f}%")
    print(f"   • Output base directory: {output_base_dir}")
    
    print(f"\n📁 Folder-wise Breakdown:")
    for folder, files in results.items():
        folder_name = folder if folder else "Root Directory"
        total_paragraphs = sum(f['paragraphs'] for f in files)
        total_sentences = sum(f['sentences'] for f in files)
        print(f"   • {folder_name}: {len(files)} files, {total_paragraphs} paragraphs, {total_sentences} sentences")
    
    print(f"{'='*80}")
    
    return results

def process_multiple_files(file_paths: List[str], output_dir: str = "output"):
    """Process multiple local files."""
    results = []
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n\n{'#'*60}")
        print(f"📄 File {i}/{len(file_paths)}")
        print(f"{'#'*60}")
        
        try:
            result = process_local_file_pipeline(file_path)
            if result:
                results.append(result)
                save_results(result, output_dir)
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
            continue
    
    print(f"\n\n{'='*60}")
    print(f"✅ File processing complete!")
    print(f"📊 Successfully processed {len(results)}/{len(file_paths)} files")
    print(f"💾 Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return results

# ==========================================
# 🏃 ENHANCED EXAMPLE USAGE FOR DIRECTORY PROCESSING
# ==========================================

if __name__ == "__main__":
    # Option 1: Process entire directory structure
    root_directory = ""  # Change this to your directory path
    
    # Option 2: Process specific files
    specific_files = [
        "/Users/habibaadawi/Documents/projects/graduation_project/data_collection-/Books/Egyptian civilization/A century of excavation in the land of the Pharaohs by James Baikie.txt",
        # "path/to/specific/file2.pdf",
    ]
    
    # Option 3: Process directory structure (RECOMMENDED)
    if os.path.exists(root_directory):
        results = process_directory_structure(root_directory, "processed_books_output")
    elif specific_files:
        # Process specific files
        file_results = process_multiple_files(specific_files, "processed_files")
        print("\n📋 File Summary:")
        for r in file_results:
            print(f"  • {r['title']}: {r['total_paragraphs']} paragraphs")
    else:
        print("ℹ️  Please set the root_directory variable to your books directory path.")
        print("   Example: root_directory = '/home/user/my_books'")