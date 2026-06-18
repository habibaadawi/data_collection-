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

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
NGRAM_THRESHOLD = 0.65
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
# 📘 STEP 1: Extraction & Initial Clean
# ==========================================

def fetch_and_clean_html(url: str) -> str:
    """Download HTML and perform initial cleaning."""
    print(f"⬇️  Downloading: {url}")
    
    try:
        if "ebooks" in url and not url.endswith(('.html', '.htm')):
            ebook_id = re.search(r'/ebooks/(\d+)', url)
            if ebook_id:
                url = f"https://www.gutenberg.org/files/{ebook_id.group(1)}/{ebook_id.group(1)}-h/{ebook_id.group(1)}-h.htm"
                print(f"🔄 Converted to: {url}")
        
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")
    
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    body = soup.find("body") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    text = "\n".join(paragraphs) if paragraphs else soup.get_text(" ", strip=True)

    # Remove Project Gutenberg boilerplate
    text = re.sub(r"\*\*\*\s*START.*?\*\*\*", "", text, flags=re.I | re.S)
    text = re.sub(r"\*\*\*\s*END.*", "", text, flags=re.I | re.S)
    text = re.sub(r"Project Gutenberg.*?eBook", "", text, flags=re.I | re.S)
    
    # Remove common noise
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"\[Footnote[^\]]*\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    
    # Fix broken OCR patterns
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[_]{3,}", "", text)
    text = re.sub(r"[\*]{3,}", "", text)
    
    # Remove common OCR artifacts
    text = re.sub(r"[|]{2,}", "", text)
    text = re.sub(r"[=]{3,}", "", text)
    text = re.sub(r"[-]{4,}", "", text)
    
    print(f"✅ Extracted {len(text)} characters")
    return text.strip()

# ==========================================
# 🧮 N-gram Probability Checker
# ==========================================

def calculate_ngram_probability(sentence: str, n: int = 2) -> float:
    """
    Calculate a simple n-gram probability score.
    Higher score = more coherent/natural text.
    """
    try:
        words = sentence.split()
        if len(words) < n:
            return 1.0
        
        vectorizer = CountVectorizer(ngram_range=(n, n), token_pattern=r'\b\w+\b')
        X = vectorizer.fit_transform([sentence])
        ngram_counts = X.toarray()[0]
        
        total_ngrams = len(words) - n + 1
        unique_ngrams = np.count_nonzero(ngram_counts)
        
        if total_ngrams == 0:
            return 1.0
        
        score = unique_ngrams / total_ngrams
        
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in sentence) / max(len(sentence), 1)
        score *= alpha_ratio
        
        return min(score, 1.0)
    
    except Exception as e:
        print(f"⚠️  N-gram calculation failed: {e}")
        return 1.0

# ==========================================
# 🤖 STEP 2: Ollama Correction Function - MODIFIED
# ==========================================

def ollama_correction_function(text_chunk: str) -> str:
    """
    Send text to local Ollama for grammatical/semantic correction.
    """
    prompt = f"""You are a professional text editor. Please correct and enhance the following text by:

                1. Fixing all grammatical errors and spelling mistakes
                2. Correcting any OCR scanning errors
                3. Improving overall readability and flow
                4. Removing any illogical number sequences or repeated digits that don't make sense
                5. Preserving the original meaning while making it more coherent
                6. If nothing was given to you, retrun nothing 

                Focus on extracting and maintaining only the semantically meaningful content. Remove any nonsensical fragments while keeping the logical narrative intact.

                Important: Return ONLY the corrected text without any explanations, notes, or introductory text.
                {text_chunk}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048
        }
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", text_chunk).strip()
    
    except Exception as e:
        print(f"⚠️  Ollama correction failed: {e}")
        return text_chunk

def process_with_selective_correction(text: str) -> List[str]:
    """
    Split text into sentences and correct ALL sentences one by one.
    """
    print("\n✂️  Splitting into sentences...")
    try:
        sentences = smart_sentence_split(text)
    except Exception as e:
        print(f"⚠️  Sentence splitting failed: {e}")
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    print(f"📊 Total sentences: {len(sentences)}")
    
    corrected_sentences = []
    
    for i, sent in enumerate(sentences):
        if len(sent.strip()) < 10:
            continue
        
        print(f"🔧 Correcting sentence {i+1}/{len(sentences)}")
        corrected = ollama_correction_function(sent)
        corrected_sentences.append(corrected)
    
    print(f"✅ Corrected {len(corrected_sentences)} sentences")
    return corrected_sentences

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
# 🚀 MAIN PIPELINE
# ==========================================

def process_book(url: str, book_title: str = None) -> dict:
    """Process a single book with enhanced filtering."""
    print(f"\n{'='*60}")
    print(f"📚 Processing: {url}")
    print(f"{'='*60}")
    
    try:
        # STEP 1: Extract and clean HTML
        raw_text = fetch_and_clean_html(url)
        if not raw_text:
            return None
        
        # STEP 2: Selective correction and sentence splitting - NOW CORRECTS ALL SENTENCES
        sentences = process_with_selective_correction(raw_text)
        
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
            "url": url,
            "title": book_title or url.split('/')[-1],
            "total_quality_sentences": len(unique_sentences),
            "total_paragraphs": len(final_paragraphs),
            "paragraphs": final_paragraphs,
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Critical error processing book: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_multiple_books(urls: List[str], output_dir: str = "output"):
    """Process multiple books from a list of URLs."""
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n\n{'#'*60}")
        print(f"📖 Book {i}/{len(urls)}")
        print(f"{'#'*60}")
        
        try:
            result = process_book(url)
            if result:
                results.append(result)
                save_results(result, output_dir)
        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")
            continue
    
    print(f"\n\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"📊 Successfully processed {len(results)}/{len(urls)} books")
    print(f"💾 Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return results

# ==========================================
# 🏃 EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # List of Project Gutenberg URLs
    book_urls = [
        "https://www.gutenberg.org/cache/epub/6624/pg6624-images.html",
        # Add more URLs here
    ]
    
    # Process all books
    results = process_multiple_books(book_urls, output_dir="processed_books")
    
    # Optional: Print summary
    print("\n📋 Summary:")
    for r in results:
        print(f"  • {r['title']}: {r['total_paragraphs']} paragraphs")