import json
import os
from pathlib import Path
from typing import List, Dict

# ==========================================
# 🔧 Configuration
# ==========================================

INPUT_DIRECTORY = "All_data_4"  # Your cleaned data folder
OUTPUT_FORMATS = {
    "jsonl": "training_data.jsonl",           # Standard JSONL
    "instruction": "training_data_instruct.jsonl",  # Instruction format
    "text": "training_data.txt",              # Plain text
    "csv": "training_data.csv"                # CSV format
}

# ==========================================
# 📊 Format Converters
# ==========================================

def format_jsonl(title: str, paragraphs: List[str]) -> str:
    """
    Format 1: Standard JSONL - Each book as one JSON line
    Best for: GPT, Claude, LLaMA fine-tuning
    """
    # Join paragraphs with double newline
    text = "\n\n".join(paragraphs)
    
    # Create a single text with title
    full_text = f"Title: {title}\n\n{text}"
    
    # Return as JSON line
    return json.dumps({"text": full_text}, ensure_ascii=False)


def format_jsonl_per_paragraph(title: str, paragraphs: List[str]) -> List[str]:
    """
    Format 2: JSONL with each paragraph as separate entry
    Best for: Smaller context windows, more granular training
    """
    lines = []
    for i, para in enumerate(paragraphs):
        entry = {
            "text": para,
            "metadata": {
                "source": title,
                "paragraph_index": i
            }
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return lines


def format_instruction(title: str, paragraphs: List[str]) -> List[str]:
    """
    Format 3: Instruction format (Q&A style)
    Best for: Chat models, instruction-following
    """
    lines = []
    text = "\n\n".join(paragraphs)
    
    # Create different instruction variations
    instructions = [
        f"Provide a detailed explanation from '{title}'",
        f"What does '{title}' say about this topic?",
        f"Summarize the content from '{title}'",
        f"Explain the historical context from '{title}'"
    ]
    
    for instruction in instructions:
        entry = {
            "instruction": instruction,
            "input": title,
            "output": text[:2000]  # Limit for manageable size
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    
    return lines


def format_plain_text(title: str, paragraphs: List[str], separator: str = "\n\n") -> str:
    """
    Format 4: Plain text with special tokens
    Best for: Simple models, custom tokenization
    """
    text = separator.join(paragraphs)
    return f"<|book|>{title}<|content|>{text}<|endoftext|>\n\n"


def format_csv_row(title: str, paragraphs: List[str]) -> str:
    """
    Format 5: CSV format
    Best for: Easy inspection, spreadsheet analysis
    """
    text = " ".join(paragraphs).replace('"', '""')  # Escape quotes
    return f'"{title}","{text}"\n'


# ==========================================
# 📁 File Processing
# ==========================================

def process_json_file(filepath: Path) -> Dict:
    """Extract title and paragraphs from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        title = data.get('title', filepath.stem)
        paragraphs = data.get('paragraphs', [])
        
        if not paragraphs:
            return None
        
        return {
            'title': title,
            'paragraphs': paragraphs,
            'filepath': str(filepath)
        }
    
    except Exception as e:
        print(f"Error processing {filepath.name}: {e}")
        return None


def find_all_json_files(root_dir: str) -> List[Path]:
    """Recursively find all JSON files."""
    root_path = Path(root_dir)
    return list(root_path.rglob("*.json"))


# ==========================================
# 🎯 Main Merging Functions
# ==========================================

def merge_to_jsonl(input_dir: str, output_file: str, per_paragraph: bool = False):
    """
    Merge all JSON files into JSONL format.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Output JSONL file path
        per_paragraph: If True, each paragraph becomes a separate entry
    """
    print(f"\n{'='*70}")
    print(f"📄 Merging to JSONL: {output_file}")
    print(f"{'='*70}\n")
    
    json_files = find_all_json_files(input_dir)
    total_books = 0
    total_paragraphs = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in json_files:
            book_data = process_json_file(filepath)
            if not book_data:
                continue
            
            total_books += 1
            total_paragraphs += len(book_data['paragraphs'])
            
            if per_paragraph:
                # One line per paragraph
                lines = format_jsonl_per_paragraph(
                    book_data['title'], 
                    book_data['paragraphs']
                )
                for line in lines:
                    out.write(line + '\n')
            else:
                # One line per book
                line = format_jsonl(
                    book_data['title'], 
                    book_data['paragraphs']
                )
                out.write(line + '\n')
            
            if total_books % 10 == 0:
                print(f"  Processed {total_books} books...")
    
    print(f"\n✅ Complete!")
    print(f"  📚 Books processed: {total_books}")
    print(f"  📝 Total paragraphs: {total_paragraphs:,}")
    print(f"  💾 Output: {output_file}\n")


def merge_to_instruction(input_dir: str, output_file: str):
    """Merge all JSON files into instruction format."""
    print(f"\n{'='*70}")
    print(f"📄 Merging to Instruction Format: {output_file}")
    print(f"{'='*70}\n")
    
    json_files = find_all_json_files(input_dir)
    total_books = 0
    total_entries = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in json_files:
            book_data = process_json_file(filepath)
            if not book_data:
                continue
            
            total_books += 1
            lines = format_instruction(
                book_data['title'], 
                book_data['paragraphs']
            )
            
            for line in lines:
                out.write(line + '\n')
                total_entries += 1
    
    print(f"\n✅ Complete!")
    print(f"  📚 Books processed: {total_books}")
    print(f"  📝 Instruction entries: {total_entries:,}")
    print(f"  💾 Output: {output_file}\n")


def merge_to_text(input_dir: str, output_file: str):
    """Merge all JSON files into plain text format."""
    print(f"\n{'='*70}")
    print(f"📄 Merging to Plain Text: {output_file}")
    print(f"{'='*70}\n")
    
    json_files = find_all_json_files(input_dir)
    total_books = 0
    total_chars = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in json_files:
            book_data = process_json_file(filepath)
            if not book_data:
                continue
            
            total_books += 1
            text = format_plain_text(
                book_data['title'], 
                book_data['paragraphs']
            )
            out.write(text)
            total_chars += len(text)
    
    print(f"\n✅ Complete!")
    print(f"  📚 Books processed: {total_books}")
    print(f"  📝 Total characters: {total_chars:,}")
    print(f"  💾 Output: {output_file}\n")


def merge_to_csv(input_dir: str, output_file: str):
    """Merge all JSON files into CSV format."""
    print(f"\n{'='*70}")
    print(f"📄 Merging to CSV: {output_file}")
    print(f"{'='*70}\n")
    
    json_files = find_all_json_files(input_dir)
    total_books = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        # Write header
        out.write('"title","text"\n')
        
        for filepath in json_files:
            book_data = process_json_file(filepath)
            if not book_data:
                continue
            
            total_books += 1
            row = format_csv_row(
                book_data['title'], 
                book_data['paragraphs']
            )
            out.write(row)
    
    print(f"\n✅ Complete!")
    print(f"  📚 Books processed: {total_books}")
    print(f"  💾 Output: {output_file}\n")


# ==========================================
# 📊 Statistics & Analysis
# ==========================================

def analyze_dataset(input_dir: str):
    """Analyze the dataset and provide statistics."""
    print(f"\n{'='*70}")
    print(f"📊 DATASET ANALYSIS")
    print(f"{'='*70}\n")
    
    json_files = find_all_json_files(input_dir)
    
    total_books = 0
    total_paragraphs = 0
    total_words = 0
    total_chars = 0
    
    book_sizes = []
    
    for filepath in json_files:
        book_data = process_json_file(filepath)
        if not book_data:
            continue
        
        total_books += 1
        num_paras = len(book_data['paragraphs'])
        total_paragraphs += num_paras
        
        book_text = " ".join(book_data['paragraphs'])
        book_words = len(book_text.split())
        book_chars = len(book_text)
        
        total_words += book_words
        total_chars += book_chars
        
        book_sizes.append({
            'title': book_data['title'],
            'paragraphs': num_paras,
            'words': book_words
        })
    
    # Sort by size
    book_sizes.sort(key=lambda x: x['words'], reverse=True)
    
    print(f"📚 Total Books: {total_books}")
    print(f"📝 Total Paragraphs: {total_paragraphs:,}")
    print(f"💬 Total Words: {total_words:,}")
    print(f"📄 Total Characters: {total_chars:,}")
    print(f"\n📏 Average per Book:")
    print(f"   • Paragraphs: {total_paragraphs/total_books:.1f}")
    print(f"   • Words: {total_words/total_books:,.0f}")
    print(f"\n🔝 Top 10 Largest Books:")
    for i, book in enumerate(book_sizes[:10], 1):
        print(f"   {i}. {book['title'][:60]}")
        print(f"      {book['words']:,} words, {book['paragraphs']} paragraphs")
    
    print(f"\n{'='*70}\n")


# ==========================================
# 🚀 Main Function
# ==========================================

def main():
    """Main function to merge JSON files into various formats."""
    
    print("\n" + "="*70)
    print("🚀 JSON TO LLM TRAINING DATA CONVERTER")
    print("="*70)
    
    # Analyze the dataset first
    analyze_dataset(INPUT_DIRECTORY)
    
    print("\n📦 Generating Training Data in Multiple Formats...\n")
    
    # Format 1: Standard JSONL (one book per line)
    merge_to_jsonl(
        INPUT_DIRECTORY, 
        OUTPUT_FORMATS['jsonl'], 
        per_paragraph=False
    )
    
    # Format 2: JSONL with paragraphs as separate entries
    merge_to_jsonl(
        INPUT_DIRECTORY, 
        "training_data_paragraphs.jsonl", 
        per_paragraph=True
    )
    
    # Format 3: Instruction format
    merge_to_instruction(
        INPUT_DIRECTORY, 
        OUTPUT_FORMATS['instruction']
    )
    
    # Format 4: Plain text
    merge_to_text(
        INPUT_DIRECTORY, 
        OUTPUT_FORMATS['text']
    )
    
    # Format 5: CSV
    merge_to_csv(
        INPUT_DIRECTORY, 
        OUTPUT_FORMATS['csv']
    )
    
    print("\n" + "="*70)
    print("✅ ALL FORMATS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output Files:")
    for format_name, filename in OUTPUT_FORMATS.items():
        size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"   • {filename} ({size:.2f} MB)")
    print()
    
    print("💡 Recommendations:")
    print("   • For GPT/Claude: Use training_data.jsonl")
    print("   • For LLaMA: Use training_data.txt")
    print("   • For instruction models: Use training_data_instruct.jsonl")
    print("   • For smaller contexts: Use training_data_paragraphs.jsonl")
    print()


if __name__ == "__main__":
    main()