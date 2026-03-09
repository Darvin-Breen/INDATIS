#!/usr/bin/env python3
"""
Translate all English documents from INDATIS ENGLISH folder to Italian
"""

import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import time

# Configuration
INPUT_FOLDER = "/Users/Neng/Desktop/INDATIS ENGLISH"
OUTPUT_FOLDER = "translated_italian"
MODEL_ID = "puettmann/LlaMaestra-3.2-1B-Translation"

def setup_folders():
    """Create output folder if it doesn't exist"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"📁 Output folder: {OUTPUT_FOLDER}")

def load_model():
    """Load the translation model"""
    print(f"\n🔄 Loading model: {MODEL_ID}")
    print("   This may take a few minutes for first download...")
    
    # Load tokenizer and model (simplified approach)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    print("   ✅ Model loaded successfully")
    return tokenizer, model

def translate_chunk(text, tokenizer, model, max_length=30720):
    """Translate a single chunk of text"""
    prompt = f"Translate from English to Italian: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,  # Lower temperature for more accurate translation
            do_sample=False   # Deterministic output
        )
    
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt part from the output
    if translated.startswith(prompt):
        translated = translated[len(prompt):].strip()
    
    return translated

def translate_document(file_path, tokenizer, model):
    """Translate an entire document, splitting if necessary"""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Original size: {len(text)} characters")
    
    # If document is very long, split into paragraphs
    if len(text) > 25000:
        print(f"   Document is long, splitting into paragraphs...")
        paragraphs = text.split('\n\n')
        translated_paragraphs = []
        
        for para in tqdm(paragraphs, desc="   Translating paragraphs", leave=False):
            if para.strip():
                try:
                    translated = translate_chunk(para, tokenizer, model)
                    translated_paragraphs.append(translated)
                except Exception as e:
                    print(f"   ⚠️ Error translating paragraph: {e}")
                    translated_paragraphs.append(para)  # Keep original on error
            else:
                translated_paragraphs.append("")  # Keep empty paragraphs
        
        return '\n\n'.join(translated_paragraphs)
    else:
        # Translate whole document
        return translate_chunk(text, tokenizer, model)

def main():
    print("="*60)
    print("🇬🇧 → 🇮🇹 ENGLISH TO ITALIAN TRANSLATOR")
    print("="*60)
    print(f"Input folder: {INPUT_FOLDER}")
    
    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder not found: {INPUT_FOLDER}")
        print("   Please make sure the folder exists on your Desktop")
        return
    
    # Get all text files
    txt_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    print(f"Found {len(txt_files)} text files to translate")
    
    if len(txt_files) == 0:
        print("❌ No .txt files found in the folder")
        return
    
    # List files found
    print("\n📄 Files to translate:")
    for i, file in enumerate(txt_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # Setup output folder
    setup_folders()
    
    # Load model
    tokenizer, model = load_model()
    
    # Translate each file
    print("\n" + "="*60)
    print("🔄 TRANSLATING DOCUMENTS")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for i, file_path in enumerate(txt_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(txt_files)}] Translating: {filename}")
        
        try:
            # Translate document
            translated_text = translate_document(file_path, tokenizer, model)
            
            # Save translated document
            output_path = os.path.join(OUTPUT_FOLDER, filename.replace('.txt', '_italian.txt'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            print(f"   ✅ Saved to: {output_path}")
            successful += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRANSLATION COMPLETE")
    print("="*60)
    print(f"Successfully translated: {successful} files")
    print(f"Failed: {failed} files")
    print(f"\nTranslated files saved in: {OUTPUT_FOLDER}/")
    print("\nTo view translated files:")
    print(f"  ls -la {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    main()
