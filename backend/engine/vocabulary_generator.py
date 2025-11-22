import json
import os
import argparse
from tqdm import tqdm
from collections import Counter

class CharacterVocabularyGenerator:
    """
    UPGRADED: CognitiveNode Aware.
    Scans directories and handles both flat JSONL and nested CognitiveNode structures.
    """
    def __init__(self):
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        print("Dynamic Character Vocabulary Generator initialized.")

    def _extract_text_recursive(self, data):
        """Recursively finds string values in a dictionary to build vocab."""
        text = ""
        if isinstance(data, dict):
            for key, value in data.items():
                # Skip metadata keys that don't contain learnable language
                if key in ['node_id', 'node_type', 'source_lessons', 'mict_metadata', 'complexity']:
                    continue
                text += self._extract_text_recursive(value)
        elif isinstance(data, list):
            for item in data:
                text += self._extract_text_recursive(item)
        elif isinstance(data, str):
            text += data
        return text

    def generate_from_corpus(self, curriculum_path: str):
        print(f"1. Scanning corpus '{curriculum_path}' to build vocabulary...")
        
        files_to_process = []
        if os.path.isdir(curriculum_path):
            for root, _, files in os.walk(curriculum_path):
                for file in files:
                    if file.endswith(('.jsonl', '.json', '.txt')):
                        files_to_process.append(os.path.join(root, file))
        elif os.path.isfile(curriculum_path):
            files_to_process.append(curriculum_path)

        if not files_to_process:
            print("FATAL: No valid files found.")
            return None

        total_char_counts = Counter()

        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Try line-by-line JSON (JSONL)
                    content = f.read()
                    try:
                        lines = content.splitlines()
                        for line in lines:
                            if not line.strip(): continue
                            try:
                                item = json.loads(line)
                                # Recursive extraction handles CognitiveNodes and flat JSON
                                text_chunk = self._extract_text_recursive(item)
                                total_char_counts.update(text_chunk.lower())
                            except json.JSONDecodeError:
                                total_char_counts.update(line.lower())
                    except Exception:
                        total_char_counts.update(content.lower())
            except Exception as e:
                print(f"   [WARN] Could not read file {file_path}: {e}")

        unique_chars = sorted(total_char_counts.keys())
        
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        for char in unique_chars:
            if char not in vocab:
                vocab[char] = len(vocab)
        
        print(f"2. Vocabulary created with {len(vocab)} unique tokens.")
        return vocab

    def save_vocab(self, vocab_dict: dict, output_filepath: str):
        print(f"3. Writing vocabulary to '{output_filepath}'...")
        try:
            with open(output_filepath, 'w') as f:
                json.dump(vocab_dict, f, indent=2)
            print("   -> Vocabulary generation complete.")
        except Exception as e:
            print(f"FATAL: Could not write vocabulary to file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a character-level vocabulary from a corpus.")
    parser.add_argument('--curriculum-path', help="Path to the source .jsonl corpus file or directory.")
    parser.add_argument('--output', help="Path for the output vocab.json file.")
    args = parser.parse_args()

    print("--- Running Dynamic Vocabulary Generation ---")
    generator = CharacterVocabularyGenerator()
    new_vocab = generator.generate_from_corpus(curriculum_path=args.curriculum_path)
    if new_vocab:
        generator.save_vocab(new_vocab, output_filepath=args.output)