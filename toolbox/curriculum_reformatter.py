import json
import argparse
import re
import hashlib
from tqdm import tqdm
import os

class CurriculumReformatter:
    """
    REVERSE-ENGINEERING EDITION.
    Parses a flattened 'question/answer' dataset, detects the original source type
    via text heuristics, and reconstructs rich CognitiveNodes.
    """
    def __init__(self):
        self.stats = {
            "total_processed": 0, "reformatted": 0, "skipped": 0,
            "types": {
                "boolq": 0, "piqa": 0, "arc_easy": 0, "gsm8k": 0, 
                "swe_bench": 0, "generic_qa": 0
            }
        }
        print("--- Curriculum Forensics Engine Initialized ---")

    def _generate_node_id(self, content_string):
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()

    def _generate_label(self, text, limit=60):
        if not text: return "Untitled Node"
        clean = str(text).replace("\n", " ").strip()
        # Remove common prefixes for cleaner labels
        clean = clean.replace("Question:", "").replace("Context:", "").strip()
        return (clean[:limit] + '..') if len(clean) > limit else clean

    def _parse_boolq(self, q, a):
        """
        Fingerprint: Starts with "Context:" ... contains "Based on the context..."
        """
        try:
            # Split context from the actual question
            parts = q.split("Based on the context, is the following statement true or false?")
            context = parts[0].replace("Context:", "").strip()
            actual_question = parts[1].replace("Question:", "").strip() if len(parts) > 1 else "Unknown Question"
            
            unique_str = f"{context}{actual_question}"
            return {
                "node_id": self._generate_node_id(unique_str),
                "label": self._generate_label(actual_question),
                "node_type": "concept.verification",
                "properties": {
                    "question": actual_question,
                    "context": context,
                    "answer": a.lower(),
                    "mict_metadata": {"complexity": "low", "modality": "text"}
                },
                "source_lessons": ["BoolQ"]
            }
        except Exception:
            return None

    def _parse_piqa(self, q, a):
        """
        Fingerprint: Starts with "To achieve the goal of"
        """
        try:
            # Extract Goal
            goal_match = re.search(r"To achieve the goal of '(.*?)',", q)
            goal = goal_match.group(1) if goal_match else "Unknown Goal"
            
            # Extract Solutions
            sol_a_match = re.search(r"Solution A: (.*?)\.\.", q)
            sol_b_match = re.search(r"Solution B: (.*?)\.\.", q)
            
            sol_a = sol_a_match.group(1) if sol_a_match else "Unknown"
            sol_b = sol_b_match.group(1) if sol_b_match else "Unknown"
            
            return {
                "node_id": self._generate_node_id(q+a),
                "label": self._generate_label(goal),
                "node_type": "concept.physical_reasoning",
                "properties": {
                    "goal": goal,
                    "solution_1": sol_a,
                    "solution_2": sol_b,
                    "correct_solution": a,
                    "mict_metadata": {"complexity": "medium", "modality": "physical_logic"}
                },
                "source_lessons": ["PIQA"]
            }
        except Exception:
            return None

    def _parse_arc(self, q, a):
        """
        Fingerprint: Contains "Which of the following is the correct answer?"
        """
        try:
            parts = q.split("Which of the following is the correct answer?")
            main_question = parts[0].replace("Question:", "").strip()
            options = parts[1].strip() if len(parts) > 1 else ""
            
            return {
                "node_id": self._generate_node_id(q+a),
                "label": self._generate_label(main_question),
                "node_type": "concept.scientific_knowledge",
                "properties": {
                    "question": main_question,
                    "options_text": options,
                    "correct_answer": a,
                    "mict_metadata": {"complexity": "medium", "modality": "science"}
                },
                "source_lessons": ["ARC-Easy"]
            }
        except Exception:
            return None

    def _parse_swe_bench(self, q, a):
        """
        Fingerprint: Starts with "Bug Report:"
        """
        try:
            # Extract the file path from the answer usually
            return {
                "node_id": self._generate_node_id(q+a),
                "label": self._generate_label(q, limit=40),
                "node_type": "skill.debugging",
                "properties": {
                    "bug_report": q.replace("Bug Report:", "").strip(),
                    "culprit_file": a,
                    "mict_metadata": {"complexity": "extreme", "modality": "code"}
                },
                "source_lessons": ["SWE-Bench"]
            }
        except Exception:
            return None

    def _parse_gsm8k(self, q, a):
        """
        Fingerprint: Answer contains "calculated as" or math logic
        """
        return {
            "node_id": self._generate_node_id(q+a),
            "label": self._generate_label(q),
            "node_type": "concept.chain_of_thought",
            "properties": {
                "question": q,
                "reasoning_trace": a, # The answer field here is the full CoT
                "mict_metadata": {"complexity": "high", "modality": "math"}
            },
            "source_lessons": ["GSM8K"]
        }

    def _parse_generic(self, q, a):
        """Fallback for anything else."""
        return {
            "node_id": self._generate_node_id(q+a),
            "label": self._generate_label(q),
            "node_type": "concept.general_knowledge",
            "properties": {
                "question": q,
                "answer": a,
                "mict_metadata": {"complexity": "unknown", "modality": "text"}
            },
            "source_lessons": ["General"]
        }

    def reformat_corpus(self, input_file, output_file):
        print(f"--- Starting Forensics Reformatting ---")
        print(f"  Input:  '{input_file}'")
        print(f"  Output: '{output_file}'")
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc="Reconstructing Nodes"):
                self.stats["total_processed"] += 1
                try:
                    item = json.loads(line)
                    q = item.get('question', '')
                    a = item.get('answer', '')
                    
                    if not q or not a:
                        self.stats["skipped"] += 1
                        continue

                    node = None
                    
                    # --- CONTENT FINGERPRINTING ---
                    
                    # 1. Check for BoolQ (Specific Context phrase)
                    if "Context:" in q and "Based on the context" in q:
                        node = self._parse_boolq(q, a)
                        if node: self.stats["types"]["boolq"] += 1

                    # 2. Check for PIQA (Goal phrase)
                    elif "To achieve the goal of" in q:
                        node = self._parse_piqa(q, a)
                        if node: self.stats["types"]["piqa"] += 1

                    # 3. Check for ARC (Multiple choice phrase)
                    elif "Which of the following is the correct answer?" in q:
                        node = self._parse_arc(q, a)
                        if node: self.stats["types"]["arc_easy"] += 1

                    # 4. Check for SWE-Bench (Bug Report phrase)
                    elif q.strip().startswith("Bug Report:") or "most likely source of the bug" in q:
                        node = self._parse_swe_bench(q, a)
                        if node: self.stats["types"]["swe_bench"] += 1

                    # 5. Check for GSM8K (Calculation keywords in answer)
                    elif "calculated as" in a or "final answer is" in a:
                        node = self._parse_gsm8k(q, a)
                        if node: self.stats["types"]["gsm8k"] += 1

                    # 6. Fallback
                    else:
                        node = self._parse_generic(q, a)
                        self.stats["types"]["generic_qa"] += 1

                    if node:
                        f_out.write(json.dumps(node) + '\n')
                        self.stats["reformatted"] += 1
                    else:
                        self.stats["skipped"] += 1

                except json.JSONDecodeError:
                    self.stats["skipped"] += 1
                    continue
        
        self.print_summary()

    def print_summary(self):
        print("\n" + "="*50)
        print(" " * 10 + "Forensics Complete!")
        print("="*50)
        print(f"  Total Lines: {self.stats['total_processed']}")
        print(f"  Nodes Created: {self.stats['reformatted']}")
        print("-"*50)
        print("  Detected Types:")
        for key, value in self.stats['types'].items():
            if value > 0:
                print(f"    - {key.upper():<15}: {value}")
        print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reverse-engineer flat datasets into CognitiveNodes.")
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', default='sota_clean_formatted.jsonl')
    args = parser.parse_args()
    reformatter = CurriculumReformatter()
    reformatter.reformat_corpus(args.input_file, args.output_file)