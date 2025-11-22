import json; import random; import os; import pickle; import argparse; import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

class GroundingDatasetGenerator:
    def __init__(self, graph_filepath: str):
        self.nodes = self._load_graph(graph_filepath)
        print("Grounding Dataset Generator (Smart Q&A) initialized.")

    def _load_graph(self, filepath: str):
        if not os.path.exists(filepath): sys.exit(f"FATAL ERROR: Knowledge Graph not found at '{filepath}'")
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: sys.exit(f"FATAL ERROR: Could not load KG: {e}")

    def generate_qa_pairs(self) -> list:
        if not self.nodes:
            print("Knowledge graph is empty. No Q&A pairs to generate.")
            return []

        qa_pairs = []
        print(f"Generating dataset from {len(self.nodes)} nodes...")

        for node_id, node_obj in list(self.nodes.items()):
            props = node_obj.properties
            
            # --- STRATEGY 1: Direct Passthrough (SOTA Data) ---
            # If the node already holds a Q/A pair, use it.
            if 'question' in props and 'answer' in props:
                qa_pairs.append({"question": props['question'], "answer": props['answer']})
                continue
            
            if 'question' in props and 'correct_answer_text' in props: # ARC style fallback
                qa_pairs.append({"question": props['question'], "answer": props['correct_answer_text']})
                continue

            if 'goal' in props and 'correct_solution' in props: # PIQA style
                qa_pairs.append({"question": props['goal'], "answer": props['correct_solution']})
                continue

            if 'bug_report' in props and 'culprit_file' in props: # SWE-Bench
                q = f"Bug Report: {props['bug_report']}\nWhich file is likely the culprit?"
                qa_pairs.append({"question": q, "answer": props['culprit_file']})
                continue

            # --- STRATEGY 2: Generative (Original Logic) ---
            # Use this for raw concepts/definitions
            display_name = node_obj.label
            if not display_name: continue

            if 'definition' in props:
                qa_pairs.append({"question": f"What is the definition of {display_name}?", "answer": str(props['definition']).lower()})
            
            if 'summary' in props:
                qa_pairs.append({"question": f"Summarize {display_name}.", "answer": str(props['summary']).lower()})

        print(f"   -> Generated {len(qa_pairs)} Q&A pairs.")
        return qa_pairs

    def write_dataset_to_file(self, qa_pairs: list, output_filepath: str):
        print(f"Writing to '{output_filepath}'...")
        random.shuffle(qa_pairs)
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for pair in qa_pairs:
                    f.write(json.dumps(pair) + '\n')
            print("Dataset generation complete.")
        except Exception as e:
            print(f"FATAL: Could not write dataset: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    print("--- Running Smart Q&A Dataset Generation ---")
    generator = GroundingDatasetGenerator(graph_filepath=args.graph_path)
    qa_pairs = generator.generate_qa_pairs()
    if qa_pairs:
        generator.write_dataset_to_file(qa_pairs, output_filepath=args.output_path)