import os; import json; import argparse; import sys
class IngestionSystem:
    def __init__(self, curriculum_root_path: str, output_filepath: str):
        self.curriculum_root = curriculum_root_path; self.output_filepath = output_filepath; self.lessons = []
    def discover_and_ingest(self):
        print(f"Scanning '{self.curriculum_root}'...")
        if not os.path.isdir(self.curriculum_root): sys.exit(f"FATAL: Curriculum root not found.")
        for dirpath, _, filenames in os.walk(self.curriculum_root):
            for filename in filenames:
                if filename.endswith(('.json', '.jsonl')):
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        try:
                            if filename.endswith('.jsonl'): [self.lessons.append(json.loads(line)) for line in f]
                            else: self.lessons.append(json.load(f))
                        except json.JSONDecodeError as e: print(f"  -> WARNING: Skipping '{filename}' due to JSON error: {e}")
        print(f"Ingested {len(self.lessons)} total lesson objects.")
    def write_pre_graph_to_file(self):
        print(f"Writing pre-graph to '{self.output_filepath}'...")
        with open(self.output_filepath, 'w') as f: json.dump(self.lessons, f, indent=2)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('curriculum_root'); parser.add_argument('output_filepath')
    args = parser.parse_args()
    ingest = IngestionSystem(args.curriculum_root, args.output_filepath)
    ingest.discover_and_ingest(); ingest.write_pre_graph_to_file()