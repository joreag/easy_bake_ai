import os
import sys
import subprocess
import argparse

def run_command(command, cwd, project_root):
    """Executes a shell command with a modified environment."""
    print("\n" + "="*80 + f"\nExecuting: {' '.join(command)}\n" + f"Working Directory: {cwd}\n" + "="*80)
    
    env = os.environ.copy()
    # Ensure project root is in PYTHONPATH
    env['PYTHONPATH'] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        env=env
    )
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
        sys.stdout.flush()
        
    process.wait()
    if process.returncode != 0:
        print(f"\n--- ERROR DETECTED in sub-process. Build failed. ---")
        sys.exit(1)
    print("--- Sub-process completed successfully. ---")

def main():
    parser = argparse.ArgumentParser(description="Unified Build Engine for a full HCTS-Transformer v1 Forge.")
    parser.add_argument('--curriculum-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--callback-url', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-seq-length', type=int, default=256)
    parser.add_argument('--d-model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--dim-feedforward', type=int, default=2048)
    parser.add_argument('--arch-type', type=str, default='standard')
    args = parser.parse_args()

    ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_ROOT = os.path.dirname(os.path.dirname(ENGINE_DIR))
    PROJECT_ROOT = APP_ROOT 

    ingestion_script = os.path.join(ENGINE_DIR, 'ingestion_system.py')
    kg_builder_script = os.path.join(ENGINE_DIR, 'knowledge_graph_builder.py')
    dataset_gen_script = os.path.join(ENGINE_DIR, 'dataset_generator.py')
    trainer_script = os.path.join(ENGINE_DIR, 'trainer.py')
    vocab_script = os.path.join(ENGINE_DIR, 'vocabulary_generator.py')
    
    os.makedirs(args.output_dir, exist_ok=True)
    pre_graph_path = os.path.join(args.output_dir, 'pre_graph.json')
    kg_path = os.path.join(args.output_dir, 'knowledge_graph.pkl')
    dataset_path = os.path.join(args.output_dir, 'grounding_dataset.jsonl')
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    model_path = os.path.join(args.output_dir, 'model.pth')

    CWD = PROJECT_ROOT

    print("\n### STAGE 1: INGESTING CURRICULUM ###")
    run_command([sys.executable, ingestion_script, args.curriculum_dir, pre_graph_path], cwd=CWD, project_root=PROJECT_ROOT)

    print("\n### STAGE 2: BUILDING KNOWLEDGE GRAPH ###")
    run_command([sys.executable, kg_builder_script, pre_graph_path, kg_path], cwd=CWD, project_root=PROJECT_ROOT)

    print("\n### STAGE 3: GENERATING Q&A DATASET ###")
    run_command([sys.executable, dataset_gen_script, kg_path, dataset_path,], cwd=CWD, project_root=PROJECT_ROOT)
    
    print("\n### STAGE 4: GENERATING VOCABULARY ###")
    run_command([sys.executable, vocab_script, '--output', vocab_path, '--curriculum-path', args.curriculum_dir], cwd=CWD, project_root=PROJECT_ROOT)

    print("\n### STAGE 5: TRAINING THE MODEL ###")
    trainer_command = [
        sys.executable, trainer_script,
        '--arch-type', args.arch_type,
        '--dataset-path', dataset_path,
        '--vocab-path', vocab_path,
        '--output-model', model_path,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--max-seq-length', str(args.max_seq_length),
        '--d-model', str(args.d_model),
        '--nhead', str(args.nhead),
        '--num-encoder-layers', str(args.num_encoder_layers),
        '--num-decoder-layers', str(args.num_decoder_layers),
        '--dim-feedforward', str(args.dim_feedforward)
    ]
    if args.callback_url:
        trainer_command.extend(['--callback-url', args.callback_url])
        
    # --- FIX: Correctly passed project_root and cwd ---
    run_command(trainer_command, cwd=CWD, project_root=PROJECT_ROOT)

    print("\n" + "#"*80 + "\n# Easy Bake AI: Full Forge Complete! #\n" + "#"*80)

if __name__ == '__main__':
    main()