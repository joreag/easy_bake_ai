WELCOME TO EASY BAKE AI (HCTS FORGE V1)
=======================================

This tool allows you to train your own HCTS-Architecture AI models directly from your desktop.
Our architecture is based on the HCTS-Transformer v1 (Hierarchical Contextual Transformation System).

PREREQUISITES:
--------------
1. You must have Python installed (Version 3.10 or 3.11 recommended).
   - Windows: Download from https://www.python.org/downloads/
   - IMPORTANT: When installing, check the box "Add Python to PATH".

-------------------------------------------------------------------------------
PART 1: THE FORGE (Web Interface)
-------------------------------------------------------------------------------
This is the main interface for building and training your AI.

Windows Users:
1. Double-click "start_windows.bat".
2. Wait for the setup to finish (the first time takes a few minutes).
3. Once the dashboard is running, open your browser to http://127.0.0.1:5555

Mac / Linux Users:
1. Open your terminal in this folder.
2. Run: chmod +x start_mac_linux.sh
3. Run: ./start_mac_linux.sh

HOW TO USE:
1. Place your training data (.jsonl files) inside the "curriculum" folder.  We have provided a mostly clean .jsonl file based on the SOTA BoolQ, PIQA, ARC-Easy and GSM8K for demonstration purposes.  SWE Bench Lite available on request.
2. Go to the web dashboard.
3. Configure your settings (Epochs, Learning Rate, etc).
4. Click "Forge AI". (Note: Using the Included SOTA Dataset will use 6+GB VRAM or RAM, please make sure you have double that available for either your video card or your system RAM for regular system operations.)

Your trained models will appear in the "builds" folder.

-------------------------------------------------------------------------------
PART 2: CHOOSING YOUR ARCHITECTURE
-------------------------------------------------------------------------------
The Forge allows you to toggle between two distinct brain structures:

1. STANDARD HCTS (v1)
   The foundational Hierarchical Cognitive Transport System.
   - Structure: A single unified stack with specialized attention heads.
   - Best For: General purpose learning, robust pattern recognition.
   - Status: Stable & Reliable.

2. PASCAL-GUIDED TRANSFORMER ("Model Z")
   An experimental architecture derived from biological cognition theories.
   - Structure: Distinct Transformer Stacks (Syntax -> Semantics -> Reasoning) 
     physically separated by learnable "Twist Matrices".
   - Mechanism: The "Twist Matrix" acts as a translator, rotating the vector 
     space as data moves from raw language processing to abstract reasoning.
   - Best For: Complex logical deduction and multi-step problem solving.
   - Status: Experimental / Research Grade.

-------------------------------------------------------------------------------
PART 3: THE ADVANCED TOOLBOX (CLI Only)
-------------------------------------------------------------------------------
For power users, we have included a suite of command-line tools to analyze,
diagnose, and chat with your models.

To use these, you must open a terminal/command prompt in this folder and 
activate the environment first:

   Windows:   .venv\Scripts\activate
   Mac/Linux: source .venv/bin/activate

1. THE CHAT INTERFACE
   Talk to your specific model build.
   Command: python toolbox/chat_cli.py --build my_first_forge
   (Or just double-click "chat_with_ai.bat" on Windows)

2. THE BRAIN SCANNER (Latent Space Mapper)
   Run an "fMRI" on your model to see how it associates concepts.
   Command: python toolbox/brain_scan.py --build my_first_forge

3. THE DATA REFORMATTER
   Clean raw SOTA datasets (BoolQ, PIQA, GSM8K) into Cognitive Nodes.
   Command: python curriculum_reformatter.py --input-file raw_data.jsonl --output-file curriculum/clean.jsonl

4. THE DOJO (Active Reinforcement Laboratory)
   A "Glass Box" training tool that quizzes your model to ensure mastery.
   - Mechanism: The Dojo runs through a dataset like a deck of flashcards. 
     If the AI answers incorrectly, the Dojo pauses and performs a "Cognitive Jolt" 
     (immediate remedial training) on that specific fact until the AI gets it right.
   - Best For: Fixing hallucinations, reinforcing weak concepts, or fine-tuning 
     on difficult data without re-running the entire build process.
   
   Standard Command:
   python toolbox/dojo.py --build my_first_forge

   Advanced Command (Target specific weak points):
   python toolbox/dojo.py --build my_first_forge --dataset curriculum/hard_questions.jsonl   

5. THE INTERACTIVE DOJO (Interactive Reinforcement)
   A real-time "Teaching Mode" where you can chat with your AI and correct 
   it on the fly. This uses "One-Shot Reinforcement" to update the model weights
   instantly based on your feedback.

   How it works:
   1. Chat normally. If the AI hallucinates or answers incorrectly...
   2. Type 'teach'.
   3. Enter the correct answer.
   4. Watch the AI re-train itself on that specific fact in seconds.

   Command: python toolbox/interactive_dojo_v1.py --build my_first_forge

   Use Case: "Potty Training" your AI, fixing logic errors, or teaching it
   facts about yourself that weren't in the training data.   

-------------------------------------------------------------------------------
ABOUT THE ARCHITECTURE
-------------------------------------------------------------------------------
The HCTS-Transformer v1 is designed for structured hierarchical learning.
Unlike standard Transformers, it emphasizes structural cognition over 
pure statistical likelihood.

Created by www.Ulshe.AI 
Contact: joreag@ulshe.AI