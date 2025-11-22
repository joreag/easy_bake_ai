@echo off
TITLE Easy Bake AI - Chat Interface
call .venv\Scripts\activate.bat
python toolbox/chat_cli.py --build my_first_forge
pause