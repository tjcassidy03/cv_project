@echo off
echo Running craft_run.py using .craft virtual environment...
call .craft\Scripts\python.exe craft_run.py

echo Running craft_ocr_run.py using .craft_ocr virtual environment...
call .craft_ocr\Scripts\python.exe craft_ocr_run.py

echo All tasks completed.
pause
