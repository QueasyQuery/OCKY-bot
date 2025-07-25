@echo off
setlocal

REM check if venv exists. create if not
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%

    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo Installing requirements...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt

    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
)

REM activate and run
call venv\Scripts\activate.bat
python source\main.py
pause