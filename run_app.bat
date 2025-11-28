@echo off
echo Installing required packages...
python -m pip install streamlit pandas plotly numpy -q
if errorlevel 1 (
    echo Error: Failed to install packages. Make sure Python is installed and in PATH.
    pause
    exit /b 1
)
echo.
echo Starting Streamlit app...
python -m streamlit run app.py
if errorlevel 1 (
    echo Error: Failed to start Streamlit app.
    pause
    exit /b 1
)

