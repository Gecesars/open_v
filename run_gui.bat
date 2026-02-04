@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\\Scripts\\python.exe" (
  echo Ambiente virtual nao encontrado. Execute a instalacao primeiro.
  pause
  exit /b 1
)

echo Iniciando GUI...
".venv\\Scripts\\python.exe" gui.py
echo.
echo Se a janela nao abriu, veja a mensagem acima.
pause
