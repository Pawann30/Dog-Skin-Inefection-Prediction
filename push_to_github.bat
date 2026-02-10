@echo off
echo ========================================================
echo   Pushing Dog Skin Disease Detection Project to GitHub
echo ========================================================

:: Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in your PATH.
    echo Please install Git from https://git-scm.com/download/win
    echo After installing, restart your terminal and run this script again.
    pause
    exit /b
)

:: Initialize repo if not exists
if not exist .git (
    echo [INFO] Initializing Git repository...
    git init
    git branch -M main
)

:: Add files
echo [INFO] Adding files...
git add .

:: Commit
echo [INFO] Committing files...
git commit -m "Initial commit for Dog Skin Disease Detection App"

:: Add remote
echo [INFO] Adding remote repository...
git remote remove origin >nul 2>nul
git remote add origin https://github.com/Pawann30/Dog-Skin-Inefection-Prediction.git

:: Push
echo [INFO] Pushing to GitHub...
echo.
echo NOTE: A browser window may open asking you to sign in to GitHub.
echo.
git push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Project pushed successfully! 🚀
) else (
    echo.
    echo [ERROR] Failed to push. Please check your internet connection or GitHub credentials.
)

pause
