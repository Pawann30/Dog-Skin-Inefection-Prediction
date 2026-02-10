@echo off
echo ========================================================
echo   Pushing Dog Skin Disease Detection Project to GitHub
echo ========================================================

set GIT_CMD=git

:: Check if system git is installed
where git >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] System Git found.
) else (
    echo [INFO] System Git not found. Checking for Portable Git...
    if exist "git-portable\cmd\git.exe" (
        set GIT_CMD="git-portable\cmd\git.exe"
        echo [INFO] Portable Git found at git-portable\cmd\git.exe
    ) else (
        echo [ERROR] Git is not installed and Portable Git is missing.
        echo Please install Git from https://git-scm.com/download/win
        pause
        exit /b
    )
)

:: Initialize repo if not exists
if not exist .git (
    echo [INFO] Initializing Git repository...
    %GIT_CMD% init
    %GIT_CMD% branch -M main
)

:: Configure user if not set
%GIT_CMD% config user.email >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Configuring default user identity...
    %GIT_CMD% config user.email "user@example.com"
    %GIT_CMD% config user.name "PawScan AI User"
)

:: Add files
echo [INFO] Adding files...
%GIT_CMD% add .

:: Commit
echo [INFO] Committing files...
%GIT_CMD% commit -m "Initial commit for Dog Skin Disease Detection App"

:: Add remote
echo [INFO] Adding remote repository...
%GIT_CMD% remote remove origin >nul 2>nul
%GIT_CMD% remote add origin https://github.com/Pawann30/Dog-Skin-Inefection-Prediction.git

:: Push
echo [INFO] Pushing to GitHub...
echo.
echo NOTE: A browser window may open asking you to sign in to GitHub.
echo.
%GIT_CMD% push -u origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Project pushed successfully! 🚀
    goto :end
)

echo.
echo [WARNING] The push failed. This usually happens if the remote repository is not empty.
echo.
set /p FORCE="Do you want to FORCE overwrite the remote repository? (Y/N): "
if /i "%FORCE%"=="Y" (
    echo.
    echo [INFO] Force pushing...
    %GIT_CMD% push -f origin main
    if %ERRORLEVEL% equ 0 (
        echo.
        echo [SUCCESS] Project force pushed successfully! 🚀
    ) else (
        echo.
        echo [ERROR] Force push failed. Check your connection or permissions.
    )
) else (
    echo.
    echo [INFO] cancelled.
)

:end
pause
