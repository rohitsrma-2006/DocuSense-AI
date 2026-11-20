@echo off
title DocuSense AI - Content Detection & PPTX Analysis System
echo =====================================================================
echo  DocuSense AI - Content Detection & Slide Quality Analysis System
echo  Industrial Training Project - Novem Controls Pvt. Ltd.
echo  Student: Rohit Sharma (Roll: 23010603054)
echo =====================================================================
echo.
echo [*] Starting DocuSense AI Server...
echo.
echo Credentials for Login:
echo   Username: @Rohit
echo   Password: rohit123
echo.
echo [*] Opening your browser at http://127.0.0.1:5000/login ...
start "" "http://127.0.0.1:5000/login" 2>nul || explorer "http://127.0.0.1:5000/login"
echo.
python app.py
pause
