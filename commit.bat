@echo off
set /p commitMessage= enter the commit message:
echo %commitMessage% > commitMessage.txt
git add *
git commit -F commitMessage.txt
git push
