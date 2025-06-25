@echo off
chcp 65001 >nul
cd /d "C:\YvesProject\中央\線上評論\momo_crawler-main"

REM 這裡列出你要爬的所有關鍵字
set keywords=口罩 益生菌 葉黃素 維他命 膠原蛋白 雞精 寵物

for %%k in (%keywords%) do (
    echo ==========================
    echo 正在爬取關鍵字：%%k
    python crawler.py --keyword %%k
    echo 關鍵字 %%k 完成
    echo.
    timeout /t 5 /nobreak >nul
)

echo 所有關鍵字都已完成
pause