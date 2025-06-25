@echo off
chcp 65001 >nul
echo 開始執行 momo 爬蟲程式 - 多關鍵字版本...
echo 執行時間: %date% %time%

cd /d "C:\YvesProject\中央\線上評論\momo_crawler-main"

REM 定義關鍵字陣列（用空格分隔）
set keywords=益生菌 口罩 維他命

for %%k in (%keywords%) do (
    echo.
    echo ========================================
    echo 正在處理關鍵字: %%k
    echo ========================================
    python crawler.py --keyword %%k
    echo 關鍵字 %%k 處理完成
    timeout /t 30 /nobreak >nul
)

echo.
echo 所有關鍵字處理完成
echo 完成時間: %date% %time%
pause 