@echo off
chcp 65001 >nul
echo 開始執行 momo 爬蟲程式...
echo 關鍵字: %1
echo 執行時間: %date% %time%

cd /d "C:\YvesProject\中央\線上評論\momo_crawler-main"

python crawler.py --keyword %1

echo 爬蟲程式執行完成
echo 完成時間: %date% %time%
pause 