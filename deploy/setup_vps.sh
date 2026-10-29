#!/usr/bin/env bash
set -e

echo "=== Memulai Setup SORTARA di VPS Ubuntu/Debian ==="

# 1. Update system & install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git nginx certbot python3-certbot-nginx

# 2. Setup Swap memory 2GB (mencegah OOM saat inferensi TensorFlow)
if [ ! -f /swapfile ]; then
    echo "Membuat Swap File 2GB..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Membuat virtual environment..."
    python3 -m venv .venv
fi

echo "Menginstall dependencies python..."
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# 4. Setup Systemd Service
echo "Memasang systemd service..."
sudo cp deploy/sorta.service /etc/systemd/system/sorta.service
sudo systemctl daemon-reload
sudo systemctl enable sorta
sudo systemctl restart sorta

echo "=== Setup Selesai! Service SORTARA aktif. ==="
echo "Status service:"
sudo systemctl status sorta --no-pager
