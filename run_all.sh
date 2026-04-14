#!/bin/bash
# 依次跑两次训练：第一次无 IoU reward，第二次加 IoU reward
# 前台运行，进度条可见
set -e

cd /home/work/tmp/trainer

# ========== 0. 确保 SAM3 服务器在 GPU 7 运行（只有 with_iou 需要） ==========
if ! curl -sf http://127.0.0.1:5000/health > /dev/null 2>&1; then
    echo "[SAM3] Starting server on GPU 7..."
    env CUDA_VISIBLE_DEVICES=7 nohup conda run -n sam3 \
        env CUDA_VISIBLE_DEVICES=7 SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
        python sam3_http_server.py --port 5000 \
        > sam3_server.log 2>&1 &
    echo "[SAM3] PID: $!"
    for i in $(seq 1 30); do
        sleep 5
        if curl -sf http://127.0.0.1:5000/health > /dev/null 2>&1; then
            echo "[SAM3] Ready after ${i}x5s"
            break
        fi
        [ $i -eq 30 ] && { echo "[SAM3] FAILED, see sam3_server.log"; exit 1; }
    done
else
    echo "[SAM3] Already running."
fi

# ========== 1. 无 IoU reward 训练（7 卡 GPU 0-6）==========
echo ""
echo "=========================================="
echo "[1/2] Training WITHOUT IoU reward on GPU 0-6"
echo "=========================================="
rm -f loss_no_iou.csv
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PATH=/home/work/conda/envs/trainer/bin:$PATH \
    /home/work/conda/envs/trainer/bin/accelerate launch \
        --config_file accelerate_config.yaml \
        train_qwen_edit_lora_v402.py \
        --config cfg_no_iou.yaml

# ========== 2. 加 IoU reward 训练（7 卡 GPU 0-6）==========
echo ""
echo "=========================================="
echo "[2/2] Training WITH IoU reward on GPU 0-6"
echo "=========================================="
rm -f loss_with_iou.csv
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PATH=/home/work/conda/envs/trainer/bin:$PATH \
    /home/work/conda/envs/trainer/bin/accelerate launch \
        --config_file accelerate_config.yaml \
        train_qwen_edit_lora_v402.py \
        --config cfg_with_iou.yaml

# ========== 3. 画 loss 曲线 ==========
echo ""
echo "=========================================="
echo "[3/3] Plotting loss curves..."
echo "=========================================="
env PATH=/home/work/conda/envs/trainer/bin:$PATH \
    /home/work/conda/envs/trainer/bin/python plot_loss.py

echo ""
echo "Done. See: loss_curves.png"
