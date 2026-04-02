# Sam3

## install
conda create -n sam3 python=3.12
source activate sam3
pip install -r requirements_sam3_http.txt

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -e .
pip install -e ".[notebooks]"
python -m pip install --force-reinstall "setuptools<82"
pip install pandas

## usage
(sam3)>python sam3_http_server.py &

# Trainer

## install
conda create -n trainer python==3.10
source trainer flymyai
pip install -r requirements.txt
pip install -r requirements.txt
pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers
accelerate launch train_qwen_edit_lora_v402.py --config cfg.yaml

# Note
因为这俩个环境不一样所以需要分开跑