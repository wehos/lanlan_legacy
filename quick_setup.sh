pip3 install torch torchvision torchaudio torchtext
cd /workspace
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y --no-install-recommends git-lfs ffmpeg libsox-dev gcc g++ cmake parallel aria2 vim #tzdata
pip install packaging ninja 
pip install flash-attn --no-build-isolation
pip install transformers datasets[audio] accelerate pickleshare ffmpeg ffmpeg-python gradio optimum  openai langchain langchain-openai 
pip install -U rembg 
git lfs install
cd /workspace/GPT-SoVITS/
pip install -r requirements.txt
cd /workspace/Whisper-Finetune
pip install -r requirements.txt
pip install -U gradio==4.17
cd /workspace/pycord
python3 -m pip install -U .[voice,speed]
cd /workspace
#apt install ./tensorrt.deb
#cp /var/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0/nv-tensorrt-local-9A1EDFBA-keyring.gpg /usr/share/keyrings/
#pip install tensorrt torch-tensorrt
#pip install onnx onnxconverter-common
#pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
# Optional touch ~/.no_auto_tmux
# Launch whisper server with --port
# Edit port variables and launch dcbot / chatbot
