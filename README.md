```bash
conda create -n cs776 python=3.10 numpy scipy -y
conda activate cs776

conda install pytorch torchvision torchaudio cpuonly nomkl -c pytorch -y

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

conda install scikit-learn -y