sudo apt install tmux neovim python3.11-venv git -y
python3 -m venv ./env
source ./env/bin/activate
export trainDataExists=False
export useSamplePgn=False
export trainModel=True
git clone https://github.com/EthanTCruz/Chess_Model.git
cd Chess_Model
git checkout cnn
pip install -r tf15_requirements.txt
cd ../
tmux new -s train
