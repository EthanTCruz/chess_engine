sudo apt install tmux neovim  git -y
export trainDataExists=False
export useSamplePgn=False
export trainModel=True
git clone https://github.com/EthanTCruz/Chess_Model.git
cd Chess_Model
git checkout cnn
pip install -r requirements.txt
cd ../
tmux new -s train
