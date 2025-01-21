sudo apt install tmux neovim sqlite3 python3.11-venv git -y
python3 -m venv ./env
source ./env/bin/activate
export trainDataExists=False
export useSamplePgn=False
export trainModel=True
git clone https://github.com/EthanTCruz/chess_engine.git
cd chess_engine
pip install -r pi_requirements.txt
tmux new -s train
sqlite3 src/model/data/sampleGameData.db
mkdir src/model/data/testing
mkdir src/model/data/training
mkdir src/model/data/validation