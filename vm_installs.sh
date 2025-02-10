git clone https://github.com/EthanTCruz/chess_engine.git
cd chess_engine
sudo apt install tmux neovim sqlite3 python3.11-venv git -y
python3 -m venv ./env
source ./env/bin/activate

pip install -r pi_requirements.txt


tmux new -s train

# mkdir src/model/data/sample/
# mkdir src/model/data/sample/testing
# mkdir src/model/data/sample/training
# mkdir src/model/data/sample/validation
# mkdir src/model/chess_model/autoencoder
cp src/model/data/sample/ src/model/data/autoencoder_dataset/ -r
sqlite3 src/model/data/sampleGameData.db
