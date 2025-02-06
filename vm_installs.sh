sudo apt install tmux neovim sqlite3 python3.11-venv git -y
python3 -m venv ./env
source ./env/bin/activate
git clone https://github.com/EthanTCruz/chess_engine.git
cd chess_engine
pip install -r pi_requirements.txt

export MAIN_DATABASE_URL="sqlite:///src/model/data/gameData.db"
export MODEL_NUM_WORKERS="2"
export AE_MODEL_NUM_EPOCHS="64"
export MODEL_EPOCHS="64"
export MODEL_DATALOADER_BATCH_SIZE="256"
export MAIN_USE_SAMPLE_PGN="True"
export MAIN_GET_DATA="False"
export MAIN_PREPROCESS_DATA="False"
export MAIN_PROCESS_DATA="False"
export MAIN_TRAIN_MODEL="False"
export MAIN_TRAIN_ENCODER="True"

tmux new -s train

mkdir src/model/data/sample/
mkdir src/model/data/sample/testing
mkdir src/model/data/sample/training
mkdir src/model/data/sample/validation
mkdir src/model/chess_model/autoencoder

sqlite3 src/model/data/sampleGameData.db
