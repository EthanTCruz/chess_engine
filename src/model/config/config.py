from pydantic_settings import BaseSettings
from dotenv import load_dotenv





load_dotenv()




class Settings(BaseSettings): 

    

    DATABASE_URL: str = "sqlite:///src/model/data/gameData.db"

    SRC_MODEL_DIR: str = './src/model'

        
    USE_SAMPLE_DATASET: bool = False
    USE_AUTOENCODER_DATASET: bool = False
    USE_FULL_DATASET: bool = False
    
    SAMPLE_PGN_DIR_NAME: str = "sample_dataset"
    AUTOENCODER_PGN_DIR_NAME: str = "autoencoder_dataset"
    FULL_PGN_DIR_NAME: str = "full_dataset"

    PGN_DIR_NAME: str = "sample_dataset"


    PGN_DIR: str = f"{SRC_MODEL_DIR}/pgn/"

    PGN_DATASET: str = f"{PGN_DIR}{PGN_DIR_NAME}/"
    


    SCORE_DEPTH: int = 1
    PLAYER: str = 'w'
    ENDGAME_TABLE: str = f"{SRC_MODEL_DIR}/data/EndgameTbl/"
    MINIMUM_ENDGAME_PIECES: int = 5

    UCB_CONSTANT: float = 0.1

    SAVE_TO_BUCKET: bool = False
    GOOGLE_APPLICATION_CREDENTIALS: str = "C:\\Users\\ethan\\git\\Full_Chess_App\\chess_engine\\terraform\\secret.json"
    BUCKET_NAME: str = "chess-model-weights"


    





    EXTRACT_DATA: bool = False
    PREPROCESS_DATA: bool = False
    PROCESS_DATA: bool = False
    TRAIN_ENCODER: bool = True
    TRAIN_MODEL: bool = False
    
    

    class Config:
        env_prefix = 'MAIN_'



def init_settings():
    settings = Settings()
    if settings.USE_SAMPLE_DATASET + settings.USE_AUTOENCODER_DATASET + settings.USE_FULL_DATASET > 1:
        raise ValueError("Only one dataset can be used at a time")
    elif settings.USE_SAMPLE_DATASET + settings.USE_AUTOENCODER_DATASET + settings.USE_FULL_DATASET == 0:
            ValueError("No dataset selected")
    if settings.USE_SAMPLE_DATASET:
        settings.PGN_DIR_NAME = settings.SAMPLE_PGN_DIR_NAME
    elif settings.USE_AUTOENCODER_DATASET:
        settings.PGN_DIR_NAME = settings.AUTOENCODER_PGN_DIR_NAME
    elif settings.USE_FULL_DATASET:
        settings.PGN_DIR_NAME = settings.FULL_PGN_DIR_NAME

    settings.PGN_DIR = f"{settings.SRC_MODEL_DIR}/pgn/"

    settings.PGN_DATASET = f"{settings.PGN_DIR}{settings.PGN_DIR_NAME}/"

    return settings

settings = init_settings()

        
class DataLoaderSettings(BaseSettings):
    MAX_CACHE_SIZE: int = 5
    BATCH_SIZE: int = 5096
    BATCH_FILE_SIZE: int = 10000000
    H5PY_CHUNK_SIZE: int = 1
    DATA_DIR: str = f'{settings.SRC_MODEL_DIR}/data/{settings.PGN_DIR_NAME}/'
    

        
    TRAINING_DIR: str = f"{DATA_DIR}training"
    TESTING_DIR: str = f"{DATA_DIR}testing"
    VALIDATION_DIR: str = f"{DATA_DIR}validation"
    class Config:
        env_prefix = 'DATALOADER_'
        
data_settings = DataLoaderSettings()


class ModelSettings(BaseSettings):
    
    DIR: str =f"{settings.SRC_MODEL_DIR}/chess_model/"
    MODEL_FILENAME: str = "torch_model.pth"
    FULL_MODEL_PATH: str = f"{DIR}{MODEL_FILENAME}"
    SELF_PLAY_MODEL_FILENAME: str ="self_play_model"
    CHECKPOINT_DIR: str = f"{DIR}checkpoints/"
    DATA: str = f"{settings.SRC_MODEL_DIR}/data"
    


    EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    TEST_SET_SIZE: float = 0.02
    VALIDATION_SET_SIZE: float  = 0.02
    TRAIN_SET_SIZE: float = 1.0 - TEST_SET_SIZE - VALIDATION_SET_SIZE

    DATALOADER_BATCH_SIZE: int = data_settings.BATCH_SIZE
    # Will only work as 0 while on windows
    NUM_WORKERS: int = 0

    
    MODEL_TYPE: str = "ChessEvalCNN"
    # SkipChessEvalCNN
    # ChessEvalDeepCNN
    # ChessEvalResNet
    class Config:
        env_prefix = 'MODEL_'

model_settings = ModelSettings()

class AutoEncoderSettings(BaseSettings):
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 17
    DATALOADER_BATCH_SIZE: int = model_settings.DATALOADER_BATCH_SIZE
    NUM_WORKERS: int = 0
    LATENT_DIMS: list = [700,600,400,300,200,100]
    
    MODEL_FILE_DIR: str = f"{model_settings.DIR}autoencoder/"
    PERSIST_WORKERS: bool = False
    class Config:
        env_prefix = 'AE_MODEL_'

ae_settings = AutoEncoderSettings()

class DeepChessModelSettings(ModelSettings):
    input_dim: int = 836
    latent_dim: int = 128
    class Config:
        env_prefix = 'DEEP_MODEL_'
