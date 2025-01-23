from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()




class Settings(BaseSettings): 

    BatchSize: int = 521

    database_url: str = "sqlite:///src/model/data/gameData.db"

    srcModelDirectory: str = './src/model'
    pgn_file: str = f"{srcModelDirectory}/pgn/full_dataset/"
    samplePgn: str = f"{srcModelDirectory}/pgn/sample_dataset/"
    nnLogDir: str = "./chess_engine/logs/"

    score_depth: int = 1
    player: str = 'w'
    endgame_table: str = f"{srcModelDirectory}/data/EndgameTbl/"
    minimumEndgamePieces: int = 5

    UCB_Constant: float = 0.1

    GOOGLE_APPLICATION_CREDENTIALS: str = "C:\\Users\\ethan\\git\\Full_Chess_App\\chess_engine\\terraform\\secret.json"
    BUCKET_NAME: str = "chess-model-weights"


    trainModel: bool = False
    selfTrain: bool = False
    trainDataExists: bool = True
    
    saveToBucket: bool = False
    tuneParameters: bool = False
    
    useSamplePgn: bool = False
    getData: bool = False
    preprocessData: bool = False
    processData: bool = True
    trainModel: bool = True
    
    

    class Config:
        env_prefix = ''
        env_file = '.conf-env'

settings = Settings()



class ModelSettings(BaseSettings):
    ModelFilePath: str =f"{settings.srcModelDirectory}/chess_model/"
    ModelFilename: str = "model.h5"
    SelfPlayModelFilename: str ="self_play_model"
    nnModelCheckpoint: str = f"{ModelFilePath}checkpoints/"
    data_dir: str = f"{settings.srcModelDirectory}/data"
    
    
    Epochs: int = 2
    learning_rate: float = 0.001
    TestSize: float = 0.02
    ValidationSize: float  = 0.02
    TrainSize: float = 1.0 - TestSize - ValidationSize

    DataLoaderBatchSize: int = 512
    # Will only work as 0 while on windows
    num_workers: int = 0
    torch_model_file: str = f"{settings.srcModelDirectory}/chess_model/torch_model.pth"
    
    modelType: str = "ChessEvalCNN"
    # SkipChessEvalCNN
    # ChessEvalDeepCNN
    # ChessEvalResNet

model_settings = ModelSettings()

class DataLoaderSettings(BaseSettings):
    MaxCacheSize: int = 5
    BatchSize: int = 5096
    BatchFileSize: int = 10000000
    ChunkSize: int = 1

    DataDirectory: str = './src/model/data/'
    TrainingDirectory: str = f"{DataDirectory}training"
    TestingDirectory: str = f"{DataDirectory}testing"
    ValidationDirectory: str = f"{DataDirectory}validation"

data_settings = DataLoaderSettings()