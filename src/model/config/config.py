from pydantic_settings import BaseSettings
from math import sqrt
class Settings(BaseSettings): 

    BatchSize: int = 1024
    
    nnGenBatchSize: int = 1
    
    nnBatchSize: int = 100

    nnScalarBatchSize: int = 10
    
    nnEpochs: int = 100
    nnTestSize: float = 0.02
    nnValidationSize: float  = 0.02

    srcModelDirectory: str = './src/model'
    data_dir: str = f"{srcModelDirectory}/data"

    ModelFilePath: str =f"{srcModelDirectory}/chess_model/"
    ModelFilename: str = "model.h5"
    pgn_file: str = f"{srcModelDirectory}/pgn/full_dataset/"

    samplePgn: str = f"{srcModelDirectory}/pgn/sample_dataset/"

    SelfPlayModelFilename: str ="self_play_model"
    

    nnLogDir: str = "./chess_engine/logs/"
    nnModelCheckpoint: str = f"{ModelFilePath}checkpoints/"

    #should run under assumption score depth will always be greater than mate depth
    score_depth: int = 1
    player: str = 'w'
    endgame_table: str = f"{srcModelDirectory}/data/EndgameTbl/"
    minimumEndgamePieces: int = 5


    #MCST parameters:
    UCB_Constant: float = 0.1



    GOOGLE_APPLICATION_CREDENTIALS: str = "C:\\Users\\ethan\\git\\Full_Chess_App\\Chess_Model\\terraform\\secret.json"
    BUCKET_NAME: str = "chess-model-weights"
    matrixScalerFile: str = f"{srcModelDirectory}/data/matrixScaler.joblib"

    # Mongo settings

    validation_collection_key: str = "validation_data"
    testing_collection_key: str = "testing_data"
    training_collection_key: str = "training_data"
    main_collection_key: str = "main_collection"

    db_name: str = "mydatabase"
    metadata_key: str = 'metadata'
    bitboards_key: str = 'positions_data'
    results_key: str = 'game_results'

    np_means_file: str = f"{data_dir}/means.npy"
    np_stds_file: str = f"{data_dir}/stds.npy"

    torch_model_file: str = f"{srcModelDirectory}/chess_model/torch_model.pth"

    mongo_host: str = "localhost"
    mongo_port: int = 27017

    mongo_url: str = f"mongodb://{mongo_host}:{mongo_port}/"

    redis_host: str = "192.168.68.50"
    redis_port: int = 6379
    redis_db: int = 1
    
    num_workers: int = 0

    halfMoveBin: int = 25

    trainModel: bool = False
    selfTrain: bool = False
    trainDataExists: bool = True
    useSamplePgn: bool = False
    saveToBucket: bool = False
    tuneParameters: bool = False

    
    
    class Config:
        env_prefix = ''

