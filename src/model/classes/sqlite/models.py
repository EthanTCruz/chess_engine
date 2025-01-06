from sqlalchemy import Column, String, Integer, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
import hashlib
import json
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict

Base = declarative_base()


class WinBucketsMixin:
    white_wins = Column(Integer, default=0)
    stalemates = Column(Integer, default=0)
    black_wins = Column(Integer, default=0)

    @property
    def total_wins(self):
        return self.white_wins + self.black_wins + self.stalemates

    @property
    def win_buckets(self):
        total_wins = self.total_wins
        if total_wins > 0:
            mean_w = self.white_wins / total_wins
            mean_s = self.stalemates / total_wins
            mean_b = self.black_wins / total_wins
            return [mean_w,  mean_s, mean_b]
        else:
            return [0, 0, 0]
        
class GamePositions(Base,WinBucketsMixin):
    __tablename__ = "GamePositions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    piece_positions = Column(String, index=True)
    castling_rights = Column(String, index=True)
    en_passant = Column(String, index=True)
    turn = Column(String, index=True)
    white_wins = Column(Integer)
    black_wins = Column(Integer)
    stalemates = Column(Integer)

    @property
    def get_hash(self):
        game_string = (
            self.piece_positions
            + self.castling_rights
            + self.en_passant
            + self.turn
        )
        hash_object = hashlib.sha256(game_string.encode())
        return hash_object.hexdigest()

# Define a function to create standalone models without inheritance
def create_standalone_model(class_name, attributes_dict):
    columns = {
        '__tablename__': class_name.lower(),
        'id': Column(Integer, primary_key=True, autoincrement=True),
        'fen': Column(String, index=True),
        'piece_positions': Column(String, index=True),
        'castling_rights': Column(String, index=True),
        'en_passant': Column(String, index=True),
        'turn': Column(String, index=True),
        'white_wins': Column(Integer),
        'black_wins': Column(Integer),
        'stalemates': Column(Integer),
        'is_training_data': Column(Boolean),
        'is_testing_data': Column(Boolean),
        'is_validation_data': Column(Boolean),
        
    }

    # Add any additional columns from `attributes_dict`
    for key in attributes_dict.keys():
        columns[key] = Column(String)

    return type(class_name, (Base,WinBucketsMixin), columns)

# Define dynamic models as standalone tables
GamePositionRollup = create_standalone_model("GamePositionRollup", sample_bitboard_dict)

