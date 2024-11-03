from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
import json
import hashlib
from chess_engine.src.model.config.config import settings
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import Bitboard_Creator

Base = declarative_base()

bc = Bitboard_Creator()
bitboards_dict = bc.get_all_bitboards()


def get_hash(piece_positions,castling_rights,en_passant,turn):
        game_string = (
            piece_positions
            + castling_rights
            + en_passant
            + turn
            
        )
        hash_object = hashlib.sha256(game_string.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

def create_dynamic_model(class_name, attributes_dict,backref_name):
    # Define a dictionary to hold the columns
    columns = {
        '__tablename__': class_name.lower(),
    }

    columns['id'] = Column(Integer, ForeignKey('GamePositions.id'), primary_key=True)
    

    for key in attributes_dict.keys():
        columns[key] = Column(Integer)
        
    columns['game_position'] = relationship("GamePositions", backref=backref_name)

    return type(class_name, (GamePositions,), columns)

class GamePositions(Base):
    __tablename__ = "GamePositions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    piece_positions = Column(String, index=True)
    castling_rights = Column(String, index=True)
    en_passant = Column(String, index=True)
    turn = Column(String, index=True)

    white_wins = Column(Integer)
    black_wins = Column(Integer)
    stalemates = Column(Integer)

    fen = f"{piece_positions} {turn} {castling_rights} {en_passant} {0} {1}"

    

    @property
    def get_hash(self):
        game_string = (
            self.piece_positions
            + self.castling_rights
            + self.en_passant
            + self.turn
            
        )
        hash_object = hashlib.sha256(game_string.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    @property
    def total_wins(self):
        total_wins = self.white_wins + self.black_wins + self.stalemates
        return total_wins

    @property
    def win_buckets(self):
        total_wins = self.total_wins
        if total_wins > 0:
            mean_w = self.white_wins / total_wins
            mean_b = self.black_wins / total_wins
            mean_s = self.stalemates / total_wins
            return [mean_w, mean_b, mean_s]
        else:
            return [0, 0, 0]

    @staticmethod
    def from_json(json_str, win_buckets):
        data = json.loads(json_str)
        win_buckets = json.loads(win_buckets)
        return GamePositions(
            piece_positions=data["piece_positions"],
            castling_rights=data["castling_rights"],
            en_passant=data["en_passant"],
            turn=data["turn"],
            white_wins=win_buckets["white_wins"],
            black_wins=win_buckets["black_wins"],
            stalemates=win_buckets["stalemates"]
        )
    
    
GamePositionRollup = create_dynamic_model("GamePositionRollup",bitboards_dict,"rollup_position")
TrainGamePositions = create_dynamic_model("TrainGamePositions",bitboards_dict,"train_position")
TestGamePositions = create_dynamic_model("TestGamePositions",bitboards_dict,"test_position")
ValidationGamePositions = create_dynamic_model("ValidationGamePositions",bitboards_dict,"validation_position")




