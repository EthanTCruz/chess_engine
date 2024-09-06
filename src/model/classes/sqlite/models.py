from pydantic import BaseModel
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import declarative_base
import json

Base = declarative_base()

class GamePositions(Base):
    __tablename__="GamePositions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    piece_positions = Column(String, index=True)
    castling_rights = Column(String, index=True)
    en_passant = Column(String, index=True)
    turn = Column(String, index=True)
    #0 for false, 1 for true
    greater_than_n_half_moves = Column(Integer, index=True)


    white_wins = Column(Integer)
    black_wins = Column(Integer)
    stalemates = Column(Integer)

    @property
    def total_wins(self):
        total_wins = self.white_wins + self.black_wins + self.stalemates
        return total_wins
    
    @property
    def win_buckets(self):
        total_wins = self.total_wins
        if total_wins > 0:
            mean_w = self.white_wins/total_wins
            mean_b = self.black_wins/total_wins
            mean_s = self.stalemates/total_wins
            return [mean_w,mean_b,mean_s]
        else: 
            return [0,0,0]
        


    @staticmethod
    def from_json(json_str,win_buckets):
        data = json.loads(json_str)
        win_buckets = json.loads(win_buckets)
        return GamePositions(

            piece_positions=data["piece_positions"],
            castling_rights=data["castling_rights"],
            en_passant=data["en_passant"],
            turn=data["turn"],
            greater_than_n_half_moves=data["greater_than_n_half_moves"],
            white_wins=win_buckets["white_wins"],
            black_wins=win_buckets["black_wins"],
            stalemates=win_buckets["stalemates"]
        )
class GamePositionRollup(Base):
    __tablename__ = "GamePositionRollup"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    piece_positions = Column(String, index=True)
    castling_rights = Column(String, index=True)
    en_passant = Column(String, index=True)
    turn = Column(String, index=True)
    greater_than_n_half_moves = Column(Integer, index=True)
    white_wins = Column(Integer)
    black_wins = Column(Integer)
    stalemates = Column(Integer)

    @property
    def total_wins(self):
        total_wins = self.white_wins + self.black_wins + self.stalemates
        return total_wins
    
    @property
    def win_buckets(self):
        total_wins = self.total_wins
        if total_wins > 0:
            mean_w = self.white_wins/total_wins
            mean_b = self.black_wins/total_wins
            mean_s = self.stalemates/total_wins
            return [mean_w,mean_b,mean_s]
        else: 
            return [0,0,0]
        
