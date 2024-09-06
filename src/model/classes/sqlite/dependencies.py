
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.sqlite.models import GamePositions,GamePositionRollup
from chess_engine.src.model.config.config import Settings
from sqlalchemy.orm import Session
from typing import List, Tuple
from sqlalchemy import or_, and_, func
import chess
import re
import pandas as pd
import numpy as np
import json 

n_half_moves = Settings().halfMoveBin


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def delete_all_game_positions(db: Session = next(get_db())):
    try:

        # Delete all records in the GamePositions table
        db.query(GamePositions).delete()

        # Commit the changes to the database
        db.commit()

        print("All records in GamePositions have been deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()

def delete_all_rollup_game_positions(db: Session = next(get_db())):
    try:

        # Delete all records in the GamePositions table
        db.query(GamePositionRollup).delete()

        # Commit the changes to the database
        db.commit()

        print("All records in RollupGamePositions have been deleted.")

    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()



def insert_board_into_db(victor: str,board: chess.Board,db: Session = next(get_db())):
    game_info = board_to_GamePostition(board=board, victor=victor)
    gamesInDB = find_game(game_info=game_info, db=db)

    if gamesInDB is None:
        db.add(game_info)
    else:
        update_game(game_info=gamesInDB, new_game_info=game_info)

    db.commit()
    db.refresh(game_info if gamesInDB is None else gamesInDB)
    return 0

def find_game(game_info: GamePositions,db: Session = get_db()):
    results = db.query(GamePositions).filter(
            and_(GamePositions.piece_positions == game_info.piece_positions,
                GamePositions.castling_rights == game_info.castling_rights,
                GamePositions.en_passant == game_info.en_passant,
                GamePositions.turn == game_info.turn,
                GamePositions.greater_than_n_half_moves == game_info.greater_than_n_half_moves)
            ).first()
    return results

def find_board_rollup(board):
    game = board_to_GamePostitionRollup(board=board)
    results = find_game_rollup(game_info=game)
    return results

def find_game_rollup(game_info: GamePositionRollup,db: Session = get_db()):
    results = db.query(GamePositionRollup).filter(
            and_(GamePositionRollup.piece_positions == game_info.piece_positions,
                GamePositionRollup.castling_rights == game_info.castling_rights,
                GamePositionRollup.en_passant == game_info.en_passant,
                GamePositionRollup.turn == game_info.turn,
                GamePositionRollup.greater_than_n_half_moves == game_info.greater_than_n_half_moves)
            ).first()
    return results

def update_game(game_info: GamePositions,new_game_info: GamePositions):
    game_info.white_wins += new_game_info.white_wins
    game_info.black_wins += new_game_info.black_wins
    game_info.stalemates += new_game_info.stalemates
    return game_info

def get_game_info(game_info: GamePositions,db: Session = get_db()):
    game = find_game(game_info=game_info,db=db)
    white_wins = game.white_wins
    black_wins = game.black_wins
    stalemates = game.stalemates
    return white_wins,black_wins,stalemates

def remove_bracketed_portion(s):
    # This regex finds a portion enclosed in square brackets
    return re.sub(r'\[.*?\]', '', s)

def board_to_GamePostition(board: chess.Board,victor: str = "NA"):
    fen = board.fen()
    fen_components = fen.split(" ")
    piece_positions = remove_bracketed_portion(fen_components[0])
    turn = fen_components[1]
    castling_rights = fen_components[2]
    en_passant = fen_components[3]
    half_move_clock = int(fen_components[4])
    half_move_bin =  1 if half_move_clock >= n_half_moves else  0

    white_wins = 0
    black_wins = 0
    stalemates = 0
    if victor == 'w':
        white_wins = 1
    elif victor == 'b':
        black_wins = 1
    elif victor == 's':
        stalemates = 1

    game = GamePositions(
        piece_positions = piece_positions,
        castling_rights = castling_rights,
        en_passant = en_passant,
        turn = turn,
        greater_than_n_half_moves = half_move_bin,
        white_wins = white_wins,
        black_wins = black_wins,
        stalemates = stalemates
    )

    return game


def board_to_GamePostitionRollup(board: chess.Board):
    fen = board.fen()
    fen_components = fen.split(" ")
    piece_positions = remove_bracketed_portion(fen_components[0])
    turn = fen_components[1]
    castling_rights = fen_components[2]
    en_passant = fen_components[3]
    half_move_clock = int(fen_components[4])
    half_move_bin =  1 if half_move_clock >= n_half_moves else  0

    white_wins = 0
    black_wins = 0
    stalemates = 0


    game = GamePositionRollup(
        piece_positions = piece_positions,
        castling_rights = castling_rights,
        en_passant = en_passant,
        turn = turn,
        greater_than_n_half_moves = half_move_bin,
        white_wins = white_wins,
        black_wins = black_wins,
        stalemates = stalemates
    )

    return game


def find_rollup_move(board:chess.Board,
                      db: Session = next(get_db()),
                      scoring_weights = [1,1,0.5],
                      score_threshold = 1):
    boards = generate_all_possible_boards(board=board)
    gpr_boards = []
    for value in boards:
        board = value['board']
        gp = board_to_GamePostitionRollup(board=board)
        value['gp'] = gp
        gpr_boards.append(gp)
    
    conditions = []
    for gp in gpr_boards:
        
        condition = and_(GamePositionRollup.piece_positions == gp.piece_positions, 
                GamePositionRollup.castling_rights == gp.castling_rights, 
                GamePositionRollup.turn == gp.turn,
                GamePositionRollup.en_passant == gp.en_passant, 
                GamePositionRollup.greater_than_n_half_moves == gp.greater_than_n_half_moves)
        conditions.append(condition)        


    results  = db.query(GamePositionRollup).filter(or_(*conditions)).all()
    if results == []:
        return 0
    game_positions_data = [extract_attributes(gp) for gp in results]

    # Step 3: Convert to a DataFrame
    df = pd.DataFrame(game_positions_data)


    df = add_score_column(df,scoring_weights = scoring_weights)
    max_score_index = df['score'].idxmax()

    # col = ["mean_w","mean_b","mean_s","score","total_wins","log_total_wins"]
    best_position = df.loc[max_score_index]
    if best_position.score < score_threshold:
        return 0
    
    move,position = find_move_for_position(boards,best_position=best_position)

    return move

def add_score_column(df,scoring_weights = [1,1,0.5]):

    df[['mean_w', 'mean_b', 'mean_s']] = pd.DataFrame(df['win_buckets'].tolist(), index=df.index)


    df['log_total_wins'] = np.log10(df['total_wins']+9)

    if df.head(1).turn.values[0] == 'b':

        df['score'] = ( df['mean_w']*scoring_weights[0] \
                       - df['mean_b']*scoring_weights[1] \
                         - df['mean_s']*scoring_weights[2]) \
                            * df['log_total_wins']
    else:
        df['score'] = (df['mean_b']*scoring_weights[1] \
                        - df['mean_w']*scoring_weights[0] \
                          - df['mean_s']*scoring_weights[2]) \
                              * df['log_total_wins']
    return df


def find_move_for_position(boards,best_position):
    for value in boards:
        game_position = value['gp']
        if (best_position['piece_positions'] == game_position.piece_positions and
            best_position['castling_rights'] == game_position.castling_rights and
            best_position['turn'] == game_position.turn and
            best_position['en_passant'] == game_position.en_passant and
            best_position['greater_than_n_half_moves'] == game_position.greater_than_n_half_moves):
            return value['move'], best_position  # Return the matching board and its position data


def extract_attributes(game_position):
    return {
        "id": game_position.id,
        "piece_positions": game_position.piece_positions,
        "castling_rights": game_position.castling_rights,
        "en_passant": game_position.en_passant,
        "turn": game_position.turn,
        "greater_than_n_half_moves": game_position.greater_than_n_half_moves,
        "white_wins": game_position.white_wins,
        "black_wins": game_position.black_wins,
        "stalemates": game_position.stalemates,
        "total_wins": game_position.total_wins,
        "win_buckets": game_position.win_buckets,
    }

def generate_all_possible_boards(board):
    all_possible_boards = []
    
    for move in board.legal_moves:
        new_board = board.copy()  # Make a copy of the board
        new_board.push(move)  # Make the move
        all_possible_boards.append({'move':move, 'board':new_board})
    
    return all_possible_boards
    
def times_position_repeated(board: chess.Board):
    rep = 0
    for i in range(0,5):
        if board.is_repetition(i):
            rep = i
    return rep

def fetch_all_game_positions(db: Session = next(get_db())):
    try:
        # Querying all records in GamePositions
        for game_position in db.query(GamePositions).yield_per(1):
            yield game_position
    finally:
        db.close()
        yield None

def fetch_one_game_position(db: Session = next(get_db())):
    try:
        # Querying the first record in GamePositions
        game_position = db.query(GamePositions).first()
        return game_position
    finally:
        db.close()

class GamePositionWithWinBuckets:
    def __init__(self, piece_positions, castling_rights, en_passant, turn, greater_than_n_half_moves, white_wins, black_wins, stalemates):
        self.piece_positions = piece_positions
        self.castling_rights = castling_rights
        self.en_passant = en_passant
        self.turn = turn
        self.greater_than_n_half_moves = greater_than_n_half_moves
        self.white_wins = white_wins
        self.black_wins = black_wins
        self.stalemates = stalemates
        self.total_wins = white_wins + black_wins + stalemates
        self.win_buckets = [white_wins/self.total_wins,black_wins/self.total_wins,stalemates/self.total_wins]

    def to_json(self):
        return json.dumps({
            "piece_positions": self.piece_positions,
            "castling_rights": self.castling_rights,
            "en_passant": self.en_passant,
            "turn": self.turn,
            "greater_than_n_half_moves": self.greater_than_n_half_moves,
        })

def create_rollup_table(yield_size: int = 200,db: Session = next(get_db())):
    try:
        # Constructing the query with group by and sum
        query = db.query(
            GamePositions.piece_positions, 
            GamePositions.castling_rights, 
            GamePositions.en_passant, 
            GamePositions.turn, 
            GamePositions.greater_than_n_half_moves, 
            func.sum(GamePositions.white_wins).label('white_wins'),
            func.sum(GamePositions.black_wins).label('black_wins'),
            func.sum(GamePositions.stalemates).label('stalemates'),

        ).group_by(
            GamePositions.piece_positions, 
            GamePositions.castling_rights, 
            GamePositions.en_passant, 
            GamePositions.turn, 
            GamePositions.greater_than_n_half_moves
        )

        gen = query.yield_per(yield_size)

        for result in gen:
            game = GamePositionRollup(
                piece_positions=result.piece_positions,
                castling_rights=result.castling_rights,
                en_passant=result.en_passant,
                turn=result.turn,
                greater_than_n_half_moves=result.greater_than_n_half_moves,
                white_wins=result.white_wins,
                black_wins=result.black_wins,
                stalemates=result.stalemates
            )
            db.add(game)
        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def fetch_all_game_positions_rollup(yield_size: int = 200,db: Session = next(get_db())):
    try:
        # Constructing the query with group by and sum
        query = db.query(
            GamePositionRollup.piece_positions, 
            GamePositionRollup.castling_rights, 
            GamePositionRollup.en_passant, 
            GamePositionRollup.turn, 
            GamePositionRollup.greater_than_n_half_moves, 
            GamePositionRollup.white_wins.label('white_wins'),
            GamePositionRollup.black_wins.label('black_wins'),
            GamePositionRollup.stalemates.label('stalemates'),

        ).group_by(
            GamePositionRollup.piece_positions, 
            GamePositionRollup.castling_rights, 
            GamePositionRollup.en_passant, 
            GamePositionRollup.turn, 
            GamePositionRollup.greater_than_n_half_moves
        )

        gen = query.yield_per(yield_size)

        for result in gen:
            game = GamePositionWithWinBuckets(
                piece_positions=result.piece_positions,
                castling_rights=result.castling_rights,
                en_passant=result.en_passant,
                turn=result.turn,
                greater_than_n_half_moves=result.greater_than_n_half_moves,
                white_wins=result.white_wins,
                black_wins=result.black_wins,
                stalemates=result.stalemates
            )

            yield game

    except GeneratorExit:

        return

    finally:
        db.close()
        yield None


def calculate_win_buckets(white_wins, black_wins, stalemates):
    total_wins = white_wins + black_wins + stalemates
    if total_wins > 0:
        mean_w = white_wins / total_wins
        mean_b = black_wins / total_wins
        mean_s = stalemates / total_wins
    else:
        mean_w = mean_b = mean_s = 0  # Default values if no wins
    return [mean_w, mean_b, mean_s]

def get_row_count(db: Session = next(get_db())):
    try:
        # Counting the rows in the GamePositions table
        count = db.query(func.count(GamePositions.id)).scalar()
        return count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        db.close()

def get_rollup_row_count(db: Session = next(get_db())):
    try:
        # Counting the rows in the GamePositions table
        count = db.query(
            GamePositions.piece_positions, 
            GamePositions.castling_rights, 
            GamePositions.en_passant, 
            GamePositions.turn, 
            GamePositions.greater_than_n_half_moves, 
            GamePositions.white_wins.label('white_wins'),
            GamePositions.black_wins.label('black_wins'),
            GamePositions.stalemates.label('stalemates'),

        ).group_by(
            GamePositions.piece_positions, 
            GamePositions.castling_rights, 
            GamePositions.en_passant, 
            GamePositions.turn, 
            GamePositions.greater_than_n_half_moves
        ).count()
        return count
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        db.close()

def get_GamePositionRollup_row_size(db: Session = next(get_db())):
    try:
        # Counting the rows in the GamePositions table
        count = db.query(
            GamePositionRollup.piece_positions, 
            GamePositionRollup.castling_rights, 
            GamePositionRollup.en_passant, 
            GamePositionRollup.turn, 
            GamePositionRollup.greater_than_n_half_moves, 
            GamePositionRollup.white_wins.label('white_wins'),
            GamePositionRollup.black_wins.label('black_wins'),
            GamePositionRollup.stalemates.label('stalemates'),

        ).group_by(
            GamePositionRollup.piece_positions, 
            GamePositionRollup.castling_rights, 
            GamePositionRollup.en_passant, 
            GamePositionRollup.turn, 
            GamePositionRollup.greater_than_n_half_moves
        ).count()
        return count
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        db.close()

def insert_bulk_boards_into_db(board_victors: List[Tuple[chess.Board, str]], db: Session = next(get_db())):
    games = []
    for board, victor in board_victors:
        games.append(board_to_GamePostition(board=board, victor=victor))
    
    try:
        db.bulk_save_objects(games)
        db.commit()
        return len(games)  # Return the number of inserted records
    except Exception as e:
        db.rollback()
        raise e