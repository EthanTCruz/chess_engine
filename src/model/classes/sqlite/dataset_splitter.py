
from chess_engine.src.model.classes.sqlite.dependencies import get_db
from chess_engine.src.model.classes.sqlite.models import (GamePositions,
                                                          GamePositionRollup)
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.orm import load_only
import random
from chess_engine.src.model.config.config import settings
from chess_engine.src.model.classes.sqlite.models import (GamePositions,
                                                          GamePositionRollup)
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import get_all_bitboards_dict
from sqlalchemy.orm import Session
import chess
from tqdm import tqdm


def create_rollup_table(
    yield_size: int = 200,
    train_pct: float = settings.nnTrainSize,
    test_pct: float = settings.nnTestSize,
    validation_pct: float = settings.nnValidationSize,
    db: Session = next(get_db())
):
    try:
        # Validate the input percentages add up to 1.0
        if not (train_pct + test_pct + validation_pct == 1.0):
            raise ValueError("The sum of train, test, and validation percentages must be 1.0")

        # Constructing the query with group by and sum
        query = db.query(
            GamePositions.piece_positions,
            GamePositions.castling_rights,
            GamePositions.en_passant,
            GamePositions.turn,
            func.sum(GamePositions.white_wins).label('white_wins'),
            func.sum(GamePositions.black_wins).label('black_wins'),
            func.sum(GamePositions.stalemates).label('stalemates'),
        ).group_by(
            GamePositions.piece_positions,
            GamePositions.castling_rights,
            GamePositions.en_passant,
            GamePositions.turn,
        )

        # Get the total count of rows
        total_count = query.count()

        # Yield results in batches with tqdm progress bar
        gen = query.yield_per(yield_size)
        for result in tqdm(gen, total=total_count, desc="Processing GamePositions"):
            fen = f"{result.piece_positions} {result.turn} {result.castling_rights} {result.en_passant} 0 1"
            board = chess.Board(fen)
            results_dict = get_all_bitboards_dict(board=board)

            # Randomly assign the data split
            rand_val = random.random()
            is_training_data = rand_val < train_pct
            is_testing_data = train_pct <= rand_val < (train_pct + test_pct)
            is_validation_data = not is_training_data and not is_testing_data  # Remaining data

            # Create the rollup entry
            game = GamePositionRollup(
                piece_positions=result.piece_positions,
                castling_rights=result.castling_rights,
                en_passant=result.en_passant,
                turn=result.turn,
                fen=fen,
                **results_dict,
                white_wins=result.white_wins,
                black_wins=result.black_wins,
                stalemates=result.stalemates,
                is_training_data=is_training_data,
                is_testing_data=is_testing_data,
                is_validation_data=is_validation_data
            )
            db.add(game)

        db.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
