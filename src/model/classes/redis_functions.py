import chess
import redis
import hiredis
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.sqlite.dependencies import (
    fetch_all_game_positions_rollup,
    get_rollup_row_count,
    board_to_GamePostitionRollup,
    board_to_GamePostition)
from tqdm import tqdm
import json


class redis_pipe():

    def __init__(self,**kwargs):
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()

        self.r = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_db))
    
    def clear_redis(self):
        self.r.flushall()

    def redis_key_generator(self, match='*', batch_size=100):
        cursor = 0
        while True:
            cursor, keys = self.r.scan(cursor=cursor, match=match, count=batch_size)
            if keys:
                # Fetch values for the current batch of keys
                values = self.r.mget(keys)
                values = [value.decode('utf-8') for value in values if value]
                # Yield pairs of (key, value)
                yield zip(keys, values)
            
            # Break the loop if cursor returns to 0, indicating the scan is complete
            if cursor == 0:
                break


    def get_db_size(self):
        num_keys = self.r.dbsize()
        return num_keys

    def sqlite_to_redis(self,batch_size: int=1000):
        batch = fetch_all_game_positions_rollup()
        row_count = get_rollup_row_count()
        pipe = self.r.pipeline()
        for i, game in enumerate(tqdm(batch, total=row_count, desc="SQLite to Redis")):
            if game:
                results = {
                    "white_wins": game.white_wins,
                    "black_wins": game.black_wins,
                    "stalemates": game.stalemates,
                    "total_wins": game.total_wins,
                    "white_mean": game.win_buckets[0],
                    "black_mean": game.win_buckets[0],
                    "stalemate_mean": game.win_buckets[0]
                }
                # Add the set command to the pipeline
                pipe.set(game.to_json(), json.dumps(results))
                
                # Execute the pipeline every `batch_size` commands
                if (i + 1) % batch_size == 0:
                    pipe.execute()
        
        # Execute any remaining commands in the pipeline
        pipe.execute()
        return 1
    
    def get_board_results(self,board):
        game = board_to_GamePostition(board=board)
        search = game.to_json()
        results = self.r.get(search)
        return results
