with cte as (select piece_positions, castling_rights, en_passant, turn, sum(white_wins), sum(black_wins), sum(stalemates)
from GamePositions
group by piece_positions, castling_rights, en_passant, turn)
select count(*) from cte;

0|id|INTEGER|1||1
1|fen|VARCHAR|0||0
2|piece_positions|VARCHAR|0||0
3|castling_rights|VARCHAR|0||0
4|en_passant|VARCHAR|0||0
5|turn|VARCHAR|0||0
6|white_wins|INTEGER|0||0
7|black_wins|INTEGER|0||0
8|stalemates|INTEGER|0||0
9|is_training_data|BOOLEAN|0||0
10|is_testing_data|BOOLEAN|0||0
11|is_validation_data|BOOLEAN|0||0
12|white pawn|VARCHAR|0||0
13|white knight|VARCHAR|0||0
14|white bishop|VARCHAR|0||0
15|white rook|VARCHAR|0||0
16|white queen|VARCHAR|0||0
17|white king|VARCHAR|0||0
18|black pawn|VARCHAR|0||0
19|black knight|VARCHAR|0||0
20|black bishop|VARCHAR|0||0
21|black rook|VARCHAR|0||0
22|black queen|VARCHAR|0||0
23|black king|VARCHAR|0||0

with cte as (select piece_positions, castling_rights, en_passant, turn, sum(white_wins), sum(black_wins), sum(stalemates)
from gamepositionrollup
group by piece_positions, castling_rights, en_passant, turn)
select count(*) from cte;
