import bz2
import pickle
import sys
from multiprocessing import Pool
from typing import Iterable

import chess.engine
import chess.pgn
import numpy as np

engine = chess.engine.SimpleEngine.popen_uci('../stockfish_14_win_x64_avx2/stockfish_14_x64_avx2.exe')


def parse_time_control(time_control_str: str) -> tuple[int, int]:
    s = time_control_str.split('+')
    return int(s[0]), int(s[1])


def extract_features(game: chess.pgn.Game):
    mainline = list(game.mainline())

    move_num = np.random.randint(2, len(mainline))

    starting_time, time_increment = parse_time_control(game.headers['TimeControl'])
    x = (
        [
            int(game.headers['WhiteElo']),
            int(game.headers['BlackElo']),
            starting_time,
            time_increment
        ],
        []
    )

    board = chess.Board()

    MATE_SCORE = 100000

    prev_prev_node_clk = starting_time - time_increment
    prev_node_clk = starting_time - time_increment

    for i in range(move_num):
        node = mainline[i]
        analysis = engine.analyse(board, chess.engine.Limit(time=0.1), game=game)
        x[1].append([
            analysis['score'].relative.score(mate_score=MATE_SCORE),
            len(list(board.legal_moves)),
            len(analysis['pv']),
            prev_prev_node_clk - node.clock() + time_increment
        ])

        board.push(node.move)
        prev_prev_node_clk = prev_node_clk
        prev_node_clk = node.clock()

    analysis = engine.analyse(board, chess.engine.Limit(time=0.1), game=game)
    x[0].extend([
        analysis['score'].relative.score(mate_score=MATE_SCORE),
        len(list(board.legal_moves)),
        len(analysis['pv']),
    ])

    return x, mainline[move_num].clock()


def game_filter(game: chess.pgn.Game) -> bool:
    mainline = list(game.mainline())
    min_elo = 2400
    return len(mainline) > 2 and 'Rated Blitz' in game.headers['Event'] and \
           int(game.headers['WhiteElo']) >= min_elo and int(game.headers['BlackElo']) >= min_elo


def multi_files_games_generator(file_names: Iterable[str], max_games: int = 24000) -> Iterable[chess.pgn.Game]:
    cnt = 0
    for filename in file_names:
        print(f'loading from file {filename}')
        for game in games_generator(filename):
            yield game
            cnt += 1
            print(cnt)
            if cnt == max_games:
                return


def games_generator(filename: str, max_games: int = 24000) -> Iterable[chess.pgn.Game]:
    with bz2.open(filename, 'rt') as f:
        cnt = 0
        while True:
            game = chess.pgn.read_game(f)
            while game is not None and not game_filter(game):
                game = chess.pgn.read_game(f)
            if game is None or cnt == max_games:
                break
            yield game
            cnt += 1


def process_and_save_games_parallel(games: Iterable[chess.pgn.Game], output_path: str):
    POOL_SIZE = 6
    with Pool(POOL_SIZE) as p:
        res = list(p.imap_unordered(extract_features, games, chunksize=1000))

    with bz2.open(output_path, 'wb') as output_file:
        pickle.dump(res, output_file)


def main():
    sys.setrecursionlimit(5000)
    with engine:
        games = multi_files_games_generator('../lichess/lichess_db_standard_rated_2021-08.pgn.bz2')
        process_and_save_games_parallel(games, '../lichess/lichess_24k.pickel.bz2')


if __name__ == '__main__':
    main()
