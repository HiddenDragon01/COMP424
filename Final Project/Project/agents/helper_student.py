# Student agent: Add your own agent here
from random import shuffle
from agents.agent import Agent
from store import register_agent
import sys
import copy
import numpy as np
import time
import traceback


class TimeUpException(Exception):
    pass


class HelperAgent:
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.zoneTiles = []
        self.start_time = 0
        self.time_limit = 1.95
        self.panic_move = None
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer
        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.
        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        self.start_time = time.time()
        # find a cheap panic move in case we lack time (one of the 4 directions will always be a valid move, otherwise the game would be done).
        r,c = my_pos
        for key in self.dir_map:
            if not chess_board[r, c, self.dir_map[key]]:
                self.panic_move = tuple(((r,c), self.dir_map[key]))
                break
        answer = self.find_next_step(chess_board, my_pos, adv_pos, max_step)
        return answer

    
    def find_next_step(self, chess_board, my_pos, adv_pos, max_step):
        try:
            nbRows = len(chess_board)
            nbCols = len(chess_board[0])

            valid_moves_sorted = self.organize_by_best_move(chess_board, my_pos, adv_pos, max_step, nbRows, nbCols)

            (x,y,d) = valid_moves_sorted[0]
            return tuple(((x,y), d))

        except Exception as e:
            #print('An exception forced the panic move:')
            traceback.print_exc()
            return self.panic_move

    def get_player_area_size(self, chess_board, my_pos, adv_pos, max_step, nbRows: int, nbCols: int):
        """
        Check if the game ends and compute the current score of the agents.
        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(nbRows):
            for c in range(nbCols):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(nbRows):
            self.check_time()
            for c in range(nbCols):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(nbRows):
            for c in range(nbCols):
                find((r, c))

        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))

        if p0_r == p1_r:
            return False, 0

        my_score = list(father.values()).count(p0_r)
        adv_score = list(father.values()).count(p1_r)

        return True, my_score - adv_score


    def check_valid_step(self, chess_board, start_pos, end_pos, barrier_dir, adv_pos, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).
        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier, is boarder or is current pos
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            self.check_time()
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                
                next_pos = cur_pos[0] + move[0],cur_pos[1] + move[1]
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached


    def get_all_simple_steps(self, chess_board, my_pos, adv_pos, max_step, nbRows: int, nbCols: int):
        all_valid_steps = []
        for r in range(max(0, my_pos[0] - max_step), 
        min(nbRows, my_pos[0] + max_step)):
            for c in range(max(0, my_pos[1] - max_step), 
        min(nbCols, my_pos[1] + max_step)):
                if (abs(my_pos[0] - r) + abs(my_pos[1] - c)) in range(0,max_step + 1):
                    for key in list(self.dir_map):
                        dir = self.dir_map[key]
                        if self.check_valid_step(chess_board, my_pos, (r, c), dir, adv_pos, max_step):
                            all_valid_steps.append((r,c,dir))
        return all_valid_steps
    

    def set_barrier(self, chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True


    def may_be_finishing_move(self, chess_board, nbRows: int, nbCols: int, move):
        # if the move reunites at least 2 barriers together, it may be a 'finishing' move. In other words, it might end the game
        r,c,d = move
        if d == 0 or d == 2:
            hori_left = c == 0 or chess_board[r, c - 1, d]
            hori_right = c == nbCols - 1 or chess_board[r, c + 1, d]
            if d == 0: # up
                vert_left = chess_board[r, c, 3] or chess_board[r - 1, c, 3]
                vert_right = chess_board[r, c, 1] or chess_board[r - 1, c, 1]
            else: # down
                vert_left = chess_board[r, c, 3] or chess_board[r + 1, c, 3]
                vert_right = chess_board[r, c, 1] or chess_board[r + 1, c, 1]
                
            return (vert_left or hori_left) and (vert_right or hori_right) 
        else:
            vert_up = r == 0 or chess_board[r - 1, c, d]
            vert_down = r == nbRows - 1 or chess_board[r + 1, c, d]
            if d == 1: # right
                hori_up = chess_board[r, c, 0] or chess_board[r, c + 1, 0]
                hori_down = chess_board[r, c, 2] or chess_board[r, c + 1, 2]
            else: # left
                hori_up = chess_board[r, c, 0] or chess_board[r, c - 1, 0]
                hori_down = chess_board[r, c, 2] or chess_board[r, c - 1, 2]

            return (hori_up or vert_up) and (hori_down or vert_down) 




    def organize_by_best_move(self, chess_board, my_pos, adv_pos, max_step, nbRows: int, nbCols: int):
        winning_moves = []
        no_end_moves = []
        tie_moves = []
        losing_moves = []

        def my_temp_func(move):
            r,c,d = move
            potential_board = copy.deepcopy(chess_board)
            self.set_barrier(potential_board, r, c, d)
            rep = None
            
            if self.may_be_finishing_move(chess_board, nbRows, nbCols, move):
                rep = self.get_player_area_size(potential_board, (r,c), adv_pos, max_step, nbRows, nbCols)

            if rep == None:
                no_end_moves.append(move)
            else:
                if rep[0]:
                    if rep[1] > 0:
                        # just immediately take any winning move
                        return True, [move]
                    elif rep[1] == 0:
                        tie_moves.append(move)
                    else:
                        losing_moves.append(move)
                else:
                    no_end_moves.append(move)

            return False, None

        output = self.do_on_all_simple_steps(chess_board, my_pos, adv_pos, max_step, nbRows, nbCols, my_temp_func)

        if output != None:
            return output

        if winning_moves:
            return winning_moves

        # sort losing moves in ascending order (smallest difference first)
        losing_moves.sort(key=lambda y: -y[1])
        # sort the neutral moves randomly
        no_end_moves, tie_is_better = self.sort_by_least_dangerous_move(chess_board, adv_pos, max_step, nbRows, nbCols, no_end_moves)

        if tie_is_better:
            return tie_moves + no_end_moves + losing_moves
        else:
            return no_end_moves + tie_moves + losing_moves


    

    # check if the ally has winning moves or if he will force the adv to suicide.
    def has_winning_moves(self, chess_board, my_pos, adv_pos, max_step, nbRows: int, nbCols: int):

        def my_temp_func(move):
            r,c,d = move
            potential_board = copy.deepcopy(chess_board)
            self.set_barrier(potential_board, r, c, d)

            if not self.may_be_finishing_move(chess_board, nbRows, nbCols, move):
                return False, (False, 0)

            rep = self.get_player_area_size(potential_board, (r,c), adv_pos, max_step, nbRows, nbCols)
            if rep[0]:
                if rep[1] > 0:
                    return True, (True, 0)
                else:
                    # game is over
                    return False, (False, 0)

            # here we check if 'we', 'us' do not lose by forced suicide.
            does_not_lose = self.does_not_lose(potential_board, adv_pos, (r,c), max_step, nbRows, nbCols)
            if not does_not_lose:
                return True, (True, 0)
            else:
                return False, (False, 0)

        output = self.do_on_all_simple_steps(chess_board, my_pos, adv_pos, max_step, nbRows, nbCols, my_temp_func)

        if output == None:
            return False, 0
        else:
            return output



    def sort_by_least_dangerous_move(self, chess_board, adv_pos, max_step, nbRows: int, nbCols: int, move_list: list):
        # The given moves must be valid

        # Act as if you did the move. Check if it enables
        # the adversary to beat you

        dangerous_moves = []
        not_dangerous = []
        potential_board = copy.deepcopy(chess_board)

        for move in move_list:

            r,c,d = move
            # place the wall
            potential_board = copy.deepcopy(chess_board)
            self.set_barrier(potential_board, r, c, d)

            # change position
            my_potential_pos = tuple((r,c))

            # finds the number of winning moves of the ennemy (we give our potential position)
            enemy_can_win, nb_turn = self.has_winning_moves(potential_board, adv_pos, my_potential_pos, max_step, nbRows, nbCols)

            if enemy_can_win and nb_turn == 0:
                dangerous_moves.append(move)
            else:
                if nb_turn == 1:
                    # change the panic move to something we know isn't instantly dangerous
                    self.panic_move = tuple(((r,c),d))
                    not_dangerous.append(move)
                else:
                    # change the panic move to something we know isn't dangerous
                    self.panic_move = tuple(((r,c),d))
                    return [move], False

        return not_dangerous + dangerous_moves, True



    def does_not_lose(self, chess_board, my_pos, adv_pos, max_step, nbRows: int, nbCols: int):
        def my_temp_func(move):
            r,c,d = move
            potential_board = copy.deepcopy(chess_board)
            self.set_barrier(potential_board, r, c, d)
            rep = None

            if self.may_be_finishing_move(chess_board, nbRows, nbCols, move):
                rep = self.get_player_area_size(potential_board, (r,c), adv_pos, max_step, nbRows, nbCols)

            if rep == None or not rep[0] or rep[1] >= 0:
                # If there's 1 win, 1 non ending or 1 tie, it's fine
                return True, True

            return False, False

        output = self.do_on_all_simple_steps(chess_board, my_pos, adv_pos, max_step, nbRows, nbCols, my_temp_func)

        if output == None:
            return False
        else:
            return output

    
    def do_on_all_simple_steps(self, chess_board, my_pos, adv_pos, max_step, nbRows: int, nbCols: int, func):
        output = None
        for r in range(max(0, my_pos[0] - max_step), 
        min(nbRows, my_pos[0] + max_step)):
            for c in range(max(0, my_pos[1] - max_step), 
        min(nbCols, my_pos[1] + max_step)):
                if (abs(my_pos[0] - r) + abs(my_pos[1] - c)) in range(0,max_step + 1):
                    for key in list(self.dir_map):
                        dir = self.dir_map[key]
                        if self.check_valid_step(chess_board, my_pos, (r, c), dir, adv_pos, max_step):
                            should_return, output = func((r,c,dir))
                            if should_return:
                                return output
        return output


    def check_time(self):
        if time.time() - self.start_time > self.time_limit:
            #print("Time breach")
            raise TimeUpException