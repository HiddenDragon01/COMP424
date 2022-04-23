# Student agent: Add your own agent here
from agents.agent import Agent
from agents import helper_student
from store import register_agent
import sys
import numpy as np
from copy import deepcopy


import time





@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
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

        # dummy return
        #  return my_pos, self.dir_map["u"]

        helper = helper_student.HelperAgent()

        n = len(chess_board[0])

        steps = self.get_all_simple_steps(chess_board, my_pos, adv_pos, max_step, n)

        steplen = len(steps)     
        #print("Steps length: " + str(len(steps)))

        if (n == 9 and steplen > 120) or (n == 10 and steplen > 100) or (n == 11 and steplen > 90) or (n == 12 and steplen > 80):
            return helper.step(chess_board, my_pos, adv_pos, max_step)

        pos,dir = self.monteCarloCalcBest(chess_board, my_pos, adv_pos, max_step, steps)

        return pos,dir
    



    def monteCarloCalcBest(self, chess_board, my_pos, adv_pos, max_step, steps):

        start_time = time.time()
        seconds = 1.97
       

        # Make copies of things so the originals don't change
       
        
        #print("In monte carlo")

        dict = {}

        i = 0
        

        
        while True:

            current_time = time.time()
            elapsed_time = current_time - start_time


            if elapsed_time > seconds:
                break

            for move in steps:

                current_time = time.time()
                elapsed_time = current_time - start_time

                if elapsed_time > seconds:
                    break

            

                i += 1

                copy_board = deepcopy(chess_board)

                r,c,dir = move

          
                self.set_barrier(copy_board, r, c, dir)
                score = self.randomSimulation(copy_board, (r,c), adv_pos, 1, max_step)
                move = ((r,c), dir)
                dict[move] = dict.get(move, 0) + score

                
        max_move = max(dict, key=dict.get)
                
        elapsed_time = current_time - start_time
        #print("Finished in " + str(elapsed_time) + " seconds.")
        return max_move
        


    # Calculate all possible moves as well as barrier placements 
    def get_all_simple_steps(self, chess_board, my_pos, adv_pos, max_step, grid_dim):
        all_valid_steps = []
        for r in range(max(0, my_pos[0] - max_step), 
        min(grid_dim, my_pos[0] + max_step)):
            for c in range(max(0, my_pos[1] - max_step), 
        min(grid_dim, my_pos[1] + max_step)):

                if (abs(my_pos[0] - r) + abs(my_pos[1] - c)) in range(0,max_step + 1):
                    for key in list(self.dir_map):
                        dir = self.dir_map[key]
                        if self.check_valid_step(chess_board, my_pos, (r, c), dir, adv_pos, max_step):
                            all_valid_steps.append((r,c,dir))
        return all_valid_steps    

    # Run randomSimulations for n iterations and return fraction of number of games won
    def randomSimulations(self, chess_board_2, my_pos_2, adv_pos_2, max_step):
        turn = 0
        sum = 0
        numSims = 5
        for i in range(numSims):
            chess_board_4 = deepcopy(chess_board_2)
            sum += self.randomSimulation(chess_board_4, my_pos_2, adv_pos_2, turn, max_step)

        
        return sum


    # check if move is valid
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




    # Given turn = 0 (my turn) or turn = 1 (adv turn), run a random simulation
    def randomSimulation(self, chess_board, my_pos_2, adv_pos_2, turn, max_step):

        chess_board_2 = deepcopy(chess_board)

        board_dim = len(chess_board_2[0])

        
        res = self.finished(chess_board_2, my_pos_2, adv_pos_2, max_step, board_dim)

        # will lose in one move
        if res[1] == -10000:
            return -10000000
        # tie
        elif res[1] == -1:
            return -150
            # can win in one move
        elif res[1] == 100:
            return 10000000

        loseInOne = True


        # While the simulation game has not finished, make a move 
        while not res[0]:
                
            if turn == 0:
                random = self.randomMove(chess_board_2, my_pos_2, adv_pos_2, max_step)
                my_pos_2 = random[0] 
                my_dir = random[1]
                self.set_barrier(chess_board_2, my_pos_2[0], my_pos_2[1], my_dir)
                turn = 1
                loseInOne = False
            elif turn == 1:
                random = self.randomMove(chess_board_2, adv_pos_2, my_pos_2, max_step)
                adv_pos_2 = random[0]
                adv_dir = random[1] 
                self.set_barrier(chess_board_2, adv_pos_2[0], adv_pos_2[1], adv_dir)
                turn = 0

            res = self.finished(chess_board_2, my_pos_2, adv_pos_2, max_step, board_dim)

        if loseInOne and res[1] == -10000:
            return -10000000

        return res[1]


    def set_barrier(self, chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True


    # Returns a random square and a barier position
    def randomMove(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)
        n = len(chess_board[0])
       

        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir


    

    def finished(self, chess_board_3, my_pos_3, adv_pos_3, max_step, board_dim):
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
        for r in range(board_dim):
            for c in range(board_dim):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_dim):
            for c in range(board_dim):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board_3[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_dim):
            for c in range(board_dim):
                find((r, c))

        p0_r = find(my_pos_3)
        p1_r = find(adv_pos_3)
       
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:

            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
            return True, -25
       

        if (p0_score - p1_score) < 0:
             return True, -10000
        else:
            return True, 100

    