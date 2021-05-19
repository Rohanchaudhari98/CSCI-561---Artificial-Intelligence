import copy
import time
import math
start_time_time = time.time()
start_time = time.process_time()

outputFile = 'output.txt'
file = open(outputFile,'w')
piececolor = ""
initialelement = ""
tiles = {'1':'a','2':'b','3':'c','4':'d','5':'e','6':'f','7':'g','8':'h'}
tiles2 = {'a':'8','b':'7','c':'6','d':'5','e':'4','f':'3','g':'2','h':'1'}




class Node:
    def __init__(self, board, move=None, parent=None, value=None):
        self.board = board
        self.value = value
        self.move = move 
        self.parent = parent

    def get_children(self, maximizing_player, mandatory_jumping):
        global our_choice
        current_state = deepcopy(self.board)
        available_moves = []
        children_states = []
        king_piece = ""
        king_row = 0
        if (our_choice == "b" or our_choice == "B"):
            available_moves = Check.find_black_available_moves(current_state, mandatory_jumping)
            king_piece = "B"
            king_row = 7
        elif (our_choice == "w" or our_choice == "W"):
            available_moves = Check.find_white_available_moves(current_state, mandatory_jumping)
            king_piece = "W"
            king_row = 0
        for i in range(len(available_moves)):
            old_i = available_moves[i][0]
            old_j = available_moves[i][1]
            new_i = available_moves[i][2]
            new_j = available_moves[i][3]
            state = deepcopy(current_state)
            Check.make_a_move(state, old_i, old_j, new_i, new_j, king_piece, king_row)
            # self.print_matrix()
            children_states.append(Node(state, [old_i, old_j, new_i, new_j]))
        return children_states

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_board(self):
        return self.board

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent


def get_moves(board, row, col, is_sorted = False):
    down, up = [(+1, -1), (+1, +1)], [(-1, -1), (-1, +1)]
    # print("Here")
    length = board.get_length()
    piece = board.get(row, col)
    # print(piece,row,col)
    if piece:
    	# for x,y in down:
    	# 	print(x,y,row,col,row+x,col+y,length)
    	# 	# print(board.is_free(row+x,col+y))
    	# 	if (0 <= (row + x) < length) and (0 <= (col + y) < length) and board.is_free(row + x, col + y):
    	# 		print("Here")
    	
    	bottom = [no_to_str(row + x, col + y) for (x, y) in down \
    				if (0 <= (row + x) < length) \
    				and (0 <= (col + y) < length) \
    				and board.is_free(row + x, col + y)]
    	# print("bottom",bottom)
    	top = [no_to_str(row + x, col + y) for (x, y) in up \
    			if (0 <= (row + x) < length) \
    			and (0 <= (col + y) < length) \
    			and board.is_free(row + x, col + y)]
    	# print("top",top)
    	
    	return (sorted(bottom + top) if piece.is_king() else \
                (sorted(bottom) if piece.is_black() else sorted(top))) \
                    if is_sorted else (bottom + top if piece.is_king() else \
                                       (bottom if piece.is_black() else top))
    return []

def get_jumps(board, row, col, is_sorted = False):
	global initialelement
	down, up = [(+1, -1), (+1, +1)], [(-1, -1), (-1, +1)]
	length = board.get_length()
	piece = board.get(row, col)
	# print(piece,row,col)
	if piece:
		if (initialelement == "w" and row == 0) or (initialelement == "b" and row == 7):
			return []
		else:
			top = [no_to_str(row + 2 * x, col + 2 * y) for (x, y) in up \
					if (0 <= (row + 2 * x) < length) \
					and (0 <= (col + 2 * y) < length) \
					and board.is_free(row + 2 * x, col + 2 * y) \
					and (not board.is_free(row + x, col + y)) \
					and (board.get(row + x, col + y).color().lower() != piece.color().lower())]
			bottom = [no_to_str(row + 2 * x, col + 2 * y) for (x, y) in down \
					if (0 <= (row + 2 * x) < length) \
					and (0 <= (col + 2 * y) < length) \
					and board.is_free(row + 2 * x, col + 2 * y) \
					and (not board.is_free(row + x, col + y)) \
					and (board.get(row + x, col + y).color().lower() != piece.color().lower())]
		if top != [] or bottom != []:
			initialelement = str(piece)
		# print("top",top)
		# print("bottom",bottom)
		return (sorted(bottom + top) if piece.is_king() else \
				(sorted(bottom) if piece.is_black() else sorted(top))) \
					if is_sorted else (bottom + top if piece.is_king() else \
						(bottom if piece.is_black() else top))
	return []

def search(board, row, col, path, paths, is_sorted = False):
	# print("search")
	path.append(no_to_str(row, col))
	jumps = get_jumps(board, row, col, is_sorted)
	# print("jumps",jumps)
	if not jumps:
		paths.append(path)
	else:
		# print("jumps",jumps)
		for position in jumps:
			(row_to, col_to) = pos(position)
			piece = copy.copy(board.get(row, col))
			board.remove(row, col)
			board.place(row_to, col_to, piece)
			if (piece.color() == 'black' \
				and row_to == board.get_length() - 1) \
					or (piece.color() == 'white' \
						and row_to == 0) \
							and (not piece.is_king()):
								piece.turn_king()
			row_mid = row + 1 if row_to > row else row - 1
			col_mid = col + 1 if col_to > col else col - 1
			capture = board.get(row_mid, col_mid)
			board.remove(row_mid, col_mid)
			search(board, row_to, col_to, copy.copy(path), paths)
			board.place(row_mid, col_mid, capture)
			board.remove(row_to, col_to)
			board.place(row, col, piece)
			# print('jumps end',jumps)
 

def find_black_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n][0] == "b":
                    if check.check_black_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_black_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_black_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_black_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
                elif board[m][n][0] == "B":
                    if Checkers.check_black_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_black_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_black_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_black_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_black_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_black_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_black_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if Checkers.check_black_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])

        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

def check_black_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == "...":
            return False
        if board[via_i][via_j][0] == "B" or board[via_i][via_j][0] == "b":
            return False
        if board[new_i][new_j] != "...":
            return False
        if board[old_i][old_j] == "...":
            return False
        if board[old_i][old_j][0] == "w" or board[old_i][old_j][0] == "W":
            return False
        return True

def check_black_moves(board, old_i, old_j, new_i, new_j):

        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == "...":
            return False
        if board[new_i][new_j] != "...":
            return False
        if board[old_i][old_j][0] == "w" or board[old_i][old_j][0] == "W":
            return False
        if board[new_i][new_j] == "...":
            return True



def get_jump(board, row, col, is_sorted = False):
	# print("get captures")
	paths = []
	board_ = copy.copy(board)
	search(board_, row, col, [], paths, is_sorted)
	# print("paths",paths,len(paths),len(paths[0]))
	if len(paths) == 1 and len(paths[0]) == 1:
		paths = []
	return paths


class Checkers(object):
    def __init__(self, length = 8):
        if length > 1:
            self._length = length
            self._cell = [[None for c in range(self._length)] \
                                for r in range(self._length)]
        else:
            raise ValueError("The minimum allowed length of a board is 2.")

    
    
    def get_length(self):
        return self._length
    
    def get_cells(self):
        return self._cell
    
    def is_free(self, row, col):
    	# print("isfree",self._cell[row][col],"here",row,col)
    	# print("isfree",'.',"here",row,col)
    	# file.write(self.row_cell[row][col])
    	# print(len(self._cell[row][col]))
    	# print(str(self._cell[row][col]) == ".")
    	return str(self._cell[row][col]) == "."
        
    def place(self, row, col, piece):
    	# print(piece)
    	self._cell[row][col] = piece
        
    def get(self, row, col):
        return self._cell[row][col]
    
    def remove(self, row, col):
        # print(row,col)
        self._cell[row][col] = Piece(".")
        
    def is_empty(self):
        for r in range(self._length):
            for c in range(self._length):
                if not self.is_free(r,c):
                    return False
        return True
    
    def is_full(self):
        for r in range(self._length):
            for c in range(self._length):
                if self.is_free(r, c):
                    return False
        return True
    
    def display(self, count = None):
        print(self)
        if count is not None:
            print("  Black: {:d}, White: {:d}"\
                  .format(count[0], count[1]))
            
    def __str__(self):
        vline = '\n' + (' ' * 2) + ('+---' * self._length) + '+' + '\n'
        numline = ' '.join([(' ' + str(i) + ' ') \
                            for i in range(1, self._length + 1)])
        str_ = (' ' * 3) + numline + vline
        for r in range(0, self._length):
            str_ += chr(97 + r) + ' |'
            for c in range(0, self._length):
            	# print(self._cell[r][c] is ' ')
            	str_ += ' ' + \
                    (str(self._cell[r][c]) \
                         if self._cell[r][c] is not None else ' ') + ' |'
            str_ += vline
        return str_
    
    def __repr__(self):
        return self.__str__()
    
class Piece(object):
	# print("higvbujknfkdsjfnlrsefnwekld")
	# print("object",object())
	symbols = ['b', 'w', '.']
	# blank = [' ']
	# print(symbols[2])
	_is_king = False
	symbols_king = ['B', 'W']
	# def __init__(self, color = 'black', is_king = False):
	# 	if color.isalpha():
	# 		color = color.lower()
	# 		if color == 'black' or color == 'white':
	# 			self._color = color
	# 			self._is_king  = is_king
	# 		else:
	# 			raise ValueError("A piece must be \'black\' or \'white\'.")
	# 	else:
	# 		raise ValueError("A piece must be \'black\' or \'white\'.")
        
	def __init__(self, color, is_king = False):
		# print("color",color)
		if color == 'black' or color == 'white':
			# print("small")
			self._color = color
			self._is_king  = is_king
		elif color == "BLACK" or color == "WHITE":
			# print(color)
			self._color = color
			self._is_king = True
			# print(self._color)
		elif color == '.':
			# print("Here")
			self._color = '.'
			self._is_king = is_king
			# print("Here",self._color)
		# print("Here")
		# print(self._color,self._is_king)
	# # print(color,is_king)
	# print("here")

	def color(self):
		# print("here1")
		# print("Here",self._color)
		return self._color

	def is_black(self):
		# print("here2")
		return self._color == 'black'

	def is_white(self):
		# print("here3")
		return self._color == 'white'

	def is_blank(self):
		return self._color == ' '

	def is_king(self):
		# print("here4")
		return self._is_king

	def turn_king(self):
		# print("here5")
		self._is_king = True

	def turn_pawn(self):
		# print("here6")
		self.is_king = False

	def __str__(self):
		# print("here1")
		if self._is_king:
			return self.symbols_king[0] if self._color == 'black' or self._color == "BLACK" else self.symbols_king[1]
		else:
			if self._color == "black":
				return self.symbols[0]
			elif self._color == "white":
				return self.symbols[1]
			elif self._color == ".":
				return '.'
			# return self.symbols[0] if self._color == 'black' else self.symbols[1]

	def __repr__(self):
		return self.__str__()
	# print("here")

def calculate_heuristics(state):
    board = state[0]
    # print("Heuristics")
    turn = state[1]
    length = board.get_length()
    bp, wp = 0, 0
    bk, wk = 0, 0
    bc, wc = 0, 0
    bkd, wkd = 0, 0
    bsd, wsd = 0.0, 0.0
    for row in range(length):
        for col in range(length):
            piece = board.get(row, col)
            if piece:
                r = row if row > (length - (row + 1)) else (length - (row + 1))
                c = col if col > (length - (col + 1)) else (length - (col + 1))
                d = int(((r ** 2.0 + c ** 2.0) ** 0.5) / 2.0)
                # if piece.color() != '.':
                    # print(piece.color())
                if piece.color().lower() == 'black':
                    # print("first")
                    bc += sum([len(v) for v in \
                               get_jump(board, row, col)])
                    if piece.is_king():
                        bk += 1
                    else:
                        bp += 1
                        bkd += row + 1
                        bsd += d
                elif piece.color().lower() == "white":
                    # print("second")
                    wc += sum([len(v) for v in \
                               get_jump(board, row, col)])
                    if piece.is_king():
                        wk += 1
                    else:
                        wp += 1
                        wkd += length - (row + 1)
                        wsd += d
    if turn == 'black':
        black_count_heuristics = \
                3.125 * (((bp + bk * 2.0) - (wp + wk * 2.0)) \
                    / 1.0 + ((bp + bk * 2.0) + (wp + wk * 2.0)))
        black_capture_heuristics = 1.0417 * ((bc - wc)/(1.0 + bc + wc))
        black_kingdist_heuristics = 1.429 * ((bkd - wkd)/(1.0 + bkd + wkd))
        black_safe_heuristics = 5.263 * ((bsd - wsd)/(1.0 + bsd + wsd))
        return black_count_heuristics + black_capture_heuristics \
                    + black_kingdist_heuristics + black_safe_heuristics
    else:
        white_count_heuristics = \
                3.125 * (((wp + wk * 2.0) - (bp + bk * 2.0)) \
                    / 1.0 + ((bp + bk * 2.0) + (wp + wk * 2.0)))
        white_capture_heuristics = 1.0416 * ((wc - bc)/(1.0 + bc + wc))
        white_kingdist_heuristics = 1.428 * ((wkd - bkd)/(1.0 + bkd + wkd))
        white_safe_heuristics = 5.263 * ((wsd - bsd)/(1.0 + bsd + wsd))
        return white_count_heuristics + white_capture_heuristics \
                    + white_kingdist_heuristics + white_safe_heuristics
                    
def terminal_state(state, maxdepth = None):
    board = state[0]
    turn = state[1]
    # print("isterminal")
    depth = state[2]
    (moves, captures) = get_possible_moves(board, turn)
    # print("terminal_state",move,captures)
    # print("terminal_state")
    if maxdepth is not None:
        # print((not moves) and (not captures) or depth >= maxdepth)
        return ((not moves) and (not captures)) or depth >= maxdepth
    else:
        # print("Here")
        return ((not moves) and (not captures))

def transition(state, action, ttype):
	# print("transition")
	# print(state[0])
	board = copy.deepcopy(state[0])
	turn = state[1]
	depth = state[2]
	# print("action",action)
	if ttype == "move":
		play_move(board, action)
	elif ttype == "jump":
		play_jump(board, action)
	depth += 1
	# print(depth)
	# print(board,turn,depth)
	return (board, turn, depth)

def maxvalue(state, maxdepth, alpha = None, beta = None):
    board = state[0]
    # print(board)
    turn = state[1]
    # print("maxvalue",maxdepth)
    if terminal_state(state, maxdepth):
    	# print("Here")
    	return calculate_heuristics(state)
    else:
        v = float('-inf')
        (moves, captures) = get_possible_moves(board, turn)
        # print("maxvalue",moves,captures)
        if captures:
            for a in captures:
                v = max(v, minvalue(transition(state, a, "jump"), \
                        maxdepth, alpha, beta))
                if alpha is not None and beta is not None:
                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
            return v
        elif moves:
            for a in moves:
                v = max(v, minvalue(transition(state, a, "move"), \
                        maxdepth, alpha, beta))
                if alpha is not None and beta is not None:
                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
            return v            

def minvalue(state, maxdepth, alpha = None, beta = None):
    board = state[0]
    turn = state[1]
    # print("minvalue",maxdepth)
    if terminal_state(state, maxdepth):
    	# print("terminal")
    	return calculate_heuristics(state)
    else:
    	# print("not terminal")
    	v = float('inf')
    	(moves, captures) = get_possible_moves(board, turn)
    	if captures:
            # print("here")
            for a in captures:
                v = min(v, maxvalue(transition(state, a, "jump"), maxdepth, alpha, beta))
                if alpha is not None and beta is not None:
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
            return v
    	elif moves:
            # print("moves")
            for a in moves:
                v = min(v, maxvalue(transition(state, a, "move"), maxdepth, alpha, beta))
                if alpha is not None and beta is not None:
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
            return v


def alphabeta(state, maxdepth = None):
    board = state[0]
    # print(board)
    # print("alphabeta")
    turn = state[1]
    # print(maxdepth)
    (moves, captures) = get_possible_moves(board, turn)
    # print("alphabeta",moves,captures)
    # print("alphabeta")
    # print("alphabeta",moves, captures)
    # return (moves,captures)
    alpha = float('-inf')
    beta = float('inf')
    # print(alpha,beta)
    if captures:
    	# print("here")
    	return max([(a, minvalue(transition(state, a, "jump"), \
    				maxdepth, alpha, beta)) \
    				for a in captures], key = lambda v: v[1])
    elif moves:
    	# print("Here")
    	return max([(a, minvalue(transition(state, a, "move"), \
    				maxdepth, alpha, beta)) \
    				for a in moves], key = lambda v: v[1])

    # print(moves,capture)
        # return ("pass", -1)

def make_a_move(board, turn):
    state = (board, turn, 0)
    # print(state)
    move,captures = alphabeta(state, 4)
    # print(move,captures)
    return move,captures





def pos(position):
    return (ord(position[0])-ord('a'),int(position[1:])-1)

def no_to_str(row, col):
	return chr(row+97)+str(col+1)

def initialize(board):
    row = col = board.get_length()
    # print(row,col)
    inputFile = 'input.txt'
    outputFile = 'output.txt'
    info = list()
    with open(inputFile) as f:
    	for line in f.readlines():
    		info.append(line.strip())
    for i in range(3,11):
    	# print(info[i])
    	# print()
    	k = 0
    	for j in info[i]:
    		# print(j)
    		if j == "w":
    			# print("white",Piece('white'))
    			board.place(i-3, k, Piece('white'))
    		elif j == "b":
    			# print("black",Piece('black'))
    			board.place(i-3, k, Piece('black'))

    		elif j == "B":
    			# print("King",Piece("Black"))
    			board.place(i-3, k, Piece("BLACK"))
    		elif j == "W":
    			board.place(i-3, k, Piece("WHITE"))
    		elif j == ".":
    			# print("Piece",Piece("."))
    			board.place(i-3, k, Piece("."))
    		k = k+1
    # initrows = (row // 2) - 1
    # print(initrows)
    # for r in range(row - 1, row - (initrows + 1), -1):
    #     for c in range(0 if r % 2 == 1 else 1, col, 2):
    #     	print(r,c)
    #     	board.place(r, c, Piece('white'))
    # for r in range(0, initrows):
    #     for c in range(0 if r % 2 == 1 else 1, col, 2):
    #     	print(r,c)
    #     	board.place(r, c, Piece())

def count_pieces(board):
    row = col = board.get_length()
    black, white = 0, 0
    for r in range(row):
        for c in range(col):
            piece = board.get(r, c)
            if piece:
                if piece.is_black():
                    black += 1
                if piece.is_white():
                    white += 1
    return (black, white)

def get_all_moves(board, color, is_sorted = False):
	# print("get_all_moves")
    row = col = board.get_length()
    # print("get_all_moves")
    final_list = []
    for r in range(row):
        for c in range(col):
            piece = board.get(r, c)
            # print("piece",piece,r,c)
            if piece:
            	# print("Here",piece,color.upper(),piece.color())
            	if piece.color() == color or piece.color() == color.upper():
            		# print("Here")
            		path_list = get_moves(board, r, c, is_sorted)
            		path_start = no_to_str(r, c)
            		# print(path_list)
            		for path in path_list:
            			final_list.append((path_start, path))
    # print("out")
    if is_sorted == True:
        final_list.sort()
    # print(final_list)
    return final_list

def sort_captures(all_captures,is_sorted=False):
	# print("sorted captures",sorted(all_captures, key = lambda x: (-len(x), x[0]))if is_sorted else all_captures)
	return sorted(all_captures, key = lambda x: (-len(x), x[0])) if is_sorted else all_captures

def get_all_jumps(board, color, is_sorted = False):
    row = col = board.get_length()
    final_list = []
    # print(row,col)
    for r in range(row):
        for c in range(col):
            piece = board.get(r, c)
            # print(piece,r,c)
            if piece:
            	# print(piece)
            	# print("Here2",color.upper(),piece.color())
            	if piece.color() == color or piece.color() == color.upper():
            		# print("Here3")
            		path_list = get_jump(board, r, c, is_sorted)
            		for path in path_list:
            			# print("get all captures",path)
            			final_list.append(path)
    # print("get all captures",final_list)
    # print(sort_captures(final_list, is_sorted))
    return sort_captures(final_list, is_sorted)

def play_move(board, move):
	# print(move)
    row,col = pos(move[0])
    # print(move)
    row_end,col_end = pos(move[1])
    path_list = get_moves(board, row, col, is_sorted = False)
    # print("play_move",path_list)
    
    if move[1] in path_list:
        piece = board.get(row, col)
        if piece.is_black() and row_end == board.get_length()-1 \
        or piece.is_white() and row_end == 0:
            piece.turn_king()
        board.remove(row, col)
        board.place(row_end, col_end, piece)
        # print(row_end,col_end)
        # print(board)
   

def play_jump(board, capture_path):
    counter = 0
    while counter < len(capture_path)-1:
        path = [capture_path[counter], capture_path[counter + 1]]
        counter += 1
        row,col = pos(path[0])
        row_end,col_end = pos(path[1])
        # print("Here",row,col)
        path_list = get_jumps(board, row, col, is_sorted = False)
        # print(path_list)
        if path[1] in path_list:
            piece = board.get(row, col)
            if piece.is_black() and row_end == board.get_length()-1 \
            or piece.is_white() and row_end == 0:
                piece.turn_king()
            board.remove(row, col)
            row_eat, col_eat = max(row, row_end)-1, max(col, col_end)-1
            board.remove(row_eat, col_eat)
            board.place(row_end, col_end, piece)
            # print(row_end,col_end)
            # print(board)
        # else:
        #     raise RuntimeError("Invalid jump/capture, please type" \
        #                      + " \'hints\' to get suggestions.")
            
def get_possible_moves(board, color, is_sorted = False):
	# print("get_possible_moves")
	move = get_all_moves(board, color, is_sorted)
	# print(move)
	jump = get_all_jumps(board, color, is_sorted)
	# print("get_ossible_moves",move,jump)
	if jump:
		# print('get hints jump',jump)
		return ([], jump)
	else:
		# print("get hints move")
		return (move, jump)
       

    
def play(piececolor):
    finalmove = []
    Piece.symbols = ['b', 'w', '.']
    Piece.symbols_king = ['B', 'W']
    # print(piececolor.lower())
    # print(banner)
   
    if piececolor == "BLACK":
    	opp_color = "white"
    elif piececolor == "WHITE":
    	opp_color = "black"
    (my_color, opponent_color) = (piececolor.lower(),opp_color)
    # print(my_color,opponent_color)
    
    board = Checkers(8)
    initialize(board)
    piece_count = count_pieces(board)
    # print(piece_count)
    # print("Current board:")
    # board.display(piece_count)
    turn = my_color
    
    
    move = make_a_move(board, turn)
    # print(move[0])
    if type(move[0]) == list:
        k = 0
        for i in move[0]:
            resstr = ""
            # print(tiles2[i[0]],tiles[i[1]])
            resstr = resstr + tiles[i[1]] + tiles2[i[0]]
            finalmove.append(resstr)
        # print(finalmove)
        for j in range(len(finalmove)-1):
            if k != (len(finalmove)-2):
                file.write("J" + " " + finalmove[j] + " " + finalmove[j+1] + "\n")
            else:
                file.write("J" + " " + finalmove[j] + " " + finalmove[j+1])
            k = k + 1
            # print(j)
    if type(move[0]) == tuple:
        for i in move[0]:
            resstr = ""
            resstr = resstr + tiles[i[1]] + tiles2[i[0]]
            # print(resstr)
            finalmove.append(resstr)
        # print(finalmove)
        file.write("E" + " " + finalmove[0] + " " + finalmove[1])
        # print(finalmove)
        # print(resstr)
        # print(i[0],i[1],type(i[0]),type(i[1]))
        # print(i,type(i))
def evaluate_states(self):
        t1 = time.time()
        global flg
        global a
        global b
        global first_move
        global newline
        global character
        global jump
        global first_letter
        temp = 0
        current_state = Node(deepcopy(self.matrix))

        first_moves = current_state.get_children(True, self.mandatory_jumping)
        dict = {}
        for i in range(len(first_moves)):
            child = first_moves[i]
            value = Checkers.minimax(child.get_board(), 4, -math.inf, math.inf, False, self.mandatory_jumping)
            dict[value] = child
        Checkers.EVALUATED_STATES = 0
        new_board = dict[max(dict)].get_board()
        # self.print_matrix()
        move = dict[max(dict)].move
        # print(move)
        if first_move == 1:
            first_letter = self.matrix[move[0]][move[1]][0]
            first_move = 0
            first_king = 1
        else:
            if(move[0] != a or move[1] !=b):
                temp = 2
            else:
                if abs(move[0]-move[2]) == 1 and abs(move[1]-move[3]) == 1:
                    flg = 0
                    end = time.process_time()
                    print(end-start)
                    exit()

        self.matrix = new_board
        t2 = time.time()
        diff = t2 - t1
            
        
        # print(first_letter)
        # print(move[0],move[1],move[2])
        if (abs(move[0]-move[2]) > 1 and abs(move[1]-move[3]) > 1):
            if (move[2] != 7 and move[2] !=0):
                if temp !=2:
                    flg = 1
                    a = move[2]
                    b = move[3]
            else:
                # print(character)
                if character == "B" or character == "W":
                    # print('here')
                    # print(move[0],move[1],a,b)
                    if move[0] !=a or move[1]!=b:
                        # print(first_king)
                        # if first_king == 1:
                            # flg = 0
                        # print("here")
                        if first_letter != "b" and first_letter != "w":
                            if temp !=2:
                                flg = 1
                                a = move[2]
                                b = move[3]
                        # else:
                        #     flg = 0
                    else:
                        # print("here")
                        if move[2] == 7 or move[2] == 0:
                            # if first_king == 1:
                                # flg = 0
                            # print("here1",first_letter)
                            if first_letter != "b" and first_letter != "w":
                                # print('here3')
                                if temp !=2:
                                    flg = 1
                                    a = move[2]
                                    b = move[3]
                            else:
                                # print("Here3")
                                flg = 0
                elif character == "b" or character == "w":
                    flg = 0
                    if move[0] !=a or move[1] !=b:
                        if temp !=2:
                            flg = 1
                            a = move[2]
                            b = move[3]
                    else:
                        if move[2] == 7 and move[2] == 0:
                            flg = 0
            if newline == 0:
                if temp != 2:
                    jump = 1
                    # print("J" + " " + (tiles[move[1]]+tiles2[move[0]]) + " " + (tiles[move[3]]+tiles2[move[2]]))
                    file.write("J" + " " + (tiles[move[1]]+tiles2[move[0]]) + " " + (tiles[move[3]]+tiles2[move[2]]))
                    newline = 1
            else:
                if temp !=2:
                    jump = 1
                    # print("J" + " " + (tiles[move[1]]+tiles2[move[0]]) + " " + (tiles[move[3]]+tiles2[move[2]]))
                    file.write('\n')
                    file.write("J" + " " + (tiles[move[1]]+tiles2[move[0]]) + " " + (tiles[move[3]]+tiles2[move[2]]))
        else:
            flg = 0
            if jump == 0:
                # print("E" + " " + (tiles[move[1]]+tiles2[move[0]]) + " " + (tiles[move[3]]+tiles2[move[2]]))
                file.write("E" + " " + (tiles[move[1]]+tiles2[move[0]]) + " " + (tiles[move[3]]+tiles2[move[2]]))
        


def main():
	inputFile = 'input.txt'
	info = list()
	with open(inputFile) as f:
		for line in f.readlines():
			info.append(line.strip())
	gametype = info[0]
	piececolor = info[1]
	# print(gametype,piececolor,type(piececolor))
	play(piececolor)


if __name__ == "__main__":
    main()
end_time = time.process_time()
end_time_time = time.time()
print("process_time",end_time-start_time)
print("time",end_time_time-start_time_time)