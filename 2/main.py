import os
import pickle
import random
from itertools import product

class Generation:
    def gen(self, n, c_k_n):
        self.table = [
            [int((i*n + i/n + j) % (n*n) + 1) for j in range(n*n)]
            for i in range(n*n)]
        Generation.mix(self)
        flook = [
            [0 for j in range(self.size*self.size)]
            for i in range(self.size*self.size)]
        iterator = 0
        difficult = self.size ** 4
        while iterator < self.size ** 4 and difficult > c_k_n:
            i, j = random.randrange(0, self.size*self.size, 1),\
                random.randrange(0, self.size*self.size, 1)
            if flook[i][j] == 0:
                iterator += 1
                flook[i][j] = 1
                temp = self.table[i][j]
                self.table[i][j] = 0
                difficult -= 1
                table_solution = []
                for copy_i in range(0, self.size*self.size):
                    table_solution.append(self.table[copy_i][:])
                i_solution = sum(1 for _ in find_solve_sudoku(
                    (self.size, self.size), table_solution))
                if i_solution != 1:
                    self.table[i][j] = temp
                    difficult += 1
    def transposing(self):
        self.table = list(map(list, zip(*self.table)))
    def swap_rows(self):
        area = random.randrange(0, self.size, 1)
        line1 = random.randrange(0, self.size, 1)
        N1 = area*self.size + line1
        line2 = random.randrange(0, self.size, 1)
        while line1 == line2:
            line2 = random.randrange(0, self.size, 1)
        N2 = area*self.size + line2
        self.table[N1], self.table[N2] = self.table[N2], self.table[N1]
    def swap_colums(self):
        Generation.transposing(self)
        Generation.swap_rows(self)
        Generation.transposing(self)
    def swap_rows_area(self):
        area1 = random.randrange(0, self.size, 1)
        area2 = random.randrange(0, self.size, 1)
        while area1 == area2:
            area2 = random.randrange(0, self.size, 1)
        for i in range(0, self.size):
            N1, N2 = area1*self.size + i, area2*self.size + i
            self.table[N1], self.table[N2] = self.table[N2], self.table[N1]
    def swap_colums_small(self):
        Generation.transposing(self)
        Generation.swap_rows_area(self)
        Generation.transposing(self)
    def mix(self):
        f = [
            'Generation.transposing(self)',
            'Generation.swap_rows(self)',
            'Generation.swap_colums(self)',
            'Generation.swap_rows_area(self)',
            'Generation.swap_colums_small(self)'
        ]
        for _ in range(random.randint(10, 50)):
            exec(f[random.randint(0, 4)])


class Sudoku:
    def __init__(self, *, n=3, count=40):
        self.size = n
        self.count = count
        Generation.gen(self, n, count)
    def show(self):
        for line in self.table:
            for elem in line:
                print(' '+(str(elem) if elem != 0 else '*').center(5), end='')
            print(' ')
    def check(self, index):
        return 0 <= index < self.size*2
    def input(self, row, col, value, trig=False):
        if (self.table[row][col] != 0 or not self.check(row) and not self.check(col)) and not trig:
            raise ValueError
        self.table[row][col] = value
    def save(self, path):
        with open(path+'.pkl', 'wb') as f:
            pickle.dump(self, f)
    def end(self):
        s = sum(1 for i in range(self.size**2)
        for j in range(self.size**2) if self.table[i][j] != 0)
        return s == self.size ** 4

def find_solve_sudoku(size, grid):
    R, C = size
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])
    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C)
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))
    for solution in solve(X, Y, []):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid

def exact_cover(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y

def solve(X, Y, solution):
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

def find_near_unknown(size, table):
    for i, j in product(range(size**2), range(size**2)):
        if table[i][j] == 0:
            return i, j

def set_(sudoku, size, i, j):
    k, m = i // size, j // size
    l = list(range(1, 10))
    for _ in range(size**2):
        if sudoku[_][j] != 0 and sudoku[_][j] in l:
            l.remove(sudoku[_][j])
        if sudoku[i][_] != 0 and sudoku[i][_] in l:
            l.remove(sudoku[i][_])
    for i, j in product(range(size), range(size)):
        if sudoku[k*size+i][m*size+j] != 0 and sudoku[k*size+i][m*size+j] in l:
            l.remove(sudoku[k*size+i][m*size+j])
    return l

def solving_sudoku(size, sudoku):
    solve, ind, replace, d = [], 0, False, {}
    while sum(1 for line in sudoku for _ in line if _ != 0) != size**4:
        if not replace:
            i, j = find_near_unknown(size, sudoku)
            if (i, j) not in d.keys():
                d[(i, j)] = ind
                solve.append(((i, j), set_(sudoku, size, i, j)))
            else:
                solve[d[(i, j)]] = (i, j), set_(sudoku, size, i, j)
        else:
            try:
                solve[ind][1].pop(0)
            except IndexError:
                continue
            replace = False
        if not len(solve[ind][1]):
            sudoku[solve[ind][0][0]][solve[ind][0][1]], replace = 0, True
            ind -= 1
            continue
        sudoku[solve[ind][0][0]][solve[ind][0][1]] = solve[ind][1][0]
        yield solve[ind][0], solve[ind][1][0]
        ind += 1

path = './/saves//'
if not os.path.exists('.//saves'):
    os.mkdir('.//saves')

print('Type:', 'load <file name>', 'or', 'start game', sep='\n')
c = input().strip().lower().split()
if c == ['start', 'game']:
    print('Input count known numbers')
    count_known_number = int(input().strip())
    game = Sudoku(count=count_known_number)
elif c[0] == 'load' and os.path.exists(path+c[1]+'.pkl'):
    with open(path+c[1]+'.pkl', 'rb') as file:
        game = pickle.load(file)
print('Select game type:', '1 (you play)', '2 (computer play)', sep='\n')
input_game_type = int(input().strip())
if input_game_type == 1:
    playing = True
elif input_game_type == 2:
    playing = False

def command():
    print('Print command:')
    i = input().strip().split()
    if i[0] == 'save':
        return None, i[1], None
    elif len(i) == 3 and all(map(str.isdigit, i)):
        return int(i[0]), int(i[1]), int(i[2])
    raise ValueError

if playing:
    game.show()
    print('Input', 'save name_file', 'Or 3 number (row) (column) (value)',sep='\n')
    while not game.end():
        try:
            i = command()
            if i[0] is None:
                game.save(path+i[1])
                break
            else:
                game.input(*i)
                game.show()
        except ValueError:
            print('\tType again')
            continue
else:
    game.show()
    for (i, j), value in solving_sudoku(game.size, game.table):
        try:
            game.input(i, j, value, trig=True)
            print()
            game.show()
        except ValueError:
            pass
