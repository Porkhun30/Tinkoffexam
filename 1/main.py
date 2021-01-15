import csv
import os
import datetime
import collections
import copy
import itertools

csvs = os.listdir('.//')
cost_shares, short_shares = collections.defaultdict(list), []
dates, c = set(), 0
for file_csv in csvs:
    with open('.//' + file_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                date = datetime.datetime(*map(int, [row[0][:4], row[0][4:6], row[0][6:], row[1][:2], row[1][2:4]]))
                if date.date() not in dates:
                    short_shares.append((date.date(), float(row[2]), c))
                    dates.add(date.date())
                    c += 1
                cost_shares[date.date()].append((date, float(row[2])))
            except (ValueError, IndexError):
                continue

def profit(trans):
    return sum(t['s'][1] - t['b'][1] for t in trans)

def sort(t):
    return sorted(list(t), key=lambda x: x['b'][2])

def information(trans):
    profit_ = profit(trans)
    for i, t in enumerate(trans, 1):
        print(f'{i}. Buy ', t['b'][0], 'Price:', t['b'][1], '| Sell', t['s'][0], 'Price:', t['s'][1], '| Profit:', t['s'][1] - t['b'][1])
    print('Overall profit - ', profit_)

def check(ind):
    ind = list(ind)
    lenght = len(ind)
    ind.sort(key=lambda x: x[0])
    start = [x[0] for x in ind]
    end = [x[1] for x in ind]
    if lenght != len(set(start)) and lenght != len(set(end)):
        return False
    if any([i for i in range(lenght - 1) if start[i+1] - end[i] < 0]):
        return False
    return True

def first(shares, short_shares):
    short_find = find_one_best_trans(short_shares)
    res = []
    if short_find['b'] is not None or short_find['s'] is not None:
        buy = short_find['b'][0]
        sell = short_find['s'][0]
        res.append(find_one_best_trans(shares[buy] + shares[sell]))
        res[0]['b'] += (short_find['b'][2],)
        res[0]['s'] += (short_find['s'][2],)
    return res

def find_one_best_trans(shares):
    shares = copy.deepcopy(shares)
    shares.sort(key=lambda _: _[1])
    res = {'b': None, 's': None}
    max_ = 0
    for i in range(len(shares)):
        for j in range(len(shares) - 1, i, -1):
            if shares[j][1] - shares[i][1] > max_ and shares[j][0] - shares[i][0] > datetime.timedelta():
                res['b'], res['s'] = shares[i], shares[j]
                max_ = shares[j][1] - shares[i][1]
    return res

def second(shares, short_shares):
    one = first(shares, short_shares)
    start, end = one[0]['b'][2], one[0]['s'][2]
    other = []
    if start > 15:
        other.extend(first(shares, short_shares[:start]))
    if end < len(short_shares) - 15:
        other.extend(first(shares, short_shares[end:]))
    t = list(zip(one * len(other), other)) + [one]
    ind = split(short_shares, start, end)
    if ind is not None:
        t.extend([first(shares, short_shares[ind[0]: ind[1]]), first(shares, short_shares[ind[2]: ind[3]])])
    max_, res = 0, None
    for x in t:
        inc = profit(x)
        if max_ < inc:
            max_ = inc
            res = x
    return sort(res)

def split(shares, start, end):
    max_cost = shares[end][1] - shares[start][1]
    info = [x[1] for x in shares]
    index = None
    for i in range(start+10, end-10):
        l1 = info[start:i]
        l2 = info[i:end+1]
        price = max(l1) - min(l1) + max(l2) - min(l2)
        if price - 50 > max_cost:
            max_cost = price
            index = start, i-1, i, end
    return index

def third(info, short_info, *, k=1):
    one = first(info, short_info)
    start, end = one[0]['b'][2], one[0]['s'][2]
    ind = [(start, end)]
    if start > 10:
        ind.append((0, start))
    if end < len(short_info) - 10:
        ind.append((end, len(short_info)-1))
    i = 0
    while len(ind) < k + 4:
        ind_split = split(short_info, *ind[i])
        if ind is not None:
            ind.extend([ind_split[0:2], ind_split[2:]])
        i += 1
        if i == len(ind):
            break
    max_ = 0
    best = None
    if len(ind) < k:
        k = len(ind)
    for i in range(k, 2, -1):
        for t in itertools.combinations(ind, i):
            cost = []
            if check(t):
                for ind in t:
                    cost.extend(first(info, short_info[ind[0]: ind[1]]))
                p = profit(cost)
                if p > max_:
                    max_ = p
                    best = cost
    return sort(best)


if __name__ == '__main__':
    print('Input count transactions:\n1. One\n2. Two\n3. k')
    count = int(input())
    if count == 3:
        print('Input K')
        k = int(input())
    if count == 1:
        information(first(cost_shares, short_shares))
    elif count == 2:
        information(second(cost_shares, short_shares))
    elif count == 3:
        information(third(cost_shares, short_shares, k=k))
