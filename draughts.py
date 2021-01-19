#!/usr/bin/python3

from itertools import product
import scipy
from scipy.special import comb

H, W = 10,10        # height and width
R = W // 2          # number of pieces on a row
C = H // 2 - 1      # number of rows in the initial position
B = (H * W) // 2    # number of squares on the board
M = R * C           # number of pieces in the initial position

FRISIAN = 0
MIN_N = 2           # 1 when using Chinook counting, 2 otherwise
MIN_P = MIN_N // 2  # 0 when using Chinook counting, 1 otherwise
WDL, DTM = 2, 8     # number of bits per position
FORMAT = WDL        # required format to generate

def choose(n, k):
    return comb(n, k, exact=True)

def frisian(num_pawns, num_kings):
    """In Frisian draughts, a king can make at most 3 consecutive moves if the same side also has pawns."""
    return 1 + 3 * num_kings if FRISIAN and num_pawns and num_kings else 1

class db3S:
    """A database slice, further subdivided by leading pawn squares for black and white."""
    def __init__(self, n, b, w, bp, wp, bsq, wsq):
        self.n = n
        assert MIN_N <= self.n
        
        self.b = b
        self.w = w
        assert MIN_P <= self.b and MIN_P <= self.w
        assert self.b + self.w == self.n
        
        self.bp = bp
        self.wp = wp
        assert 0 <= self.bp <= self.b and 0 <= self.wp <= self.w
        self.bk = b - bp
        self.wk = w - wp

        self.bsq = bsq
        self.wsq = wsq
        assert self.bsq <= B - self.bp and self.wsq <= B - self.wp

        self.symmetry = 1 if self.b == self.w and self.bp == self.wp and self.bsq == self.wsq else 2

    def gapped(self):
        """Gapped indexing, where black and white pawns can overlap (these are illegal positions)."""
        wfactor = 1 if not self.wp else choose(B - (self.wsq + 1), self.wp - 1)
        bfactor = 1 if not self.bp else choose(B - (self.bsq + 1), self.bp - 1)
        return wfactor * bfactor * choose(B - (self.wp + self.bp), self.wk) * choose(B - (self.wp + self.bp + self.wk), self.bk) * frisian(self.wp, self.wk) * frisian(self.bp, self.bk)

class db3R:
    """A database slice, further subdivided by leading pawns ranks for black and white."""
    def __init__(self, n, b, w, bp, wp, br, wr):
        self.n = n
        assert MIN_N <= self.n
        
        self.b = b
        self.w = w
        assert MIN_P <= self.b and MIN_P <= self.w
        assert self.b + self.w == self.n
        
        self.bp = bp
        self.wp = wp
        assert 0 <= self.bp <= self.b and 0 <= self.wp <= self.w
        self.bk = b - bp
        self.wk = w - wp

        self.br = br
        self.wr = wr
        assert self.br < H and self.wr < H

        self.symmetry = 1 if self.b == self.w and self.bp == self.wp and self.br == self.wr else 2

    def gapped(self):
        """Gapped indexing, where black and white pawns can overlap (these are illegal positions)."""
        wfactor = 1 if not self.wp else choose(B - R * self.wr, self.wp) - choose(B - R * (self.wr + 1), self.wp)
        bfactor = 1 if not self.bp else choose(B - R * self.br, self.bp) - choose(B - R * (self.br + 1), self.bp)
        return wfactor * bfactor * choose(B - (self.wp + self.bp), self.wk) * choose(B - (self.wp + self.bp + self.wk), self.bk) * frisian(self.wp, self.wk) * frisian(self.bp, self.bk)

class db2:
    """A database slice: this is the largest unit that can be built independently."""
    def __init__(self, n, b, w, bp, wp):
        self.n = n
        assert MIN_N <= self.n
        
        self.b = b
        self.w = w
        assert MIN_P <= self.b and MIN_P <= self.w
        assert self.b + self.w == self.n
        
        self.bp = bp
        self.wp = wp
        assert 0 <= self.bp <= self.b and 0 <= self.wp <= self.w
        self.bk = b - bp
        self.wk = w - wp

        self.symmetry = 1 if self.b == self.w and self.bp == self.wp else 2

    def subs3R(self):
        bminr = 0 if not self.bp else 1
        bmaxr = 1 if not self.bp else H - (self.bp - 1) // R
        wminr = 0 if not self.wp else 1
        wmaxr = 1 if not self.wp else H - (self.wp - 1) // R
        return [
            (self.n, self.b, self.w, self.bp, self.wp, i_br, i_wr)
            for i_r in range(bminr + wminr, bmaxr + wmaxr - 1)
            for (i_br, i_wr) in product(range(bminr, bmaxr), range(wminr, wmaxr))
            if i_br + i_wr == i_r
        ]

    def subs3S(self):
        bminsq = 0 if not self.bp else R
        bmaxsq = 1 if not self.bp else B - (self.bp - 1)
        wminsq = 0 if not self.wp else R
        wmaxsq = 1 if not self.wp else B - (self.wp - 1)
        return [
            (self.n, self.b, self.w, self.bp, self.wp, i_bsq, i_wsq)
            for i_sq in range(bminsq + wminsq, bmaxsq + wmaxsq - 1)
            for (i_bsq, i_wsq) in product(range(bminsq, bmaxsq), range(wminsq, wmaxsq))
            if i_bsq + i_wsq == i_sq
        ]

    def gapped(self):
        """Gapped indexing, where black and white pawns can overlap (these are illegal positions)."""
        return choose(B - R, self.wp) * choose(B - R, self.bp) * choose(B - (self.wp + self.bp), self.wk) * choose(B - (self.wp + self.bp + self.wk), self.bk) * frisian(self.wp, self.wk) * frisian(self.bp, self.bk)
        
    def gapless(self):
        """Gapless indexing, where black and white pawns don't overlap."""
        return sum([
            choose(R, i_wp0) * choose(R, i_bp0) * choose(B - 2 * R, self.wp - i_wp0) * choose((B - 2 * R) - (self.wp - i_wp0), self.bp - i_bp0)
            for (i_wp0, i_bp0) in product(range(min(R, self.wp) + 1), range(min(R, self.bp) + 1))
        ]) * choose(B - (self.wp + self.bp), self.wk) * choose(B - (self.wp + self.bp + self.wk), self.bk) * frisian(self.wp, self.wk) * frisian(self.bp, self.bk)

class db1:
    def __init__(self, n, b, w):
        self.n = n
        assert MIN_N <= self.n

        self.b = b
        self.w = w
        assert MIN_P <= self.b and MIN_P <= self.w
        assert self.b + self.w == self.n
        
    def subs2(self):
        return [ 
            (self.n, self.b, self.w, i_bp, i_wp) 
            for i_p in range(self.n + 1)
            for (i_bp, i_wp) in product(range(self.b + 1), range(self.w + 1))
            if i_bp + i_wp == i_p          
        ]

    def subs3R(self):
        return [
            s3r
            for s2 in self.subs2()
            for s3r in db2(*s2).subs3R()
        ]

    def subs3S(self):
        return [
            s3s
            for s2 in self.subs2()
            for s3s in db2(*s2).subs3S()
        ]

    def deps1(self):
        return [
            (i_n, i_b, i_w)
            for i_n in range(MIN_N, self.n)
            for (i_b, i_w) in product(range(MIN_P, self.b + 1), range(MIN_P, self.w + 1))
            if i_b + i_w == i_n
        ]

    def deps2(self):
        return [
            s2
            for d1 in self.deps1()
            for s2 in db1(*d1).subs2()
        ]

    def deps3R(self):
        return [
            s3r 
            for d2 in self.deps2()
            for s3r in db2(*d2).subs3R()
        ]

    def deps3S(self):
        return [
            s3s 
            for d2 in self.deps2()
            for s3s in db2(*d2).subs3S()
        ]

class db0:
    def __init__(self, n):
        self.n = n
        assert MIN_N <= self.n
        
    def subs1(self):
        return [ 
            (self.n, i_b, i_w) 
            for (i_b, i_w) in product(range(MIN_P, self.n + 1 - MIN_P), range(MIN_P, self.n + 1 - MIN_P)) 
            if i_b + i_w == self.n
        ]

    def subs2(self):
        return [
            s2 
            for s1 in self.subs1()
            for s2 in db1(*s1).subs2()
        ]

    def subs3R(self):
        return [
            s3r
            for s2 in self.subs2()
            for s3r in db2(*s2).subs3R()
        ]

    def subs3S(self):
        return [
            s3s
            for s2 in self.subs2()
            for s3s in db2(*s2).subs3S()
        ]

    def deps0(self):
        return [
            i_n
            for i_n in range(MIN_N, self.n)
        ]

    def deps1(self):
        return [
            s1
            for d0 in self.deps0()
            for s1 in db0(d0).subs1()
        ]

    def deps2(self): 
        return [
            s2
            for d1 in self.deps1()
            for s2 in db1(*d1).subs2()
        ]

    def deps3R(self):
        return [
            s3r
            for d2 in self.deps2()
            for s3r in db2(*d2).subs3R()
        ]

    def deps3S(self):
        return [
            s3s
            for d2 in self.deps2()
            for s3s in db2(*d2).subs3S()
        ]

# Basic machine characteristics
year = 365 * 24 * 60 * 60
core_speed = 500e3
core_year = core_speed * year
cores = 128

#db_generate = db1(2 * M, M, M)
db_generate = db0(10)
def db_filter(t):
    return (
        (t[0] <= 7) or 
        (t[0] ==  8 and t[1] >= 3 and t[2] >= 3) or
        (t[0] ==  9 and t[1] >= 4 and t[2] >= 4 ) or 
        (t[0] == 10 and t[1] == 5 and t[2] == 5)
    )
#    return True

# targets = [
#     (t, db2(*t).gapless(), db2(*t).symmetry)
#     for t in db_generate.deps2() + db_generate.subs2()
#     if db_filter(t)
# ]

# targets = [
#     (t, db2(*t).gapped(), db2(*t).symmetry)
#     for t in db_generate.deps2() + db_generate.subs2()
#     if db_filter(t)
# ]

targets = [
    (t, db3S(*t).gapped(), db3S(*t).symmetry)
    for t in db_generate.deps3S() + db_generate.subs3S()
    if db_filter(t) 
]

# targets = [
#     (t, db3S(*t).gapped(), db3S(*t).symmetry)
#     for t in db_generate.deps3S() + db_generate.subs3S()
#     if db_filter(t) 
# ]

targets = list(sorted(targets, key=lambda t: t[1]))
for t in targets:
    print("%s = %s" % (str(t[0]), "{:,}".format(t[1]).rjust(15)))
num_positions = sum(list(map(lambda t: t[1], targets))) 
print("Total number of positions: %s" % "{:,}".format(num_positions))
num_subdivisions = len(targets)
print("Total number of subdivisions: %s" % "{:,}".format(num_subdivisions))
avg_size = (sum(list(map(lambda t: t[1] * t[2], targets))) // num_subdivisions) // ((8 // FORMAT) * 2**20)
print("Average memory per subdivision (MiB): %s" % "{:,}".format(avg_size))
max_size = max(list(map(lambda t: t[1] * t[2], targets))) // ((8 // FORMAT) * 2**20)
print("Maximum memory per subdivision (MiB): %s" % "{:,}".format(max_size))
time = num_positions * 365 / (cores * core_year)
print("Estimated time to build (days): %s" % "{:,}".format(time))
