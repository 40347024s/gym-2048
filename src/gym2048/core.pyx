# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.stdint cimport uint16_t, uint64_t

cdef inline uint64_t xorshift64star(uint64_t x) nogil:
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    return x * <uint64_t>2685821657736338717

cdef inline int move_line(uint16_t* line, uint16_t* out, int* reward) nogil:
    cdef uint16_t tmp0, tmp1, tmp2, tmp3
    cdef uint16_t tmp[4]
    cdef int t = 0
    cdef int i = 0
    cdef int out_i = 0

    tmp0 = line[0]
    tmp1 = line[1]
    tmp2 = line[2]
    tmp3 = line[3]

    if tmp0 != 0:
        tmp[t] = tmp0
        t += 1
    if tmp1 != 0:
        tmp[t] = tmp1
        t += 1
    if tmp2 != 0:
        tmp[t] = tmp2
        t += 1
    if tmp3 != 0:
        tmp[t] = tmp3
        t += 1

    while i < t:
        if i + 1 < t and tmp[i] == tmp[i + 1]:
            out[out_i] = <uint16_t>(tmp[i] * 2)
            reward[0] += out[out_i]
            i += 2
        else:
            out[out_i] = tmp[i]
            i += 1
        out_i += 1

    while out_i < 4:
        out[out_i] = 0
        out_i += 1

    if out[0] != line[0] or out[1] != line[1] or out[2] != line[2] or out[3] != line[3]:
        return 1
    return 0

cdef class Board:
    cdef np.ndarray board
    cdef uint64_t rng_state

    def __cinit__(self):
        self.board = np.zeros(16, dtype=np.uint16)
        self.rng_state = <uint64_t>0x9e3779b97f4a7c15

    cdef inline uint64_t _rand_u64(self) nogil:
        self.rng_state = xorshift64star(self.rng_state)
        return self.rng_state

    cdef inline void _seed(self, uint64_t seed) nogil:
        if seed == 0:
            self.rng_state ^= <uint64_t>0x9e3779b97f4a7c15
        else:
            self.rng_state = seed

    cpdef reset(self, uint64_t seed=0):
        self.board.fill(0)
        with nogil:
            self._seed(seed)
        self._add_random_tile()
        self._add_random_tile()

    cdef int _add_random_tile(self) except -1:
        cdef uint16_t[:] b = self.board
        cdef int empty = 0
        cdef int i
        for i in range(16):
            if b[i] == 0:
                empty += 1

        if empty == 0:
            return 0

        cdef uint64_t r
        with nogil:
            r = self._rand_u64()
        cdef int target = <int>(r % empty)

        for i in range(16):
            if b[i] == 0:
                if target == 0:
                    break
                target -= 1

        with nogil:
            r = self._rand_u64()
        if r % 10 == 0:
            b[i] = 4
        else:
            b[i] = 2
        return 1

    cpdef tuple step(self, int action):
        cdef int reward = 0
        cdef int moved = self._move(action, &reward)
        if moved:
            self._add_random_tile()
        cdef int done = self._is_terminal()
        return reward, done, moved

    cdef int _move(self, int action, int* reward) except -1:
        cdef uint16_t[:] b = self.board
        cdef uint16_t line[4]
        cdef uint16_t out[4]
        cdef int moved_any = 0
        cdef int r, c
        cdef int moved

        if action < 0 or action > 3:
            raise ValueError("action must be 0=up,1=down,2=left,3=right")

        if action == 2:  # left
            for r in range(4):
                line[0] = b[r * 4 + 0]
                line[1] = b[r * 4 + 1]
                line[2] = b[r * 4 + 2]
                line[3] = b[r * 4 + 3]
                moved = move_line(&line[0], &out[0], reward)
                if moved:
                    b[r * 4 + 0] = out[0]
                    b[r * 4 + 1] = out[1]
                    b[r * 4 + 2] = out[2]
                    b[r * 4 + 3] = out[3]
                    moved_any = 1

        elif action == 3:  # right
            for r in range(4):
                line[0] = b[r * 4 + 3]
                line[1] = b[r * 4 + 2]
                line[2] = b[r * 4 + 1]
                line[3] = b[r * 4 + 0]
                moved = move_line(&line[0], &out[0], reward)
                if moved:
                    b[r * 4 + 3] = out[0]
                    b[r * 4 + 2] = out[1]
                    b[r * 4 + 1] = out[2]
                    b[r * 4 + 0] = out[3]
                    moved_any = 1

        elif action == 0:  # up
            for c in range(4):
                line[0] = b[0 * 4 + c]
                line[1] = b[1 * 4 + c]
                line[2] = b[2 * 4 + c]
                line[3] = b[3 * 4 + c]
                moved = move_line(&line[0], &out[0], reward)
                if moved:
                    b[0 * 4 + c] = out[0]
                    b[1 * 4 + c] = out[1]
                    b[2 * 4 + c] = out[2]
                    b[3 * 4 + c] = out[3]
                    moved_any = 1

        else:  # down
            for c in range(4):
                line[0] = b[3 * 4 + c]
                line[1] = b[2 * 4 + c]
                line[2] = b[1 * 4 + c]
                line[3] = b[0 * 4 + c]
                moved = move_line(&line[0], &out[0], reward)
                if moved:
                    b[3 * 4 + c] = out[0]
                    b[2 * 4 + c] = out[1]
                    b[1 * 4 + c] = out[2]
                    b[0 * 4 + c] = out[3]
                    moved_any = 1

        return moved_any

    cdef int _is_terminal(self) nogil:
        cdef uint16_t[:] b = self.board
        cdef int i
        for i in range(16):
            if b[i] == 0:
                return 0

        cdef int r, c
        for r in range(4):
            for c in range(4):
                if c < 3 and b[r * 4 + c] == b[r * 4 + c + 1]:
                    return 0
                if r < 3 and b[r * 4 + c] == b[(r + 1) * 4 + c]:
                    return 0
        return 1

    cpdef np.ndarray get_board_view(self):
        return self.board.reshape((4, 4))

    cpdef np.ndarray copy_board(self):
        return self.board.reshape((4, 4)).copy()

    cpdef set_board(self, np.ndarray board_in):
        cdef np.ndarray arr = np.asarray(board_in, dtype=np.uint16)
        if arr.size != 16:
            raise ValueError("board must have 16 elements")
        self.board[:] = arr.reshape(16)
