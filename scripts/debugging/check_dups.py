import struct
import random

def uniform_int(rng):
    return rng.randint(1, 0xFFFFFFFE)

class MT19937:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        
    def gen(self):
        # Python's random uses MT19937. Let's just use C++ to be exact since C++ std::mt19937 is standard.
        pass
