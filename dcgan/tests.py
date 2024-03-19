from model import *
import torch

def print_gen():
    g = Generator(100, 64, 3)
    g.apply(weights_init)
    print(g)


def print_dis():
    d = Discriminator(3, 64)
    d.apply(weights_init)
    print(d)


if __name__ == "__main__":
    print_gen()
    print_dis()