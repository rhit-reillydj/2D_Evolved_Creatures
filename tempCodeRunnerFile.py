import sys
import pickle
import numpy as np
import pygame
import Box2D.b2 as b2
from EvolvedCreatures import LimbNode, simulate_creature


if __name__ == "__main__":
    SEED = 0
    filename = f"Creatures/best_genome_seed_{SEED}.pkl"
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
    except Exception as e:
        print(f"Error loading genome from '{filename}': {e}")
        sys.exit(1)

    simulate_creature(genome, visualize=True)