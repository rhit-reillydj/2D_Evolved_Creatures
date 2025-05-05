import sys
import pickle
from EvolvedCreatures_2D import LimbNode, simulate_creature

# 1 + 38: Timed bounce on platform
# 2: front wheel
# 3: back wheel
# 42: Roller

SEED = 38

if __name__ == "__main__":
    filename = f"Creatures/best_genome_seed_{SEED}.pkl"
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
    except Exception as e:
        print(f"Error loading genome from '{filename}': {e}")
        sys.exit(1)

    print(f"Fitness: {simulate_creature(genome, visualize=True)}")