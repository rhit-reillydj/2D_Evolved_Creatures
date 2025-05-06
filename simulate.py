import sys
import pickle
from EvolvedCreatures import LimbNode, simulate_creature

# 2: front wheel;                           Fitness: ~7
# 79: back wheel, 85: back wheel standing   Fitness: ~14
# 42: Roller                                Fitness: ~9
# 1 + 38: Bouncer;                          Fitness: ~120-150

SEED = 85

if __name__ == "__main__":
    filename = f"Creatures/best_genome_seed_{SEED}.pkl"
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
    except Exception as e:
        print(f"Error loading genome from '{filename}': {e}")
        sys.exit(1)

    print(f"Seed {SEED}, Fitness: {simulate_creature(genome, visualize=True)}")