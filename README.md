# 2D Evolved Virtual Creatures

A 2D re‐implementation of Karl Sims’ **Evolved Virtual Creatures** using Box2D and neuro‐evolution.

## Overview

This project brings Sims’ classic algorithm into a 2D physics world. Creatures are represented as tree‐structured articulated bodies. Each joint is controlled by a tiny neural network, and a genetic algorithm evolves morphology and controller parameters to maximize forward locomotion.

## Features

- **2D Physics**: Powered by [Box2D](https://github.com/pybox2d/pybox2d)
- **Neuro‐evolution**: Joint motor controllers encoded as small MLPs
- **Morphological Mutation**: Attachment points, joint torques, speeds, and limb count
- **Early Stopping**: Automatic termination on fitness stagnation
- **Visualization**: Real‐time rendering via Pygame
- **Seeded Runs**: Reproducible experiments with different random seeds
- **Save/Load**: Best genome per seed persisted as a pickle file

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/2d-evolved-creatures.git
   cd 2d-evolved-creatures
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

**requirements.txt**
```
numpy
pygame
box2d-py
```

## Usage

The best genome for each run is saved to:
```
Creatures/best_genome_seed_{SEED}.pkl
```

## Project Structure

```
.
├── main.py              # CLI entry point & parameter parsing
├── sims.py              # Core classes: LimbNode, mutation, simulation, evolution
├── simulate.py          # Run simulations of best outcomes
├── requirements.txt     # Install dependencies
├── Creatures/           # Saved best‐of‐run genomes
└── README.md
```

## Example Outcomes

You’ll see a rich variety of locomotive strategies emerge:

- **Wheel‐like structures** at the front or rear  
- **Rolling creatures** that form circular bodies  
- **Bouncing/hopping** morphologies leveraging springy joints  
