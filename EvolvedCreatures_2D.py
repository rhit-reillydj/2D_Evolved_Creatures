import numpy as np
import random
import pygame
import pickle
import Box2D.b2 as b2


SEED = 0  # Placeholder for the seed, to be set in the main function



class LimbNode:
    def __init__(self, parent=None, attachment=(0,0), max_torque=10.0, speed=1.0, nn_weights=None):
        self.parent = parent
        self.attachment = attachment
        self.max_torque = max_torque
        self.speed = speed
        self.children = []
        
        if nn_weights is None:
            self.nn_input_weights = np.random.randn(2, 4) * 0.5
            self.nn_hidden_weights = np.random.randn(4, 1) * 0.5
            self.nn_input_bias = np.random.randn(4) * 0.5
            self.nn_hidden_bias = np.random.randn(1) * 0.5
        else:
            self.nn_input_weights, self.nn_hidden_weights, self.nn_input_bias, self.nn_hidden_bias = nn_weights

    def add_child(self, child):
        self.children.append(child)

    def copy(self, parent=None):
        new_node = LimbNode(
            parent=parent,
            attachment=self.attachment,
            max_torque=self.max_torque,
            speed=self.speed,
            nn_weights=(
                self.nn_input_weights.copy(),
                self.nn_hidden_weights.copy(),
                self.nn_input_bias.copy(),
                self.nn_hidden_bias.copy()
            )
        )
        new_node.children = [child.copy(parent=new_node) for child in self.children]
        return new_node

# Mutation functions

def mutate_attachment(node):
    node.attachment = (
        node.attachment[0] + np.random.normal(0, 0.1),
        node.attachment[1] + np.random.normal(0, 0.1)
    )

def mutate_joint_params(node):
    node.max_torque = max(0.1, node.max_torque + np.random.normal(0, 1.0))
    node.speed = max(0.1, node.speed + np.random.normal(0, 0.1))

def mutate_nn_weights(node):
    node.nn_input_weights += np.random.normal(0, 0.1, node.nn_input_weights.shape)
    node.nn_hidden_weights += np.random.normal(0, 0.1, node.nn_hidden_weights.shape)
    node.nn_input_bias += np.random.normal(0, 0.1, node.nn_input_bias.shape)
    node.nn_hidden_bias += np.random.normal(0, 0.1, node.nn_hidden_bias.shape)

def mutate_genome(root, mutation_rate=0.3):
    nodes = [root]
    while nodes:
        current = nodes.pop(0)
        nodes.extend(current.children)

        if np.random.rand() < mutation_rate:
            mutate_attachment(current)
        if np.random.rand() < mutation_rate:
            mutate_joint_params(current)
        if np.random.rand() < mutation_rate:
            mutate_nn_weights(current)
        if np.random.rand() < mutation_rate/2 and len(current.children) < 3:
            current.add_child(LimbNode(parent=current))
        if np.random.rand() < mutation_rate/2 and current.children:
            current.children.pop(np.random.randint(len(current.children)))

#--------------------------------------------------
# Simulation Function
#--------------------------------------------------
def simulate_creature(genome, visualize=False, steps=700):
    world = b2.world(gravity=(0, -9.8))
    ground = world.CreateStaticBody(position=(0, 0))
    ground.CreatePolygonFixture(box=(50, 1))

    bodies = {}
    joints = []

    root_body = world.CreateDynamicBody(position=(0, 5))
    root_body.CreatePolygonFixture(box=(0.5, 0.5), density=1, friction=0.3)
    bodies[genome] = root_body

    def build_limb(node, parent_body):
        for child in node.children:
            child_body = world.CreateDynamicBody(
                position=parent_body.position + b2.vec2(*child.attachment)
            )
            child_body.CreatePolygonFixture(box=(0.4, 0.4), density=1, friction=0.3)
            bodies[child] = child_body

            joint = world.CreateRevoluteJoint(
                bodyA=parent_body,
                bodyB=child_body,
                anchor=parent_body.position + b2.vec2(*child.attachment),
                enableMotor=True,
                maxMotorTorque=child.max_torque,
                motorSpeed=0
            )
            joints.append((joint, child))
            build_limb(child, child_body)

    build_limb(genome, root_body)

    initial_x = root_body.position.x
    #----------------------
    # Visualization Setup
    #----------------------
    if visualize:
        pygame.init()
        screen_width, screen_height = 800, 600
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()

    # Simulation Loop
    for _ in range(steps):
        for joint, child in joints:
            angle = joint.angle
            speed = joint.speed
            nn_input = np.array([angle, speed])
            hidden = np.tanh(nn_input @ child.nn_input_weights + child.nn_input_bias)
            output = np.tanh(hidden @ child.nn_hidden_weights + child.nn_hidden_bias)
            joint.motorSpeed = float(output[0]) * child.speed
            joint.maxMotorTorque = child.max_torque
        world.Step(0.016, 6, 2)

        if visualize:
            # Calculate camera offset so creature is centered
            offset_x = root_body.position.x * 30 - screen_width / 2
            
            # Clear screen
            screen.fill((255, 255, 255))

            # Draw background grid lines for speed reference
            for meter in range(-100, 200):
                x_px = int(meter * 30 - offset_x)
                if 0 <= x_px <= screen_width:
                    pygame.draw.line(screen, (200, 200, 200), (x_px, 0), (x_px, screen_height))
            # Draw darker lines every 5 meters
            for meter in range(-100, 200):
                if meter % 5 == 0:
                    x_px = int(meter * 30 - offset_x)
                    if 0 <= x_px <= screen_width:
                        pygame.draw.line(screen, (150, 150, 150), (x_px, 0), (x_px, screen_height), 2)

            # Draw bodies with camera follow
            for body in world.bodies:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [body.transform * v for v in shape.vertices]
                    pygame_vertices = [
                        (
                            int(v.x * 30 - offset_x),
                            int(screen_height / 2 - v.y * 30)
                        )
                        for v in vertices
                    ]
                    pygame.draw.polygon(screen, (0, 0, 255), pygame_vertices)

            # Update display
            pygame.display.flip()
            clock.tick(60)

    final_x = root_body.position.x
    distance = final_x - initial_x
    return distance

#--------------------------------------------------
# Evolutionary Loop with Stagnation Check
#--------------------------------------------------
def create_initial_genome():
    root = LimbNode()
    root.add_child(LimbNode(parent=root))
    return root


def evolve(pop_size=150, generations=100, stagnation_thresh=0.01, stagnation_gens=10):
    population = [create_initial_genome() for _ in range(pop_size)]
    best_history = []
    stagnant_count = 0

    for gen in range(generations):
        fitness = [simulate_creature(ind) for ind in population]
        best_fitness = max(fitness)
        best_history.append(best_fitness)
        print(f"Generation {gen}: Best fitness {best_fitness:.2f}")

        # Check for stagnation
        if len(best_history) > 1:
            improvement = best_history[-1] - best_history[-2]
            if improvement < stagnation_thresh:
                stagnant_count += 1
            else:
                stagnant_count = 0
        if stagnant_count >= stagnation_gens:
            print(f"Stagnation detected: <{stagnation_thresh} improvement over {stagnation_gens} gens. Stopping early.")
            break

        # Selection & reproduction
        pairs = sorted(zip(fitness, population), key=lambda p: p[0], reverse=True)
        keep = int(pop_size * 0.3)
        next_gen = [genome.copy() for _, genome in pairs[:keep]]
        while len(next_gen) < pop_size:
            parent = random.choice(pairs[:keep])[1]
            offspring = parent.copy()
            mutate_genome(offspring)
            next_gen.append(offspring)
        population = next_gen

    # Save best
    best_genome = pairs[0][1]
    filename = f"Creatures/best_genome_seed_{SEED}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(best_genome, f)
    print(f"Saved best genome to {filename}")
    
    return best_history[-1]

if __name__ == "__main__":
    seed_fitness = {}
    for seed in range(100):
        SEED = seed
        random.seed(SEED)
        np.random.seed(SEED)
        fitness = evolve()
        print(f"Fitness for seed {SEED}: {fitness:.2f}")
        seed_fitness[SEED] = fitness
    print("Fitness for each seed:")
    for seed, fitness in seed_fitness.items():
        print(f"Seed {seed}: Fitness {fitness:.2f}")