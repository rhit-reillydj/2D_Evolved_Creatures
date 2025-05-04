import sys
import pickle
import numpy as np
import pygame
import Box2D.b2 as b2

#--------------------------------------------------
# LimbNode Class (must match pickled genome definition)
#--------------------------------------------------
class LimbNode:
    def __init__(self, parent=None, attachment=(0,0), max_torque=10.0, speed=1.0, nn_weights=None):
        self.parent = parent
        self.attachment = attachment
        self.max_torque = max_torque
        self.speed = speed
        self.children = []
        
        if nn_weights is None:
            # random initialization for new nodes (not used here)
            self.nn_input_weights = np.random.randn(2, 4) * 0.5
            self.nn_hidden_weights = np.random.randn(4, 1) * 0.5
            self.nn_input_bias = np.random.randn(4) * 0.5
            self.nn_hidden_bias = np.random.randn(1) * 0.5
        else:
            # load stored weights
            (self.nn_input_weights,
             self.nn_hidden_weights,
             self.nn_input_bias,
             self.nn_hidden_bias) = nn_weights

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

#--------------------------------------------------
# Simulation Function
#--------------------------------------------------
def simulate_creature(genome, visualize=False, steps=300):
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
    if visualize:
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        clock = pygame.time.Clock()

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
            screen.fill((255, 255, 255))
            for body in world.bodies:
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [body.transform * v for v in shape.vertices]
                    pygame_vertices = [(int(v.x*30 + 400), int(300 - v.y*30)) for v in vertices]
                    pygame.draw.polygon(screen, (0,0,255), pygame_vertices)
            pygame.display.flip()
            clock.tick(60)

    final_x = root_body.position.x
    distance = final_x - initial_x
    print(f"Distance traveled: {distance:.2f}")
    return distance

#--------------------------------------------------
# Main: load and simulate a saved genome
#--------------------------------------------------
if __name__ == "__main__":
    SEED = 42
    filename = f"Creatures/best_genome_seed_{SEED}.pkl"
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
    except Exception as e:
        print(f"Error loading genome from '{filename}': {e}")
        sys.exit(1)

    simulate_creature(genome, visualize=True)