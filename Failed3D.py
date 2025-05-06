"""
prototype_creature_evolution.py
--------------------------------
A minimal Python prototype for evolving simple articulated "creatures"
via genetic algorithms, inspired by Sims (1994). This script co-evolves:

  1. Morphology: a directed graph of 3D box segments, allowing branching.
  2. Local "brains": per-node multilayer perceptrons (MLPs) control each joint.
  3. Structural operators: add/remove nodes, subtree crossover.
  4. Parametric operators: Gaussian tweaks to all gene values.

Key sections:
  • Utility: infers number of nodes from genome length.
  • Genotype → Graph: decodes node sizes/masses and adjacency relationships.
  • Graph → Simulation: spawns rigid bodies & joints in PyBullet.
  • Control: constructs per-node MLPs from genome weights.
  • Evaluation: runs physics sim, applies torques, measures forward travel.
  • Mutation & Crossover: structural (add/remove nodes) and parametric.
  • GA Setup: DEAP population, selection, mating, mutation.
  • Main Loop: runs generations, logs best fitness & node count.

Run:
    python prototype_creature_evolution.py

Dependencies:
    pip install numpy pybullet networkx deap
"""

import random
import time
from typing import Tuple, List
import numpy as np
import pybullet as p
import pybullet_data
import networkx as nx
from dataclasses import dataclass
from deap import base, creator, tools
import copy

import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)
MAX_LINKS = 120         # Bullet hard limit is 128; leave head‑room
MAX_MASS  = 5.0         # clamp masses so Bullet’s inertial calc can’t overflow

PHYSICS_CLIENT = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# ──────────────────────────────────────────────────────────────
#  Utility: Inferring dynamic node count from genome length
# ──────────────────────────────────────────────────────────────

def get_n_nodes(genome: List[float]) -> int:
    """
    Our encoding uses 24 genes per node (4 size+mass  + 3 offset-to-parent
    + 17 controller weights) except that the **root node has no offset**.
    Genome length:  L = 4 + (N-1)*7 + 17*N  = 24*N - 3
    →  N = (L + 3) // 24
    """
    return max(1, (len(genome) + 3) // 24)

# ──────────────────────────────────────────────────────────────
#  1. GENOTYPE → MORPHOLOGY GRAPH
# ──────────────────────────────────────────────────────────────

def build_morphology_graph(genome: np.ndarray) -> nx.DiGraph:
    """
    Decode a directed graph G from the genome:
      - First 4*N genes: for each node i:
          * dx,dy,dz = abs(genome[4*i : 4*i+3]) → box half-sizes
          * mass = max(genome[4*i+3], 0.1)
      - Next section (optional) supplies adjacency flags & offsets for every unordered node pair:
          * 1 flag gene: >0 means edge i→j, <0 means j→i
          * 3 offset genes: x,y,z of child's attachment relative to parent
      - If no adjacency edges are added, fallback to a simple linear chain: i→i+1.

    Returns:
      G: networkx.DiGraph where each node has 'size', 'mass', 'brain_dims';
         each edge has 'joint_type','joint_axis','parent_offset'.
    """
    
    
    G = nx.DiGraph()
    n = get_n_nodes(genome)
    # --- decode node attributes ---
    
    idx = 0
    for nid in range(n):
        # clamp sizes to [MIN_SIZE, MAX_SIZE]
        MIN_SIZE = 0.05
        MAX_SIZE = 1.0          # half-extents in m
        raw = genome[idx:idx+4]; idx += 4
        dx, dy, dz = np.clip(np.abs(raw[:3]), MIN_SIZE, MAX_SIZE)
        # clamp mass to [0.1, MAX_MASS]
        mass = float(np.clip(raw[3], 0.1, MAX_MASS))
        G.add_node(nid, size=(dx,dy,dz), mass=mass)
        
        # ── connect to previous node with a revolute joint ─────────
        if nid > 0:
            # dynamic clamp: no parent → child offset can exceed sum of half-sizes
            parent_half = sum(G.nodes[nid-1]['size']) / 2  # Use full parent size
            child_half = sum((dx, dy, dz)) / 2  # Use full child size
            limit = parent_half + child_half
            raw_off = genome[idx:idx+3]; idx += 3
            ox, oy, oz = np.clip(raw_off, -limit * 0.5, limit * 0.5)

            G.add_edge(
                nid-1, nid,
                joint_type=p.JOINT_REVOLUTE,
                joint_axis=(0, 1, 0),
                parent_offset=(ox, oy, oz)
            )
    return G

# ──────────────────────────────────────────────────────────────
#  2. PHENOTYPE: Spawn Graph into PyBullet and Map Joints
# ──────────────────────────────────────────────────────────────

@dataclass
class CreatureIDs:
    """
    Holds simulation IDs:
      body_uid: base link ID in PyBullet
      joint_map: list mapping each link index to its controlling node ID
    """
    body_uid: int
    joint_map: List[int]


def spawn_creature(client_id: int, G: nx.DiGraph) -> CreatureIDs:
    """
    Instantiate the directed graph G as an articulated multibody:
      - Create collision & visual shapes per node (boxes)
      - For each edge (parent→child), add a link with:
         * mass = child.mass
         * joint type, axis, and attachment offset

    Returns CreatureIDs with the new body UID and joint_map.
    """
    # --- create shapes ---
    col, vis = {}, {}
    for nid, data in G.nodes(data=True):
        dx, dy, dz = data['size']; half=(dx,dy,dz)
        col[nid] = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client_id)
        vis[nid] = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                       rgbaColor=[0.6,0.6,0.8,1], physicsClientId=client_id)
    
    # --- gather link parameters ---
    root = 0
    link_masses, link_cols, link_vis = [], [], []
    link_pos, link_orn = [], []
    link_inert_pos, link_inert_orn = [], []
    parent_nodes, children = [], []
    link_joint_types, link_joint_axes = [], []
    
    for parent in nx.topological_sort(G):
        for child in G.successors(parent):
            ed = G[parent][child]
            link_masses.append(G.nodes[child]['mass'])
            link_cols.append(col[child]); link_vis.append(vis[child])
            link_pos.append(ed['parent_offset'])
            link_orn.append(p.getQuaternionFromEuler((0,0,0)))
            link_inert_pos.append((0,0,0))
            link_inert_orn.append(p.getQuaternionFromEuler((0,0,0)))
            parent_nodes.append(parent)
            children.append(child)
            # **collect joint info in lockstep** with the rest
            link_joint_types.append(ed['joint_type'])
            link_joint_axes.append(ed['joint_axis'])
    
    # --- map child→link index ---
    child_to_link = {c: i for i, c in enumerate(children)}
    link_parents = [child_to_link[p] if p != root else -1 for p in parent_nodes]
    joint_map = children.copy()
    
    # --- create multibody ---
    uid = p.createMultiBody(
        baseMass=G.nodes[root]['mass'],
        baseCollisionShapeIndex=col[root],
        baseVisualShapeIndex=vis[root],
        basePosition=(0,0,G.nodes[root]['size'][2]),
        baseOrientation=p.getQuaternionFromEuler((0,0,0)),
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_cols,
        linkVisualShapeIndices=link_vis,
        linkPositions=link_pos,
        linkOrientations=link_orn,
        linkInertialFramePositions=link_inert_pos,
        linkInertialFrameOrientations=link_inert_orn,
        linkParentIndices=link_parents,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
        physicsClientId=client_id)
    
    return CreatureIDs(body_uid=uid, joint_map=joint_map)

# ──────────────────────────────────────────────────────────────
#  3. LOCAL BRAIN: Per-node MLP Extraction
# ──────────────────────────────────────────────────────────────

def local_mlp(genome: np.ndarray, nid: int):
    """
    Extract weight matrices and biases for node nid from genome:
      - morph_genes = 7*N - 3
      - each node has 17 ctrl genes:
          * W1: 8 weights (4×2), b1: 4 biases
          * W2: 4 weights (1×4), b2: 1 bias
    Returns tuple (W1,b1,W2,b2).
    """
    n = get_n_nodes(genome)
    morph_genes = 7*n - 3          # (4 morph + 3 offset) × (n-1)  + 4 root
    base = morph_genes + nid*17

    # 17 controller genes per node
    W1 = genome[base          : base+8 ].reshape((4, 2))   # 2-input → 4-hidden
    b1 = genome[base+8        : base+12]
    W2 = genome[base+12       : base+16].reshape((1, 4))
    b2 = genome[base+16]
    return W1, b1, W2, b2

# ──────────────────────────────────────────────────────────────
#  4. FITNESS: Simulation Loop & Torque Application
# ──────────────────────────────────────────────────────────────

SIM_DT = 1/240
SIM_TIME = 2.0
GRAV    = -9.81
MAX_T   = 50.0

def evaluate(ind) -> Tuple[float]:
    """
    Given an Individual (list of genes), decode morphology+brains,
    run a PyBullet sim, apply torques from local MLPs each step,
    and return forward displacement of the base as fitness.
    """
    
    n = get_n_nodes(ind)
    if n-1 > MAX_LINKS:            # n‑1 child links
        # Huge penalty so GA quickly abandons oversize genomes
        return (-1e6,)
    
    genome = np.array(ind, dtype=np.float32)
    # ─┤ DEBUG: dump genome before anything else ├───────────────────────
    print(f"[DEBUG] Evaluating genome (len={len(ind)}): {ind}")
    # decode graph & spawn
    G = build_morphology_graph(genome)
    
    CLIENT_ID = PHYSICS_CLIENT
    p.resetSimulation(physicsClientId=CLIENT_ID)
    p.setGravity(0, 0, GRAV, physicsClientId=CLIENT_ID)
    p.loadURDF("plane.urdf", physicsClientId=CLIENT_ID)

    # ── sandbox the native call so we never hit heap-corruption directly ──
    try:
        cr = spawn_creature(CLIENT_ID, G)
    except Exception as e:
        print(f"[DEBUG] spawn_creature failed: {e}")
        return (-1e6,)

    # initialize contact sensors: 6 faces per node
    contact_sensors = {nid: [-1.0]*6 for nid in range(get_n_nodes(ind))}
    
    # Ensure joint_map aligns with actual joints
    num_joints = p.getNumJoints(cr.body_uid, physicsClientId=CLIENT_ID)
    if len(cr.joint_map) > num_joints:
        # Trim any excess mappings
        cr.joint_map = cr.joint_map[:num_joints]
    elif len(cr.joint_map) < num_joints:
        # Pad with root node for missing joints
        cr.joint_map += [0] * (num_joints - len(cr.joint_map))
    
    # build per-node torque functions
    ctrl_funcs = {}
    for nid in range(n):
        W1, b1, W2, b2 = local_mlp(genome, nid)
        ctrl_funcs[nid] = lambda x, W1=W1, b1=b1, W2=W2, b2=b2: float(
            W2.dot(np.tanh(W1.dot(x)+b1)) + b2
        )
    
    # simulate
    for _ in range(int(SIM_TIME/SIM_DT)):
        # reset sensors each step
        for arr in contact_sensors.values():
            for i in range(6): arr[i] = -1.0
        # gather contacts
        
        for pt in p.getContactPoints(bodyA=cr.body_uid, physicsClientId=CLIENT_ID):
            link = pt[3]
            # skip the base (link == -1) or anything out of range
            if link < 0 or link >= num_joints:
                continue
            normB = pt[7]
            # now safe to query link‐frame
            ls = p.getLinkState(cr.body_uid, link, physicsClientId=CLIENT_ID)
            inv_pos, inv_ori = p.invertTransform(ls[0], ls[1])
            local_norm, _ = p.multiplyTransforms((0,0,0), inv_ori, normB, (0,0,0,1))
            # determine face index
            axis = max(range(3), key=lambda i: abs(local_norm[i]))
            sign = 0 if local_norm[axis] < 0 else 1
            face = axis*2 + sign
            sensor_node = cr.joint_map[link]
            contact_sensors[sensor_node][face] = 1.0
        
        for jidx, nid in enumerate(cr.joint_map):
            ang, vel, *_ = p.getJointState(cr.body_uid, jidx, physicsClientId=CLIENT_ID)
            # construct input vector: angle, velocity, then 6 contact sensors
            x = np.array([ang, vel], dtype=np.float32)  # 2-element input
            ang, vel, *_ = p.getJointState(cr.body_uid, jidx, physicsClientId=CLIENT_ID)
            torque = ctrl_funcs[nid](x)
            torque = np.clip(torque, -MAX_T, MAX_T)
            p.setJointMotorControl2(
                cr.body_uid, jidx, p.TORQUE_CONTROL, force=torque,
                physicsClientId=CLIENT_ID
            )
        p.stepSimulation(physicsClientId=CLIENT_ID)
    
    # compute fitness = +X displacement
    pos, _ = p.getBasePositionAndOrientation(cr.body_uid, physicsClientId=CLIENT_ID)
    return (pos[0],)

# ──────────────────────────────────────────────────────────────
#  5. MUTATION & CROSSOVER OPERATORS
# ──────────────────────────────────────────────────────────────

def param_mutation(ind, indpb=0.2, sigma=0.1):
    """
    Gaussian mutation on every gene with independent probability indpb.
    """
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] += random.gauss(0, sigma)
    return (ind,)

def add_node_mutation(ind):
    """
    Append a new node to the genome:
      - 4 morph genes: size (0.1,0.1,0.1) + mass 0.5
      - 3 offset genes: (0,0,0.2)
      - 17 ctrl genes: zeros
    """
    ind.extend([0.1,0.1,0.1,0.5] + [0,0,0.2] + [0.0]*17)
    return (ind,)

def remove_node_mutation(ind):
    """
    Remove the last node if more than one remains.
    Deletes 4 node genes + 3 edge genes + 17 ctrl genes.
    """
    n = get_n_nodes(ind)
    if n <= 1:
        return (ind,)
    remove_count = (4 + 3) + 17
    del ind[-remove_count:]
    return (ind,)

def subtree_crossover(ind1, ind2):
    """
    Swap tail gene blocks at random cutpoints in two parents.
    Ensures both morphology and ctrl genes of subtrees are exchanged.
    """
    # skip crossover if either parent has fewer than 2 nodes
    n1 = get_n_nodes(ind1)
    n2 = get_n_nodes(ind2)
    if n1 < 2 or n2 < 2:
        return ind1, ind2
    
    n1 = get_n_nodes(ind1); n2 = get_n_nodes(ind2)
    cut1 = random.randint(1, n1-1); cut2 = random.randint(1, n2-1)
    blk1 = 24 * (n1 - cut1); blk2 = 24 * (n2 - cut2)
    tail1 = ind1[-blk1:]; tail2 = ind2[-blk2:]
    ind1[-blk1:] = tail2; ind2[-blk2:] = tail1
    return ind1, ind2

def structural_mutation(ind, add_prob=0.4):
    if random.random() < add_prob:
        ind, = add_node_mutation(ind)
    elif get_n_nodes(ind) > 1:            # keep at least one
        ind, = remove_node_mutation(ind)
    return ind,

def mutate(ind):
    # ❶ parametric
    ind, = param_mutation(ind, indpb=0.2, sigma=0.1)
    # ❷ structural
    ind, = structural_mutation(ind, add_prob=0.5)
    return ind,


# ──────────────────────────────────────────────────────────────
#  6. GA SETUP
# ──────────────────────────────────────────────────────────────
POP        = 30        # population size
GEN        = 20        # number of generations
INIT_NODES = 2         # starting chain length

# register DEAP classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# DEAP toolbox registration
toolbox = base.Toolbox()
toolbox.register("attr", random.uniform, -1.0, 1.0)
# initial genome length = 24*INIT_NODES - 3
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr,
    n=(24 * INIT_NODES - 3)
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", subtree_crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# ──────────────────────────────────────────────────────────────
#  7. MAIN EVOLUTION LOOP
# ──────────────────────────────────────────────────────────────
def main(verbose=True):
    random.seed(0)
    pop = toolbox.population(POP)
    # initial evaluation
    for ind in pop:
        ind.fitness.values = evaluate(ind)
    # iterate generations
    print("GEN =", GEN)
    for gen in range(1, GEN+1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]
        # crossover
        for i in range(0, len(offspring), 2):
            if random.random() < 0.5 and i+1 < len(offspring):
                toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values
        # mutation
        for ind in offspring:
            if random.random() < 0.2:
                toolbox.mutate(ind)
                del ind.fitness.values
        # evaluation
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"→ Generation {gen:02d}: evaluating {len(invalid)} individuals…", flush=True)
        for i, ind in enumerate(invalid, start=1):
            print(f"  eval {i}/{len(invalid)}…", end="\r", flush=True)
            ind.fitness.values = evaluate(ind)
        pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        if verbose:
            print(f"Gen {gen:02d} | fitness={best.fitness.values[0]:.3f} | nodes={get_n_nodes(best)}")
    best = tools.selBest(pop, 1)[0]
    if verbose:
        print(f"Done! Best fitness={best.fitness.values[0]:.3f}, nodes={get_n_nodes(best)}")
    return best

# ──────────────────────────────────────────────────────────────
#  8. VISUALIZATION: Replay Best Individual in GUI
# ──────────────────────────────────────────────────────────────
def visualize(best_genome: List[float]):
    """
    Launch PyBullet GUI, spawn the best creature, and replay its behavior.
    """
    genome = np.array(best_genome, dtype=np.float32)
    G = build_morphology_graph(genome)
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0,0,GRAV, client)
    p.loadURDF("plane.urdf", physicsClientId=client)
    cr = spawn_creature(client, G)
    # Build controllers
    num_joints = p.getNumJoints(cr.body_uid, physicsClientId=client)
    if len(cr.joint_map) > num_joints:
        cr.joint_map = cr.joint_map[:num_joints]
    elif len(cr.joint_map) < num_joints:
        cr.joint_map += [0]*(num_joints - len(cr.joint_map))

    n = get_n_nodes(best_genome)
    ctrl_funcs = {}
    for nid in range(n):
        W1, b1, W2, b2 = local_mlp(genome, nid)
        ctrl_funcs[nid] = lambda x, W1=W1, b1=b1, W2=W2, b2=b2: float(
            W2.dot(np.tanh(W1.dot(x)+b1)) + b2
        )

    # Simulate with GUI
    for _ in range(int(SIM_TIME/SIM_DT)):
        for jidx, nid in enumerate(cr.joint_map):
            if jidx >= num_joints:
                continue
            ang, vel, *_ = p.getJointState(cr.body_uid, jidx, physicsClientId=client)
            x = np.array([ang, vel], dtype=np.float32)  # 2-element input
            tq = ctrl_funcs.get(nid, lambda _x:0.0)(x)
            p.setJointMotorControl2(
                cr.body_uid, jidx, p.TORQUE_CONTROL, force=np.clip(tq, -MAX_T, MAX_T),
                physicsClientId=client
            )
        p.stepSimulation(physicsClientId=client)
        time.sleep(SIM_DT)
    # keep the GUI open until user closes
    while True:
        p.stepSimulation(physicsClientId=client)

if __name__ == '__main__':
    # best = main()
    # p.disconnect(PHYSICS_CLIENT)
    # visualize(best)
    
    crash_genome = [-0.417880961709292, -0.16042924429184935, -0.9074886461947174, -0.7355323791323869, -0.9589007587164466, -0.8939439089602296, -0.8535777012702783, -0.09798681320409312, 0.10155435527487566, 0.48175763974184393, -0.905219383027835, -0.15562250766116237, 0.27393207482344084, -0.8284771648841143, -0.11037768970759232, -0.26148792152040445, 1.0127564315158233, -0.7604459219938378, -0.18274755763370387, -0.1655490404075899, 0.45636100919935596, -0.34766244724834117, -0.8306858707814013, -0.659197935139119, -0.8860190400009931, 0.2761472564562053, -0.6532522703250623, 0.24157354865143524, 0.22501349578245944, 0.4098474214798735, 0.024237301222862495, -0.3725509329618734, 0.7549149078570558, -0.2938578365529727, -0.07240002748410646, 0.3669638115068915, 0.21846609594260766, 1.0321605508944323, 0.9094353548762442, 0.8595197012188527, 0.8681526993305162, 0.20393863423262334, -0.019595872539994064, 0.40823363476473773, -0.569160814029064, -0.6633704793782116, -0.831177858826834, 0.8552988598445779, 0.19575679456678707, 0.24102034611302203, -0.19377617442715872, -0.6998580453554228, 0.2039398258931755, -0.49505423992499265, 0.611789312035083, 0.4654379096108321, -0.9454656299089765, 0.9331954949101069, -0.9273679033466438, -0.8207613623385852, -0.45695992892052517, -0.5965867155447784, -0.5277098341667952, -0.2796014005235421, 0.49648581539394604, -0.19057727847031125, -0.46032049054914825, -0.10370062987002883, -0.3066793067077947, -0.6633704793782116, -0.831177858826834, 0.8552988598445779, 0.19575679456678707, 0.24102034611302203, -0.19377617442715872, -0.6998580453554228, 0.2039398258931755, -0.49505423992499265, 0.611789312035083, 0.4654379096108321, -0.9454656299089765, 0.9331954949101069, -0.9273679033466438, -0.8207613623385852, -0.45695992892052517, -0.5965867155447784, -0.5277098341667952, -0.2796014005235421, 0.49648581539394604, -0.19057727847031125, -0.46032049054914825, -0.10370062987002883, -0.3066793067077947, 0.05218289924395012, 0.1, 0.17588095948085852, 0.5, 0, 0, 0.2, 
                      0.0, 0.0, -0.0018713598689113617, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12618394686722265, 0.012141342780728526, 0.1205949777319794, 0.1, 0.1, 0.5, 0, 0, 0.2, 0.0, 0.0, 0.022886442171727545, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1060324885209406, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.5, 0, 0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print("Replaying crash genome:")
    print(crash_genome)
    print(evaluate(crash_genome))