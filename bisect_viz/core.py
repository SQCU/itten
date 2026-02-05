"""
Core data structures and pure functions for bisection visualization.

All state is serializable to JSON. No GUI dependencies.
Deterministic given same inputs.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Protocol
from dataclasses import dataclass, field, asdict
import json
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CANONICAL IMPORT - one source of truth
from spectral_ops_fast import local_fiedler_vector, expand_neighborhood, DEVICE


class GraphView(Protocol):
    """Graph you can query but not enumerate."""

    def neighbors(self, node: int) -> List[int]:
        ...

    def degree(self, node: int) -> int:
        ...

    def seed_nodes(self) -> List[int]:
        ...


@dataclass
class GraphState:
    """
    Snapshot of graph exploration state.
    Tracks discovered nodes and edges.
    """

    nodes: List[int] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    # Track which nodes have been expanded (their neighbors queried)
    expanded: Set[int] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [[a, b] for a, b in self.edges],
            "positions": {str(k): list(v) for k, v in self.positions.items()},
            "expanded": list(self.expanded),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphState":
        return cls(
            nodes=d.get("nodes", []),
            edges=[tuple(e) for e in d.get("edges", [])],
            positions={int(k): tuple(v) for k, v in d.get("positions", {}).items()},
            expanded=set(d.get("expanded", [])),
        )


@dataclass
class BisectionState:
    """
    State of spectral bisection.
    """

    partition_a: Set[int] = field(default_factory=set)
    partition_b: Set[int] = field(default_factory=set)
    fiedler_values: Dict[int, float] = field(default_factory=dict)
    lambda2: float = 0.0
    cut_edges: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition_a": list(self.partition_a),
            "partition_b": list(self.partition_b),
            "fiedler_values": {str(k): v for k, v in self.fiedler_values.items()},
            "lambda2": self.lambda2,
            "cut_edges": self.cut_edges,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BisectionState":
        return cls(
            partition_a=set(d.get("partition_a", [])),
            partition_b=set(d.get("partition_b", [])),
            fiedler_values={int(k): v for k, v in d.get("fiedler_values", {}).items()},
            lambda2=d.get("lambda2", 0.0),
            cut_edges=d.get("cut_edges", 0),
        )


@dataclass
class VisualizationState:
    """
    Complete visualization state.
    Combines graph state and bisection state.
    """

    graph: GraphState = field(default_factory=GraphState)
    bisection: BisectionState = field(default_factory=BisectionState)
    frame_number: int = 0
    title: str = "Spectral Bisection"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "bisection": self.bisection.to_dict(),
            "frame_number": self.frame_number,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualizationState":
        return cls(
            graph=GraphState.from_dict(d.get("graph", {})),
            bisection=BisectionState.from_dict(d.get("bisection", {})),
            frame_number=d.get("frame_number", 0),
            title=d.get("title", "Spectral Bisection"),
        )


def grow_graph_step(
    graph: GraphView,
    state: GraphState,
    nodes_to_expand: int = 1,
) -> GraphState:
    """
    Grow graph by expanding unexpanded nodes.

    Pure function: returns new state, does not modify input.
    Note: The graph itself may be mutated by querying neighbors
    (for generative graphs like InfiniteGraph).
    """
    new_nodes = list(state.nodes)
    new_edges = list(state.edges)
    new_expanded = set(state.expanded)

    # Find nodes to expand (frontier = unexpanded nodes)
    frontier = [n for n in new_nodes if n not in new_expanded]

    if not frontier and not new_nodes:
        # Initialize from seed if completely empty
        seeds = graph.seed_nodes()
        new_nodes.extend(seeds)
        frontier = seeds

    # If no frontier but we have nodes, re-query existing nodes
    # This triggers growth in InfiniteGraph since neighbors() can spawn new nodes
    if not frontier and new_nodes:
        # Pick nodes to re-query (prefer recently expanded)
        frontier = list(new_nodes)[-nodes_to_expand:]

    # Expand nodes
    expanded_count = 0
    edge_set = set(new_edges)
    node_set = set(new_nodes)

    for node in frontier:
        if expanded_count >= nodes_to_expand:
            break

        # Query neighbors - may spawn new nodes in generative graphs
        neighbors = graph.neighbors(node)
        new_expanded.add(node)
        expanded_count += 1

        for neighbor in neighbors:
            if neighbor not in node_set:
                new_nodes.append(neighbor)
                node_set.add(neighbor)

            edge = (min(node, neighbor), max(node, neighbor))
            if edge not in edge_set:
                new_edges.append(edge)
                edge_set.add(edge)

    return GraphState(
        nodes=new_nodes,
        edges=new_edges,
        positions=dict(state.positions),  # Preserve existing positions
        expanded=new_expanded,
    )


def compute_bisection(
    graph: GraphView,
    state: GraphState,
    num_iterations: int = 30,
    hop_expansion: int = 1,
) -> BisectionState:
    """
    Compute spectral bisection of current graph state.

    Uses local Fiedler vector computation from spectral_ops.
    """
    if len(state.nodes) < 3:
        return BisectionState(
            partition_a=set(state.nodes),
            partition_b=set(),
            fiedler_values={n: 0.0 for n in state.nodes},
            lambda2=0.0,
            cut_edges=0,
        )

    # Compute Fiedler vector
    fiedler, lambda2 = local_fiedler_vector(
        graph,
        seed_nodes=state.nodes[:5],  # Use first few nodes as seeds
        num_iterations=num_iterations,
        hop_expansion=hop_expansion,
    )

    # Ensure all current nodes have Fiedler values
    for node in state.nodes:
        if node not in fiedler:
            fiedler[node] = 0.0

    # Partition by sign
    partition_a = {n for n in state.nodes if fiedler.get(n, 0) >= 0}
    partition_b = {n for n in state.nodes if fiedler.get(n, 0) < 0}

    # Count cut edges
    cut = 0
    for a, b in state.edges:
        if (a in partition_a and b in partition_b) or (a in partition_b and b in partition_a):
            cut += 1

    return BisectionState(
        partition_a=partition_a,
        partition_b=partition_b,
        fiedler_values=fiedler,
        lambda2=lambda2,
        cut_edges=cut,
    )


def state_to_json(state: VisualizationState) -> str:
    """Serialize visualization state to JSON string."""
    return json.dumps(state.to_dict(), indent=2)


def state_from_json(json_str: str) -> VisualizationState:
    """Deserialize visualization state from JSON string."""
    return VisualizationState.from_dict(json.loads(json_str))


def create_demo_state(
    seed_size: int = 10,
    growth_steps: int = 5,
    nodes_per_step: int = 3,
    rng_seed: Optional[int] = 42,
) -> Tuple[VisualizationState, Any]:
    """
    Create a demo visualization state using InfiniteGraph.

    Returns (state, graph) tuple so caller can continue growing.
    """
    from graph_view import InfiniteGraph
    from .layout import compute_layout

    # Create graph
    graph = InfiniteGraph(
        new_node_rate=0.1,
        local_connection_prob=0.3,
        seed_size=seed_size,
        rng_seed=rng_seed,
    )

    # Initialize state
    state = GraphState(
        nodes=list(range(seed_size)),
        edges=[],
        expanded=set(),
    )

    # Build initial edges
    edge_set = set()
    for node in state.nodes:
        for neighbor in graph.neighbors(node):
            if neighbor in state.nodes:
                edge = (min(node, neighbor), max(node, neighbor))
                if edge not in edge_set:
                    state.edges.append(edge)
                    edge_set.add(edge)
        state.expanded.add(node)

    # Grow graph
    for _ in range(growth_steps):
        state = grow_graph_step(graph, state, nodes_to_expand=nodes_per_step)

    # Compute layout
    state.positions = compute_layout(
        state.nodes,
        state.edges,
        method="force",
        iterations=100,
        rng_seed=rng_seed,
    )

    # Compute bisection
    bisection = compute_bisection(graph, state)

    viz_state = VisualizationState(
        graph=state,
        bisection=bisection,
        frame_number=0,
        title="Spectral Bisection Demo",
    )

    return viz_state, graph


def generate_animation_frames(
    graph: GraphView,
    initial_state: GraphState,
    num_frames: int = 20,
    nodes_per_frame: int = 2,
    recompute_bisection_every: int = 5,
    rng_seed: Optional[int] = 42,
) -> List[VisualizationState]:
    """
    Generate a sequence of frames showing graph growth and bisection.

    Returns list of VisualizationState for each frame.
    """
    from .layout import compute_layout

    frames = []
    state = initial_state

    for frame_num in range(num_frames):
        # Grow graph
        state = grow_graph_step(graph, state, nodes_to_expand=nodes_per_frame)

        # Update layout incrementally
        state.positions = compute_layout(
            state.nodes,
            state.edges,
            method="force",
            positions=state.positions,
            iterations=20,  # Fewer iterations for incremental
            rng_seed=rng_seed,
        )

        # Compute or reuse bisection
        if frame_num % recompute_bisection_every == 0 or frame_num == num_frames - 1:
            bisection = compute_bisection(graph, state)
        else:
            # Extend existing bisection to new nodes
            # Use edges from state, NOT graph.neighbors() to avoid triggering growth
            prev_bisection = frames[-1].bisection if frames else BisectionState()
            bisection = BisectionState(
                partition_a=set(prev_bisection.partition_a),
                partition_b=set(prev_bisection.partition_b),
                fiedler_values=dict(prev_bisection.fiedler_values),
                lambda2=prev_bisection.lambda2,
                cut_edges=prev_bisection.cut_edges,
            )

            # Build adjacency from known edges
            adjacency: Dict[int, Set[int]] = {n: set() for n in state.nodes}
            for a, b in state.edges:
                adjacency[a].add(b)
                adjacency[b].add(a)

            for node in state.nodes:
                if node not in bisection.partition_a and node not in bisection.partition_b:
                    # Assign new nodes based on known neighbors
                    neighbors_a = sum(
                        1 for n in adjacency.get(node, []) if n in bisection.partition_a
                    )
                    neighbors_b = sum(
                        1 for n in adjacency.get(node, []) if n in bisection.partition_b
                    )
                    if neighbors_a >= neighbors_b:
                        bisection.partition_a.add(node)
                    else:
                        bisection.partition_b.add(node)
                    bisection.fiedler_values[node] = 0.0

        viz_state = VisualizationState(
            graph=state,
            bisection=bisection,
            frame_number=frame_num,
            title=f"Frame {frame_num + 1}/{num_frames}",
        )
        frames.append(viz_state)

    return frames
