"""
Command-line interface for bisection visualizer.

Follows pipe-friendly architecture:
- stdin JSON -> processing -> stdout JSON
- Optional --render for image output
"""

import argparse
import json
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Infinite Bisection Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo and render to PNG
  python -m bisect_viz.cli --demo --render output.png

  # Run demo and output JSON state
  python -m bisect_viz.cli --demo > state.json

  # Load state from file and render
  python -m bisect_viz.cli --input state.json --render output.png

  # Generate animation frames
  python -m bisect_viz.cli --demo --animate --frames 20 --render animation.gif
""",
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--demo",
        action="store_true",
        help="Run with demo graph (InfiniteGraph)",
    )
    input_group.add_argument(
        "--input", "-i",
        type=str,
        metavar="FILE",
        help="Load state from JSON file (or - for stdin)",
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        metavar="FILE",
        help="Write state JSON to file (default: stdout)",
    )
    parser.add_argument(
        "--render", "-r",
        type=str,
        metavar="FILE",
        help="Render visualization to image file (PNG or GIF)",
    )

    # Demo/animation options
    parser.add_argument(
        "--seed-size",
        type=int,
        default=10,
        help="Initial seed graph size (default: 10)",
    )
    parser.add_argument(
        "--growth-steps",
        type=int,
        default=5,
        help="Number of graph growth steps (default: 5)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Animation options
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate animation (requires --render with .gif)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="Number of animation frames (default: 20)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for animation (default: 5)",
    )

    # Render options
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Image width in pixels (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Image height in pixels (default: 600)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show node ID labels",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Hide statistics overlay",
    )

    # Misc
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output to stderr",
    )

    args = parser.parse_args()

    def log(msg: str):
        if not args.quiet:
            print(msg, file=sys.stderr)

    # Import core modules
    from .core import (
        VisualizationState,
        state_to_json,
        state_from_json,
        create_demo_state,
        generate_animation_frames,
        GraphState,
    )
    from .render import render_frame, render_animation

    # Get or create state
    state = None
    graph = None

    if args.demo:
        log("Creating demo graph...")
        if args.animate:
            # Create initial state for animation
            from graph_view import InfiniteGraph
            from .layout import compute_layout

            graph = InfiniteGraph(
                new_node_rate=0.2,  # Higher rate for visible growth
                local_connection_prob=0.4,
                seed_size=args.seed_size,
                rng_seed=args.rng_seed,
            )

            # Initialize state with seed nodes - only partially expanded
            # to leave room for growth animation
            initial_state = GraphState(
                nodes=list(range(args.seed_size)),
                edges=[],
                expanded=set(),  # Start with none expanded
            )

            # Build initial edges from already-known adjacencies
            # without triggering new growth yet
            edge_set = set()
            node_set = set(initial_state.nodes)
            for node in list(initial_state.nodes)[:3]:  # Only expand first few
                for neighbor in graph.neighbors(node):
                    if neighbor in node_set:
                        edge = (min(node, neighbor), max(node, neighbor))
                        if edge not in edge_set:
                            initial_state.edges.append(edge)
                            edge_set.add(edge)
                    elif neighbor not in node_set:
                        # New node discovered
                        initial_state.nodes.append(neighbor)
                        node_set.add(neighbor)
                        edge = (min(node, neighbor), max(node, neighbor))
                        initial_state.edges.append(edge)
                        edge_set.add(edge)
                initial_state.expanded.add(node)

            # Initial layout
            initial_state.positions = compute_layout(
                initial_state.nodes,
                initial_state.edges,
                method="force",
                iterations=50,
                rng_seed=args.rng_seed,
            )

            log(f"Generating {args.frames} animation frames...")
            frames = generate_animation_frames(
                graph,
                initial_state,
                num_frames=args.frames,
                nodes_per_frame=2,
                recompute_bisection_every=3,
                rng_seed=args.rng_seed,
            )

            # Use last frame as current state
            state = frames[-1]
        else:
            state, graph = create_demo_state(
                seed_size=args.seed_size,
                growth_steps=args.growth_steps,
                rng_seed=args.rng_seed,
            )
            frames = None

        log(f"Graph has {len(state.graph.nodes)} nodes, {len(state.graph.edges)} edges")
        log(f"Partition A: {len(state.bisection.partition_a)}, Partition B: {len(state.bisection.partition_b)}")
        log(f"Cut edges: {state.bisection.cut_edges}, Lambda2: {state.bisection.lambda2:.4f}")

    elif args.input:
        # Load from file or stdin
        if args.input == "-":
            json_str = sys.stdin.read()
        else:
            with open(args.input, "r") as f:
                json_str = f.read()

        state = state_from_json(json_str)
        frames = None
        log(f"Loaded state with {len(state.graph.nodes)} nodes")

    else:
        # Check for stdin
        if not sys.stdin.isatty():
            json_str = sys.stdin.read()
            if json_str.strip():
                state = state_from_json(json_str)
                frames = None
                log(f"Loaded state from stdin with {len(state.graph.nodes)} nodes")

    if state is None:
        log("No input provided. Use --demo or --input, or pipe JSON to stdin.")
        parser.print_help()
        sys.exit(1)

    # Render if requested
    if args.render:
        render_kwargs = {
            "width": args.width,
            "height": args.height,
            "show_labels": args.show_labels,
            "show_stats": not args.no_stats,
        }

        if args.animate and frames:
            if not args.render.lower().endswith(".gif"):
                log("Warning: --animate requires .gif output, appending .gif")
                args.render += ".gif"

            log(f"Rendering animation to {args.render}...")
            render_animation(
                frames,
                args.render,
                fps=args.fps,
                **render_kwargs,
            )
        else:
            log(f"Rendering to {args.render}...")
            img = render_frame(state, **render_kwargs)
            img.save(args.render)

        log(f"Saved: {args.render}")

    # Output state
    json_output = state_to_json(state)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        log(f"State saved to {args.output}")
    elif not args.render:
        # Only output JSON to stdout if not rendering
        # (to avoid mixing binary/text output)
        print(json_output)


if __name__ == "__main__":
    main()
