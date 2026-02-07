#!/usr/bin/env python3
"""
Test that lattice extrusion is spectrally-determined with theta interactions.

This test verifies that:
1. Lattice type distribution changes with theta
2. Node values show theta dependence
3. Not all geometry_type == "square"

Key insight: theta controls the spectral emphasis in lattice type selection.
- theta=0.1: expansion-dominant (high expansion -> hex in open regions)
- theta=0.5: balanced between expansion and gradient
- theta=0.9: gradient-dominant (high gradient -> triangle at bottlenecks)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

from lattice.patterns import create_islands_and_bridges, TerritoryGraph
from lattice.extrude import ExpansionGatedExtruder, ExtrusionState


def run_extrusion_at_theta(
    territory: TerritoryGraph,
    theta: float,
    max_iterations: int = 15,
    debug: bool = False
) -> ExtrusionState:
    """
    Run extrusion at a specific theta value.

    Args:
        territory: The TerritoryGraph to extrude
        theta: Spectral blend parameter (0=expansion-dominant, 1=gradient-dominant)
        max_iterations: Maximum extrusion iterations
        debug: If True, print debug info about spectral values

    Returns:
        ExtrusionState after extrusion completes
    """
    extruder = ExpansionGatedExtruder(
        territory=territory,
        expansion_threshold=1.0,  # Lower threshold to get more extrusions
        lanczos_iterations=15,
        hop_radius=3,
        theta=theta
    )

    state = extruder.run(max_iterations=max_iterations)

    if debug:
        # Print debug info about expansion and gradient values
        expansions = []
        gradients = []
        nodes = list(state.nodes.values())
        idx = 0
        while idx < len(nodes):
            node = nodes[idx]
            if node.layer > 0:
                expansions.append(node.expansion)
                gradients.append(node.fiedler_gradient_mag)
            idx += 1

        if expansions:
            exp_arr = np.array(expansions)
            grad_arr = np.array(gradients)
            print(f"     DEBUG - Expansion: min={exp_arr.min():.3f}, max={exp_arr.max():.3f}, mean={exp_arr.mean():.3f}")
            print(f"     DEBUG - Gradient:  min={grad_arr.min():.3f}, max={grad_arr.max():.3f}, mean={grad_arr.mean():.3f}")

    return state


def count_lattice_types(state: ExtrusionState) -> Dict[str, int]:
    """
    Count the distribution of lattice types in the extrusion state.

    Returns:
        Dict mapping lattice type to count
    """
    counts = {"square": 0, "triangle": 0, "hex": 0}

    nodes = list(state.nodes.values())
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.layer > 0:  # Only count extruded nodes
            geo_type = node.geometry_type
            counts[geo_type] = counts.get(geo_type, 0) + 1
        idx += 1

    return counts


def compute_node_value_stats(state: ExtrusionState) -> Dict[str, float]:
    """
    Compute statistics on node values for extruded nodes.

    Returns:
        Dict with mean, std, min, max of node values
    """
    values = []
    nodes = list(state.nodes.values())
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.layer > 0:  # Only count extruded nodes
            values.append(node.node_value)
        idx += 1

    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    values_arr = np.array(values)
    return {
        "mean": float(np.mean(values_arr)),
        "std": float(np.std(values_arr)),
        "min": float(np.min(values_arr)),
        "max": float(np.max(values_arr))
    }


def verify_distributions_differ(
    dist1: Dict[str, int],
    dist2: Dict[str, int],
    dist3: Dict[str, int]
) -> bool:
    """
    Verify that three lattice type distributions are meaningfully different.

    Returns True if distributions differ, False otherwise.
    """
    # Convert to normalized distributions
    total1 = sum(dist1.values()) or 1
    total2 = sum(dist2.values()) or 1
    total3 = sum(dist3.values()) or 1

    norm1 = {k: v / total1 for k, v in dist1.items()}
    norm2 = {k: v / total2 for k, v in dist2.items()}
    norm3 = {k: v / total3 for k, v in dist3.items()}

    # Compute pairwise distances
    types = ["square", "triangle", "hex"]

    def dist(a, b):
        return sum(abs(a.get(t, 0) - b.get(t, 0)) for t in types)

    d12 = dist(norm1, norm2)
    d13 = dist(norm1, norm3)
    d23 = dist(norm2, norm3)

    # At least one pair should differ meaningfully (>5% total variation)
    threshold = 0.05
    return d12 > threshold or d13 > threshold or d23 > threshold


def main():
    """Run lattice theta test."""
    print("=" * 70)
    print("LATTICE SPECTRAL DECISIONS TEST")
    print("=" * 70)
    print("\nTesting that lattice extrusion varies with theta parameter.")
    print("theta controls spectral emphasis in lattice type selection.\n")

    # Create territory with islands and bridges
    print("1. Creating TerritoryGraph with islands and bridges...")
    territory = create_islands_and_bridges(
        num_islands=4,
        island_radius=6,
        bridge_width=2,
        spacing=20
    )

    num_nodes = len(territory.all_nodes())
    print(f"   Created graph with {num_nodes} nodes")
    print(f"   Islands: {len(territory.islands)}")
    print(f"   Bridges: {len(territory.bridges)}")

    # Test at different theta values
    theta_values = [0.1, 0.5, 0.9]
    results = {}

    print("\n2. Running extrusion at different theta values...")
    print("-" * 70)

    for theta in theta_values:
        print(f"\n   theta = {theta}:")
        state = run_extrusion_at_theta(territory, theta, debug=True)

        # Count lattice types
        type_counts = count_lattice_types(state)
        total_extruded = sum(type_counts.values())

        print(f"     Total extruded: {total_extruded}")
        print(f"     Lattice types: {type_counts}")

        # Compute node value stats
        value_stats = compute_node_value_stats(state)
        print(f"     Node value mean: {value_stats['mean']:.4f}")
        print(f"     Node value std:  {value_stats['std']:.4f}")

        results[theta] = {
            "state": state,
            "type_counts": type_counts,
            "value_stats": value_stats,
            "total_extruded": total_extruded
        }

    # Verify distributions differ
    print("\n" + "=" * 70)
    print("3. VERIFICATION")
    print("=" * 70)

    dist1 = results[0.1]["type_counts"]
    dist2 = results[0.5]["type_counts"]
    dist3 = results[0.9]["type_counts"]

    distributions_differ = verify_distributions_differ(dist1, dist2, dist3)

    # Check if not all are "square"
    all_square = all(
        results[theta]["type_counts"]["triangle"] == 0 and
        results[theta]["type_counts"]["hex"] == 0
        for theta in theta_values
    )

    # Check node values vary with theta
    mean_values = [results[theta]["value_stats"]["mean"] for theta in theta_values]
    node_values_vary = max(mean_values) - min(mean_values) > 0.05

    print("\nTest Results:")
    print("-" * 70)

    # Test 1: Distributions differ
    status1 = "PASS" if distributions_differ else "FAIL"
    print(f"  [1] Lattice type distribution changes with theta: {status1}")
    if not distributions_differ:
        print("      WARNING: All theta values produced same distribution")

    # Test 2: Not all square
    status2 = "PASS" if not all_square else "FAIL"
    print(f"  [2] Not all geometry_type == 'square': {status2}")
    if all_square:
        print("      WARNING: All nodes have square geometry")

    # Test 3: Node values vary
    status3 = "PASS" if node_values_vary else "FAIL"
    print(f"  [3] Node values show theta dependence: {status3}")
    print(f"      Mean values: {[f'{v:.4f}' for v in mean_values]}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nLattice Type Distribution:")
    print("-" * 70)
    print(f"{'theta':<10} | {'square':<10} | {'triangle':<10} | {'hex':<10} | {'total':<10}")
    print("-" * 70)

    for theta in theta_values:
        counts = results[theta]["type_counts"]
        total = results[theta]["total_extruded"]
        print(f"{theta:<10.1f} | {counts['square']:<10} | {counts['triangle']:<10} | {counts['hex']:<10} | {total:<10}")

    print("-" * 70)

    print("\nNode Value Statistics:")
    print("-" * 70)
    print(f"{'theta':<10} | {'mean':<10} | {'std':<10} | {'min':<10} | {'max':<10}")
    print("-" * 70)

    for theta in theta_values:
        stats = results[theta]["value_stats"]
        print(f"{theta:<10.1f} | {stats['mean']:<10.4f} | {stats['std']:<10.4f} | {stats['min']:<10.4f} | {stats['max']:<10.4f}")

    print("-" * 70)

    # Overall result
    all_passed = distributions_differ and not all_square and node_values_vary

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL RESULT: PASS")
        print("Lattice extrusion is spectrally-determined with theta interactions.")
    else:
        print("OVERALL RESULT: PARTIAL")
        print("Some success criteria not met. See individual test results above.")
    print("=" * 70)

    return results, all_passed


def write_results_report(results: dict, all_passed: bool):
    """Write results to markdown report."""
    report_path = "/home/bigboi/itten/hypercontexts/lattice-spectral-decisions-results.md"

    theta_values = [0.1, 0.5, 0.9]

    lines = [
        "# Lattice Spectral Decisions Results",
        "",
        "## Overview",
        "",
        "This report documents the results of testing theta-dependent lattice type selection",
        "in the `ExpansionGatedExtruder`. The theta parameter controls the spectral emphasis:",
        "",
        "- **theta=0.1**: Expansion-dominant (high expansion -> hex in open regions)",
        "- **theta=0.5**: Balanced between expansion and gradient",
        "- **theta=0.9**: Gradient-dominant (high gradient -> triangle at bottlenecks)",
        "",
        "## Test Configuration",
        "",
        "- **Territory**: 4 islands connected by 3 bridges",
        "- **Island radius**: 6 nodes",
        "- **Bridge width**: 2 nodes",
        "- **Spacing**: 20 units between island centers",
        "- **Expansion threshold**: 1.0",
        "- **Lanczos iterations**: 15",
        "- **Hop radius**: 3",
        "",
        "## Results",
        "",
        "### Lattice Type Distribution",
        "",
        "| theta | square | triangle | hex | total |",
        "|-------|--------|----------|-----|-------|",
    ]

    for theta in theta_values:
        counts = results[theta]["type_counts"]
        total = results[theta]["total_extruded"]
        lines.append(f"| {theta:.1f} | {counts['square']} | {counts['triangle']} | {counts['hex']} | {total} |")

    lines.extend([
        "",
        "### Node Value Statistics",
        "",
        "| theta | mean | std | min | max |",
        "|-------|------|-----|-----|-----|",
    ])

    for theta in theta_values:
        stats = results[theta]["value_stats"]
        lines.append(f"| {theta:.1f} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")

    # Success criteria
    dist1 = results[0.1]["type_counts"]
    dist2 = results[0.5]["type_counts"]
    dist3 = results[0.9]["type_counts"]

    distributions_differ = verify_distributions_differ(dist1, dist2, dist3)

    all_square = all(
        results[theta]["type_counts"]["triangle"] == 0 and
        results[theta]["type_counts"]["hex"] == 0
        for theta in theta_values
    )

    mean_values = [results[theta]["value_stats"]["mean"] for theta in theta_values]
    node_values_vary = max(mean_values) - min(mean_values) > 0.05

    lines.extend([
        "",
        "## Success Criteria",
        "",
        f"1. **Lattice type distribution changes with theta**: {'PASS' if distributions_differ else 'FAIL'}",
        f"2. **Not all geometry_type == 'square'**: {'PASS' if not all_square else 'FAIL'}",
        f"3. **Node values show theta dependence**: {'PASS' if node_values_vary else 'FAIL'}",
        "",
        f"**Overall Result**: {'PASS' if all_passed else 'PARTIAL'}",
        "",
        "## Implementation Details",
        "",
        "### select_lattice_type() Function",
        "",
        "The `select_lattice_type()` function determines lattice geometry based on:",
        "",
        "```python",
        "def select_lattice_type(expansion, fiedler_gradient_mag, theta):",
        "    # Normalize inputs to [0, 1]",
        "    norm_expansion = clip((expansion - 0.5) / 2.5, 0, 1)",
        "    norm_gradient = clip(fiedler_gradient_mag, 0, 1)",
        "    ",
        "    # Compute spectral score",
        "    spectral_score = (1 - theta) * norm_expansion + theta * (1 - norm_gradient)",
        "    ",
        "    if spectral_score > 0.65:",
        "        return 'hex'       # Open region",
        "    elif spectral_score < 0.35:",
        "        return 'triangle'  # Bottleneck",
        "    else:",
        "        return 'square'    # Neutral",
        "```",
        "",
        "### Node Value Computation",
        "",
        "Node values for visualization are computed as theta-weighted blend:",
        "",
        "- theta < 0.5: Blend between expansion and Fiedler value",
        "- theta >= 0.5: Blend between Fiedler value and gradient magnitude",
        "",
        "This ensures visual properties change continuously with theta.",
        "",
        "## Interpretation",
        "",
    ])

    if all_passed:
        lines.extend([
            "The test demonstrates that lattice extrusion is now spectrally-determined:",
            "",
            "1. **Spectral properties drive geometry selection**: The combination of local",
            "   expansion (lambda_2) and Fiedler gradient determines lattice type.",
            "",
            "2. **Theta provides user control**: Adjusting theta changes the emphasis",
            "   between expansion-based and gradient-based geometry selection.",
            "",
            "3. **Visual properties vary**: Node values change with theta, enabling",
            "   theta-dependent visualization in downstream rendering.",
            "",
            "This satisfies the goal of making lattice extrusion spectrally-determined",
            "with theta interactions, rather than always using 'square' geometry.",
        ])
    else:
        lines.extend([
            "The test shows partial success. Some criteria were not fully met.",
            "This may indicate:",
            "",
            "1. The territory structure may not have sufficient spectral variation",
            "2. Threshold values may need tuning for the specific graph structure",
            "3. Additional spectral properties may be needed for differentiation",
        ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nReport written to: {report_path}")


if __name__ == '__main__':
    results, all_passed = main()
    write_results_report(results, all_passed)
