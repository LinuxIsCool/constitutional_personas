"""
Constitutional Personas: Voronoi Embedding of Democratic Constitutions

This visualization embeds democratic constitutions as "agentic personas" in a 2D space,
where proximity indicates philosophical kinship and Voronoi cells represent each
constitution's domain of influence in idea-space.

Axes:
  X: Individual Liberty ←→ Collective Welfare
  Y: Centralized Authority ←→ Distributed Power

Each constitution is characterized by:
  - Its position in philosophical space
  - A persona "voice" reflecting its foundational ethos
  - Visual styling reflecting its character
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi
import matplotlib.colors as mcolors
from dataclasses import dataclass
from typing import List, Tuple
import colorsys

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUTIONAL PERSONA DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Constitution:
    """A constitutional persona with philosophical positioning and character."""
    name: str
    country: str
    year: int
    x: float  # Individual Liberty (-1) ←→ Collective Welfare (+1)
    y: float  # Centralized (-1) ←→ Distributed (+1)
    color_hue: float  # Base hue for the Voronoi cell (0-1)
    motto: str  # The constitution's "voice"
    traits: List[str]

CONSTITUTIONS = [
    Constitution(
        name="The Founders' Covenant",
        country="United States",
        year=1787,
        x=-0.7, y=0.8,
        color_hue=0.6,  # Blue
        motto="We hold these truths to be self-evident—that liberty is the birthright of all.",
        traits=["Individualist", "Federalist", "Rights-Bearer", "Revolutionary"]
    ),
    Constitution(
        name="The Phoenix Charter",
        country="Germany",
        year=1949,
        color_hue=0.08,  # Orange
        x=0.2, y=0.6,
        motto="From the ashes of tyranny, we build dignity as the cornerstone of all law.",
        traits=["Dignitarian", "Federal", "Social-Market", "Remembering"]
    ),
    Constitution(
        name="The Republic's Voice",
        country="France",
        year=1958,
        x=-0.2, y=-0.3,
        color_hue=0.58,  # Indigo
        motto="Liberté, égalité, fraternité—the revolution continues in law.",
        traits=["Republican", "Secular", "Unitary", "Universal"]
    ),
    Constitution(
        name="The Mosaic Compact",
        country="India",
        year=1950,
        x=0.3, y=0.7,
        color_hue=0.08,  # Saffron-adjacent
        motto="Unity in diversity—a billion voices, one constitutional song.",
        traits=["Pluralist", "Federal", "Directive", "Accommodating"]
    ),
    Constitution(
        name="The Rainbow Covenant",
        country="South Africa",
        year=1996,
        x=0.5, y=0.3,
        color_hue=0.33,  # Green
        motto="Never again. From the wound of apartheid blooms transformative justice.",
        traits=["Transformative", "Rights-Expansive", "Ubuntu", "Healing"]
    ),
    Constitution(
        name="The Pacifist's Oath",
        country="Japan",
        year=1947,
        x=0.1, y=-0.2,
        color_hue=0.95,  # Red-pink
        motto="We renounce war. In peace, we find our strength.",
        traits=["Pacifist", "Parliamentary", "Harmonious", "Restrained"]
    ),
    Constitution(
        name="The Social Pact",
        country="Sweden",
        year=1974,
        x=0.7, y=0.1,
        color_hue=0.15,  # Yellow
        motto="The welfare of the people is the highest law.",
        traits=["Social-Democratic", "Egalitarian", "Transparent", "Nordic"]
    ),
    Constitution(
        name="The Alpine Concordat",
        country="Switzerland",
        year=1999,
        x=-0.3, y=0.95,
        color_hue=0.0,  # Red
        motto="The people speak directly—every voice shapes the nation.",
        traits=["Direct-Democratic", "Cantonal", "Neutral", "Consensual"]
    ),
    Constitution(
        name="The Verdant Charter",
        country="Costa Rica",
        year=1949,
        x=0.4, y=0.0,
        color_hue=0.38,  # Teal-green
        motto="We abolished the army to feed the schools. Peace is our weapon.",
        traits=["Demilitarized", "Ecological", "Educational", "Stable"]
    ),
    Constitution(
        name="The Maple Accord",
        country="Canada",
        year=1982,
        x=0.0, y=0.5,
        color_hue=0.0,  # Red
        motto="Peace, order, and good government—rights within community.",
        traits=["Bilingual", "Federal", "Charter-Rights", "Multicultural"]
    ),
    Constitution(
        name="The Digital Republic",
        country="Estonia",
        year=1992,
        x=-0.4, y=0.2,
        color_hue=0.55,  # Cyan-blue
        motto="A nation in code—democracy at the speed of light.",
        traits=["E-Governance", "Post-Soviet", "Innovative", "Resilient"]
    ),
    Constitution(
        name="The Amazonian Covenant",
        country="Brazil",
        year=1988,
        x=0.6, y=0.4,
        color_hue=0.25,  # Yellow-green
        motto="From military rule to citizen's constitution—rights bloom in the tropics.",
        traits=["Extensive-Rights", "Federal", "Social", "Indigenous-Recognizing"]
    ),
    Constitution(
        name="The Fjord Charter",
        country="Norway",
        year=1814,
        x=0.5, y=-0.1,
        color_hue=0.6,  # Blue
        motto="Europe's oldest democracy in continuous use—stability through evolution.",
        traits=["Constitutional-Monarchy", "Oil-Fund", "Transparent", "Enduring"]
    ),
    Constitution(
        name="The Morning Calm",
        country="South Korea",
        year=1987,
        x=0.1, y=-0.4,
        color_hue=0.55,  # Blue
        motto="From dictatorship to dynamism—democracy won in the streets.",
        traits=["Democratic-Transition", "Presidential", "Economic-Rights", "Reunification-Oriented"]
    ),
    Constitution(
        name="The Emerald Covenant",
        country="Ireland",
        year=1937,
        x=0.2, y=-0.5,
        color_hue=0.35,  # Green
        motto="From colonial past to European future—identity in sovereignty.",
        traits=["Post-Colonial", "Catholic-Heritage", "European", "Evolving"]
    ),
    Constitution(
        name="The Southern Cross",
        country="Australia",
        year=1901,
        x=-0.5, y=0.6,
        color_hue=0.12,  # Golden
        motto="A federation of the far continent—pragmatism under the southern sky.",
        traits=["Federal", "Westminster", "Pragmatic", "Evolving"]
    ),
    Constitution(
        name="The Tui's Song",
        country="New Zealand",
        year=1852,  # Uncodified but constitutional acts
        x=0.3, y=-0.6,
        color_hue=0.5,  # Teal
        motto="No single document contains us—our constitution lives and breathes.",
        traits=["Uncodified", "Parliamentary", "Treaty-Based", "Progressive"]
    ),
    Constitution(
        name="The Iberian Spring",
        country="Spain",
        year=1978,
        x=0.1, y=0.3,
        color_hue=0.05,  # Orange-red
        motto="From Franco's shadow into European light—autonomy within unity.",
        traits=["Post-Authoritarian", "Autonomous-Communities", "Monarchical", "European"]
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# VORONOI GENERATION WITH BOUNDED REGIONS
# ═══════════════════════════════════════════════════════════════════════════════

def bounded_voronoi(points: np.ndarray, bounds: Tuple[float, float, float, float]) -> Tuple[Voronoi, List]:
    """
    Generate Voronoi diagram with properly bounded regions.
    Returns the Voronoi object and a list of finite polygon vertices for each point.
    """
    min_x, max_x, min_y, max_y = bounds

    # Add mirror points to bound the diagram
    mirrored = []
    for px, py in points:
        mirrored.append([2*min_x - px, py])  # Left mirror
        mirrored.append([2*max_x - px, py])  # Right mirror
        mirrored.append([px, 2*min_y - py])  # Bottom mirror
        mirrored.append([px, 2*max_y - py])  # Top mirror

    all_points = np.vstack([points, mirrored])
    vor = Voronoi(all_points)

    # Extract bounded polygons for original points only
    polygons = []
    for i in range(len(points)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            polygons.append(None)
            continue

        polygon = np.array([vor.vertices[j] for j in region])

        # Clip to bounds
        polygon[:, 0] = np.clip(polygon[:, 0], min_x, max_x)
        polygon[:, 1] = np.clip(polygon[:, 1], min_y, max_y)

        polygons.append(polygon)

    return vor, polygons


def generate_gradient_color(hue: float, lightness_range: Tuple[float, float] = (0.3, 0.7)) -> str:
    """Generate a color from hue with varied saturation and lightness."""
    saturation = 0.6 + np.random.random() * 0.3
    lightness = lightness_range[0] + np.random.random() * (lightness_range[1] - lightness_range[0])
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return mcolors.rgb2hex(rgb)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_constitutional_voronoi(
    constitutions: List[Constitution],
    figsize: Tuple[int, int] = (16, 14),
    save_path: str = None
):
    """
    Create the aesthetic Voronoi visualization of constitutional personas.
    """
    # Extract points
    points = np.array([[c.x, c.y] for c in constitutions])
    bounds = (-1.2, 1.2, -1.2, 1.2)

    # Generate bounded Voronoi
    vor, polygons = bounded_voronoi(points, bounds)

    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')

    # Draw Voronoi cells with gradient fills
    for i, (const, polygon) in enumerate(zip(constitutions, polygons)):
        if polygon is None:
            continue

        # Create gradient effect with layered polygons
        base_color = generate_gradient_color(const.color_hue, (0.25, 0.4))
        edge_color = generate_gradient_color(const.color_hue, (0.5, 0.65))

        # Main cell
        cell = Polygon(polygon, closed=True,
                      facecolor=base_color,
                      edgecolor=edge_color,
                      linewidth=2.5,
                      alpha=0.85)
        ax.add_patch(cell)

        # Inner glow effect
        if len(polygon) >= 3:
            centroid = polygon.mean(axis=0)
            inner = centroid + (polygon - centroid) * 0.7
            inner_cell = Polygon(inner, closed=True,
                                facecolor=generate_gradient_color(const.color_hue, (0.35, 0.5)),
                                edgecolor='none',
                                alpha=0.3)
            ax.add_patch(inner_cell)

    # Draw constitutional points as glowing nodes
    for const in constitutions:
        # Outer glow
        for r, a in [(0.08, 0.1), (0.05, 0.2), (0.03, 0.4)]:
            glow = plt.Circle((const.x, const.y), r,
                            color=generate_gradient_color(const.color_hue, (0.6, 0.8)),
                            alpha=a)
            ax.add_patch(glow)

        # Core point
        core = plt.Circle((const.x, const.y), 0.015,
                         color='white', alpha=0.95)
        ax.add_patch(core)

        # Country label
        ax.annotate(
            f"{const.country}\n({const.year})",
            (const.x, const.y),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='white',
            alpha=0.95,
            fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a0f', alpha=0.7, edgecolor='none')
        )

    # Axis styling
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')

    # Axis labels with philosophical dimensions
    ax.set_xlabel('← Individual Liberty                    Collective Welfare →',
                 fontsize=12, color='#888888', fontweight='bold', labelpad=15)
    ax.set_ylabel('← Centralized Power                    Distributed Power →',
                 fontsize=12, color='#888888', fontweight='bold', labelpad=15)

    # Grid lines (subtle)
    ax.axhline(y=0, color='#333333', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='#333333', linewidth=0.8, linestyle='--', alpha=0.5)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Tick styling
    ax.tick_params(colors='#555555', labelsize=9)

    # Title
    ax.set_title(
        'Constitutional Personas: Democratic Architectures in Philosophical Space',
        fontsize=16,
        fontweight='bold',
        color='white',
        pad=20,
        fontfamily='sans-serif'
    )

    # Subtitle
    ax.text(0, -1.35,
            'Each Voronoi cell represents a constitution\'s domain of philosophical influence.\n'
            'Proximity indicates kinship in democratic thought.',
            ha='center', va='top',
            fontsize=10, color='#666666', style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                   facecolor='#0a0a0f', edgecolor='none')
        print(f"Saved visualization to {save_path}")

    return fig, ax


def print_constitutional_personas(constitutions: List[Constitution]):
    """Print the persona descriptions for each constitution."""
    print("\n" + "═" * 80)
    print("  CONSTITUTIONAL PERSONAS: VOICES OF DEMOCRACY")
    print("═" * 80 + "\n")

    for const in sorted(constitutions, key=lambda c: c.year):
        print(f"┌{'─' * 76}┐")
        print(f"│  {const.name:^72}  │")
        print(f"│  {const.country} ({const.year}){' ' * (66 - len(const.country) - len(str(const.year)))}  │")
        print(f"├{'─' * 76}┤")

        # Word wrap the motto
        motto = const.motto
        while len(motto) > 72:
            split_at = motto[:72].rfind(' ')
            print(f"│  \"{motto[:split_at]}{' ' * (71 - split_at)}  │")
            motto = motto[split_at+1:]
        print(f"│  \"{motto}\"{' ' * (70 - len(motto))}  │")

        print(f"│{' ' * 76}│")
        traits_str = " · ".join(const.traits)
        print(f"│  Traits: {traits_str}{' ' * (65 - len(traits_str))}  │")
        print(f"│  Position: ({const.x:+.1f}, {const.y:+.1f}){' ' * 52}  │")
        print(f"└{'─' * 76}┘")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print persona descriptions
    print_constitutional_personas(CONSTITUTIONS)

    # Generate visualization
    fig, ax = create_constitutional_voronoi(
        CONSTITUTIONS,
        figsize=(16, 14),
        save_path="constitutional_voronoi.png"
    )

    plt.show()
