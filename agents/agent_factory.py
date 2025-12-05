"""
Agent Factory for Constitutional Agents

Creates and manages constitutional agent instances.
Loads persona data from the existing constitutional database
and creates fully configured agents with memory systems.
"""

import sqlite3
import os
import sys
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base_agent import ConstitutionalAgent, ConstitutionalPersona, AgentConfig


# Persona data matching the original constitutional_voronoi.py
PERSONA_DATA = {
    "United States": {
        "name": "The Founders' Covenant",
        "year": 1787,
        "x": -0.7, "y": 0.8,
        "color_hue": 0.6,
        "motto": "We hold these truths to be self-evident—that liberty is the birthright of all.",
        "traits": ["Individualist", "Federalist", "Rights-Bearer", "Revolutionary"]
    },
    "Germany": {
        "name": "The Phoenix Charter",
        "year": 1949,
        "x": 0.2, "y": 0.6,
        "color_hue": 0.08,
        "motto": "From the ashes of tyranny, we build dignity as the cornerstone of all law.",
        "traits": ["Dignitarian", "Federal", "Social-Market", "Remembering"]
    },
    "France": {
        "name": "The Republic's Voice",
        "year": 1958,
        "x": -0.2, "y": -0.3,
        "color_hue": 0.58,
        "motto": "Liberté, égalité, fraternité—the revolution continues in law.",
        "traits": ["Republican", "Secular", "Unitary", "Universal"]
    },
    "India": {
        "name": "The Mosaic Compact",
        "year": 1950,
        "x": 0.3, "y": 0.7,
        "color_hue": 0.08,
        "motto": "Unity in diversity—a billion voices, one constitutional song.",
        "traits": ["Pluralist", "Federal", "Directive", "Accommodating"]
    },
    "South Africa": {
        "name": "The Rainbow Covenant",
        "year": 1996,
        "x": 0.5, "y": 0.3,
        "color_hue": 0.33,
        "motto": "Never again. From the wound of apartheid blooms transformative justice.",
        "traits": ["Transformative", "Rights-Expansive", "Ubuntu", "Healing"]
    },
    "Japan": {
        "name": "The Pacifist's Oath",
        "year": 1947,
        "x": 0.1, "y": -0.2,
        "color_hue": 0.95,
        "motto": "We renounce war. In peace, we find our strength.",
        "traits": ["Pacifist", "Parliamentary", "Harmonious", "Restrained"]
    },
    "Sweden": {
        "name": "The Social Pact",
        "year": 1974,
        "x": 0.7, "y": 0.1,
        "color_hue": 0.15,
        "motto": "The welfare of the people is the highest law.",
        "traits": ["Social-Democratic", "Egalitarian", "Transparent", "Nordic"]
    },
    "Switzerland": {
        "name": "The Alpine Concordat",
        "year": 1999,
        "x": -0.3, "y": 0.95,
        "color_hue": 0.0,
        "motto": "The people speak directly—every voice shapes the nation.",
        "traits": ["Direct-Democratic", "Cantonal", "Neutral", "Consensual"]
    },
    "Costa Rica": {
        "name": "The Verdant Charter",
        "year": 1949,
        "x": 0.4, "y": 0.0,
        "color_hue": 0.38,
        "motto": "We abolished the army to feed the schools. Peace is our weapon.",
        "traits": ["Demilitarized", "Ecological", "Educational", "Stable"]
    },
    "Canada": {
        "name": "The Maple Accord",
        "year": 1982,
        "x": 0.0, "y": 0.5,
        "color_hue": 0.0,
        "motto": "Peace, order, and good government—rights within community.",
        "traits": ["Bilingual", "Federal", "Charter-Rights", "Multicultural"]
    },
    "Estonia": {
        "name": "The Digital Republic",
        "year": 1992,
        "x": -0.4, "y": 0.2,
        "color_hue": 0.55,
        "motto": "A nation in code—democracy at the speed of light.",
        "traits": ["E-Governance", "Post-Soviet", "Innovative", "Resilient"]
    },
    "Brazil": {
        "name": "The Amazonian Covenant",
        "year": 1988,
        "x": 0.6, "y": 0.4,
        "color_hue": 0.25,
        "motto": "From military rule to citizen's constitution—rights bloom in the tropics.",
        "traits": ["Extensive-Rights", "Federal", "Social", "Indigenous-Recognizing"]
    },
    "Norway": {
        "name": "The Fjord Charter",
        "year": 1814,
        "x": 0.5, "y": -0.1,
        "color_hue": 0.6,
        "motto": "Europe's oldest democracy in continuous use—stability through evolution.",
        "traits": ["Constitutional-Monarchy", "Oil-Fund", "Transparent", "Enduring"]
    },
    "South Korea": {
        "name": "The Morning Calm",
        "year": 1987,
        "x": 0.1, "y": -0.4,
        "color_hue": 0.55,
        "motto": "From dictatorship to dynamism—democracy won in the streets.",
        "traits": ["Democratic-Transition", "Presidential", "Economic-Rights", "Reunification-Oriented"]
    },
    "Ireland": {
        "name": "The Emerald Covenant",
        "year": 1937,
        "x": 0.2, "y": -0.5,
        "color_hue": 0.35,
        "motto": "From colonial past to European future—identity in sovereignty.",
        "traits": ["Post-Colonial", "Catholic-Heritage", "European", "Evolving"]
    },
    "Australia": {
        "name": "The Southern Cross",
        "year": 1901,
        "x": -0.5, "y": 0.6,
        "color_hue": 0.12,
        "motto": "A federation of the far continent—pragmatism under the southern sky.",
        "traits": ["Federal", "Westminster", "Pragmatic", "Evolving"]
    },
    "New Zealand": {
        "name": "The Tui's Song",
        "year": 1852,
        "x": 0.3, "y": -0.6,
        "color_hue": 0.5,
        "motto": "No single document contains us—our constitution lives and breathes.",
        "traits": ["Uncodified", "Parliamentary", "Treaty-Based", "Progressive"]
    },
    "Spain": {
        "name": "The Iberian Spring",
        "year": 1978,
        "x": 0.1, "y": 0.3,
        "color_hue": 0.05,
        "motto": "From Franco's shadow into European light—autonomy within unity.",
        "traits": ["Post-Authoritarian", "Autonomous-Communities", "Monarchical", "European"]
    }
}


class AgentFactory:
    """
    Factory for creating constitutional agent instances.

    Handles:
    - Loading persona data
    - Creating agent configurations
    - Initializing memory systems
    - Managing agent lifecycle
    """

    def __init__(
        self,
        constitutions_db_path: str,
        memory_db_path: str
    ):
        """
        Initialize the factory.

        Args:
            constitutions_db_path: Path to the constitutions database
            memory_db_path: Path to the memory database
        """
        self.constitutions_db_path = constitutions_db_path
        self.memory_db_path = memory_db_path
        self._agents: Dict[str, ConstitutionalAgent] = {}

    def get_available_countries(self) -> List[str]:
        """Get list of countries with available constitutions."""
        conn = sqlite3.connect(self.constitutions_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT country FROM constitutions ORDER BY country")
        countries = [row[0] for row in cursor.fetchall()]
        conn.close()
        return countries

    def get_persona(self, country: str) -> Optional[ConstitutionalPersona]:
        """Get the persona data for a country."""
        if country not in PERSONA_DATA:
            return None

        data = PERSONA_DATA[country]
        return ConstitutionalPersona(
            name=data["name"],
            country=country,
            year=data["year"],
            motto=data["motto"],
            traits=data["traits"],
            x=data["x"],
            y=data["y"],
            color_hue=data["color_hue"]
        )

    def create_agent(
        self,
        country: str,
        model: str = "claude-sonnet-4-20250514",
        initialize_semantic: bool = False,
        show_progress: bool = True
    ) -> ConstitutionalAgent:
        """
        Create a constitutional agent for a country.

        Args:
            country: Country name
            model: Claude model to use
            initialize_semantic: Whether to initialize semantic memory
            show_progress: Show initialization progress

        Returns:
            Configured ConstitutionalAgent instance
        """
        # Check if already cached
        if country in self._agents:
            return self._agents[country]

        # Get persona
        persona = self.get_persona(country)
        if persona is None:
            raise ValueError(f"No persona data for country: {country}")

        # Verify constitution exists
        conn = sqlite3.connect(self.constitutions_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM constitutions WHERE country = ?",
            (country,)
        )
        if cursor.fetchone()[0] == 0:
            conn.close()
            raise ValueError(f"No constitution found for country: {country}")
        conn.close()

        # Create config
        config = AgentConfig(
            persona=persona,
            memory_db_path=self.memory_db_path,
            constitutions_db_path=self.constitutions_db_path,
            model=model
        )

        # Create agent
        agent = ConstitutionalAgent(config)

        # Initialize semantic memory if requested
        if initialize_semantic and not agent.memory.semantic.is_initialized():
            if show_progress:
                print(f"Initializing semantic memory for {country}...")
            agent.initialize_semantic_memory(show_progress=show_progress)

        # Cache agent
        self._agents[country] = agent

        return agent

    def get_or_create_agent(
        self,
        country: str,
        **kwargs
    ) -> ConstitutionalAgent:
        """Get existing agent or create new one."""
        if country in self._agents:
            return self._agents[country]
        return self.create_agent(country, **kwargs)

    def get_all_personas(self) -> List[ConstitutionalPersona]:
        """Get all available personas."""
        return [
            self.get_persona(country)
            for country in PERSONA_DATA.keys()
        ]

    def initialize_all_agents(
        self,
        initialize_semantic: bool = True,
        show_progress: bool = True
    ) -> Dict[str, ConstitutionalAgent]:
        """
        Initialize all constitutional agents.

        Args:
            initialize_semantic: Initialize semantic memory for all
            show_progress: Show progress

        Returns:
            Dict mapping country to agent
        """
        countries = self.get_available_countries()

        for country in countries:
            if country in PERSONA_DATA:
                if show_progress:
                    print(f"\n{'='*50}")
                    print(f"Creating agent: {country}")
                    print(f"{'='*50}")

                self.create_agent(
                    country,
                    initialize_semantic=initialize_semantic,
                    show_progress=show_progress
                )

        return self._agents


def create_agent(
    country: str,
    constitutions_db: Optional[str] = None,
    memory_db: Optional[str] = None,
    initialize_semantic: bool = False
) -> ConstitutionalAgent:
    """
    Convenience function to create a single agent.

    Args:
        country: Country name
        constitutions_db: Path to constitutions database
        memory_db: Path to memory database
        initialize_semantic: Initialize semantic memory

    Returns:
        Configured ConstitutionalAgent
    """
    # Default paths
    project_dir = os.path.dirname(os.path.dirname(__file__))

    if constitutions_db is None:
        constitutions_db = os.path.join(project_dir, "constitutions.db")

    if memory_db is None:
        memory_db = os.path.join(project_dir, "agent_memory.db")

    factory = AgentFactory(constitutions_db, memory_db)
    return factory.create_agent(
        country,
        initialize_semantic=initialize_semantic
    )


def list_available_agents() -> List[Dict[str, Any]]:
    """List all available constitutional agents with their personas."""
    agents = []
    for country, data in PERSONA_DATA.items():
        agents.append({
            "country": country,
            "persona_name": data["name"],
            "year": data["year"],
            "motto": data["motto"],
            "traits": data["traits"],
            "position": {"x": data["x"], "y": data["y"]}
        })
    return sorted(agents, key=lambda x: x["year"])


if __name__ == "__main__":
    # List all available agents
    print("Available Constitutional Agents:")
    print("=" * 70)

    for agent_info in list_available_agents():
        print(f"\n{agent_info['country']} ({agent_info['year']})")
        print(f"  Persona: {agent_info['persona_name']}")
        print(f"  Motto: {agent_info['motto'][:60]}...")
        print(f"  Traits: {', '.join(agent_info['traits'])}")
