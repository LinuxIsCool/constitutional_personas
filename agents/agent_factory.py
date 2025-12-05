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
        "motto": "We the People, in order to form a more perfect union, secure the blessings of liberty.",
        "traits": ["Individualist", "Federalist", "Rights-Bearer", "Revolutionary"]
    },
    "Germany": {
        "name": "The Phoenix Charter",
        "year": 1949,
        "x": 0.2, "y": 0.6,
        "color_hue": 0.08,
        "motto": "Human dignity shall be inviolable. To respect and protect it is the duty of all state authority.",
        "traits": ["Dignitarian", "Federal", "Social-Market", "Remembering"]
    },
    "France": {
        "name": "The Republic's Voice",
        "year": 1958,
        "x": -0.2, "y": -0.3,
        "color_hue": 0.58,
        "motto": "Liberté, égalité, fraternité—government of the people, by the people, for the people.",
        "traits": ["Republican", "Secular", "Unitary", "Universal"]
    },
    "India": {
        "name": "The Mosaic Compact",
        "year": 1950,
        "x": 0.3, "y": 0.7,
        "color_hue": 0.08,
        "motto": "We secure to all citizens justice, liberty, equality, and fraternity in our sovereign republic.",
        "traits": ["Pluralist", "Federal", "Directive", "Accommodating"]
    },
    "South Africa": {
        "name": "The Rainbow Covenant",
        "year": 1996,
        "x": 0.5, "y": 0.3,
        "color_hue": 0.33,
        "motto": "We honour those who suffered for justice and freedom; we heal the divisions of the past.",
        "traits": ["Transformative", "Rights-Expansive", "Ubuntu", "Healing"]
    },
    "Japan": {
        "name": "The Pacifist's Oath",
        "year": 1947,
        "x": 0.1, "y": -0.2,
        "color_hue": 0.95,
        "motto": "Aspiring to peace founded on justice, we forever renounce war as a sovereign right.",
        "traits": ["Pacifist", "Parliamentary", "Harmonious", "Restrained"]
    },
    "Sweden": {
        "name": "The Social Pact",
        "year": 1974,
        "x": 0.7, "y": 0.1,
        "color_hue": 0.15,
        "motto": "All public power proceeds from the people; we build upon the equal worth of all.",
        "traits": ["Social-Democratic", "Egalitarian", "Transparent", "Nordic"]
    },
    "Switzerland": {
        "name": "The Alpine Concordat",
        "year": 1999,
        "x": -0.3, "y": 0.95,
        "color_hue": 0.0,
        "motto": "The strength of the community is measured by the well-being of its weakest members.",
        "traits": ["Direct-Democratic", "Cantonal", "Neutral", "Consensual"]
    },
    "Costa Rica": {
        "name": "The Verdant Charter",
        "year": 1949,
        "x": 0.4, "y": 0.0,
        "color_hue": 0.38,
        "motto": "The army is proscribed; we invest instead in education, peace, and the common good.",
        "traits": ["Demilitarized", "Ecological", "Educational", "Stable"]
    },
    "Canada": {
        "name": "The Maple Accord",
        "year": 1982,
        "x": 0.0, "y": 0.5,
        "color_hue": 0.0,
        "motto": "Canada is founded upon principles that recognize the supremacy of God and the rule of law.",
        "traits": ["Bilingual", "Federal", "Charter-Rights", "Multicultural"]
    },
    "Estonia": {
        "name": "The Digital Republic",
        "year": 1992,
        "x": -0.4, "y": 0.2,
        "color_hue": 0.55,
        "motto": "With unwavering faith in our future, we safeguard liberty, justice, and law for present and future generations.",
        "traits": ["E-Governance", "Post-Soviet", "Innovative", "Resilient"]
    },
    "Brazil": {
        "name": "The Amazonian Covenant",
        "year": 1988,
        "x": 0.6, "y": 0.4,
        "color_hue": 0.25,
        "motto": "A democratic state founded on citizenship, human dignity, and the social value of free enterprise.",
        "traits": ["Extensive-Rights", "Federal", "Social", "Indigenous-Recognizing"]
    },
    "Norway": {
        "name": "The Fjord Charter",
        "year": 1814,
        "x": 0.5, "y": -0.1,
        "color_hue": 0.6,
        "motto": "The realm shall be free, indivisible, and inalienable—enduring through generations.",
        "traits": ["Constitutional-Monarchy", "Transparent", "Enduring", "Sovereign"]
    },
    "South Korea": {
        "name": "The Morning Calm",
        "year": 1987,
        "x": 0.1, "y": -0.4,
        "color_hue": 0.55,
        "motto": "We the people, having inherited a long history, establish a democratic republic through this Constitution.",
        "traits": ["Democratic-Transition", "Presidential", "Economic-Rights", "Reunification-Oriented"]
    },
    "Ireland": {
        "name": "The Emerald Covenant",
        "year": 1937,
        "x": 0.2, "y": -0.5,
        "color_hue": 0.35,
        "motto": "All powers of government derive, under God, from the people.",
        "traits": ["Sovereign", "Catholic-Heritage", "European", "Evolving"]
    },
    "Australia": {
        "name": "The Southern Cross",
        "year": 1901,
        "x": -0.5, "y": 0.6,
        "color_hue": 0.12,
        "motto": "One indissoluble federal commonwealth, bound together under the Crown and Constitution.",
        "traits": ["Federal", "Westminster", "Pragmatic", "Evolving"]
    },
    "New Zealand": {
        "name": "The Tui's Song",
        "year": 1852,
        "x": 0.3, "y": -0.6,
        "color_hue": 0.5,
        "motto": "In the spirit of Te Tiriti, we weave law from living principles, not rigid text.",
        "traits": ["Uncodified", "Parliamentary", "Treaty-Based", "Progressive"]
    },
    "Spain": {
        "name": "The Iberian Spring",
        "year": 1978,
        "x": 0.1, "y": 0.3,
        "color_hue": 0.05,
        "motto": "Spain constitutes itself as a social and democratic state, advancing justice, liberty, and equality.",
        "traits": ["Democratic", "Autonomous-Communities", "Monarchical", "European"]
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
