#!/usr/bin/env python3
"""
Constitutional Agent CLI

Interactive command-line interface for conversing with constitutional agents.

Usage:
    python run_agent.py                    # List available agents
    python run_agent.py --country USA      # Talk to US Constitution
    python run_agent.py --init USA         # Initialize semantic memory for USA
    python run_agent.py --init-all         # Initialize all agents
    python run_agent.py --status USA       # Check agent status
"""

import argparse
import os
import sys
import readline  # Enable command history

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from agents.agent_factory import AgentFactory, list_available_agents, PERSONA_DATA
from db.memory_schema import create_memory_database

# Project paths
PROJECT_DIR = os.path.dirname(__file__)
CONSTITUTIONS_DB = os.path.join(PROJECT_DIR, "constitutions.db")
MEMORY_DB = os.path.join(PROJECT_DIR, "agent_memory.db")


def ensure_databases():
    """Ensure all databases exist."""
    if not os.path.exists(CONSTITUTIONS_DB):
        print(f"Error: Constitutions database not found at {CONSTITUTIONS_DB}")
        print("Please run: python populate_constitutions.py")
        sys.exit(1)

    if not os.path.exists(MEMORY_DB):
        print("Creating memory database...")
        create_memory_database(MEMORY_DB)


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CONSTITUTIONAL PERSONAS: AGENTIC DEMOCRACY                      ║
║                                                                              ║
║   Converse with constitutions as living, thinking entities.                  ║
║   Each agent embodies the values, principles, and voice of their nation.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def list_agents():
    """List all available constitutional agents."""
    print_banner()
    print("Available Constitutional Agents:")
    print("=" * 75)

    agents = list_available_agents()

    for agent in agents:
        print(f"\n┌─ {agent['country']} ({agent['year']}) ──")
        print(f"│  Persona: {agent['persona_name']}")
        print(f"│  \"{agent['motto'][:65]}...\"" if len(agent['motto']) > 65 else f"│  \"{agent['motto']}\"")
        print(f"│  Traits: {', '.join(agent['traits'])}")
        print(f"└{'─' * 72}")

    print(f"\nTotal: {len(agents)} constitutional agents available")
    print("\nUsage: python run_agent.py --country <COUNTRY_NAME>")
    print("Example: python run_agent.py --country 'United States'")


def show_status(factory: AgentFactory, country: str):
    """Show agent status."""
    try:
        agent = factory.create_agent(country, initialize_semantic=False)
        status = agent.get_status()

        print(f"\n{'='*50}")
        print(f"Agent Status: {country}")
        print(f"{'='*50}")
        print(f"Persona: {status['persona']}")
        print(f"Session: {status['session_id'][:8]}...")
        print(f"Semantic Memory Initialized: {status['semantic_initialized']}")
        print(f"\nMemory Statistics:")
        print(f"  Persistent memories: {status['memory_stats']['persistent_count']}")
        print(f"  Episodic memories: {status['memory_stats']['episodic_count']}")
        print(f"  Procedures: {status['memory_stats']['procedural_count']}")
        print(f"  Semantic chunks: {status['memory_stats']['semantic_count']}")

    except Exception as e:
        print(f"Error getting status: {e}")


def initialize_semantic(factory: AgentFactory, country: str):
    """Initialize semantic memory for a country."""
    print(f"\nInitializing semantic memory for {country}...")
    print("This will chunk and embed the constitutional text using Ollama.")
    print("Make sure Ollama is running with an embedding model (e.g., nomic-embed-text)")
    print()

    try:
        agent = factory.create_agent(country, initialize_semantic=True, show_progress=True)
        print(f"\n✓ Semantic memory initialized for {country}")
        print(f"  Chunks: {agent.memory.semantic.count()}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        print("  ollama pull nomic-embed-text")


def initialize_all(factory: AgentFactory):
    """Initialize semantic memory for all countries."""
    print("\nInitializing semantic memory for all constitutional agents...")
    print("This may take a while.\n")

    countries = factory.get_available_countries()
    success = 0
    failed = []

    for country in countries:
        if country in PERSONA_DATA:
            print(f"\n{'='*50}")
            print(f"Processing: {country}")
            print(f"{'='*50}")

            try:
                agent = factory.create_agent(
                    country,
                    initialize_semantic=True,
                    show_progress=True
                )
                success += 1
            except Exception as e:
                print(f"✗ Error: {e}")
                failed.append(country)

    print(f"\n{'='*50}")
    print("Initialization Complete")
    print(f"{'='*50}")
    print(f"Success: {success}/{len(PERSONA_DATA)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


def interactive_session(factory: AgentFactory, country: str):
    """Run an interactive session with an agent."""
    print(f"\nLoading {country} agent...")

    try:
        agent = factory.create_agent(country, initialize_semantic=False)
    except Exception as e:
        print(f"Error creating agent: {e}")
        return

    # Check if semantic memory is initialized
    if not agent.memory.semantic.is_initialized():
        print("\n⚠ Warning: Semantic memory not initialized.")
        print("The agent won't be able to retrieve constitutional text.")
        print("Run with --init flag first, or continue without RAG support.")
        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            return

    # Print greeting
    print("\n" + "=" * 70)
    print(agent.get_greeting())
    print("=" * 70)
    print("\nCommands: /quit, /status, /search <query>, /help")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                if cmd == 'quit' or cmd == 'exit':
                    agent.end_session()
                    print("\nFarewell. May constitutional wisdom guide your path.")
                    break

                elif cmd == 'status':
                    status = agent.get_status()
                    print(f"\nSession turns: {status['turn_count']}")
                    print(f"Semantic initialized: {status['semantic_initialized']}")
                    print(f"Memory stats: {status['memory_stats']}")

                elif cmd == 'search' and cmd_arg:
                    results = agent.search_constitution(cmd_arg, top_k=3)
                    if results:
                        print(f"\nRelevant passages for '{cmd_arg}':")
                        for i, r in enumerate(results, 1):
                            print(f"\n[{i}] {r['section']} (similarity: {r['similarity']:.1%})")
                            print(f"    {r['content'][:200]}...")
                    else:
                        print(f"No results found for: {cmd_arg}")

                elif cmd == 'history':
                    episodes = agent.recall_past_interactions(limit=5)
                    if episodes:
                        print("\nRecent interactions:")
                        for e in episodes:
                            print(f"  - [{e['timestamp'] or 'unknown'}] {e['summary']}")
                    else:
                        print("No past interactions recorded.")

                elif cmd == 'help':
                    print("\nAvailable commands:")
                    print("  /quit     - End the session")
                    print("  /status   - Show agent status")
                    print("  /search <query> - Search constitution")
                    print("  /history  - Show past interactions")
                    print("  /help     - Show this help")

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")

                continue

            # Prepare the query (this updates memory and gets context)
            query_context = agent.prepare_query(user_input)

            # In a full implementation, we would send this to Claude via the Agent SDK
            # For now, we'll show what would be sent
            print(f"\n[Debug] Turn {query_context['turn_count']}")
            print(f"[Debug] Active procedures: {query_context['active_procedures']}")
            print(f"[Debug] Semantic results: {len(query_context['semantic_results'])} passages found")

            # Show relevant constitutional passages
            if query_context['semantic_results']:
                print("\n─── Relevant Constitutional Text ───")
                for r in query_context['semantic_results'][:2]:
                    print(f"[{r['section']}] {r['content'][:150]}...")
                print("────────────────────────────────────\n")

            # Placeholder response (in production, this would come from Claude)
            # You would use:
            # from claude_agent_sdk import query, ClaudeSDKClient
            # async for message in query(prompt=user_input, system=query_context['system_prompt']):
            #     print(message)

            print(f"\n{agent.persona.name}:")
            print("(To generate actual responses, integrate with Claude Agent SDK)")
            print(f"[System prompt prepared with {len(query_context['system_prompt'])} chars]")

            # Record a placeholder interaction
            agent.process_response(
                user_input,
                "[Response would be generated here via Claude Agent SDK]",
                importance=0.5
            )

        except KeyboardInterrupt:
            print("\n\nSession interrupted.")
            agent.end_session()
            break
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Constitutional Agent CLI - Converse with constitutions as living entities"
    )

    parser.add_argument(
        '--country', '-c',
        type=str,
        help='Country name to converse with (e.g., "United States", "Germany")'
    )

    parser.add_argument(
        '--init', '-i',
        type=str,
        metavar='COUNTRY',
        help='Initialize semantic memory for a country'
    )

    parser.add_argument(
        '--init-all',
        action='store_true',
        help='Initialize semantic memory for all countries'
    )

    parser.add_argument(
        '--status', '-s',
        type=str,
        metavar='COUNTRY',
        help='Show status for a country agent'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available agents'
    )

    args = parser.parse_args()

    # Ensure databases exist
    ensure_databases()

    # Create factory
    factory = AgentFactory(CONSTITUTIONS_DB, MEMORY_DB)

    # Handle commands
    if args.list or (not args.country and not args.init and not args.init_all and not args.status):
        list_agents()

    elif args.init_all:
        initialize_all(factory)

    elif args.init:
        initialize_semantic(factory, args.init)

    elif args.status:
        show_status(factory, args.status)

    elif args.country:
        interactive_session(factory, args.country)


if __name__ == "__main__":
    main()
