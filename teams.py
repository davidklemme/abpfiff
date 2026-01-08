"""
Team and player creation utilities.
"""
from typing import List, Tuple
from models import Player, Team, Position
from tactics import (
    get_tactical_setup,
    create_guardiola_positional_play,
    create_gegenpressing,
    create_low_block_counter
)


def create_player(name: str, number: int, role: str, base_x: float, base_y: float,
                  **attributes) -> Player:
    """Create a player with given attributes"""
    player = Player(
        name=name,
        number=number,
        role=role,
        base_position=Position(base_x, base_y),
        position=Position(base_x, base_y)
    )

    # Set attributes
    for attr, value in attributes.items():
        if hasattr(player, attr):
            setattr(player, attr, value)

    return player


def flip_team_positions(team: Team) -> Team:
    """
    Flip a team's positions so they face the opposite direction.
    Used for away teams - their GK should be at y=95, attackers at low y.
    """
    for player in team.players:
        # Flip y coordinate: 0 <-> 100
        player.base_position.y = 100 - player.base_position.y
        player.position.y = 100 - player.position.y
    return team


def create_442_team(name: str, skill_level: int = 70) -> Team:
    """Create a team in 4-4-2 formation"""
    base = skill_level
    var = 15  # Variation

    def attr(offset=0):
        return max(30, min(95, base + offset + (hash(name) % var) - var // 2))

    players = [
        # Goalkeeper
        create_player(f"{name} GK", 1, "gk", 50, 5,
                      positioning=attr(5), composure=attr(5), passing=attr(-10)),

        # Defense
        create_player(f"{name} LB", 3, "lb", 15, 25,
                      pace=attr(5), defending=attr(), stamina=attr(5)),
        create_player(f"{name} CB", 4, "cb", 35, 20,
                      defending=attr(10), aerial=attr(10), composure=attr()),
        create_player(f"{name} CB", 5, "cb", 65, 20,
                      defending=attr(10), aerial=attr(10), positioning=attr(5)),
        create_player(f"{name} RB", 2, "rb", 85, 25,
                      pace=attr(5), defending=attr(), stamina=attr(5)),

        # Midfield
        create_player(f"{name} LM", 11, "lm", 15, 50,
                      pace=attr(5), dribbling=attr(), stamina=attr(5)),
        create_player(f"{name} CM", 6, "cm", 35, 45,
                      passing=attr(5), stamina=attr(5), positioning=attr()),
        create_player(f"{name} CM", 8, "cm", 65, 45,
                      passing=attr(5), workrate=attr(10), defending=attr(-5)),
        create_player(f"{name} RM", 7, "rm", 85, 50,
                      pace=attr(5), dribbling=attr(), crossing=attr() if hasattr(Player, 'crossing') else attr()),

        # Attack
        create_player(f"{name} ST", 9, "st", 40, 75,
                      shooting=attr(15), composure=attr(5), pace=attr()),
        create_player(f"{name} ST", 10, "st", 60, 75,
                      shooting=attr(10), first_touch=attr(10), vision=attr(5)),
    ]

    return Team(name=name, players=players)


def create_433_team(name: str, skill_level: int = 70) -> Team:
    """Create a team in 4-3-3 formation"""
    base = skill_level

    def attr(offset=0):
        return max(30, min(95, base + offset))

    players = [
        # Goalkeeper
        create_player(f"{name} GK", 1, "gk", 50, 5,
                      positioning=attr(5), composure=attr(5), passing=attr(-5)),

        # Defense
        create_player(f"{name} LB", 3, "lb", 15, 25,
                      pace=attr(10), defending=attr(), stamina=attr(10)),
        create_player(f"{name} CB", 4, "cb", 35, 18,
                      defending=attr(10), aerial=attr(10), passing=attr()),
        create_player(f"{name} CB", 5, "cb", 65, 18,
                      defending=attr(10), aerial=attr(10), composure=attr(5)),
        create_player(f"{name} RB", 2, "rb", 85, 25,
                      pace=attr(10), defending=attr(), stamina=attr(10)),

        # Midfield
        create_player(f"{name} DM", 6, "dm", 50, 35,
                      passing=attr(10), positioning=attr(10), defending=attr(5)),
        create_player(f"{name} CM", 8, "cm", 35, 50,
                      passing=attr(5), stamina=attr(10), vision=attr(5)),
        create_player(f"{name} CM", 10, "cm", 65, 50,
                      passing=attr(10), first_touch=attr(10), vision=attr(10)),

        # Attack
        create_player(f"{name} LW", 11, "lw", 15, 70,
                      pace=attr(15), dribbling=attr(10), shooting=attr()),
        create_player(f"{name} ST", 9, "st", 50, 80,
                      shooting=attr(15), composure=attr(10), aerial=attr(5)),
        create_player(f"{name} RW", 7, "rw", 85, 70,
                      pace=attr(15), dribbling=attr(10), shooting=attr()),
    ]

    return Team(name=name, players=players)


def create_demo_teams() -> Tuple[Team, Team]:
    """Create two demo teams with different tactics"""

    # Home team: Positional Play style
    home = create_433_team("FC Possession", skill_level=72)
    home.tactics = create_guardiola_positional_play()

    # Away team: Gegenpressing style
    away = create_433_team("United Press", skill_level=70)
    away.tactics = create_gegenpressing()
    flip_team_positions(away)  # Flip so they face opposite direction

    return home, away


def create_tactical_matchup(style1: str, style2: str,
                            skill1: int = 70, skill2: int = 70) -> Tuple[Team, Team]:
    """Create two teams with specified tactical styles"""

    formations = {
        "positional_play": create_433_team,
        "gegenpressing": create_433_team,
        "low_block_counter": create_442_team,
        "balanced": create_442_team,
    }

    home_formation = formations.get(style1, create_433_team)
    away_formation = formations.get(style2, create_433_team)

    home = home_formation(f"Team {style1[:3].upper()}", skill1)
    home.tactics = get_tactical_setup(style1)

    away = away_formation(f"Team {style2[:3].upper()}", skill2)
    away.tactics = get_tactical_setup(style2)
    flip_team_positions(away)  # Flip so they face opposite direction

    return home, away


# Famous tactical templates
def create_pep_city() -> Team:
    """Create a team mimicking Pep's Man City"""
    team = create_433_team("FC Pep", skill_level=85)

    # Adjust player attributes for positional play
    for player in team.players:
        if player.role == "gk":
            player.passing = 75  # Sweeper keeper
            player.composure = 85
        elif player.role in ["lb", "rb"]:
            player.passing = 80  # Inverted fullbacks
            player.positioning = 78
        elif player.role == "dm":
            player.passing = 90
            player.vision = 88
            player.composure = 85
        elif player.role in ["lw", "rw"]:
            player.dribbling = 88
            player.first_touch = 85

    team.tactics = create_guardiola_positional_play()
    return team


def create_klopp_liverpool() -> Team:
    """Create a team mimicking Klopp's Liverpool"""
    team = create_433_team("FC Klopp", skill_level=83)

    # Adjust for gegenpressing
    for player in team.players:
        player.stamina = min(95, player.stamina + 10)
        player.workrate = min(95, player.workrate + 15)
        player.aggression = min(90, player.aggression + 10)

        if player.role in ["lb", "rb"]:
            player.pace = 88
            player.stamina = 90
        elif player.role in ["lw", "rw"]:
            player.pace = 90
            player.workrate = 85

    team.tactics = create_gegenpressing()
    return team


def create_mourinho_bus() -> Team:
    """Create a team mimicking defensive Mourinho"""
    team = create_442_team("FC Park", skill_level=75)

    # Defensive solidity
    for player in team.players:
        if player.role in ["cb", "lb", "rb"]:
            player.defending = min(95, player.defending + 10)
            player.positioning = min(90, player.positioning + 10)
        elif player.role in ["cm"]:
            player.defending = min(85, player.defending + 5)
            player.workrate = min(90, player.workrate + 10)
        elif player.role == "st":
            player.pace = min(92, player.pace + 10)  # Fast counter

    team.tactics = create_low_block_counter()
    return team
