"""
Spatial control and influence mapping system.
Models which team controls different areas of the pitch.
"""
from typing import List, Tuple, Optional
from models import Position, Player, Team, MatchState
import math


class SpaceControl:
    """
    Calculates spatial control across the pitch.
    This is the foundation for tactical interactions.
    """

    def __init__(self, resolution: int = 10):
        """
        resolution: Grid resolution (10 = 10x10 grid = 100 cells)
        """
        self.resolution = resolution
        self.cell_size = 100.0 / resolution

    def calculate_control_grid(self, home: Team, away: Team) -> List[List[float]]:
        """
        Calculate control values for entire pitch.
        Returns grid where:
          > 0.5 = home team controls
          < 0.5 = away team controls
          = 0.5 = contested
        """
        grid = []
        for row in range(self.resolution):
            grid_row = []
            for col in range(self.resolution):
                # Center of this cell
                pos = Position(
                    (col + 0.5) * self.cell_size,
                    (row + 0.5) * self.cell_size
                )
                control = self.control_at(pos, home, away)
                grid_row.append(control)
            grid.append(grid_row)
        return grid

    def control_at(self, pos: Position, home: Team, away: Team) -> float:
        """
        Calculate which team controls a specific position.
        Returns 0-1 (0 = away, 1 = home)
        """
        home_influence = sum(p.influence_at(pos) for p in home.players)
        away_influence = sum(p.influence_at(pos) for p in away.players)

        total = home_influence + away_influence
        if total == 0:
            return 0.5

        return home_influence / total

    def find_open_spaces(self, attacking_team: Team, defending_team: Team,
                          min_control: float = 0.6) -> List[Position]:
        """Find positions where attacking team has good control"""
        open_spaces = []
        grid = self.calculate_control_grid(attacking_team, defending_team)

        for row in range(self.resolution):
            for col in range(self.resolution):
                # Attacking team needs > min_control
                if grid[row][col] >= min_control:
                    pos = Position(
                        (col + 0.5) * self.cell_size,
                        (row + 0.5) * self.cell_size
                    )
                    open_spaces.append(pos)

        return open_spaces

    def find_passing_lanes(self, passer: Player, teammates: List[Player],
                           opponents: List[Player]) -> List[Tuple[Player, float]]:
        """
        Find available passing lanes and their quality.
        Returns list of (target_player, lane_quality) tuples.
        """
        lanes = []

        for teammate in teammates:
            if teammate == passer:
                continue

            quality = self._evaluate_passing_lane(passer, teammate, opponents)
            if quality > 0.1:  # Minimum threshold
                lanes.append((teammate, quality))

        return sorted(lanes, key=lambda x: x[1], reverse=True)

    def _evaluate_passing_lane(self, passer: Player, target: Player,
                                opponents: List[Player]) -> float:
        """
        Evaluate quality of a passing lane.
        Considers: distance, opponent proximity to lane, target space
        """
        distance = passer.position.distance_to(target.position)

        # Very short or very long passes are harder
        distance_factor = 1.0
        if distance < 5:
            distance_factor = 0.8  # Too close
        elif distance > 40:
            distance_factor = max(0.3, 1.0 - (distance - 40) * 0.02)

        # Check for opponents in the passing lane
        lane_blocked = 0.0
        for opp in opponents:
            block_factor = self._point_to_line_distance(
                opp.position, passer.position, target.position
            )
            if block_factor < 5:  # Within 5 units of the lane
                # Closer to lane = more blocking
                lane_blocked += (5 - block_factor) / 5 * 0.3

        lane_factor = max(0.1, 1.0 - lane_blocked)

        # Is the target in space?
        target_space = 1.0
        for opp in opponents:
            opp_dist = opp.position.distance_to(target.position)
            if opp_dist < 10:
                target_space -= (10 - opp_dist) / 10 * 0.2

        target_space = max(0.2, target_space)

        # Passer's passing ability
        passer_ability = passer.effective_attribute('passing') / 100

        return distance_factor * lane_factor * target_space * passer_ability

    def corridor_openness(self, start: Position, end: Position,
                          opponents: List[Player],
                          radius: float = 6.0,
                          press_bubble: float = 8.0) -> float:
        """How clear the corridor from `start` to `end` actually is
        (0..1). Each opponent near the corridor multiplies in a
        continuous penalty - threading a ball through pressuring
        defenders is expensive, threading it through two is much more
        so. Defenders inside the passer's immediate bubble ramp in
        gently: their harassment is already priced as pressure on the
        release (execution), and a ball is played around a man at two
        meters - it is the bodies ALONG the path that close a lane.
        Pure geometry: ability, on either end, is priced elsewhere."""
        openness = 1.0
        for opponent in opponents:
            distance, along = self._corridor_projection(opponent.position,
                                                        start, end)
            threat = max(0.0, 1.0 - distance / radius)
            threat *= min(1.0, along / press_bubble)
            openness *= 1.0 - 0.55 * threat
        return openness

    def _corridor_projection(self, point: Position, line_start: Position,
                             line_end: Position):
        """(perpendicular distance to the segment, distance along it)."""
        dx = line_end.x - line_start.x
        dy = line_end.y - line_start.y
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return point.distance_to(line_start), 0.0
        t = max(0.0, min(1.0, ((point.x - line_start.x) * dx +
                               (point.y - line_start.y) * dy) / len_sq))
        closest = Position(line_start.x + t * dx, line_start.y + t * dy)
        return point.distance_to(closest), t * math.sqrt(len_sq)

    def _point_to_line_distance(self, point: Position,
                                 line_start: Position, line_end: Position) -> float:
        """Calculate perpendicular distance from point to line segment"""
        # Vector from start to end
        dx = line_end.x - line_start.x
        dy = line_end.y - line_start.y

        # Length squared
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return point.distance_to(line_start)

        # Project point onto line
        t = max(0, min(1, (
            (point.x - line_start.x) * dx +
            (point.y - line_start.y) * dy
        ) / len_sq))

        # Closest point on line
        closest = Position(
            line_start.x + t * dx,
            line_start.y + t * dy
        )

        return point.distance_to(closest)

    def pressing_effectiveness(self, pressing_team: Team, ball_pos: Position,
                                radius: float = 15) -> Tuple[int, float]:
        """
        Calculate how effective a press is around the ball.
        Returns (num_pressers, total_pressure)
        """
        pressers = 0
        total_pressure = 0.0

        for player in pressing_team.players:
            dist = player.position.distance_to(ball_pos)
            if dist < radius:
                pressers += 1
                # Pressure based on proximity and workrate
                pressure = (radius - dist) / radius
                pressure *= player.effective_attribute('workrate') / 100
                pressure *= player.effective_attribute('aggression') / 100
                total_pressure += pressure

        return pressers, total_pressure

    def numerical_superiority(self, pos: Position, team_a: Team, team_b: Team,
                               radius: float = 15) -> int:
        """
        Calculate numerical difference around a position.
        Positive = team_a has more players, negative = team_b has more
        """
        team_a_count = sum(1 for p in team_a.players
                          if p.position.distance_to(pos) < radius)
        team_b_count = sum(1 for p in team_b.players
                          if p.position.distance_to(pos) < radius)
        return team_a_count - team_b_count


class PassingGraph:
    """
    Represents available passing options as a graph.
    Used for tactical analysis and AI decision making.
    """

    def __init__(self, space_control: SpaceControl):
        self.space_control = space_control

    def build_graph(self, team: Team, opponents: Team) -> dict:
        """
        Build a graph of passing options.
        Returns dict: player -> [(target, quality, is_progressive), ...]
        """
        graph = {}

        for player in team.players:
            lanes = self.space_control.find_passing_lanes(
                player, team.players, opponents.players
            )

            graph[player] = []
            for target, quality in lanes:
                # Is this pass progressive (moves ball forward)?
                is_progressive = target.position.y > player.position.y + 5
                graph[player].append((target, quality, is_progressive))

        return graph

    def find_combination_play(self, graph: dict, start_player: Player,
                               depth: int = 3) -> List[List[Player]]:
        """
        Find possible combination plays (one-twos, third man runs).
        Returns possible passing sequences.
        """
        sequences = []
        self._dfs_combinations(graph, [start_player], depth, sequences)
        return sequences

    def _dfs_combinations(self, graph: dict, current_path: List[Player],
                          remaining_depth: int, results: List):
        if remaining_depth == 0:
            if len(current_path) > 2:
                results.append(current_path.copy())
            return

        current = current_path[-1]
        if current not in graph:
            return

        for target, quality, _ in graph[current]:
            if target not in current_path and quality > 0.3:
                current_path.append(target)
                self._dfs_combinations(graph, current_path, remaining_depth - 1, results)
                current_path.pop()
