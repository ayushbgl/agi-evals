"""
ASCII Board Renderer for Catan

Renders the game board as ASCII art for LLM consumption.
Shows tiles, numbers, buildings, roads, and robber.
"""

from typing import Dict, List, Optional, Tuple
from catanatron.game import Game
from catanatron.models.board import Board
from catanatron.models.map import LandTile, Port, Coordinate
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE, SETTLEMENT, CITY


# Resource symbols
RESOURCE_SYMBOLS = {
    WOOD: "🌲",      # or "W"
    BRICK: "🧱",     # or "B"
    SHEEP: "🐑",     # or "S"
    WHEAT: "🌾",     # or "H" (harvest)
    ORE: "�ite",      # or "O"
    None: "🏜️",      # Desert
}

RESOURCE_LETTERS = {
    WOOD: "Wd",
    BRICK: "Bk",
    SHEEP: "Sh",
    WHEAT: "Wh",
    ORE: "Or",
    None: "De",  # Desert
}

# Color symbols
COLOR_SYMBOLS = {
    "RED": "R",
    "BLUE": "B",
    "ORANGE": "O",
    "WHITE": "W",
}


def render_board_ascii(game: Game, use_emoji: bool = False) -> str:
    """
    Render the game board as ASCII art.

    Args:
        game: The Catan game instance
        use_emoji: If True, use emoji symbols; otherwise use letters

    Returns:
        ASCII representation of the board
    """
    board = game.state.board
    catan_map = board.map

    lines = []
    lines.append("=" * 60)
    lines.append("CATAN BOARD")
    lines.append("=" * 60)

    # Get all land tiles sorted by coordinate for consistent display
    land_tiles = list(catan_map.land_tiles.items())

    # Group tiles by "row" (based on coordinate)
    # For hex grids, we can use the sum of first two coords as row indicator
    rows = {}
    for coord, tile in land_tiles:
        # Use y coordinate (second element) as primary row
        row_key = coord[1]
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append((coord, tile))

    # Sort rows
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])

    # Render each row of tiles
    res_map = RESOURCE_LETTERS if not use_emoji else RESOURCE_SYMBOLS

    lines.append("")
    lines.append("TILES (Resource:Number):")
    lines.append("-" * 40)

    for row_idx, (_, tiles) in enumerate(sorted_rows):
        # Sort tiles in row by x coordinate
        tiles.sort(key=lambda x: x[0][0])

        # Add indentation for hex layout
        indent = "  " * (len(sorted_rows) // 2 - row_idx) if row_idx < len(sorted_rows) // 2 else "  " * (row_idx - len(sorted_rows) // 2)
        indent = "  " * abs(len(sorted_rows) // 2 - row_idx)

        tile_strs = []
        for coord, tile in tiles:
            res = res_map.get(tile.resource, "??")
            num = tile.number if tile.number else "--"

            # Check if robber is here
            robber = "🦹" if coord == board.robber_coordinate else "  "
            if not use_emoji:
                robber = "*" if coord == board.robber_coordinate else " "

            tile_strs.append(f"[{res}:{num:>2}{robber}]")

        lines.append(indent + " ".join(tile_strs))

    lines.append("")
    lines.append(f"Robber location: {board.robber_coordinate}")

    # Render buildings
    lines.append("")
    lines.append("BUILDINGS:")
    lines.append("-" * 40)

    if board.buildings:
        buildings_by_color = {}
        for node_id, (color, building_type) in board.buildings.items():
            color_name = color.name
            if color_name not in buildings_by_color:
                buildings_by_color[color_name] = {"settlements": [], "cities": []}

            if building_type == SETTLEMENT:
                buildings_by_color[color_name]["settlements"].append(node_id)
            else:
                buildings_by_color[color_name]["cities"].append(node_id)

        for color_name, buildings in buildings_by_color.items():
            settlements = buildings["settlements"]
            cities = buildings["cities"]
            lines.append(f"  {color_name}:")
            if settlements:
                lines.append(f"    Settlements at nodes: {settlements}")
            if cities:
                lines.append(f"    Cities at nodes: {cities}")
    else:
        lines.append("  (no buildings yet)")

    # Render roads
    lines.append("")
    lines.append("ROADS:")
    lines.append("-" * 40)

    if board.roads:
        roads_by_color = {}
        seen_edges = set()
        for edge, color in board.roads.items():
            # Avoid duplicates (edges stored both ways)
            normalized = tuple(sorted(edge))
            if normalized in seen_edges:
                continue
            seen_edges.add(normalized)

            color_name = color.name
            if color_name not in roads_by_color:
                roads_by_color[color_name] = []
            roads_by_color[color_name].append(list(normalized))

        for color_name, roads in roads_by_color.items():
            lines.append(f"  {color_name}: {roads}")
    else:
        lines.append("  (no roads yet)")

    # Render ports
    lines.append("")
    lines.append("PORTS:")
    lines.append("-" * 40)

    ports = [t for t in catan_map.tiles.values() if isinstance(t, Port)]
    if ports:
        for port in ports:
            port_type = res_map.get(port.resource, "3:1 Any")
            if port.resource is None:
                port_type = "3:1 Any"
            nodes = list(port.nodes.values())
            lines.append(f"  {port_type} port at nodes {nodes}")

    lines.append("=" * 60)

    return "\n".join(lines)


def render_board_compact(game: Game) -> str:
    """
    Render a compact board representation focusing on strategic info.
    """
    board = game.state.board
    catan_map = board.map

    lines = []
    lines.append("BOARD STATE:")

    # Tiles with resources and numbers
    lines.append("\nTiles (Coord → Resource:Number):")
    for coord, tile in sorted(catan_map.land_tiles.items(), key=lambda x: (x[0][1], x[0][0])):
        res = RESOURCE_LETTERS.get(tile.resource, "??")
        num = tile.number if tile.number else "D"
        robber = " *ROBBER*" if coord == board.robber_coordinate else ""
        lines.append(f"  {coord} → {res}:{num}{robber}")

    # Node to tile mapping (which resources each node touches)
    lines.append("\nKey Settlement Spots (Node → Adjacent Resources):")
    node_resources = {}
    for coord, tile in catan_map.land_tiles.items():
        for node_id in tile.nodes.values():
            if node_id not in node_resources:
                node_resources[node_id] = []
            if tile.resource and tile.number:
                node_resources[node_id].append(f"{RESOURCE_LETTERS[tile.resource]}:{tile.number}")

    # Show top spots (nodes touching 3 tiles with good numbers)
    good_spots = []
    for node_id, resources in node_resources.items():
        if len(resources) >= 2:  # At least 2 resource tiles
            # Check if node is buildable
            if node_id in board.board_buildable_ids:
                good_spots.append((node_id, resources))

    good_spots.sort(key=lambda x: -len(x[1]))
    for node_id, resources in good_spots[:10]:
        lines.append(f"  Node {node_id}: {', '.join(resources)}")

    # Current buildings
    if board.buildings:
        lines.append("\nBuildings:")
        for node_id, (color, btype) in board.buildings.items():
            bname = "Settlement" if btype == SETTLEMENT else "City"
            lines.append(f"  {color.name}: {bname} at node {node_id}")

    # Current roads
    if board.roads:
        lines.append("\nRoads:")
        seen = set()
        for edge, color in board.roads.items():
            norm = tuple(sorted(edge))
            if norm not in seen:
                seen.add(norm)
                lines.append(f"  {color.name}: {list(norm)}")

    return "\n".join(lines)


def render_tile_info(game: Game) -> str:
    """
    Render detailed tile information for strategic analysis.
    """
    board = game.state.board
    catan_map = board.map

    lines = []
    lines.append("TILE DETAILS:")
    lines.append("-" * 50)

    # Probability of each number
    DICE_PROBS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    # Group by resource
    by_resource = {}
    for coord, tile in catan_map.land_tiles.items():
        res = tile.resource
        if res not in by_resource:
            by_resource[res] = []
        by_resource[res].append((coord, tile))

    for resource, tiles in by_resource.items():
        res_name = RESOURCE_LETTERS.get(resource, "Desert")
        lines.append(f"\n{res_name}:")
        for coord, tile in tiles:
            if tile.number:
                prob = DICE_PROBS.get(tile.number, 0)
                robber = " [ROBBER]" if coord == board.robber_coordinate else ""
                lines.append(f"  {coord}: Number {tile.number} (prob: {prob}/36){robber}")
                lines.append(f"    Nodes: {list(tile.nodes.values())}")

    return "\n".join(lines)


def get_node_resource_info(game: Game) -> Dict[int, List[str]]:
    """
    Get a mapping of node_id to list of adjacent resources with numbers.
    Useful for LLMs to evaluate settlement spots.
    """
    board = game.state.board
    catan_map = board.map

    node_resources = {}
    for coord, tile in catan_map.land_tiles.items():
        for node_id in tile.nodes.values():
            if node_id not in node_resources:
                node_resources[node_id] = []
            if tile.resource and tile.number:
                is_robber = coord == board.robber_coordinate
                node_resources[node_id].append({
                    "resource": RESOURCE_LETTERS[tile.resource],
                    "number": tile.number,
                    "blocked_by_robber": is_robber,
                })

    return node_resources
