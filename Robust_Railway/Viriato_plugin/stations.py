from typing import Any

from Robust_Railway.event_activity_graph_multitracks import (
    EARailwayNetwork,
    Station,
)


def get_stations_and_junctions(
    node_codes: list[str], junction_codes: list[str], api: Any
) -> tuple[list[int], list[int], dict[int, str], dict[str, int]]:
    """
    Retrieve station and junction node information from the API.

    Args:
        node_codes [list[str]]: List of station codes.
        junction_codes [list[str]]: List of junction codes.
        api [Any]: API client.

    Returns:
        tuple: (station_ids, junction_ids, id_to_code, code_to_id)
    """
    stations, junctions = [], []
    id_to_code, code_to_id = {}, {}
    for node in api.get_all_nodes():
        node_id, node_code = node.id, node.code
        if node_code in node_codes:
            stations.append(node_id)
            id_to_code[node_id] = node_code
            code_to_id[node_code] = node_id
        if node_code in junction_codes:
            junctions.append(node_id)
            id_to_code[node_id] = node_code
            code_to_id[node_code] = node_id
    return stations, junctions, id_to_code, code_to_id


def create_stations(
    EAG: EARailwayNetwork, station_ids: list[int], junction_ids: list[int], node_tracks: dict[int, list[int]]
) -> EARailwayNetwork:
    """
    Adds stations and junctions to the event-activity graph.

    Args:
        EAG [EARailwayNetwork]: Graph object.
        station_ids [list[int]]: Station IDs.
        junction_ids [list[int]]: Junction IDs.
        node_tracks [dict[int, list[int]]]: Node tracks per station.

    Returns:
        EARailwayNetwork: Updated graph.
    """
    for station_id in station_ids + junction_ids:
        shunting_capacity = False if station_id in junction_ids else True
        EAG.add_station(
            Station(
                station_id=station_id,
                code=EAG.id_to_code[station_id],
                node_tracks=node_tracks.get(station_id, []),
                shunting_yard_capacity=shunting_capacity,
                junction=station_id in junction_ids,
            )
        )
    return EAG
