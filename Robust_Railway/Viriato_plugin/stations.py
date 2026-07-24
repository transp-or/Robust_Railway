from typing import Any

from py_client.aidm import RoutingPoint

from Robust_Railway.event_activity_graph_multitracks import EARailwayNetwork, NodeTrack, Station


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
    EAG: EARailwayNetwork,
    station_ids: list[int],
    junction_ids: list[int],
    node_tracks: dict[int, list[NodeTrack]],
    api: Any,
) -> EARailwayNetwork:
    """
    Adds stations and junctions to the event-activity graph.

    Args:
        EAG [EARailwayNetwork]: Graph object.
        station_ids [list[int]]: Station IDs.
        junction_ids [list[int]]: Junction IDs.
        node_tracks [dict[int, list[NodeTrack]]]: Node tracks per station.
        api [Any]: API client.
    Returns:
        EARailwayNetwork: Updated graph.
    """
    for station_id in station_ids + junction_ids:
        shunting_capacity = False if station_id in junction_ids else True
        nt = []
        for n in node_tracks.get(station_id, []):
            routing_point = RoutingPoint(station_id, n.id)
            incoming_routes = api.get_incoming_routing_edges(routing_point)
            outgoing_routes = api.get_outgoing_routing_edges(routing_point)
            incoming_section_tracks, outgoing_section_tracks = [], []
            for ir in incoming_routes:
                incoming_section_tracks.append(ir.start_section_track_id)
            for outr in outgoing_routes:
                outgoing_section_tracks.append(outr.end_section_track_id)
            nt.append(NodeTrack(n.id, str(n.code), incoming_section_tracks, outgoing_section_tracks))
        EAG.add_station(
            Station(
                station_id=station_id,
                code=EAG.id_to_code[station_id],
                node_tracks=nt,
                shunting_yard_capacity=shunting_capacity,
                junction=station_id in junction_ids,
            )
        )
    return EAG
