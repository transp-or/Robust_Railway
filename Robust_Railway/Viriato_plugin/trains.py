from collections import defaultdict
from typing import Any

from Robust_Railway.event_activity_graph_multitracks import (
    EARailwayNetwork,
    Train,
)


def create_trains(
    EAG: EARailwayNetwork, api: Any, trains: list[Any], TRAINS_TO_IGNORE: set[str]
) -> tuple[EARailwayNetwork, dict, dict]:
    """
    Creates train objects in the event-activity graph.

    Args:
        EAG [EARailwayNetwork]: Graph object.
        api [Any]: API client.
        trains [list]: List of train objects.
        TRAINS_TO_IGNORE [set[str]]: Train codes to ignore.

    Returns:
        tuple: (EAG, travel_times [dict], stations_visited [dict])
    """
    travel_times: defaultdict[tuple, dict] = defaultdict(dict)
    stations_visited: defaultdict[Any, list] = defaultdict(list)
    for train in trains:
        visited_stations = {}
        print("train code", train.code, "in TRAINS_TO_IGNORE?", train.code in TRAINS_TO_IGNORE)
        if train.code in TRAINS_TO_IGNORE:
            continue
        begin_at, end_at = get_valid_train_segment(train, EAG)
        print(f"train {train.code} begins at {begin_at} and ends at {end_at}")
        if end_at is None or begin_at is None or end_at - begin_at <= 1:
            continue
        to_exclude = False
        for i, node_idx in enumerate(range(begin_at, end_at + 1)):
            node = train.train_path_nodes[node_idx]
            station = EAG.get_station_by_id(node.node_id)
            if not station:
                to_exclude = True
                print(f"exclude train {train.code} at node {node_idx}")
            else:
                visited_stations[station] = i
        print(f"Train {train.code} to exclude ? {to_exclude}")
        if not to_exclude:
            capacity = get_train_capacity(train, api)
            train_path_nodes = sorted([node for node in train.train_path_nodes], key=lambda node: node.arrival_time)
            new_train = Train(train.id, capacity, train_path_nodes, begin_at, end_at, train.code, visited_stations)
            print(f"Train {train.code} has capacity {capacity}")
            EAG.add_train(new_train)
            prev_node = train.train_path_nodes[0]
            stations_visited[new_train].append(EAG.get_station_by_id(prev_node.node_id))
            for node in train.train_path_nodes[1:]:
                if node.section_track_id:
                    stations_visited[new_train].append(EAG.get_station_by_id(node.node_id))
                    travel_times[(prev_node.node_id, node.node_id)][new_train] = max(
                        0.1, node.minimum_run_time.total_seconds() / 60
                    )
                    travel_times[(node.node_id, prev_node.node_id)][new_train] = max(
                        0.1, node.minimum_run_time.total_seconds() / 60
                    )
                prev_node = node
    remove_trains_splitting(EAG)
    return EAG, travel_times, stations_visited


def get_train_capacity(train: Any, api: Any) -> int:
    """
    Returns the total capacity of a train (first + second class seats).
    """
    formation = api.get_formation(train.train_path_nodes[0].formation_id)
    return formation.places_first_class + formation.places_second_class


def remove_trains_splitting(EAG: EARailwayNetwork) -> None:
    """
    Detect and print trains that split at some point (same station and time).
    """
    for t1 in EAG.trains:
        for t2 in EAG.trains:
            if t1 != t2:
                prec_equi = False
                if t1.train_path_nodes is not None and t2.train_path_nodes is not None:
                    for n1 in t1.train_path_nodes[t1.begin_at : t1.end_at]:
                        for n2 in t2.train_path_nodes[t2.begin_at : t2.end_at]:
                            time_n1 = n1.arrival_time.hour * 60 + n1.arrival_time.minute
                            time_n2 = n2.arrival_time.hour * 60 + n2.arrival_time.minute
                            if n1.node_id == n2.node_id and time_n1 == time_n2:
                                if prec_equi:
                                    print("Trains splitting", t1.id, t1.code, t2.id, t2.code, n1.node_id)
                                    prec_equi = True
                                else:
                                    prec_equi = True


def get_valid_train_segment(train, EAG):
    """
    Finds the valid train path segment inside RER Vaud.

    Args:
        train: Train object.
        EAG: Event-Activity Graph.

    Returns:
        tuple: (begin_at, end_at)
    """
    stations = {s.id for s in EAG.stations}

    def get_scheduled_time(node, event_type):
        time_of_event = node.arrival_time if event_type == "arrival" else node.departure_time
        return time_of_event.hour * 60 + time_of_event.minute

    def find_first_valid(idx_range, beginning):
        if beginning:
            return next(
                (
                    i
                    for i in idx_range
                    if (
                        train.train_path_nodes[i].node_id in stations
                        and get_scheduled_time(train.train_path_nodes[i], "departure")
                        >= EAG.start_time_window + EAG.waiting_time
                        and not EAG.get_station_by_id(train.train_path_nodes[i].node_id).junction
                    )
                ),
                None,
            )
        else:
            return next(
                (
                    i
                    for i in idx_range
                    if (
                        train.train_path_nodes[i].node_id in stations
                        and get_scheduled_time(train.train_path_nodes[i], "arrival")
                        <= EAG.end_time_window - EAG.waiting_time
                        and not EAG.get_station_by_id(train.train_path_nodes[i].node_id).junction
                    )
                ),
                None,
            )

    begin_at = find_first_valid(range(len(train.train_path_nodes)), True)
    end_at = find_first_valid(reversed(range(len(train.train_path_nodes))), False)

    if begin_at is None or end_at is None:
        print(f"Train {train.code}, {train.id} has no valid stations in RER Vaud.")
        # raise ValueError(f"Train {train.id} has no valid stations in RER Vaud.")

    return begin_at, end_at
