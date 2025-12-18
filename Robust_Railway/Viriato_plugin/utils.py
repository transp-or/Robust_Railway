import ast
import csv
from typing import Any

import numpy as np
from py_client.aidm import RoutingPoint

from Robust_Railway.event_activity_graph_multitracks import Activity, EARailwayNetwork, Event, Train


def load_csv_column(file_path: str, column_name: str) -> list[str]:
    """Loads a specific column from a CSV file into a list."""
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        return [row[column_name] for row in csv.DictReader(csvfile)]


def get_section_codes(station_ids: list[int], api: Any) -> set[str]:
    """Search for all outgoing sections of all stations in a list."""
    sections: set[str] = set()
    for s_id in station_ids:
        sections.update(st.section_code for st in api.get_section_tracks_from(s_id))
    return sections


def get_sections_tracks_per_station_node_tracks(
    api: Any, station_id: int, node_track: int, sections_code: set[str]
) -> tuple[list[int], list[int]]:
    """Get section tracks entering and leaving a station node track."""
    section_tracks_entering = []
    section_tracks_leaving = []
    routing_point = RoutingPoint(station_id, node_track)
    incoming_routes = api.get_incoming_routing_edges(routing_point)
    outcoming_routes = api.get_outgoing_routing_edges(routing_point)
    for incoming_route in incoming_routes:
        if api.get_section_track(incoming_route.start_section_track_id).section_code in sections_code:
            section_tracks_entering.append(incoming_route.start_section_track_id)
    for outcoming_route in outcoming_routes:
        if api.get_section_track(outcoming_route.end_section_track_id).section_code in sections_code:
            section_tracks_leaving.append(outcoming_route.end_section_track_id)
    return section_tracks_entering, section_tracks_leaving


def find_section_track_code(api: Any, section_track_id_lst: list[int]) -> list[str]:
    """Get section track codes from IDs."""
    return [api.get_section_track(st_id).code for st_id in section_track_id_lst]


def convert_to_EAG_obj(EAG, results):
    """
    Converts raw event and activity arrays into EAG objects and updates the graph.

    Args:
        EAG: Event-Activity Graph object.
        results: List of results containing event/activity arrays and dictionaries.

    Returns:
        tuple: (EAG, event_id_to_obj)
    """
    tot_events, tot_activities = np.empty(shape=[0, 11]), np.empty(shape=[0, 10])

    for result in results:
        if result is None:
            continue  # Skip if a train was ignored
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            local_events,
            local_activities,
        ) = result

        tot_events = np.append(tot_events, local_events, axis=0)
        tot_activities = np.append(tot_activities, local_activities, axis=0)

    event_id_to_obj, activity_id_to_obj = {}, {}

    # Add events to event-activity graph
    # Replace None with a default value (e.g., -1)
    tot_events = np.array([[val if val is not None else -1 for val in row] for row in tot_events])
    tot_events = tot_events[np.lexsort((tot_events[:, 3], tot_events[:, 5], tot_events[:, 6]))]
    previous_station_id, previous_node_track_id, previous_train_id, previous_node_track_planned_id = (
        None,
        None,
        None,
        None,
    )
    for event in tot_events:
        if event[3] != previous_station_id:
            station = EAG.get_station_by_id(event[3])
        if event[4] != previous_node_track_id:
            node_track = EAG.get_node_track_by_id(event[4], station)
        if event[5] != previous_node_track_planned_id:
            node_track_planned = EAG.get_node_track_by_id(event[5], station)
        if event[6] != previous_train_id:
            train = EAG.get_train_by_id(event[6])
        event_to_add = Event(
            event[0],
            event[1],
            float(event[2]),
            station,
            node_track,
            node_track_planned,
            train,
            event[7],
            str(event[8]) == "True",
            str(event[9]) == "True",
            event[10],
        )
        EAG.add_event(event_to_add)
        previous_station_id = event[3]
        previous_node_track_id = event[4]
        previous_node_track_planned_id = event[5]
        previous_train_id = event[6]
        event_id_to_obj[int(event[0])] = event_to_add
        if not event_to_add.aggregated:
            EAG.add_E_train_item(train, event_to_add)
        else:
            EAG.add_E_train_aggregated_item(train, event_to_add)

    # Replace None with a default value (e.g., -1)
    tot_activities = np.array([[val if val is not None else -1 for val in row] for row in tot_activities])
    col1 = tot_activities[:, 1].astype(int)
    col2 = tot_activities[:, 2].astype(int)
    col3 = tot_activities[:, 3].astype(int)
    tot_activities = tot_activities[np.lexsort((col1, col2, col3))]
    previous_origin_id, previous_dest_id, previous_group_id = None, None, None
    for activity in tot_activities:
        if activity[1] != previous_origin_id:
            origin = EAG.get_event_by_id(activity[1])
        if activity[2] != previous_dest_id:
            destination = EAG.get_event_by_id(activity[2])
        if activity[3] == "-1":
            group = None
        elif activity[3] != previous_group_id:
            group = EAG.get_group_by_id(activity[3])
        if activity[5] == "-1":
            section_track = None
        else:
            section_track = EAG.get_section_track_by_id(activity[5])
        if str(activity[8]) == "-1":
            intermediate_stations = None
        else:
            intermediate_stations = []
            intermediate_stations_ids = ast.literal_eval(activity[8])
            for station_id in intermediate_stations_ids:
                intermediate_stations.append(EAG.get_station_by_id(station_id))
        if activity[9]:
            section_track_planned = EAG.get_section_track_by_id(int(activity[9]))
        else:
            section_track_planned = None

        activity_to_add = Activity(
            activity[0],
            origin,
            destination,
            group,
            activity[4],
            section_track,
            str(activity[6]) == "True",
            str(activity[7]) == "True",
            intermediate_stations,
            section_track_planned,
        )

        EAG.add_activities(activity_to_add)
        previous_origin_id = activity[1]
        previous_dest_id = activity[2]
        previous_group_id = activity[3]
        activity_id_to_obj[int(activity[0])] = activity_to_add
        if not activity_to_add.aggregated:
            EAG.add_A_train_item(origin.train, activity_to_add)
        else:
            EAG.add_A_train_aggregated_item(origin.train, activity_to_add)

    for result in results:
        if result is None:
            continue  # Skip if a train was ignored
        (
            local_events_with_different_node_tracks,
            local_train_running_similar,
            local_A_plus,
            local_A_minus,
            local_A_minus_agg,
            local_A_plus_agg,
            local_agg_to_disagg_events,
            local_disagg_to_agg_events,
            local_agg_to_disagg_activities,
            local_disagg_to_agg_activities,
            local_nb_pass_through_activities,
            local_nb_waiting_activities,
            _,
            _,
        ) = result

        for events in local_events_with_different_node_tracks:
            events_lst = []
            for event in events:
                events_lst.append(EAG.get_event_by_id(int(event)))
            EAG.add_events_with_different_node_tracks(events_lst)
        for key, val in local_train_running_similar.items():
            key_obj = (
                EAG.get_station_by_id(key[0]),
                EAG.get_station_by_id(key[1]),
                int(key[2]),
                EAG.get_train_by_id(key[3]),
            )
            val_obj = []
            for v in val:
                val_obj.append(activity_id_to_obj[int(v)])
            EAG.add_A_train_running_similar_item(key_obj, val_obj)
        for key, val in local_A_plus.items():
            key_obj = event_id_to_obj[int(key)]
            val_obj = []
            for v in val:
                val_obj.append(activity_id_to_obj[int(v)])
            EAG.add_A_plus_item(key_obj, val_obj)
        for key, val in local_A_minus.items():
            key_obj = event_id_to_obj[int(key)]
            val_obj = []
            for v in val:
                val_obj.append(activity_id_to_obj[int(v)])
            EAG.add_A_minus_item(key_obj, val_obj)
        for key, val in local_A_minus_agg.items():
            key_obj = event_id_to_obj[int(key)]
            val_obj = []
            for v in val:
                val_obj.append(activity_id_to_obj[int(v)])
            EAG.add_A_minus_agg_item(key_obj, val_obj)
        for key, val in local_A_plus_agg.items():
            key_obj = event_id_to_obj[int(key)]
            val_obj = []
            for v in val:
                val_obj.append(activity_id_to_obj[int(v)])
            EAG.add_A_plus_agg_item(key_obj, val_obj)
        for key, val in local_agg_to_disagg_events.items():
            key_obj = event_id_to_obj[int(key)]
            val_obj = []
            for v in val:
                val_obj.append(event_id_to_obj[int(v)])
            EAG.add_agg_to_disagg_events_item(key_obj, val_obj)
        for key, val in local_disagg_to_agg_events.items():
            key_obj = event_id_to_obj[int(key)]
            val_obj = event_id_to_obj[int(val)]
            EAG.add_disagg_to_agg_events_item(key_obj, val_obj)
        for key, val in local_agg_to_disagg_activities.items():
            key_obj = activity_id_to_obj[int(key)]
            val_obj = []
            for v in val:
                val_obj.append(activity_id_to_obj[int(v)])
            EAG.add_agg_to_disagg_activities_item(key_obj, val_obj)
        for key, val in local_disagg_to_agg_activities.items():
            key_obj = activity_id_to_obj[int(key)]
            val_obj = activity_id_to_obj[int(val)]
            EAG.add_disagg_to_agg_activities_item(key_obj, val_obj)
        EAG.nb_pass_through_activities += local_nb_pass_through_activities
        EAG.nb_waiting_activities += local_nb_waiting_activities

    return EAG, event_id_to_obj


def get_node_tracks(station_ids: list[int], api: Any) -> dict[int, list[int]]:
    """Returns a dictionary mapping stations to their node tracks."""
    return {s_id: [nt.id for nt in api.get_node(s_id).node_tracks] for s_id in station_ids}


def train_turning(train: Train, EAG: EARailwayNetwork) -> bool:
    """Checks if a train is turning (changing direction) in its path."""
    stations_visited = []
    previous_station = None
    turning = False
    for a in EAG.get_ordered_activities_train(train):
        if a.origin.station not in stations_visited:
            stations_visited.append(a.origin.station)
        else:
            if previous_station != a.origin.station:
                turning = True
                break
        previous_station = a.origin.station
    return turning
