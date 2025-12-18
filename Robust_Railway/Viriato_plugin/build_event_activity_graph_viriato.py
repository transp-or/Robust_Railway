import csv
import pickle
from collections import defaultdict

import pandas as pd
from joblib import Parallel, delayed
from py_client.algorithm_interface import algorithm_interface_factory

from Robust_Railway.event_activity_graph_multitracks import (
    Activity,
    Disruption,
    EARailwayNetwork,
    Event,
)
from Robust_Railway.Viriato_plugin.events_activities import (
    add_emergency_buses,
    find_all_transfering_events_pairs,
    process_short_turning,
    process_train,
)
from Robust_Railway.Viriato_plugin.passengers import create_passenger_groups, find_all_simple_paths
from Robust_Railway.Viriato_plugin.sections import create_section_tracks
from Robust_Railway.Viriato_plugin.stations import create_stations, get_stations_and_junctions
from Robust_Railway.Viriato_plugin.trains import create_trains
from Robust_Railway.Viriato_plugin.utils import (
    convert_to_EAG_obj,
    get_node_tracks,
    get_section_codes,
    get_sections_tracks_per_station_node_tracks,
    load_csv_column,
    train_turning,
)


def create_events_and_activities(
    api,
    EAG,
    all_paths,
    INTERSECTIONS,
    intermediate_stations,
    incoming_tracks,
    outgoing_tracks,
):
    """Creates events and activities for all trains in the given window."""

    results = Parallel(n_jobs=12)(
        delayed(process_train)(
            train,
            EAG,
            all_paths,
            intermediate_stations,
            api,
            incoming_tracks,
            outgoing_tracks,
        )
        for train in EAG.trains
    )

    EAG, _ = convert_to_EAG_obj(EAG, results)

    # Create passenger origin and destination events, and access, egress, penalty, and transfer activities
    arrival_events = defaultdict(list)
    departure_events = defaultdict(list)
    for e in EAG.events:
        if e.event_type == "arrival" and e.aggregated:
            arrival_events[e.station].append(e)
        elif e.event_type == "departure" and e.aggregated:
            departure_events[e.station].append(e)

    idx_events = 0
    idx_activities = idx_events * 100_000
    for group in EAG.passengers_groups:
        # Create access and egress nodes
        access_node = Event(
            idx_events, None, None, group.origin, None, None, None, "passenger origin", True, False, None
        )
        idx_events += 1
        egress_node = Event(
            idx_events, None, None, group.destination, None, None, None, "passenger destination", True, False, None
        )
        idx_events += 1
        EAG.add_event(access_node)
        EAG.add_event(egress_node)

        # Create access, egress, and penalty activities
        activity = Activity(idx_activities, access_node, egress_node, group, "penalty", None, True, False)
        EAG.add_activities(activity)
        idx_activities += 1

        for node in EAG.events:
            if node.node_type == "regular" and node.aggregated:
                if node.station.id == group.origin.id and node.event_type == "arrival":
                    activity = Activity(idx_activities, access_node, node, group, "access", None, True, False)
                    EAG.add_activities(activity)
                    EAG.A_minus_agg[access_node].append(activity)
                    EAG.A_plus_agg[node].append(activity)
                    idx_activities += 1
                elif node.station.id == group.destination.id and node.event_type == "departure":
                    activity = Activity(idx_activities, node, egress_node, group, "egress", None, True, False)
                    EAG.add_activities(activity)
                    EAG.A_minus_agg[node].append(activity)
                    EAG.A_plus_agg[egress_node].append(activity)
                    idx_activities += 1

    for train1 in EAG.trains:
        for train2 in EAG.trains:
            if train1 == train2:
                continue
            transfering_pairs = find_all_transfering_events_pairs(EAG, train1, train2, INTERSECTIONS, False)
            for event1, event2 in transfering_pairs:
                activity = Activity(idx_activities, event1, event2, None, "transferring", None, True, False)
                EAG.add_activities(activity)
                EAG.A_minus_agg[event1].append(activity)
                EAG.A_plus_agg[event2].append(activity)
                idx_activities += 1

    return EAG, idx_events, idx_activities


def add_short_turning(EAG, idx_events, idx_activities, all_paths, intermediate_stations, INTERSECTIONS):
    incoming_tracks = EAG.incoming_tracks
    outgoing_tracks = EAG.outgoing_tracks
    d_stations = {t.origin for t in EAG.disruption_scenario.section_tracks}
    d_stations.update({t.destination for t in EAG.disruption_scenario.section_tracks})

    results = Parallel(n_jobs=12)(
        delayed(process_short_turning)(
            train,
            EAG,
            d_stations,
            all_paths,
            intermediate_stations,
            incoming_tracks,
            outgoing_tracks,
        )
        for train in EAG.trains
    )

    shunting_stations = {}
    first_events_all = {}
    return_lst_all = []
    for result in results:
        train_id = result["train"]
        shunting_stations[train_id] = result["shunting_station"]
        first_events_all[train_id] = result["first_events_idx"]
        return_lst_all.append(result["return_lst"])

    EAG, _ = convert_to_EAG_obj(EAG, return_lst_all)

    # Add short-turning activities:
    for train in EAG.trains:
        if train.id not in shunting_stations or train.id not in first_events_all:
            print(f"train {train.id} not passing in disruption")
            print("train obj", train)
            continue
        if train_turning(train, EAG):
            continue

        shunting_station = shunting_stations[train.id]
        first_events = first_events_all[train.id]
        similar_activities = []
        for e1 in EAG.E_train[train]:
            if e1.station.id == shunting_station and e1.event_type == "departure" and e1.node_type == "regular":
                for e2_id in first_events:
                    e2 = EAG.get_event_by_id(e2_id)
                    if e2.event_type != "arrival":
                        continue
                    if e1.node_track == e2.node_track:
                        activity_to_add = Activity(idx_activities, e1, e2, None, "short-turning", None, False, False)
                        similar_activities.append(activity_to_add)
                        EAG.add_activities(activity_to_add)
                        EAG.add_A_train_item(e1.train, activity_to_add)
                        EAG.add_A_plus_item(e2, activity_to_add)
                        EAG.add_A_minus_item(e1, activity_to_add)
                        idx_activities += 1
        EAG.add_similar_short_turning(similar_activities)

    # Create access and egress nodes for short-turning
    for group in EAG.passengers_groups:
        access_node, egress_node = None, None
        for e in EAG.events:
            if e.aggregated and e.node_type == "passenger origin" and e.station == group.origin:
                access_node = e
            elif e.aggregated and e.node_type == "passenger destination" and e.station == group.destination:
                egress_node = e
        for node in EAG.events:
            if node.node_type == "short-turning" and node.aggregated:
                if node.station.id == group.origin.id and node.event_type == "arrival":
                    activity = Activity(idx_activities, access_node, node, group, "access", None, True, False)
                    EAG.add_activities(activity)
                    EAG.A_minus_agg[access_node].append(activity)
                    EAG.A_plus_agg[node].append(activity)
                    idx_activities += 1
                elif node.station.id == group.destination.id and node.event_type == "departure":
                    activity = Activity(idx_activities, node, egress_node, group, "egress", None, True, False)
                    EAG.add_activities(activity)
                    EAG.A_minus_agg[node].append(activity)
                    EAG.A_plus_agg[egress_node].append(activity)
                    idx_activities += 1

    # Add transfer activities (excluding emergency buses)
    for train1 in EAG.trains:
        for train2 in EAG.trains:
            if train1 == train2:
                continue
            transfering_pairs = find_all_transfering_events_pairs(EAG, train1, train2, INTERSECTIONS, True)
            for event1, event2 in transfering_pairs:
                activity = Activity(idx_activities, event1, event2, None, "transferring", None, True, False)
                EAG.add_activities(activity)
                EAG.A_minus_agg[event1].append(activity)
                EAG.A_plus_agg[event2].append(activity)
                idx_activities += 1

    # Add transfer activities for emergency buses
    for bus in EAG.buses:
        a1 = next((a for a in EAG.activities if a.activity_type == "emergency bus" and a.origin.train == bus), None)
        if a1:
            for a2 in EAG.activities:
                if (
                    a2.activity_type == "dwelling"
                    and a2.origin.node_type == "short-turning"
                    and a2.origin.station == a1.destination.station
                ):
                    new_activity = Activity(
                        idx_activities, a1.destination, a2.origin, None, "transferring", None, True, False
                    )
                    EAG.add_activities(new_activity)
                    EAG.add_A_minus_agg_item(a1.destination, new_activity)
                    EAG.add_A_plus_agg_item(a2.origin, new_activity)
                    idx_activities += 1
        else:
            raise ValueError(f"No emergency bus activity defined for bus {bus.id}")
    return EAG


def build_graph_from_viriato(
    api_url,
    RER_Vaud_junctions_file,
    links_file,
    RER_Vaud_stations_file,
    RER_Vaud_travel_time_by_bus,
    df,
    probabilities,
    initial_timetable_and_graph_ID,
    solve_init_timetable,
    MANUAL_LINK_CORRECTIONS,
    TRAINS_TO_IGNORE,
    INTERSECTIONS,
    **kwargs,
):
    """
    Build the event-activity graph from Viriato data and configuration files.
    Handles both initial timetable and disruption scenarios.
    """
    # Load precomputed graph if available
    if initial_timetable_and_graph_ID:
        with open(
            f"results_event_activity/Viriato_network/EAG_updated_{initial_timetable_and_graph_ID}.pkl",
            "rb",
        ) as f:
            EAG = pickle.load(f)
        idx_events = max(EAG.events_ids) + 1
        idx_activities = max(EAG.activities_ids) + 1
        all_paths_dict = EAG.all_paths_dict
        intermediate_stations = EAG.intermediate_stations
        start_time_window = EAG.start_time_window
        end_time_window = EAG.end_time_window
        print("Start time window", start_time_window)
        print("End time window", end_time_window)
    else:
        # Build graph from scratch using API
        with algorithm_interface_factory.create(api_url) as api:
            time_window = api.get_time_window_algorithm_parameter("timeWindowParameterMandatory")
            start_time_window = time_window.from_time.hour * 60 + time_window.from_time.minute
            end_time_window = time_window.to_time.hour * 60 + time_window.to_time.minute
            print("Start time window", start_time_window)
            print("End time window", end_time_window)

            EAG = EARailwayNetwork(start_time_window, end_time_window, **kwargs)

            # Load station and junction codes
            station_codes = load_csv_column(RER_Vaud_stations_file, "station_code")
            junction_codes = load_csv_column(RER_Vaud_junctions_file, "node_code")
            stations, junctions, id_to_code, code_to_id = get_stations_and_junctions(station_codes, junction_codes, api)
            EAG.code_to_id = code_to_id
            EAG.id_to_code = id_to_code

            # Get node tracks for all stations and junctions
            station_tracks = get_node_tracks(stations + junctions, api)
            EAG = create_stations(EAG, stations, junctions, station_tracks)
            for station in EAG.stations:
                node_tracks = [t.id for t in station.node_tracks]
                print("station:", station.id, id_to_code[station.id], node_tracks)

            sections_code = get_section_codes(stations, api)
            with open(links_file, encoding="utf-8") as f:
                links = [(row["origin_code"], row["destination_code"]) for row in csv.DictReader(f)]

            # Build incoming/outgoing tracks dictionaries
            incoming_tracks = {}
            outgoing_tracks = {}
            for station in stations + junctions:
                for node_track in station_tracks[station]:
                    section_tracks_entering, section_tracks_leaving = get_sections_tracks_per_station_node_tracks(
                        api, station, node_track, sections_code
                    )
                    incoming_tracks[(station, node_track)] = section_tracks_entering
                    outgoing_tracks[(station, node_track)] = section_tracks_leaving
            EAG.add_incoming_outcoming_tracks(incoming_tracks, outgoing_tracks)

            # Junction tracks
            incoming_junction_tracks = {junction: api.get_section_tracks_to(junction) for junction in junctions}
            outgoing_junction_tracks = {junction: api.get_section_tracks_from(junction) for junction in junctions}

            # Get trains from API
            trains = api.get_trains_driving_any_node(time_window, stations)
            print("Number of trains in API", len(trains))

            EAG, travel_times, stations_visited = create_trains(EAG, api, trains, TRAINS_TO_IGNORE)
            print("Number of trains in EAG", len(EAG.trains))

            # Build section tracks
            EAG, intermediate_stations, tt_links = create_section_tracks(
                api,
                EAG,
                links,
                junctions,
                station_tracks,
                incoming_tracks,
                outgoing_tracks,
                incoming_junction_tracks,
                outgoing_junction_tracks,
                MANUAL_LINK_CORRECTIONS,
                travel_times,
                stations_visited,
            )
            print("Number of section tracks:", len(EAG.section_tracks))
            EAG.intermediate_stations = intermediate_stations

            nb_station_tracks = sum(len(station.node_tracks) for station in EAG.stations)
            print("Number of station tracks:", nb_station_tracks)

            # Load passenger groups if available, else create them
            if initial_timetable_and_graph_ID:
                with open(
                    f"results_event_activity/Viriato_network/EAG_updated_{initial_timetable_and_graph_ID}.pkl",
                    "rb",
                ) as f:
                    EAG_prime = pickle.load(f)
                    for g in EAG_prime.passengers_groups:
                        EAG.add_passengers_group(g)
            else:
                EAG = create_passenger_groups(EAG, df, probabilities, links, tt_links)

            for group in EAG.passengers_groups:
                print(
                    f"Group {group.id} going from {group.origin.id} to {group.destination.id} "
                    f"at desired starting time {group.time}"
                )

            # Find all simple paths for passengers
            all_paths_dict, _ = find_all_simple_paths(EAG, links, tt_links)
            EAG.all_paths_dict = all_paths_dict

            # Create events and activities for trains and passengers
            EAG, idx_events, idx_activities = create_events_and_activities(
                api,
                EAG,
                all_paths_dict,
                INTERSECTIONS,
                intermediate_stations,
                incoming_tracks,
                outgoing_tracks,
            )

    # Handle disruption scenario if required
    if not solve_init_timetable:
        with algorithm_interface_factory.create(api_url) as api:
            time_window = api.get_time_window_algorithm_parameter("timeWindowParameterMandatory")
            print("Add a disruption scenario")
            st_closures = api.get_section_track_closures(time_window)
            nt_closures = api.get_node_track_closures(time_window)

            if st_closures and nt_closures:
                raise ValueError("Simultaneous section track and node track closure not implemented")

            section_tracks_disruption = set()
            node_tracks_disruption = set()
            start_time_disruption = None
            end_time_disruption = None

            # Process section track closures
            for c in st_closures:
                disruption_start = (
                    c.closure_time_window_from_node.from_time.hour * 60
                    + c.closure_time_window_from_node.from_time.minute
                )
                disruption_end = (
                    c.closure_time_window_to_node.to_time.hour * 60 + c.closure_time_window_to_node.to_time.minute
                )
                if start_time_disruption and disruption_start != start_time_disruption:
                    raise ValueError("All disruptions should have the same start time")
                if end_time_disruption and disruption_end != end_time_disruption:
                    raise ValueError("All disruptions should have the same end time")
                if (
                    c.closure_time_window_from_node.from_time != c.closure_time_window_to_node.from_time
                    or c.closure_time_window_from_node.to_time != c.closure_time_window_to_node.to_time
                ):
                    raise ValueError("Disruption times mismatch at section track ends")
                start_time_disruption = disruption_start
                end_time_disruption = disruption_end
                st = EAG.get_section_track_by_id(c.section_track_id)
                section_tracks_disruption.add(st)

            # Process node track closures
            for c in nt_closures:
                disruption_start = (
                    c.closure_time_window_from_node.from_time.hour * 60
                    + c.closure_time_window_from_node.from_time.minute
                )
                disruption_end = (
                    c.closure_time_window_to_node.to_time.hour * 60 + c.closure_time_window_to_node.to_time.minute
                )
                if start_time_disruption and disruption_start != start_time_disruption:
                    raise ValueError("All disruptions should have the same start time")
                if end_time_disruption and disruption_end != end_time_disruption:
                    raise ValueError("All disruptions should have the same end time")
                if (
                    c.closure_time_window_from_node.from_time != c.closure_time_window_to_node.from_time
                    or c.closure_time_window_from_node.to_time != c.closure_time_window_to_node.to_time
                ):
                    raise ValueError("Disruption times mismatch at section track ends")
                start_time_disruption = disruption_start
                end_time_disruption = disruption_end
                nt = EAG.get_node_track_by_id(c.node_track_id)
                node_tracks_disruption.add(nt)

            # If there is a disruption scenario, add it to the graph
            if st_closures or nt_closures:
                EAG.add_disruption_scenario(
                    Disruption(
                        start_time_disruption,
                        end_time_disruption,
                        node_tracks_disruption,
                        section_tracks_disruption,
                    )
                )

                if disruption_start != EAG.start_time_window:
                    raise ValueError("Start of the disruption should be equal to the start of the time window")

                print("Disruption scenario")
                print("Node tracks_disruption:", [nt.id for nt in node_tracks_disruption])
                print("Section tracks_disruption:", [st.id for st in section_tracks_disruption])
                print("disruption start time:", start_time_disruption)
                print("disruption end time:", end_time_disruption)

                # Load bus travel times and distances
                df_bus = pd.read_csv(RER_Vaud_travel_time_by_bus)
                travel_time_dict = {
                    (row["origin"], row["destination"]): row["travel_time_bus (min)"] for _, row in df_bus.iterrows()
                }
                distance_dict = {(row["origin"], row["destination"]): row["km"] for _, row in df_bus.iterrows()}
                travel_time_id_dict = {
                    (EAG.code_to_id[orig], EAG.code_to_id[dest]): time
                    for (orig, dest), time in travel_time_dict.items()
                }
                distance_id_dict = {
                    (EAG.code_to_id[orig], EAG.code_to_id[dest]): dist for (orig, dest), dist in distance_dict.items()
                }

                # Add emergency buses and short-turning activities if full closure
                if len(section_tracks_disruption) > 1:
                    print("Adding emergency buses and short-turning activities")
                    idx_events, idx_activities = add_emergency_buses(
                        EAG, travel_time_id_dict, distance_id_dict, idx_events, idx_activities
                    )
                    EAG = add_short_turning(
                        EAG, idx_events, idx_activities, all_paths_dict, intermediate_stations, INTERSECTIONS
                    )
            else:
                print("No disruption scenario provided")

    # ------------------------------------------------------------------------------------------
    #                                   Precompute info
    # ------------------------------------------------------------------------------------------

    events = EAG.events
    activities = EAG.activities
    stations = EAG.stations

    # Precompute common conditions
    regular_events = [e for e in events if e.node_type == "regular"]
    rerouting_events = [e for e in events if e.node_type == "rerouting"]
    turning_events = [e for e in events if e.node_type == "short-turning"]

    # Sets definitions
    regular_aggregated_events = [e for e in regular_events if e.aggregated]
    regular_rerouting_turning_aggregated_events = [
        e for e in regular_events + rerouting_events + turning_events if e.aggregated
    ]
    regular_disaggregated_events = [e for e in regular_events if not e.aggregated]
    regular_rerouting_turning_events = [
        e for e in regular_events + rerouting_events + turning_events if not e.aggregated
    ]

    activity_types = {
        "train": [
            "train running",
            "train waiting",
            "pass-through",
            "starting",
            "ending",
            "artificial",
            "short-turning",
        ],
        "group": [
            "passenger running",
            "dwelling",
            "transferring",
            "access",
            "egress",
            "penalty",
            "emergency bus",
        ],
    }

    categorized_activities = {
        key: [a for a in activities if a.activity_type in activity_types[key]] for key in activity_types
    }

    grouped_activities = {
        atype: [a for a in activities if a.activity_type == atype]
        for atype in set(activity_types["train"] + activity_types["group"])
    }

    # Precompute dictionaries
    A_waiting_plus, A_waiting_minus, arrival_time_train_end = {}, {}, {}
    A_access, A_egress = defaultdict(list), defaultdict(list)
    A_waiting_pass_through_dict = defaultdict(list)
    train_running_dict = defaultdict(list)

    for a in categorized_activities["group"]:
        if a.activity_type == "access":
            A_access[a.passenger_group].append(a)
        if a.activity_type == "egress":
            A_egress[a.passenger_group].append(a)

    for activity in grouped_activities["pass-through"]:
        A_waiting_pass_through_dict[(activity.origin.station, activity.origin.node_track)].append(activity)

    for activity in grouped_activities["train waiting"]:
        A_waiting_pass_through_dict[(activity.origin.station, activity.origin.node_track)].append(activity)

    for activity in grouped_activities["train running"]:
        train_running_dict[activity.section_track].append(activity)

    for event in events:
        A_waiting_plus[event] = [
            arc
            for arc in categorized_activities["train"]
            if arc.destination.station == event.station
            and arc.destination.event_type == event.event_type
            and arc.destination.scheduled_time == event.scheduled_time
            and arc.destination.train == event.train
            and arc.destination.node_type == event.node_type
            and (arc.activity_type == "train waiting" or arc.activity_type == "starting")
        ]
        A_waiting_minus[event] = [
            arc
            for arc in categorized_activities["train"]
            if arc.origin.station == event.station
            and arc.origin.event_type == event.event_type
            and arc.origin.scheduled_time == event.scheduled_time
            and arc.origin.train == event.train
            and arc.origin.node_type == event.node_type
            and (arc.activity_type == "train waiting" or arc.activity_type == "starting")
        ]

        if event.node_type == "train destination":
            arrival_time_train_end[event.train] = event.scheduled_time

    starting_activities_dict = {}
    for train in EAG.trains:
        starting_activities_dict[train] = [
            arc for arc in grouped_activities["starting"] if arc.origin.train.id == train.id
        ]
        if not starting_activities_dict[train]:
            raise ValueError(f"No start activities for train {train.id}")

    preprocess = {
        "categorized_activities": categorized_activities,
        "grouped_activities": grouped_activities,
        "regular_aggregated_events": regular_aggregated_events,
        "regular_rerouting_turning_aggregated_events": regular_rerouting_turning_aggregated_events,
        "regular_rerouting_turning_events": regular_rerouting_turning_events,
        "starting_activities_dict": starting_activities_dict,
        "A_waiting_plus": A_waiting_plus,
        "A_waiting_minus": A_waiting_minus,
        "regular_disaggregated_events": regular_disaggregated_events,
        "arrival_time_train_end": arrival_time_train_end,
        "A_waiting_pass_through_dict": A_waiting_pass_through_dict,
        "train_running_dict": train_running_dict,
        "A_access": A_access,
        "A_egress": A_egress,
    }

    EAG.add_preprocessing_info(preprocess)

    return EAG
