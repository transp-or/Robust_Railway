# Robust logger (console + file)
import logging
from collections import defaultdict

import numpy as np

from Robust_Railway.event_activity_graph_multitracks import (
    Activity,
    Bus,
    Event,
    SectionTrack,
)

logger = logging.getLogger(__name__)


def create_events(
    EAG,
    node,
    station,
    node_type,
    train,
    agg_to_disagg_events,
    disagg_to_agg_events,
    local_events,
    idx_events,
    time_previous_event=None,
):
    """Create arrival and departure events for a train at a station."""
    events_station, aggregated_events_station = np.empty((0, 7)), np.empty((0, 5))

    if not station.node_tracks:
        nodes_tracks = [None]
        node_track_in_timetable = None
    else:
        nodes_tracks = station.node_tracks
        node_track_in_timetable = node.node_track_id
        if node_track_in_timetable is None:
            logger.debug(f"No node track specified for train {train.id} at station {station.id}")

    diss_events = np.empty((0, 2))
    for event_type in ["arrival", "departure"]:
        for node_track in nodes_tracks:
            time_of_event = node.arrival_time if event_type == "arrival" else node.departure_time
            total_minutes = time_of_event.hour * 60 + time_of_event.minute
            total_minutes = max(EAG.start_time_window, min(total_minutes, EAG.end_time_window))
            nd_track = node_track.id if node_track else None

            if node_type in ["short-turning", "rerouting"]:
                in_timetable = False
                e_time = time_previous_event
            else:
                e_time = total_minutes
                if station.junction:
                    in_timetable = True
                elif node_track and node_track.id == node_track_in_timetable:
                    in_timetable = True
                else:
                    in_timetable = False

            new_row = np.array(
                [[
                    idx_events,
                    event_type,
                    e_time,
                    station.id,
                    nd_track,
                    node_track_in_timetable,
                    train.id,
                    node_type,
                    False,
                    in_timetable,
                    node.id,
                ]],
                dtype=object,
            )
            local_events = np.concatenate((local_events, new_row), axis=0)
            events_station = np.concatenate(
                (
                    events_station,
                    np.array(
                        [[idx_events, event_type, station.id, total_minutes, train.id, nd_track, in_timetable]],
                        dtype=object,
                    ),
                ),
                axis=0,
            )
            diss_events = np.concatenate((diss_events, np.array([[idx_events, event_type]], dtype=object)), axis=0)
            idx_events += 1

        # Add aggregated events (only for non-junction stations)
        if not station.junction:
            time_of_event = node.arrival_time if event_type == "arrival" else node.departure_time
            total_minutes = time_of_event.hour * 60 + time_of_event.minute
            e_time = time_previous_event if node_type in ["short-turning", "rerouting"] else total_minutes
            new_row = np.array(
                [[idx_events, event_type, e_time, station.id, None, None, train.id, node_type, True, False, None]],
                dtype=object,
            )
            local_events = np.concatenate((local_events, new_row), axis=0)
            aggregated_events_station = np.concatenate(
                (
                    aggregated_events_station,
                    np.array([[idx_events, event_type, station.id, total_minutes, train.id]], dtype=object),
                ),
                axis=0,
            )
            agg_to_disagg_events[idx_events] = [
                diss_event[0] for diss_event in diss_events if diss_event[1] == event_type
            ]
            for diss_event in diss_events:
                disagg_to_agg_events[diss_event[0]] = idx_events
            idx_events += 1

    return (
        EAG,
        events_station,
        aggregated_events_station,
        agg_to_disagg_events,
        disagg_to_agg_events,
        local_events,
        idx_events,
    )


def create_activities(
    EAG,
    prev_events,
    prev_agg_events,
    events_station,
    aggregated_events_station,
    agg_to_disagg_activities,
    disagg_to_agg_activities,
    all_paths,
    train_running_similar,
    A_plus,
    A_minus,
    A_minus_agg,
    A_plus_agg,
    intermediate_stations,
    local_activities,
    idx_activities,
    current_section_track,
    prev_diss_running_activities,
    prev_diss_dwelling_activities,
    incoming_tracks,
    outgoing_tracks,
    short_turning=True,
):
    """Create and organize activities for train operations and passenger movements."""
    diss_running_activities = prev_diss_running_activities
    diss_dwelling_activities = prev_diss_dwelling_activities

    # Train running and pass-through activities
    for current_event in events_station:
        for prev_event in prev_events:
            if prev_event[1] == "departure" and current_event[1] == "arrival" and prev_event[2] != current_event[2]:
                tracks = EAG.get_tracks_between(prev_event[2], current_event[2])
                if not tracks:
                    raise ValueError(f"Tracks missing between station {prev_event[2]} and station {current_event[2]}")
                for track in tracks:
                    check_prev_in_outgoing = (
                        prev_event[5] and (int(prev_event[2]), int(prev_event[5])) in outgoing_tracks
                    )
                    check_current_in_outgoing = (
                        current_event[5] and (int(current_event[2]), int(current_event[5])) in incoming_tracks
                    )
                    if (
                        check_prev_in_outgoing
                        and check_current_in_outgoing
                        and (
                            track.id not in outgoing_tracks[(int(prev_event[2]), int(prev_event[5]))]
                            or track.id not in incoming_tracks[(int(current_event[2]), int(current_event[5]))]
                        )
                    ):
                        continue
                    elif (
                        check_prev_in_outgoing
                        and not check_current_in_outgoing
                        and track.id not in outgoing_tracks[(int(prev_event[2]), int(prev_event[5]))]
                    ):
                        continue
                    elif (
                        check_current_in_outgoing
                        and not check_prev_in_outgoing
                        and track.id not in incoming_tracks[(int(current_event[2]), int(current_event[5]))]
                    ):
                        continue

                    in_timetable = (
                        False
                        if short_turning
                        else (
                            int(track.id) == int(current_section_track)
                            and str(prev_event[6]) == "True"
                            and str(current_event[6]) == "True"
                        )
                    )
                    local_activities = np.concatenate(
                        (
                            local_activities,
                            np.array(
                                [[
                                    idx_activities,
                                    int(prev_event[0]),
                                    int(current_event[0]),
                                    None,
                                    "train running",
                                    track.id,
                                    False,
                                    in_timetable,
                                    None,
                                    current_section_track,
                                ]],
                                dtype=object,
                            ),
                        ),
                        axis=0,
                    )
                    diss_running_activities.append(idx_activities)
                    key = (prev_event[2], current_event[2], prev_event[3], prev_event[4])
                    train_running_similar[key].append(idx_activities)
                    A_minus[prev_event[0]].append(idx_activities)
                    A_plus[current_event[0]].append(idx_activities)
                    idx_activities += 1

        # Pass-through and waiting activities
        for current_event_2 in events_station:
            if (
                current_event[1] == "arrival"
                and current_event_2[1] == "departure"
                and current_event[2] == current_event_2[2]
                and current_event[5] == current_event_2[5]
            ):
                station_obj = EAG.get_station_by_id(current_event[2])
                in_timetable = (
                    False
                    if short_turning
                    else (
                        current_event_2[6] == "True"
                        and current_event[6] == "True"
                        and (float(current_event_2[3]) - float(current_event[3])) < EAG.waiting_time
                    )
                )
                local_activities = np.concatenate(
                    (
                        local_activities,
                        np.array(
                            [[
                                idx_activities,
                                int(current_event[0]),
                                int(current_event_2[0]),
                                None,
                                "pass-through",
                                None,
                                False,
                                in_timetable,
                                None,
                                None,
                            ]],
                            dtype=object,
                        ),
                    ),
                    axis=0,
                )
                diss_dwelling_activities.append(idx_activities)
                A_minus[current_event[0]].append(idx_activities)
                A_plus[current_event_2[0]].append(idx_activities)
                idx_activities += 1

                # Waiting activities (not at junctions)
                if not station_obj.junction:
                    in_timetable = (
                        current_event_2[6] == "True"
                        and current_event[6] == "True"
                        and (float(current_event_2[3]) - float(current_event[3])) >= EAG.waiting_time
                    )
                    local_activities = np.concatenate(
                        (
                            local_activities,
                            np.array(
                                [[
                                    idx_activities,
                                    int(current_event[0]),
                                    int(current_event_2[0]),
                                    None,
                                    "train waiting",
                                    None,
                                    False,
                                    in_timetable,
                                    None,
                                    None,
                                ]],
                                dtype=object,
                            ),
                        ),
                        axis=0,
                    )
                    diss_dwelling_activities.append(idx_activities)
                    A_minus[current_event[0]].append(idx_activities)
                    A_plus[current_event_2[0]].append(idx_activities)
                    idx_activities += 1

    # Passenger running and dwelling activities (aggregated)
    if len(aggregated_events_station) > 0:
        valid_current_events = [e for e in aggregated_events_station if e[1] == "arrival"]
        valid_prev_agg_events = [e for e in prev_agg_events if e[1] == "departure"]

        for current_agg_event in valid_current_events:
            for prev_agg_event in valid_prev_agg_events:
                inter_stations_str = None
                key_tuple = (int(prev_agg_event[2]), int(current_agg_event[2]), int(prev_agg_event[4]))
                if key_tuple in intermediate_stations:
                    inter_stations_str = str(intermediate_stations[key_tuple])
                local_activities = np.concatenate(
                    (
                        local_activities,
                        np.array(
                            [[
                                idx_activities,
                                prev_agg_event[0],
                                current_agg_event[0],
                                None,
                                "passenger running",
                                None,
                                True,
                                False,
                                inter_stations_str,
                                None,
                            ]],
                            dtype=object,
                        ),
                    ),
                    axis=0,
                )
                agg_to_disagg_activities[idx_activities] = diss_running_activities
                for a in diss_running_activities:
                    disagg_to_agg_activities[a] = idx_activities
                A_minus_agg[prev_agg_event[0]].append(idx_activities)
                A_plus_agg[current_agg_event[0]].append(idx_activities)
                idx_activities += 1

            valid_aggregated_events_station_2 = [e for e in aggregated_events_station if e[1] == "departure"]
            for current_agg_event_2 in valid_aggregated_events_station_2:
                local_activities = np.concatenate(
                    (
                        local_activities,
                        np.array(
                            [[
                                idx_activities,
                                current_agg_event[0],
                                current_agg_event_2[0],
                                None,
                                "dwelling",
                                None,
                                True,
                                False,
                                None,
                                None,
                            ]],
                            dtype=object,
                        ),
                    ),
                    axis=0,
                )
                agg_to_disagg_activities[idx_activities] = diss_dwelling_activities
                for a in diss_dwelling_activities:
                    disagg_to_agg_activities[a] = idx_activities
                A_minus_agg[current_agg_event[0]].append(idx_activities)
                A_plus_agg[current_agg_event_2[0]].append(idx_activities)
                idx_activities += 1

        diss_running_activities = []
        diss_dwelling_activities = []

    return (
        EAG,
        agg_to_disagg_activities,
        disagg_to_agg_activities,
        train_running_similar,
        A_plus,
        A_minus,
        A_minus_agg,
        A_plus_agg,
        local_activities,
        idx_activities,
        diss_running_activities,
        diss_dwelling_activities,
    )


def add_emergency_buses(EAG, travel_time_id_dict, distance_id_dict, idx_emergency_events, idx_emergency_activities):
    """
    Add emergency bus events and activities to the Event-Activity Graph (EAG) in case of disruption.
    This function creates bus events and activities between shunting yards before and after the disruption.
    """

    def add_events_and_activities(e1, e2):
        nonlocal idx_emergency_activities, idx_emergency_events, idx_bus, section_track_idx
        # Create a new bus and add it to the EAG
        bus = Bus(idx_bus, EAG.bus_capacity)
        EAG.add_bus(bus)
        min_duration = travel_time_id_dict[(e1.station.id, e2.station.id)]
        # Create departure and arrival events for the bus
        new_e1 = Event(
            idx_emergency_events,
            "departure",
            e1.scheduled_time + EAG.minimum_transfer_time,
            e1.station,
            None,
            None,
            bus,
            "emergency",
            True,
            False,
        )
        new_e2 = Event(
            idx_emergency_events + 1,
            "arrival",
            e1.scheduled_time + EAG.minimum_transfer_time + min_duration,
            e2.station,
            None,
            None,
            bus,
            "emergency",
            True,
            False,
        )
        EAG.add_event(new_e1)
        EAG.add_event(new_e2)
        # Add transferring activity (from train to bus)
        activity1 = Activity(idx_emergency_activities, e1, new_e1, None, "transferring", None, True, False)

        # Find or create the section track for the bus
        section_tracks = EAG.get_section_tracks(new_e1.station, new_e2.station, True)
        if len(section_tracks) == 1:
            st = section_tracks[0]
            st.travel_time[bus] = travel_time_id_dict[(new_e1.station.id, new_e2.station.id)]
        elif len(section_tracks) == 0:
            section_track_to_add = SectionTrack(
                "E" + str(section_track_idx),
                new_e1.station,
                new_e2.station,
                distance_id_dict[(new_e1.station.id, new_e2.station.id)],
                {bus: travel_time_id_dict[(new_e1.station.id, new_e2.station.id)]},
                True,
            )
            EAG.add_section_track(section_track_to_add)
            st = section_track_to_add
            section_track_idx += 1
        else:
            raise ValueError("There should not be more than one road section track between each pair of stations")

        # Add emergency bus activity (bus running between stations)
        activity2 = Activity(idx_emergency_activities + 1, new_e1, new_e2, None, "emergency bus", st, True, False)
        EAG.add_activities([activity1, activity2])
        EAG.add_A_minus_agg_item(e1, activity1)
        EAG.add_A_plus_agg_item(new_e1, activity1)
        EAG.add_A_minus_agg_item(new_e1, activity2)
        EAG.add_A_plus_agg_item(new_e2, activity2)
        idx_emergency_activities += 2
        idx_emergency_events += 2
        idx_bus += 1

        # Add transferring activities for passengers from bus to other trains at arrival station
        for a2 in EAG.activities:
            if a2.activity_type == "dwelling" and a2.origin.station == new_e2.station and a2.origin.train != e1.train:
                new_activity = Activity(
                    idx_emergency_activities, new_e2, a2.origin, None, "transferring", None, True, False
                )
                EAG.add_activities(new_activity)
                EAG.add_A_minus_agg_item(new_e2, new_activity)
                EAG.add_A_plus_agg_item(a2.origin, new_activity)
                idx_emergency_activities += 1

    # Main logic: only run if there is a disruption scenario
    if EAG.disruption_scenario.section_tracks:
        print("disrupted section_tracks")
        d_stations = set()
        idx_bus = 5000
        section_track_idx = 0
        # Collect all stations involved in the disruption
        for t in EAG.disruption_scenario.section_tracks:
            d_stations.add(t.origin)
            d_stations.add(t.destination)
        for s in d_stations:
            print("d_stations", s.id)

        # For each train, find shunting yards before and after the disruption and link them by bus
        for train in EAG.trains:
            passed_disruption = False
            last_shunting_event, first_shunting_event = None, None
            for a in EAG.get_ordered_agg_activities_train(train):
                if a.activity_type == "passenger running":
                    stations_lst = [a.origin.station, a.destination.station]
                    if a.intermediate_stations:
                        stations_lst.extend(a.intermediate_stations)
                    if a.origin.station.shunting_yard_capacity > 0 and not passed_disruption:
                        last_shunting_event = a.origin
                    if len(set(stations_lst) & set(d_stations)) >= 2:
                        passed_disruption = True
                    if (
                        passed_disruption
                        and a.destination.station.shunting_yard_capacity > 0
                        and not first_shunting_event
                    ):
                        first_shunting_event = a.destination
            if last_shunting_event and first_shunting_event:
                add_events_and_activities(last_shunting_event, first_shunting_event)
    return idx_emergency_events, idx_emergency_activities


def find_all_transfering_events_pairs(EAG, train1, train2, INTERSECTIONS, short_turning):
    if train1 == train2:
        return []
    else:
        pairs = []
        # Filter events for train1 and train2
        events_train1 = [e for e in EAG.events if e.train == train1 and e.aggregated]
        events_train2 = [e for e in EAG.events if e.train == train2 and e.aggregated]

        # Further filter by intersection stations
        events_train1 = np.array([e for e in events_train1 if EAG.id_to_code[int(e.station.id)] in INTERSECTIONS])
        events_train2 = np.array([e for e in events_train2 if EAG.id_to_code[int(e.station.id)] in INTERSECTIONS])

        # Build index for faster lookup: station_id -> list of events
        events_by_station_train2 = defaultdict(list)
        for e2 in events_train2:
            events_by_station_train2[int(e2.station.id)].append(e2)

        # Main matching
        for e1 in events_train1:
            sid = e1.station.id
            if sid in events_by_station_train2:
                for e2 in events_by_station_train2[sid]:

                    if (
                        e1.event_type == "departure"
                        and e2.event_type == "arrival"
                        and e2 != EAG.last_event_train(train2)
                        and EAG.get_incoming_station(e1) != EAG.get_outgoing_station(e2)
                    ):
                        if short_turning:
                            if e1.node_type == "rerouting" or e2.node_type == "rerouting":
                                pairs.append((e1, e2))
                        else:
                            pairs.append((e1, e2))
                    elif (
                        e1.event_type == "arrival"
                        and e2.event_type == "departure"
                        and e1 != EAG.last_event_train(train1)
                        and EAG.get_incoming_station(e2) != EAG.get_outgoing_station(e1)
                    ):
                        if short_turning:
                            if e1.node_type == "rerouting" or e2.node_type == "rerouting":
                                pairs.append((e2, e1))
                        else:
                            pairs.append((e2, e1))

        return pairs


def process_train(
    train,
    EAG,
    all_paths,
    intermediate_stations,
    api,
    incoming_tracks,
    outgoing_tracks,
):
    """
    Process a train to generate its events and activities for the Event-Activity Graph (EAG).
    Handles both disaggregated and aggregated events/activities, including pass-through and waiting.
    """
    # Initialization
    train_running_similar = defaultdict(list)
    agg_to_disagg_events, agg_to_disagg_activities = {}, {}
    disagg_to_agg_events, disagg_to_agg_activities = {}, {}
    nb_pass_through_activities, nb_waiting_activities = 0, 0
    A_plus, A_minus, A_minus_agg, A_plus_agg = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    events_with_different_node_tracks = []
    diss_running_activities = []
    diss_dwelling_activities = []
    local_events = np.empty((0, 11), dtype=object)
    local_activities = np.empty((0, 10), dtype=object)

    event_train = next((t for t in EAG.trains if t.id == train.id), None)
    begin_at, end_at = train.begin_at, train.end_at

    if event_train and end_at - begin_at > 1:
        print(
            f"Processing train {train.id} ({train.code}) with {len(train.train_path_nodes)} train path nodes, "
            f"with nodes inside the time window between indices {begin_at} - {end_at}, and capacity {train.capacity}"
        )
        idx_events = event_train.id * 10**6
        idx_activities = idx_events * 100_000

        prev_events, prev_agg_events = [], []
        for node_idx in range(begin_at, end_at + 1):
            (
                events_station,
                aggregated_events_station,
                arrival_events_nearly_identic,
                departure_events_nearly_identic,
            ) = ([], [], [], [])
            node = train.train_path_nodes[node_idx]
            station = EAG.get_station_by_id(node.node_id)

            if not station:
                print(f"Train {train.id} with a segment outside of the RER Vaud.")
                return (
                    events_with_different_node_tracks,
                    train_running_similar,
                    A_plus,
                    A_minus,
                    A_minus_agg,
                    A_plus_agg,
                    agg_to_disagg_events,
                    disagg_to_agg_events,
                    agg_to_disagg_activities,
                    disagg_to_agg_activities,
                    nb_pass_through_activities,
                    nb_waiting_activities,
                    local_events,
                    local_activities,
                )

            # Check if pass-through or waiting in initial schedule
            start_time = node.arrival_time.hour * 60 + node.arrival_time.minute
            end_time = node.departure_time.hour * 60 + node.departure_time.minute
            if end_time - start_time >= EAG.waiting_time:
                nb_waiting_activities += 1
            else:
                nb_pass_through_activities += 1

            # Add disaggregated and aggregated events
            (
                EAG,
                events_station,
                aggregated_events_station,
                agg_to_disagg_events,
                disagg_to_agg_events,
                local_events,
                idx_events,
            ) = create_events(
                EAG,
                node,
                station,
                "regular",
                event_train,
                agg_to_disagg_events,
                disagg_to_agg_events,
                local_events,
                idx_events,
            )

            arrival_events_nearly_identic = [event[0] for event in events_station if event[1] == "arrival"]
            departure_events_nearly_identic = [event[0] for event in events_station if event[1] == "departure"]

            # First event of the train
            if node_idx == begin_at:
                idx_starting_event = idx_events
                new_row = np.array(
                    [[
                        idx_starting_event,
                        None,
                        EAG.start_time_window,
                        station.id,
                        None,
                        None,
                        event_train.id,
                        "train origin",
                        False,
                        True,
                        None,
                    ]],
                    dtype=object,
                )
                local_events = np.vstack((local_events, new_row))
                idx_events += 1
                for event_dest in events_station:
                    if event_dest[1] == "arrival":
                        in_timetable = str(event_dest[6]) == "True"
                        local_activities = np.append(
                            local_activities,
                            [[
                                idx_activities,
                                idx_starting_event,
                                event_dest[0],
                                None,
                                "starting",
                                None,
                                False,
                                in_timetable,
                                None,
                                None,
                            ]],
                            axis=0,
                        )
                        A_minus[idx_starting_event].append(idx_activities)
                        A_plus[event_dest[0]].append(idx_activities)
                        idx_activities += 1

            # Last event of the train
            elif node_idx == end_at:
                idx_ending_event = idx_events
                print("end time window for event", EAG.end_time_window)
                new_row = np.array(
                    [[
                        idx_ending_event,
                        None,
                        EAG.end_time_window,
                        station.id,
                        None,
                        None,
                        event_train.id,
                        "train destination",
                        False,
                        True,
                        None,
                    ]],
                    dtype=object,
                )
                local_events = np.vstack((local_events, new_row))
                idx_events += 1
                for event_origin in events_station:
                    if event_origin[1] == "departure":
                        in_timetable = str(event_origin[6]) == "True"
                        local_activities = np.append(
                            local_activities,
                            [[
                                idx_activities,
                                int(event_origin[0]),
                                int(idx_ending_event),
                                None,
                                "ending",
                                None,
                                False,
                                in_timetable,
                                None,
                                None,
                            ]],
                            axis=0,
                        )
                        A_minus[event_origin[0]].append(idx_activities)
                        A_plus[idx_ending_event].append(idx_activities)
                        idx_activities += 1

            # Add disaggregated and aggregated activities for subsequent events
            current_section_track = node.section_track_id
            if current_section_track is None:
                logger.debug(f"No section track specified for train {train.id} at station {station.id}")

            (
                EAG,
                agg_to_disagg_activities,
                disagg_to_agg_activities,
                train_running_similar,
                A_plus,
                A_minus,
                A_minus_agg,
                A_plus_agg,
                local_activities,
                idx_activities,
                diss_running_activities,
                diss_dwelling_activities,
            ) = create_activities(
                EAG,
                prev_events,
                prev_agg_events,
                events_station,
                aggregated_events_station,
                agg_to_disagg_activities,
                disagg_to_agg_activities,
                all_paths,
                train_running_similar,
                A_plus,
                A_minus,
                A_minus_agg,
                A_plus_agg,
                intermediate_stations,
                local_activities,
                idx_activities,
                current_section_track,
                diss_running_activities,
                diss_dwelling_activities,
                incoming_tracks,
                outgoing_tracks,
            )

            # Update previous events for next iteration
            if station:
                prev_events = events_station
                if not station.junction:
                    prev_agg_events = aggregated_events_station
                if arrival_events_nearly_identic:
                    events_with_different_node_tracks.append(list(arrival_events_nearly_identic))
                if departure_events_nearly_identic:
                    events_with_different_node_tracks.append(list(departure_events_nearly_identic))

    return (
        events_with_different_node_tracks,
        train_running_similar,
        A_plus,
        A_minus,
        A_minus_agg,
        A_plus_agg,
        agg_to_disagg_events,
        disagg_to_agg_events,
        agg_to_disagg_activities,
        disagg_to_agg_activities,
        nb_pass_through_activities,
        nb_waiting_activities,
        local_events,
        local_activities,
    )


def process_short_turning(train, EAG, d_stations, all_paths, intermediate_stations, incoming_tracks, outgoing_tracks):
    """
    Process a short-turning train to generate its events and activities for the Event-Activity Graph (EAG).
    Handles the creation of events and activities for the part of the train path before the disruption.
    """
    # Initialization
    train_running_similar = defaultdict(list)
    agg_to_disagg_events, agg_to_disagg_activities = {}, {}
    disagg_to_agg_events, disagg_to_agg_activities = {}, {}
    nb_pass_through_activities, nb_waiting_activities = 0, 0
    A_plus, A_minus, A_minus_agg, A_plus_agg = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    events_with_different_node_tracks = []
    diss_running_activities = []
    diss_dwelling_activities = []
    local_events = np.empty((0, 11), dtype=object)
    local_activities = np.empty((0, 10), dtype=object)
    first_events_idx = []

    idx_events = train.id * 10**12
    idx_activities = idx_events * 100_000

    prev_events, prev_agg_events = [], []

    return_lst = (
        events_with_different_node_tracks,
        train_running_similar,
        A_plus,
        A_minus,
        A_minus_agg,
        A_plus_agg,
        agg_to_disagg_events,
        disagg_to_agg_events,
        agg_to_disagg_activities,
        disagg_to_agg_activities,
        nb_pass_through_activities,
        nb_waiting_activities,
        local_events,
        local_activities,
    )

    begin_at, end_at = train.begin_at, train.end_at
    if end_at - begin_at <= 1:
        # Not enough nodes to process short-turning
        return {"train": train.id, "shunting_station": None, "first_events_idx": None, "return_lst": return_lst}

    passed_disruption = False
    shunting_node_idx = None
    shunting_station = None
    time_previous_event = None

    # Find the shunting station and node index before the disruption
    for node_idx in range(begin_at, end_at + 1):
        node = train.train_path_nodes[node_idx]
        station = EAG.get_station_by_id(node.node_id)
        if node_idx < end_at:
            next_node = train.train_path_nodes[node_idx + 1]
            next_station = EAG.get_station_by_id(next_node.node_id)
        else:
            next_station = None
        if not station:
            continue  # Segment outside of the RER Vaud
        if station.shunting_yard_capacity > 0 and not passed_disruption:
            shunting_station = station
            shunting_node_idx = node_idx
            previous_event = EAG.get_event_by_attributes("departure", station, None, train, "regular", False)
            if previous_event:
                time_previous_event = previous_event.scheduled_time + 0.01
            else:
                raise ValueError("Previous event for shunting station not found.")
        if station in d_stations and next_station in d_stations:
            passed_disruption = True

    if shunting_node_idx is None or not passed_disruption:
        logger.debug(f"Train {train.id} not going through the disruption")
        return {"train": train.id, "shunting_station": None, "first_events_idx": None, "return_lst": return_lst}

    # Process events and activities for the short-turning part (from shunting station back to begin_at)
    for node_idx in range(shunting_node_idx, begin_at - 1, -1):
        events_station, aggregated_events_station, arrival_events_nearly_identic, departure_events_nearly_identic = (
            [],
            [],
            [],
            [],
        )

        node = train.train_path_nodes[node_idx]
        station = EAG.get_station_by_id(node.node_id)
        if not station:
            continue

        # Create events for this node
        (
            EAG,
            events_station,
            aggregated_events_station,
            agg_to_disagg_events,
            disagg_to_agg_events,
            local_events,
            idx_events,
        ) = create_events(
            EAG,
            node,
            station,
            "short-turning",
            train,
            agg_to_disagg_events,
            disagg_to_agg_events,
            local_events,
            idx_events,
            time_previous_event,
        )

        # Increment time for next event to avoid duplicates
        if time_previous_event is not None:
            time_previous_event += 0.01

        # Store first events indices for reference
        if node_idx == shunting_node_idx:
            first_events_idx = local_events[:, 0]

        arrival_events_nearly_identic = [event[0] for event in events_station if event[1] == "arrival"]
        departure_events_nearly_identic = [event[0] for event in events_station if event[1] == "departure"]

        # Last event of the short-turning train (at begin_at)
        if node_idx == begin_at:
            idx_ending_event = idx_events
            new_row = np.array(
                [[
                    idx_ending_event,
                    None,
                    EAG.end_time_window,
                    station.id,
                    None,
                    None,
                    train.id,
                    "train destination",
                    False,
                    True,
                    None,
                ]],
                dtype=object,
            )
            local_events = np.vstack((local_events, new_row))
            idx_events += 1
            for event_origin in events_station:
                if event_origin[1] == "departure":
                    in_timetable = False
                    local_activities = np.append(
                        local_activities,
                        [[
                            idx_activities,
                            int(event_origin[0]),
                            int(idx_ending_event),
                            None,
                            "ending",
                            None,
                            False,
                            in_timetable,
                            None,
                            None,
                        ]],
                        axis=0,
                    )
                    A_minus[event_origin[0]].append(idx_activities)
                    A_plus[idx_ending_event].append(idx_activities)
                    idx_activities += 1

        # Add activities for this node
        current_section_track = node.section_track_id
        if current_section_track is None:
            logger.debug(f"No section track specified for train {train.id} at station {station.id}")
        (
            EAG,
            agg_to_disagg_activities,
            disagg_to_agg_activities,
            train_running_similar,
            A_plus,
            A_minus,
            A_minus_agg,
            A_plus_agg,
            local_activities,
            idx_activities,
            diss_running_activities,
            diss_dwelling_activities,
        ) = create_activities(
            EAG,
            prev_events,
            prev_agg_events,
            events_station,
            aggregated_events_station,
            agg_to_disagg_activities,
            disagg_to_agg_activities,
            all_paths,
            train_running_similar,
            A_plus,
            A_minus,
            A_minus_agg,
            A_plus_agg,
            intermediate_stations,
            local_activities,
            idx_activities,
            current_section_track,
            diss_running_activities,
            diss_dwelling_activities,
            incoming_tracks,
            outgoing_tracks,
            short_turning=True,
        )
        # Update previous events for next iteration
        if station:
            prev_events = events_station
            if not station.junction:
                prev_agg_events = aggregated_events_station
            if arrival_events_nearly_identic:
                events_with_different_node_tracks.append(list(arrival_events_nearly_identic))
            if departure_events_nearly_identic:
                events_with_different_node_tracks.append(list(departure_events_nearly_identic))

    return_lst = (
        events_with_different_node_tracks,
        train_running_similar,
        A_plus,
        A_minus,
        A_minus_agg,
        A_plus_agg,
        agg_to_disagg_events,
        disagg_to_agg_events,
        agg_to_disagg_activities,
        disagg_to_agg_activities,
        nb_pass_through_activities,
        nb_waiting_activities,
        local_events,
        local_activities,
    )
    return {
        "train": train.id,
        "shunting_station": shunting_station.id if shunting_station else None,
        "first_events_idx": first_events_idx,
        "return_lst": return_lst,
    }
