import random
from typing import Union

import numpy as np

from ..event_activity_graph_multitracks import EARailwayNetwork, NodeTrack, SectionTrack, Train
from ..operators.destroy_operators import get_train_usage
from ..operators.repair_operators_cancel import cancel_train_completely
from ..operators.repair_operators_delay import get_earliest_start_based_on_track_occupancy
from ..operators.track_occupancy import get_junction_usage, get_track_usage


def random_order(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, trains: list[Train]):
    # Return a random permutation of the train list
    return random.sample(trains, len(trains))


def keep_order_at_disruption(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, trains: list[Train]
):
    t_disruption = {}
    non_disrupted_trains = []

    disrupted_sections = set()
    if EAG.disruption_scenario and EAG.disruption_scenario.section_tracks:
        for track in EAG.disruption_scenario.section_tracks:
            disrupted_sections.add((track.origin, track.destination))

    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            tracks.add(track)

    # Separate disrupted and non-disrupted trains
    for t in trains:
        disrupted = False
        for a in EAG.A_train[t]:
            if a.section_track_planned and a.section_track_planned in tracks:
                t_disruption[t] = Y[a.origin.id]
                disrupted = True
                break  # Only consider the first disrupted activity
        if not disrupted:
            non_disrupted_trains.append(t)

    # Order the disrupted trains by their disruption value
    ordered_trains = sorted(t_disruption, key=lambda t: t_disruption[t])

    # Shuffle non-disrupted trains for randomness
    random.shuffle(non_disrupted_trains)

    # Combine both lists: disrupted trains first, non-disrupted trains at the end
    ordered_trains.extend(non_disrupted_trains)
    return ordered_trains


def regret_2step(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, trains: list[Train]):
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    if EAG.disruption_scenario is None or not EAG.disruption_scenario.section_tracks:
        return random_order(EAG, X, Y, Z, PHI, arc_usage, trains)

    disrupted_sections = set()
    for track in getattr(EAG.disruption_scenario, "section_tracks", []):
        disrupted_sections.add((track.origin, track.destination))

    disrupted_tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            disrupted_tracks.add(track)

    trains_passing_in_disruption = []
    t_disruption = {}
    t_section = {}
    direction = {}

    def t_through_disrupted_sections(t: Train):
        # Check if train passes through any disrupted track
        return any(
            a.section_track in disrupted_tracks for a in EAG.A_train[t] if X[a.id] > 0.5 and a.section_track is not None
        )

    eligible_trains = [t for t in trains if t_through_disrupted_sections(t)]

    # Group only disrupted trains by section track
    for t in eligible_trains:
        for a in EAG.A_train[t]:
            if a.section_track_planned and a.section_track_planned in disrupted_tracks:
                trains_passing_in_disruption.append(t)
                t_disruption[t] = a.origin.scheduled_time
                t_section[t] = a.section_track_planned
                direction[t] = (a.origin.station, a.destination.station)
                break  # Only consider the first disrupted activity

    # Separate trains by direction
    directions = list(set(direction.values()))

    if len(directions) != 2:
        return random_order(EAG, X, Y, Z, PHI, arc_usage, trains)

    dir1, dir2 = directions
    trains_dir_1 = sorted(
        [t for t in trains_passing_in_disruption if direction[t] == dir1], key=lambda t: t_disruption[t]
    )
    trains_dir_2 = sorted(
        [t for t in trains_passing_in_disruption if direction[t] == dir2], key=lambda t: t_disruption[t]
    )

    sorted_trains = []

    def compute_delay_train(train: Train, X, Y):
        # Compute total delay for a train
        delay = 0
        for a in EAG.categorized_activities["train"]:
            if X[a.id] > 0.5:
                delay += Y[a.origin.id] - a.origin.scheduled_time
        return delay

    # Remove all trains from the solution initially
    for train in trains:
        Xplus, Yplus, Zplus, PHIplus = cancel_train_completely(EAG, Xplus, Yplus, Zplus, PHIplus, train)

    def find_activity_in_disruption(train):
        # Find the first activity in disruption for a train
        for a in EAG.A_train[train]:
            if a.section_track in disrupted_tracks:
                return a
        raise ValueError(f"Train {train.id} not passing through the disrupted section tracks")

    def get_similar_activities(activity):
        # Get similar activities for aggregation/disaggregation
        if activity.activity_type != "short-turning":
            agg_act = EAG.disagg_to_agg_activities[activity]
            similar_activities = [
                a
                for a in EAG.agg_to_disagg_activities[agg_act]
                if a.origin.station.id == activity.origin.station.id
                and a.destination.station.id == activity.destination.station.id
            ]
            if activity not in similar_activities:
                raise ValueError("activity missing")
            return similar_activities
        else:
            for act_list in EAG.similar_short_turning:
                if activity in act_list:
                    return act_list

    def get_start_end_times(activity, train: Train, X: dict, Z: dict, Y: dict, PHI: dict) -> tuple:
        # Compute the earliest start and end times for an activity
        start_times: dict[Union[SectionTrack, NodeTrack, None], float] = {}
        end_times: dict[Union[SectionTrack, NodeTrack, None], float] = {}
        disagg_activities = get_similar_activities(activity)
        track_usage = get_track_usage(EAG, X, Z, Y, train)
        junction_usage = get_junction_usage(EAG, X, Z, Y, train)
        at_station = activity.origin.station == activity.destination.station

        current_track = activity.origin.node_track_planned if at_station else activity.section_track_planned
        for disagg_activity in disagg_activities:
            track = disagg_activity.origin.node_track if at_station else disagg_activity.section_track
            if at_station and disagg_activity.activity_type != activity.activity_type:
                if activity.activity_type == "pass-through":
                    continue
            if track in start_times:
                continue
            # Compute the earliest slot on this track
            start_times[track], end_times[track], _ = get_earliest_start_based_on_track_occupancy(
                EAG,
                disagg_activity,
                Y[disagg_activity.origin.id],
                Y[disagg_activity.destination.id],
                track_usage,
                junction_usage,
                backward=False,
                min_stop_time=0,
                previous_activity_end_time=0,
                earliest_next_activity_time=None,
                verbose=0,
            )
        min_delay = np.inf
        min_start_time = None
        min_end_time = None
        min_delay_activity = None
        for disagg_activity in disagg_activities:
            track = disagg_activity.origin.node_track if at_station else disagg_activity.section_track
            delay = (start_times[track] - getattr(disagg_activity.origin, "scheduled_time", 0)) + (
                end_times[track] - getattr(disagg_activity.destination, "scheduled_time", 0)
            )
            if delay <= min_delay:
                if delay == min_delay and track == current_track:
                    min_delay_activity = disagg_activity
                elif delay < min_delay:
                    min_delay_activity = disagg_activity
                min_delay = delay
                min_start_time = start_times[track]
                min_end_time = end_times[track]
        return min_delay_activity, min_start_time, min_end_time, min_delay

    # Main regret-based ordering loop
    while len(trains_dir_1) > 0 and len(trains_dir_2) > 0:
        t1 = trains_dir_1[0]
        t2 = trains_dir_2[0]

        # Sequence 1: t1 then t2
        Xplus_1, Yplus_1, Zplus_1, PHIplus_1 = Xplus.copy(), Yplus.copy(), Zplus.copy(), PHIplus.copy()
        activity = find_activity_in_disruption(t1)
        min_delay_activity, min_start_time, min_end_time, min_delay_1 = get_start_end_times(
            activity, t1, Xplus_1, Zplus_1, Yplus_1, PHIplus_1
        )
        Xplus_1[min_delay_activity.id] = 1
        Yplus_1[min_delay_activity.origin.id] = min_start_time
        Yplus_1[min_delay_activity.destination.id] = min_end_time

        # Followed by t2
        Xplus_1_1, Yplus_1_1, Zplus_1_1, PHIplus_1_1 = Xplus_1.copy(), Yplus_1.copy(), Zplus_1.copy(), PHIplus_1.copy()
        activity = find_activity_in_disruption(t2)
        min_delay_activity, min_start_time, min_end_time, min_delay_1_1 = get_start_end_times(
            activity, t2, Xplus_1_1, Zplus_1_1, Yplus_1_1, PHIplus_1_1
        )
        Xplus_1_1[min_delay_activity.id] = 1
        Yplus_1_1[min_delay_activity.origin.id] = min_start_time
        Yplus_1_1[min_delay_activity.destination.id] = min_end_time

        # Sequence 2: t2 then t1
        Xplus_2, Yplus_2, Zplus_2, PHIplus_2 = Xplus.copy(), Yplus.copy(), Zplus.copy(), PHIplus.copy()
        activity = find_activity_in_disruption(t2)
        min_delay_activity, min_start_time, min_end_time, min_delay_2 = get_start_end_times(
            activity, t2, Xplus_2, Zplus_2, Yplus_2, PHIplus_2
        )
        Xplus_2[min_delay_activity.id] = 1
        Yplus_2[min_delay_activity.origin.id] = min_start_time
        Yplus_2[min_delay_activity.destination.id] = min_end_time

        # Followed by t1
        Xplus_2_1, Yplus_2_1, Zplus_2_1, PHIplus_2_1 = Xplus_2.copy(), Yplus_2.copy(), Zplus_2.copy(), PHIplus_2.copy()
        activity = find_activity_in_disruption(t1)
        min_delay_activity, min_start_time, min_end_time, min_delay_2_1 = get_start_end_times(
            activity, t1, Xplus_2_1, Zplus_2_1, Yplus_2_1, PHIplus_2_1
        )
        Xplus_2_1[min_delay_activity.id] = 1
        Yplus_2_1[min_delay_activity.origin.id] = min_start_time
        Yplus_2_1[min_delay_activity.destination.id] = min_end_time

        regret_1 = min_delay_2_1 - min_delay_1
        regret_2 = min_delay_1_1 - min_delay_2

        # Choose the sequence with lower regret
        EPS = 1e-6
        if regret_1 > regret_2:
            sorted_trains.append(t1)
            trains_dir_1.pop(0)
            Xplus, Yplus, Zplus, PHIplus = Xplus_1.copy(), Yplus_1.copy(), Zplus_1.copy(), PHIplus_1.copy()
        elif abs(regret_1 - regret_2) < EPS:
            if t_disruption[t1] <= t_disruption[t2]:
                sorted_trains.append(t1)
                trains_dir_1.pop(0)
                Xplus, Yplus, Zplus, PHIplus = Xplus_1.copy(), Yplus_1.copy(), Zplus_1.copy(), PHIplus_1.copy()
            else:
                sorted_trains.append(t2)
                trains_dir_2.pop(0)
                Xplus, Yplus, Zplus, PHIplus = Xplus_2.copy(), Yplus_2.copy(), Zplus_2.copy(), PHIplus_2.copy()
        else:
            sorted_trains.append(t2)
            trains_dir_2.pop(0)
            Xplus, Yplus, Zplus, PHIplus = Xplus_2.copy(), Yplus_2.copy(), Zplus_2.copy(), PHIplus_2.copy()

    # Add remaining trains
    for t in trains_dir_1:
        if t in trains:
            sorted_trains.append(t)
    for t in trains_dir_2:
        if t in trains:
            sorted_trains.append(t)
    for t in trains:
        if t not in sorted_trains:
            sorted_trains.append(t)

    return sorted_trains


def mean_more_used_first(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, trains: list[Train]
):
    # Order trains by their average usage (descending)
    train_usage = get_train_usage(EAG, X, Y, Z, arc_usage)
    avg_usage = {}
    for t in trains:
        usage = train_usage.get(t, [])
        avg = np.mean(usage) if len(usage) > 0 else 0
        avg_usage[t] = avg
    sorted_trains = sorted(avg_usage, key=lambda train: avg_usage[train], reverse=True)
    return sorted_trains


def group_two_by_two(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, trains: list[Train]):
    # Group disrupted trains by section track, pair them, and order by scheduled time
    section_track_groups: dict[SectionTrack, list[Train]] = {}
    non_disrupted_trains = []
    t_disruption = {}
    t_section = {}

    disrupted_sections = set()
    if EAG.disruption_scenario and EAG.disruption_scenario.section_tracks:
        for track in EAG.disruption_scenario.section_tracks:
            disrupted_sections.add((track.origin, track.destination))

    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            tracks.add(track)

    # Group only disrupted trains by section track
    for t in trains:
        disrupted = False
        for a in EAG.A_train[t]:
            if a.section_track_planned and a.section_track_planned in tracks:
                if a.section_track_planned not in section_track_groups:
                    section_track_groups[a.section_track_planned] = []
                section_track_groups[a.section_track_planned].append(t)
                t_disruption[t] = Y[a.origin.id]
                t_section[t] = a.section_track_planned
                disrupted = True
                break  # Only consider the first disrupted activity
        if not disrupted:
            non_disrupted_trains.append(t)

    # Create pairs of trains based on the closest scheduled time for each section track
    grouped_trains = []
    for _, section_trains in section_track_groups.items():
        section_trains_sorted = sorted(section_trains, key=lambda t: t_disruption[t])
        if len(section_trains_sorted) % 2 != 0:
            if random.random() < 0.5:
                grouped_trains.append([section_trains_sorted.pop()])
        while len(section_trains_sorted) > 1:
            train1 = section_trains_sorted.pop(0)
            train2 = section_trains_sorted.pop(0)
            grouped_trains.append([train1, train2])

        # If there's an odd number of trains, the last one will remain ungrouped
        if section_trains_sorted:
            grouped_trains.append([section_trains_sorted.pop()])

    # Reorder the grouped_trains based on the smallest scheduled time in each pair
    grouped_trains_sorted = sorted(grouped_trains, key=lambda group: min(t_disruption[t] for t in group))

    # Flatten the grouped_trains_sorted to a one-dimensional list
    flattened_trains = [train for group in grouped_trains_sorted for train in group]

    # Add non-disrupted trains at the end with random order
    random.shuffle(non_disrupted_trains)
    flattened_trains.extend(non_disrupted_trains)

    return flattened_trains
