import logging
import random
from collections import defaultdict

import numpy as np

from ..event_activity_graph_multitracks import EARailwayNetwork, Train

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
#                                Tools
# --------------------------------------------------------------------


def get_train_usage(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, arc_usage: dict):
    # Compute usage statistics for each train based on arc_usage
    train_usage = defaultdict(list)
    for train in EAG.trains:
        for a in EAG.A_train[train]:
            if X[a.id] < 0.5:
                continue
            if a.activity_type in ("starting", "ending"):
                continue
            if a not in EAG.disagg_to_agg_activities:  # Can happen if the train starts at a junction
                continue
            else:
                agg_a = EAG.disagg_to_agg_activities[a]
                if agg_a in arc_usage:
                    train_usage[train].append(arc_usage[agg_a])
    return train_usage


# --------------------------------------------------------------------
#                               Operators
# --------------------------------------------------------------------


def most_delayed(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select the most delayed trains probabilistically
    trains = []
    weights = []
    epsilon = 1e-10  # Small value to avoid division by zero
    for t in EAG.trains:
        delay = 0.0
        this_train_activities = [
            a
            for a in EAG.get_ordered_activities_train(t)
            if a.activity_type in ("train running", "train waiting", "pass-through", "short-turning")
        ]
        for a in this_train_activities:
            if X[a.id] >= 0.5:
                delay += float(Y[a.origin.id]) - float(a.origin.scheduled_time)
        trains.append(t)
        weights.append(delay)
    probs = (np.array(weights, dtype=float) + epsilon) / np.sum(np.array(weights, dtype=float) + epsilon)
    probs = np.maximum(epsilon, probs)
    probs = probs / np.sum(probs)  # Normalize again to ensure sum = 1
    selected_indices = np.random.choice(len(trains), size=min(size, len(EAG.trains)), replace=False, p=probs)
    return [trains[i] for i in selected_indices]


def random_selection(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select random trains
    return random.sample(EAG.trains, min(size, len(EAG.trains)))


def mean_most_used(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains with highest average usage
    train_usage = get_train_usage(EAG, X, Y, Z, arc_usage)
    trains = []
    weights = []
    for t in EAG.trains:
        usage = train_usage[t]
        avg = np.mean(usage) if len(usage) > 0 else 0
        if avg > 0:
            trains.append(t)
            weights.append(avg)
    if not trains:
        return []
    size = min(size, len(trains))
    probs = np.array(weights) / sum(weights)
    selected_indices = np.random.choice(len(trains), size=size, replace=False, p=probs)
    return [trains[i] for i in selected_indices]


def mean_less_used(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains with lowest average usage
    train_usage = get_train_usage(EAG, X, Y, Z, arc_usage)
    trains = []
    weights = []
    for t in EAG.trains:
        usage = train_usage[t]
        avg = np.mean(usage) if len(usage) > 0 else 0
        if avg > 0:
            trains.append(t)
            weights.append(avg)
    if not trains:
        return []
    size = min(size, len(trains))
    probs = (1 - np.array(weights)) / np.sum(1 - np.array(weights))
    selected_indices = np.random.choice(len(trains), size=size, replace=False, p=probs)
    return [trains[i] for i in selected_indices]


def select_station(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains passing through randomly chosen stations
    stations = random.sample(EAG.stations, min(size, len(EAG.stations)))

    def t_through_station(t: Train):
        return any(
            a.origin.station in stations or a.destination.station in stations
            for a in EAG.A_train[t]
            if X[a.id] > 0.5 and a.activity_type in ("train waiting", "pass-through")
        )

    eligible_trains = [t for t in EAG.trains if t_through_station(t)]
    return random.sample(eligible_trains, min(size, len(eligible_trains)))


def select_station_close_time(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains passing through a station, ordered by time at station
    eligible_trains: list[Train] = []
    iter = 0
    while len(eligible_trains) < 2:
        station = random.choice(EAG.stations)

        def t_through_station(t: Train):
            return any(
                (a.origin.station == station or a.destination.station == station)
                and X[a.id] > 0.5
                and a.activity_type in ("train waiting", "pass-through")
                for a in EAG.A_train[t]
            )

        def time_at_station(t: Train):
            for a in EAG.A_train[t]:
                if (
                    X[a.id] > 0.5
                    and a.origin.station == station
                    and a.activity_type in ("train waiting", "pass-through")
                ):
                    return Y[a.origin.id]
            return float("inf")  # Fallback if no matching activity found

        eligible_trains = [t for t in EAG.trains if t_through_station(t)]
        if len(eligible_trains) >= 2:
            first_train = random.choice(eligible_trains)
            eligible_trains.remove(first_train)
            eligible_trains.sort(key=time_at_station)
            return [first_train] + eligible_trains[: min(size - 1, len(eligible_trains))]
        iter += 1
        if iter > 100:
            # Not found a valid station, return random trains
            logger.debug("Not found a valid station, return random trains")
            eligible_trains = [t for t in EAG.trains]
            return eligible_trains[: min(size, len(eligible_trains))]


def most_used_station(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains passing through the most used stations
    station_to_trains = defaultdict(set)
    for t in EAG.trains:
        for a in EAG.A_train[t]:
            if X[a.id] > 0.5 and a.activity_type in ("train waiting", "pass-through"):
                station_to_trains[a.origin.station].add(t)
    station_counts = {station: len(trains) for station, trains in station_to_trains.items()}
    train_weights: defaultdict[Train, int] = defaultdict(int)
    for station, ts in station_to_trains.items():
        count = station_counts[station]
        for t in ts:
            train_weights[t] += count
    trains, weights = zip(*train_weights.items())
    size = min(size, len(trains))
    probs = np.array(weights) / sum(weights)
    selected_indices = np.random.choice(len(trains), size=size, replace=False, p=probs)
    return [trains[i] for i in selected_indices]


def select_section(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains passing through a randomly chosen section
    section = random.choice(EAG.section_tracks)

    def t_through_section(t: Train):
        return any(a.section_track == section for a in EAG.A_train[t] if X[a.id] > 0.5 if a.section_track is not None)

    eligible_trains = [t for t in EAG.trains if t_through_section(t)]
    return random.sample(eligible_trains, min(size, len(eligible_trains)))


def through_disrupted_section(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select trains passing through disrupted sections
    if EAG.disruption_scenario is None:
        raise ValueError("Disruption scenario is missing.")
    disruption = EAG.disruption_scenario
    if disruption.section_tracks is None:
        return []
    disrupted_sections = set()
    for track in disruption.section_tracks:
        disrupted_sections.add((track.origin, track.destination))
    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections:
            tracks.add(track)

    def t_through_disrupted_sections(t: Train):
        return any(a.section_track in tracks for a in EAG.A_train[t] if X[a.id] > 0.5 if a.section_track is not None)

    eligible_trains = [t for t in EAG.trains if t_through_disrupted_sections(t)]
    return random.sample(eligible_trains, min(size, len(eligible_trains)))


def through_disrupted_section_period(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1
):
    # Select trains passing through disrupted sections during the disruption period
    if EAG.disruption_scenario is None:
        raise ValueError("Disruption scenario is missing.")
    disruption = EAG.disruption_scenario
    if disruption.section_tracks is None:
        return []
    disrupted_sections = set()
    for track in disruption.section_tracks:
        disrupted_sections.add((track.origin, track.destination))
    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections:
            tracks.add(track)

    def t_through_disrupted_sections_period(t: Train):
        return any(
            (
                (disruption.end_time >= a.origin.scheduled_time)
                and (disruption.start_time < a.origin.scheduled_time)
                or (disruption.end_time >= a.destination.scheduled_time)
                and (disruption.start_time < a.destination.scheduled_time)
            )
            for a in EAG.A_train[t]
            if X[a.id] > 0.5 and a.section_track is not None and a.section_track in tracks
        )

    eligible_trains = [t for t in EAG.trains if t_through_disrupted_sections_period(t)]
    return random.sample(eligible_trains, min(size, len(eligible_trains)))


def through_disrupted_section_period_plus10(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1
):
    # Select trains passing through disrupted sections during the disruption period plus 10 minutes
    if EAG.disruption_scenario is None:
        raise ValueError("Disruption scenario is missing.")
    disruption = EAG.disruption_scenario
    if disruption.section_tracks is None:
        return []
    disrupted_sections = set()
    for track in disruption.section_tracks:
        disrupted_sections.add((track.origin, track.destination))
    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            tracks.add(track)

    def t_through_disrupted_sections_period_plus10(t: Train):
        return any(
            (
                (disruption.end_time + 10 >= a.origin.scheduled_time)
                and (disruption.start_time < a.origin.scheduled_time)
                or (disruption.end_time + 10 >= a.destination.scheduled_time)
                and (disruption.start_time < a.destination.scheduled_time)
            )
            for a in EAG.A_train[t]
            if X[a.id] > 0.5 and a.section_track is not None and a.section_track in tracks
        )

    eligible_trains = [t for t in EAG.trains if t_through_disrupted_sections_period_plus10(t)]
    return random.sample(eligible_trains, min(size, len(eligible_trains)))


def through_disrupted_section_period_plus10_close(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1
):
    # Select trains in disrupted sections (period+10), triés par premier passage
    if EAG.disruption_scenario is None:
        raise ValueError("Disruption scenario is missing.")
    disruption = EAG.disruption_scenario
    if disruption.section_tracks is None:
        return []
    disrupted_sections = set()
    for track in disruption.section_tracks:
        disrupted_sections.add((track.origin, track.destination))
    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            tracks.add(track)

    def t_through_disrupted_sections_period_plus10(t: Train):
        return any(
            (
                (disruption.end_time + 10 >= a.origin.scheduled_time)
                and (disruption.start_time < a.origin.scheduled_time)
                or (disruption.end_time + 10 >= a.destination.scheduled_time)
                and (disruption.start_time < a.destination.scheduled_time)
            )
            for a in EAG.A_train[t]
            if X[a.id] > 0.5 and a.section_track is not None and a.section_track in tracks
        )

    def get_first_disrupted_time(t: Train):
        disrupted_times = [
            a.origin.scheduled_time
            for a in EAG.A_train[t]
            if X[a.id] > 0.5
            and a.section_track in tracks
            and (disruption.start_time < a.origin.scheduled_time <= disruption.end_time + 10)
        ]
        return min(disrupted_times) if disrupted_times else float("inf")

    eligible_trains = [t for t in EAG.trains if t_through_disrupted_sections_period_plus10(t)]
    eligible_trains.sort(key=get_first_disrupted_time)
    if len(eligible_trains) > 0:
        first_train = random.choice(eligible_trains)
        idx = eligible_trains.index(first_train)
        return eligible_trains[idx : idx + size]
    else:
        eligible_trains = EAG.trains
        first_train = random.choice(eligible_trains)
        idx = eligible_trains.index(first_train)
        return eligible_trains[idx : idx + size]


def cancelled(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Select cancelled trains
    def t_cancelled(t: Train):
        return (sum(X[a.id] for a in EAG.A_train[t] if a.activity_type not in ("starting", "ending")) == 0) or any(
            Z[e.id] > 0 for e in EAG.get_events_per_train(t) if e.id in Z
        )

    cancelled_trains = [t for t in EAG.trains if t_cancelled(t)]
    return random.sample(cancelled_trains, min(size, len(cancelled_trains)))
