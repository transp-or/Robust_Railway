import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from py_client.aidm import StopStatus

from ..event_activity_graph_multitracks import Activity, Bus, EARailwayNetwork, NodeTrack, SectionTrack, Station, Train

logger = logging.getLogger(__name__)


def determine_direction(
    activity: Activity,
    track: SectionTrack | NodeTrack | None,
    previous_track: Optional[SectionTrack] = None,
    next_track: Optional[SectionTrack] = None,
) -> int:
    """Determine the direction of the activity on the section track."""
    if track is None:
        raise ValueError("Section track is None, cannot determine direction.")
    elif isinstance(track, NodeTrack):
        if previous_track is None and next_track is None:
            raise ValueError(
                f"Previous track or next track is None for activity {activity.id} on node track"
                f" {track.id}, cannot determine direction."
            )
        if previous_track and previous_track.id in track.incoming_section_tracks:
            direction = 1
        elif previous_track and previous_track.id in track.outgoing_section_tracks:
            direction = -1
        elif next_track and next_track.id in track.incoming_section_tracks:
            direction = -1
        elif next_track and next_track.id in track.outgoing_section_tracks:
            direction = 1
        else:
            raise ValueError(
                f"Problem with activity {activity.id} on node track {track.id} : previous track "
                f"does not correspond to incoming or outgoing tracks"
            )
    else:
        if (activity.origin.station == track.origin) and (activity.destination.station == track.destination):
            direction = 1
        elif (activity.origin.station == track.destination) and (activity.destination.station == track.origin):
            direction = -1
        else:
            raise ValueError(
                f"Problem with activity {activity.id} on section track {track.id} : origin "
                f" and destination do not correspond"
            )
    return direction


def get_minimum_headway(EAG: EARailwayNetwork, train: Train | Bus):
    """Get the headway required before the given train."""
    if train.capacity > 0:
        return EAG.minimum_headway_passenger_trains
    else:
        return EAG.minimum_headway_freight_trains


def get_track_usage_crossing(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict
) -> Dict[Station, List[Tuple[float, float, Any, Any, Any, Any, str, Any]]]:
    """Determines the occupation of a node track conflicting with an incoming or outcoming route

    Args:
        EAG (EARailwayNetwork): Event Activity Graph
        X (dict): X variable
        Y (dict): Y variable
        Z (dict): Z variable

    Returns:
        track_usage (Dict[Station, List[Tuple[float, float, Any, Any, Any, Any, str, Any]]]): List of tuples
        with the (start_time, end_time, direction, train, section track, node track, type ("incoming",
        "outgoing", "D", or "I"), and stop status) of each occupation of the tracks or (start_time, end_time,
        None, None, None, None, "D"", None) for a disruption. The first occupation is always the initial
        occupation of the track, with start_time = end_time = EAG.start_time_window and type "I"
    """
    track_usage: Dict[Station, List[Tuple[float, float, Any, Any, Any, Any, str, Any]]] = defaultdict(list)

    for t in EAG.trains:
        activities_train_activated = []
        for a in EAG.get_ordered_activities_train(t):
            if X[a.id] > 0.5:
                activities_train_activated.append(a)
        previous_a = activities_train_activated[0]
        for a in activities_train_activated[1:]:
            if previous_a.activity_type == "train running" and a.activity_type in ["pass-through", "train waiting"]:
                if a.activity_type == "train waiting":
                    ss = StopStatus.commercial_stop
                elif a.activity_type == "pass-through":
                    ss = StopStatus.passing
                else:
                    raise ValueError("Invalid activity type for activity")
                track_usage[a.origin.station].append((
                    Y[a.origin.id],
                    Y[a.destination.id],
                    determine_direction(previous_a, previous_a.section_track),
                    a.origin.train.id,
                    previous_a.section_track,
                    a.origin.node_track,
                    "incoming",
                    ss,
                ))
            elif previous_a.activity_type in ["pass-through", "train waiting"] and a.activity_type == "train running":
                if previous_a.activity_type == "train waiting":
                    ss = StopStatus.commercial_stop
                elif previous_a.activity_type == "pass-through":
                    ss = StopStatus.passing
                else:
                    raise ValueError("Invalid activity type for activity")
                track_usage[previous_a.origin.station].append((
                    Y[previous_a.origin.id],
                    Y[previous_a.destination.id],
                    determine_direction(a, a.section_track),
                    a.origin.train.id,
                    a.section_track,
                    previous_a.origin.node_track,
                    "outgoing",
                    ss,
                ))

    # Add disruptions
    if EAG.disruption_scenario:
        for node_track in EAG.disruption_scenario.node_tracks:  # type: ignore
            for station in EAG.stations:
                if node_track in station.node_tracks:
                    track_usage[station].append((
                        EAG.disruption_scenario.start_time,
                        EAG.disruption_scenario.end_time,
                        0,
                        "D",
                        None,
                        node_track,
                        "D",
                        None,
                    ))
    # Add initial occupation for node tracks
    for s in EAG.stations:
        for node_track in s.node_tracks:
            track_usage[s].append((EAG.start_time_window, EAG.start_time_window, 0, "I", None, node_track, "I", None))

    # Sort usage by start and end time
    for x in track_usage.values():
        x.sort(key=lambda tup: (tup[0], tup[1]))

    return track_usage


def get_track_usage(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict) -> Dict[SectionTrack | NodeTrack, List[Tuple]]:
    """Determine when each track is used by trains in the current solution of the timetable

    Args:
        EAG (EARailwayNetwork): Event Activity Graph
        X (dict): X variable
        Y (dict): Y variable
        Z (dict): Z variable

    Returns:
        track_usage (Dict[SectionTrack|NodeTrack, List[Tuple]]): List of tuples with the (start_time,
        end_time, direction, train_id, minimum_headway) of each occupation of the track
        or (start_time, end_time, 0, "D", 0) for a disruption
    """
    EPS = 1e-6
    track_usage: Dict[SectionTrack | NodeTrack, List[Tuple[Any, Any, Any, Union[int, str], Any]]] = defaultdict(list)
    for activity in (
        a
        for a in EAG.activities
        if a.activity_type in ("train running", "train waiting", "pass-through", "short-turning")
    ):
        if X[activity.id] > 0.5:
            if activity.activity_type == "train running" and activity.section_track and activity.origin.train:
                section_track: SectionTrack = activity.section_track
                start = activity.origin
                end = activity.destination
                direction = determine_direction(activity, section_track)
                headway = get_minimum_headway(EAG, activity.origin.train)
                track_usage[section_track].append(
                    (Y[start.id], Y[end.id], direction, activity.origin.train.id, headway)
                )
            else:
                node_track = activity.origin.node_track
                start = activity.origin
                end = activity.destination
                direction = 0
                headway = 0
                if node_track is not None and activity.origin.train:
                    track_usage[node_track].append(
                        (Y[start.id], Y[end.id], direction, activity.origin.train.id, headway)
                    )
                    if Y[start.id] - Y[end.id] > EPS and activity.activity_type != "ending":
                        logger.debug(EAG.print_activity_info(activity))
                        raise ValueError(
                            "Problematic activity: start time after end time",
                            Y[start.id],
                            Y[end.id],
                            EAG.print_activity_info(activity),
                        )

    # Add disruptions
    if EAG.disruption_scenario:
        for section_track in EAG.disruption_scenario.section_tracks:  # type: ignore
            track_usage[section_track].append(
                (EAG.disruption_scenario.start_time, EAG.disruption_scenario.end_time, 0, "D", 0)
            )
        for node_track in EAG.disruption_scenario.node_tracks:  # type: ignore
            track_usage[node_track].append(
                (EAG.disruption_scenario.start_time, EAG.disruption_scenario.end_time, 0, "D", 0)
            )

        # Remove activities during disruption
        for key, value in track_usage.items():
            in_section_tracks = (
                EAG.disruption_scenario.section_tracks is not None and key in EAG.disruption_scenario.section_tracks
            )
            in_node_tracks = (
                EAG.disruption_scenario.node_tracks is not None and key in EAG.disruption_scenario.node_tracks
            )
            if in_section_tracks or in_node_tracks:
                updated_value = []
                for usage in value:
                    if not (
                        usage[0] > EAG.disruption_scenario.start_time and usage[1] < EAG.disruption_scenario.end_time
                    ):
                        updated_value.append(usage)
                track_usage[key] = updated_value

    # Add initial occupation for all tracks and node tracks
    for track in EAG.section_tracks:
        direction = 0
        headway = 0
        track_usage[track].append((EAG.start_time_window, EAG.start_time_window, direction, "I", headway))

    for s in EAG.stations:
        for node_track in s.node_tracks:
            direction = 0
            headway = 0
            track_usage[node_track].append((EAG.start_time_window, EAG.start_time_window, direction, "I", headway))

    # Sort usage by start and end time
    for x in track_usage.values():
        x.sort(key=lambda tup: (tup[0], tup[1]))

    return track_usage


def get_junction_usage(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict) -> Dict[Station, List[Tuple]]:
    """Determine when each junction station is used by trains in the current solution of the timetable

    Args:
        EAG (EARailwayNetwork): Event Activity Graph
        X (dict): X variable
        Y (dict): Y variable
        Z (dict): Z variable

    Returns:
        junction_usage (Dict[Station, List[Tuple]]): List of tuples with the (start_time,
        end_time, direction, train_id, minimum_headway) of each occupation of the junctions
    """
    junction_usage: dict[Any, list[tuple[int, int, int, Any, int]]] = defaultdict(list)
    for activity in (a for a in EAG.activities if a.activity_type == "pass-through"):
        if X[activity.id] > 0.5:
            if activity.origin.station.junction and activity.origin.train:
                junction = activity.origin.station
                start = activity.origin
                end = activity.destination
                direction = 0
                headway = 0
                junction_usage[junction].append((Y[start.id], Y[end.id], direction, activity.origin.train.id, headway))

    # Add initial occupation for all junctions
    for s in EAG.stations:
        if s.junction:
            direction = 0
            headway = 0
            junction_usage[s].append((EAG.start_time_window, EAG.start_time_window, direction, "I", headway))

    # Sort usage by start and end time
    for x in junction_usage.values():
        x.sort(key=lambda tup: (tup[0], tup[1]))

    return junction_usage
