import logging

from Robust_Railway.operators.repair_operators_delay import (
    delay_train_and_retrack,
    set_train_to_timetable,
)

from ..event_activity_graph_multitracks import Activity, EARailwayNetwork, Station, Train
from .repair_operators_cancel import cancel_if_too_delayed

logger = logging.getLogger(__name__)


def set_train_to_short_turning(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train):
    """Set the train to short-turning at the first available shunting station after disruption."""
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, X, Y, Z, PHI, train)

    d_stations: set[Station] = set()
    if not EAG.disruption_scenario:
        raise ValueError("No disruption scenario defined")
    if EAG.disruption_scenario.section_tracks:
        for t in EAG.disruption_scenario.section_tracks:
            d_stations.add(t.origin)
            d_stations.add(t.destination)

    shunting_station = None
    passed_disruption = False
    if not train.train_path_nodes:
        logger.debug(f"Train {train.id} has no path nodes")
        return Xplus, Yplus, Zplus, PHIplus

    # Find the first shunting station after disruption
    for node_idx, node in enumerate(train.train_path_nodes):
        station = EAG.get_station_by_id(node.node_id)
        if node_idx < len(train.train_path_nodes) - 1:
            next_node = train.train_path_nodes[node_idx + 1]
            next_station = EAG.get_station_by_id(next_node.node_id)
        else:
            next_station = None
        if not station:  # segment outside of the RER Vaud
            continue
        if station.shunting_yard_capacity > 0 and not passed_disruption:
            shunting_station = station
        if station in d_stations and next_station in d_stations:
            passed_disruption = True

    if not passed_disruption:
        logger.debug(f"Train {train.id} not going through the disruption")
        return Xplus, Yplus, Zplus, PHIplus
    elif not shunting_station:
        logger.debug(f"Train {train.id} going through the disruption, but no available shunting station")
        return Xplus, Yplus, Zplus, PHIplus

    # Activate short-turning activities after disruption
    passed_disruption = False
    previous_activity_activated: Activity | None = None
    previous_event = None
    for a in EAG.get_ordered_disagg_activities_train(train):
        if a.activity_type == "short-turning":
            passed_disruption = True
        if passed_disruption:
            if a.activity_type == "ending":
                if a.origin.node_type == "regular":
                    Xplus[a.id] = 0
                elif (
                    previous_activity_activated
                    and previous_activity_activated.activity_type != "ending"
                    and a.origin == previous_event
                ):
                    Xplus[a.id] = 1
                    previous_activity_activated = a
                    previous_event = a.destination
            elif a.activity_type != "ending" and a.destination.node_type == "regular":
                Xplus[a.id] = 0
            else:
                # Only set one similar activity to 1
                if previous_activity_activated:
                    if (
                        previous_activity_activated.activity_type == "short-turning"
                        and a.activity_type != "short-turning"
                        and a.origin == previous_event
                        and a.activity_type != "pass-through"
                    ):
                        Xplus[a.id] = 1
                        previous_activity_activated = a
                        previous_event = a.destination
                    else:
                        if (
                            previous_activity_activated.origin.station != a.origin.station
                            or previous_activity_activated.destination.station != a.destination.station
                        ) and a.origin == previous_event:
                            if a.origin.station.junction or a.activity_type != "pass-through":
                                Xplus[a.id] = 1
                                previous_activity_activated = a
                                previous_event = a.destination
                else:
                    if a.origin == previous_event:
                        Xplus[a.id] = 1
                        previous_activity_activated = a
                        previous_event = a.destination
                    else:
                        Xplus[a.id] = 0
        else:
            Xplus[a.id] = int(a.in_timetable)
            if a.in_timetable:
                previous_event = a.destination
            else:
                Xplus[a.id] = 0

    for event in EAG.get_events_per_train(train):
        Yplus[event.id] = event.scheduled_time
        Zplus[event.id] = 0

    return Xplus, Yplus, Zplus, PHIplus


def short_turn_train(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    verbose: int = 0,
):
    """Apply short-turning to a train and repair its timing and tracks."""
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)
    Xplus, Yplus, Zplus, PHIplus = set_train_to_short_turning(EAG, Xplus, Yplus, Zplus, PHIplus, train)
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=0, station_track_change=0
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus
