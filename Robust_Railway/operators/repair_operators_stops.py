import logging
import random
from typing import cast

from ..event_activity_graph_multitracks import Activity, EARailwayNetwork, Station, Train
from .repair_operators_cancel import cancel_if_too_delayed
from .repair_operators_delay import delay_train_and_retrack, set_train_to_timetable

logger = logging.getLogger(__name__)


def add_one_stop(
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
    size = random.randint(1, 4)
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)

    A_pass_through = {}
    for e in EAG.get_events_per_train(train):
        if (
            e.event_type == "arrival"
            and not e.aggregated
            and not e.station.junction
            and not e.node_type == "short-turning"
        ):
            a_waiting = []
            a_pass = []
            for a_ in EAG.A_minus[e]:
                if a_.activity_type == "train waiting":
                    a_waiting.append(a_)
                elif a_.activity_type == "pass-through":
                    a_pass.append(a_)

            for a_ in a_pass:
                if Xplus[a_.id] > 0.5:
                    for a2 in a_waiting:
                        if a_.origin == a2.origin and a_.destination == a2.destination:
                            A_pass_through[a_] = a2

    if len(A_pass_through) == 0:
        if verbose > 0:
            logger.debug("No stop added")
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
            EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
        )
        return Xplus, Yplus, Zplus, PHIplus

    choices = list(A_pass_through.keys())
    if choices:
        a2_list = [cast(Activity, choice) for choice in random.choices(choices, k=min(size, len(choices)))]
    else:
        # Handle empty case or raise an error
        raise ValueError("No pass-through activities available.")

    for a in a2_list:
        Xplus[a.id] = 0
        activity = A_pass_through[a]
        if activity is not None:
            Xplus[activity.id] = 1
        else:
            raise ValueError(f"A_pass_through[{a}] is None")

        time_increment = max(0, EAG.waiting_time - (Yplus[activity.destination.id] - Yplus[activity.origin.id]))
        Yplus[activity.destination.id] += time_increment

        this_train_activities = [
            a2 for a2 in EAG.get_ordered_activities_train(train) if a2.id in Xplus and Xplus[a2.id] >= 0.5
        ]
        passed_added_stop = False
        for a3 in this_train_activities:
            if a3 == activity:
                passed_added_stop = True
            if passed_added_stop:
                Yplus[a3.origin.id] += time_increment
                Yplus[a3.destination.id] += time_increment

    if verbose > 0:
        logger.debug(f"Added a stop at station {activity.origin.station}")

    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
    )

    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)

    return Xplus, Yplus, Zplus, PHIplus


def remove_one_stop(
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
    size = random.randint(1, 4)
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)

    this_train_activities = [
        a
        for a in EAG.get_ordered_activities_train(train)
        if a.activity_type in ("train running", "train waiting", "pass-through")
    ]
    # Reduce this_train_activities to keep only activities in timetable
    this_train_activities = [
        a
        for a in this_train_activities
        if (a.id in Xplus) and (Xplus[a.id] >= 0.5)  # Activity is happening and is a train activity
    ]
    if len(this_train_activities) == 0:
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
            EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
        )
        return Xplus, Yplus, Zplus

    A_waiting = {}
    for e in EAG.get_events_per_train(train):
        if e.event_type == "arrival" and not e.aggregated:
            a_waiting = []
            a_pass = []
            for a in EAG.A_minus[e]:
                if a.activity_type == "train waiting":
                    a_waiting.append(a)
                elif a.activity_type == "pass-through":
                    a_pass.append(a)

            for a in a_waiting:
                if Xplus[a.id] > 0.5:
                    for a2 in a_pass:
                        if a.origin == a2.origin and a.destination == a2.destination:
                            A_waiting[a] = a2

    if len(A_waiting) == 0:
        if verbose > 0:
            logger.debug("No stop removed")
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
            EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
        )
        return Xplus, Yplus, Zplus, PHIplus

    choices = list(A_waiting)
    if choices:
        a2_list = [
            cast(Activity, choice) for choice in random.choices(choices, k=min(size, len(choices)))
        ]  # n is the number of selections you want
    else:
        # Handle empty case or raise an error
        raise ValueError("No waiting activities available.")

    for a2 in a2_list:
        Xplus[a2.id] = 0

        activity = A_waiting[a2]
        if activity is None:
            raise ValueError(f"A_waiting[{a}] is None")
        Xplus[activity.id] = 1
        Yplus[activity.destination.id] = Yplus[activity.origin.id]

        if verbose > 0:
            logger.debug(f"Removed a stop at station {a2.origin.station}")

    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=verbose
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)

    return Xplus, Yplus, Zplus, PHIplus


def change_node_track(
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
    station: Station,
    verbose: int = 0,
):

    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)

    this_train_activities = [
        a
        for a in EAG.get_ordered_activities_train(train)
        if a.activity_type in ("train running", "train waiting", "pass-through")
    ]
    # Reduce this_train_activities to keep only activities in timetable
    this_train_activities = [
        a
        for a in this_train_activities
        if (a.id in Xplus) and (Xplus[a.id] >= 0.5)  # Activity is happening and is a train activity
    ]
    if len(this_train_activities) == 0:
        logger.debug("change_node_track error : no activity planned")
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
            EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
        )
        return Xplus, Yplus, Zplus

    choices = []
    for a in EAG.A_train[train]:
        if a.activity_type in ["train waiting", "pass-through"] and a.origin.station == station:
            if not a.origin.station.junction and len(a.origin.station.node_tracks) > 1 and Xplus[a.id] > 0.5:
                choices.append(a)
    if len(choices) == 0:
        logger.debug("no change possible")
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
            EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
        )
        return Xplus, Yplus, Zplus, PHIplus  # No change possible

    found_incoming_arc, found_outcoming_arc = False, False

    while not (found_incoming_arc and found_outcoming_arc):
        activity_to_change = cast(Activity, random.choice(choices))
        Xplus[activity_to_change.id] = 0

        # Change waiting or pass-through activity
        potential_new_acts = []
        agg_act = EAG.disagg_to_agg_activities[activity_to_change]
        dis_acts = EAG.agg_to_disagg_activities[agg_act]
        for a2 in dis_acts:
            if (
                a2.origin.station == activity_to_change.origin.station
                and a2.destination.station == activity_to_change.destination.station
            ):
                if (
                    a2.origin.node_track != activity_to_change.origin.node_track
                    and a2.activity_type == activity_to_change.activity_type
                ):
                    potential_new_acts.append(a2)
        new_activity = cast(Activity, random.choice(potential_new_acts))
        Xplus[new_activity.id] = 1

        # Change incoming arcs
        for a3 in EAG.A_plus[activity_to_change.origin]:
            if Xplus[a3.id] == 1:
                in_origin_node_track = a3.origin.node_track
                in_section_track = a3.section_track
                Xplus[a3.id] = 0
                break

        found_incoming_arc = False
        for a4 in EAG.A_plus[new_activity.origin]:
            if (
                a4.origin.node_track == in_origin_node_track
                and a4.section_track == in_section_track
                and a4.destination.node_track == new_activity.origin.node_track
            ):
                Xplus[a4.id] = 1
                found_incoming_arc = True
                break

        # Change outcoming arcs
        for a5 in EAG.A_minus[activity_to_change.destination]:
            if Xplus[a5.id] == 1:
                out_destination_node_track = a5.destination.node_track
                out_section_track = a5.section_track
                Xplus[a5.id] = 0
                break

        found_outcoming_arc = False
        for a6 in EAG.A_minus[new_activity.destination]:
            if (
                a6.destination.node_track == out_destination_node_track
                and a6.section_track == out_section_track
                and a6.origin.node_track == new_activity.destination.node_track
            ):
                Xplus[a6.id] = 1
                found_outcoming_arc = True
                break

    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)

    return Xplus, Yplus, Zplus, PHIplus
