import logging
import random
from typing import Tuple, cast

import numpy as np

from Robust_Railway.event_activity_graph_multitracks import EARailwayNetwork, Train
from Robust_Railway.operators.repair_operators_cancel import cancel_if_too_delayed
from Robust_Railway.operators.repair_operators_delay import delay_train_and_retrack, set_train_to_timetable
from Robust_Railway.operators.repair_rerouting import set_train_to_short_turning

from ..passenger_assignment import passenger_assignment

logger = logging.getLogger(__name__)


def make_pass_group_better(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Try to improve the solution for the most penalized passenger groups
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    costs_per_group, _, _, _ = cast(
        Tuple[list, dict, dict, float],
        passenger_assignment(EAG, X, Y, PHI, stopping_everywhere=False, verbose=0, level_of_detail=1),
    )
    groups = np.arange(len(costs_per_group))

    # Compute hypothetical costs if all trains stop everywhere
    hypothetic_costs_per_group, _, used_trains, _ = cast(
        Tuple[list, dict, dict, float],
        passenger_assignment(EAG, X, Y, PHI, stopping_everywhere=True, verbose=0, level_of_detail=1),
    )
    cost_diff = {}
    for group in groups:
        group_obj = EAG.get_group_by_id(group)
        diff = group_obj.num_passengers * abs(costs_per_group[group] - hypothetic_costs_per_group[group])
        cost_diff[group_obj] = diff

    group_objs, diffs = zip(*cost_diff.items())
    total = np.sum(diffs)
    if total == 0:
        # Fallback: uniform probabilities
        probs = np.ones_like(diffs) / len(diffs)
    else:
        probs = diffs / total
    selected_indices = np.random.choice(len(group_objs), p=probs, size=size)
    selected_groups = [group_objs[i] for i in selected_indices]

    for selected_group in selected_groups:
        if len(used_trains[selected_group]) > 1:
            # Not implemented: group transferring
            continue
        if len(used_trains[selected_group]) == 0:
            # Passenger groups that can only take the penalty arc
            continue

        train = next(iter(used_trains[selected_group]))
        this_train_activities = [a2 for a2 in EAG.get_ordered_activities_train(train) if a2.id in X and X[a2.id] >= 0.5]

        corresponding_access, corresponding_egress = None, None
        for a in this_train_activities:
            if a.activity_type in ["train waiting", "pass-through"]:
                if a.origin.station == selected_group.origin:
                    corresponding_access = a
                if a.destination.station == selected_group.destination:
                    corresponding_egress = a
        if corresponding_access is None or corresponding_egress is None:
            raise ValueError(
                "Missing corresponding access and/or egress arcs", corresponding_access, corresponding_egress
            )

        changed_access, changed_egress, new_access, new_egress = False, False, None, None
        # Convert pass-through to waiting at access
        if corresponding_access.activity_type == "pass-through":
            X[corresponding_access.id] = 0
            access_delay = max(
                0, EAG.waiting_time - (Y[corresponding_access.destination.id] - Y[corresponding_access.origin.id])
            )
            for a in EAG.A_train[train]:
                if (
                    a.origin.station == corresponding_access.origin.station
                    and a.origin.node_track == corresponding_access.origin.node_track
                    and a.activity_type == "train waiting"
                ):
                    X[a.id] = 1
                    Y[a.destination.id] += access_delay
                    new_access = a
                    changed_access = True
            if not changed_access:
                raise ValueError("Missing waiting arc")
            else:
                passed_added_stop = False
                for a3 in this_train_activities:
                    if a3 == new_access:
                        passed_added_stop = True
                    if a3 != new_access and passed_added_stop:
                        Y[a3.origin.id] += access_delay
                        Y[a3.destination.id] += access_delay

        # Convert pass-through to waiting at egress
        if corresponding_egress.activity_type == "pass-through":
            X[corresponding_egress.id] = 0
            egress_delay = max(
                0, EAG.waiting_time - (Y[corresponding_egress.destination.id] - Y[corresponding_egress.origin.id])
            )
            for a in EAG.A_train[train]:
                if (
                    a.origin.station == corresponding_egress.origin.station
                    and a.origin.node_track == corresponding_egress.origin.node_track
                    and a.activity_type == "train waiting"
                ):
                    X[a.id] = 1
                    new_egress = a
                    changed_egress = True
            if not changed_egress:
                raise ValueError("Missing waiting arc")
            else:
                passed_added_stop = False
                for a3 in this_train_activities:
                    if a3 == new_egress:
                        passed_added_stop = True
                    if a3 != new_egress and passed_added_stop:
                        Y[a3.origin.id] += egress_delay
                        Y[a3.destination.id] += egress_delay

        # Apply delay and retrack, then cancel if too delayed
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
            EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=0
        )
        Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=0)

    return Xplus, Yplus, Zplus, PHIplus


def add_emergency_bus(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1):
    # Activate an emergency bus arc
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    potential_arcs = [a for a in EAG.grouped_activities["emergency bus"] if PHIplus[a.id] == 0]
    new_activated_arcs = random.sample(potential_arcs, min(size, len(potential_arcs)))

    if len(new_activated_arcs) == 0:
        return Xplus, Yplus, Zplus, PHIplus
    elif len(new_activated_arcs) == 1:
        new_activated_arc = new_activated_arcs[0]
    else:
        raise ValueError("There should be at most one emergency bus added at the same time")

    PHIplus[new_activated_arc.id] = 1
    Yplus[new_activated_arc.origin.id] = new_activated_arc.origin.scheduled_time
    Yplus[new_activated_arc.destination.id] = new_activated_arc.destination.scheduled_time

    return Xplus, Yplus, Zplus, PHIplus


def add_emergency_bus_and_short_turn(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, arc_usage: dict, size=1
):
    # Activate an emergency bus and short-turn a train at the destination station
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    potential_arcs = [a for a in EAG.grouped_activities["emergency bus"] if PHIplus[a.id] == 0]
    new_activated_arcs = random.sample(potential_arcs, min(size, len(potential_arcs)))

    if len(new_activated_arcs) == 0:
        return Xplus, Yplus, Zplus, PHIplus
    elif len(new_activated_arcs) == 1:
        new_activated_arc = new_activated_arcs[0]
    else:
        raise ValueError("There should be at most one emergency bus added at the same time")

    PHIplus[new_activated_arc.id] = 1
    Yplus[new_activated_arc.origin.id] = new_activated_arc.origin.scheduled_time
    Yplus[new_activated_arc.destination.id] = new_activated_arc.destination.scheduled_time

    # Find trains to short-turn at the destination station
    possible_train_short_turn = [
        a3.origin.train
        for a3 in EAG.grouped_activities["short-turning"]
        if (
            a3.origin.station == new_activated_arc.destination.station
            and a3.origin.scheduled_time - Yplus[new_activated_arc.destination.id] >= EAG.minimum_transfer_time
            and a3.origin.scheduled_time - Yplus[new_activated_arc.destination.id] < EAG.maximum_transfer_time
            and isinstance(a3.origin.train, Train)
        )
    ]

    trains_to_short_turn = random.sample(possible_train_short_turn, min(size, len(possible_train_short_turn)))

    if len(trains_to_short_turn) == 0:
        return Xplus, Yplus, Zplus, PHIplus
    elif len(trains_to_short_turn) == 1:
        train_to_short_turn = trains_to_short_turn[0]
    else:
        raise ValueError("There should be at most one train to short-turn")

    logger.debug(f"train to short-turn {train_to_short_turn.id}")
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplus, Yplus, Zplus, PHIplus, train_to_short_turn)
    Xplus, Yplus, Zplus, PHIplus = set_train_to_short_turning(EAG, Xplus, Yplus, Zplus, PHIplus, train_to_short_turn)
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train_to_short_turn, section_track_change=0, station_track_change=0
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(
        EAG, Xplus, Yplus, Zplus, PHIplus, train_to_short_turn, verbose=0
    )

    return Xplus, Yplus, Zplus, PHIplus
