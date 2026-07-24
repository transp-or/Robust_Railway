from collections import defaultdict
from typing import Any, DefaultDict, Tuple

import gurobipy as gp
from py_client.aidm import StopStatus

from Robust_Railway.find_paths import find_all_paths


# First objective function - Passenger inconvenience
def z_p(EAG, w, y, v, v2, v3, v4) -> float:
    running_costs = sum(
        v[arc, group] * group.num_passengers
        for arc in EAG.grouped_activities["passenger running"]
        for group in EAG.passengers_groups
    )
    waiting_costs = sum(
        v3[arc, group] * group.num_passengers
        for arc in EAG.grouped_activities["dwelling"]
        for group in EAG.passengers_groups
    )
    transfer_cost = sum(
        group.num_passengers * v2[arc, group]
        for arc in EAG.grouped_activities["transferring"]
        for group in EAG.passengers_groups
    )
    access_cost = sum(v4[arc] * arc.passenger_group.num_passengers for arc in EAG.grouped_activities["access"])
    penalty_cost_sol = sum(
        w[arc, arc.passenger_group] * arc.passenger_group.num_passengers * EAG.penalty_cost
        for arc in EAG.grouped_activities["penalty"]
    )
    emergency_bus_costs = sum(
        v[arc, group] * group.num_passengers
        for arc in EAG.grouped_activities["emergency bus"]
        for group in EAG.passengers_groups
    )
    return running_costs + waiting_costs + transfer_cost + access_cost + penalty_cost_sol + emergency_bus_costs


# Second objective function - Operational costs
def z_o(EAG, phi) -> float:
    return gp.quicksum(
        phi[arc] * EAG.km_cost_emergency_bus * arc.section_track.distance
        for arc in EAG.grouped_activities["emergency bus"]
    )


# Third objective function - Deviation cost
def z_d(EAG, x, z, y, delta) -> float:
    train_waiting_activities = EAG.grouped_activities["train waiting"]
    return (
        EAG.delta_1
        * gp.quicksum(
            ((EAG.arrival_time_train_end[event.train] + 10) - float(event.scheduled_time)) * z[event]
            for event in EAG.regular_disaggregated_events
        )
        + EAG.delta_2
        * gp.quicksum(
            x[a] * a.section_track.distance
            for a in EAG.grouped_activities["train running"]
            if a.origin.node_type in ["rerouting", "short-turning"]
            and a.destination.node_type in ["rerouting", "short-turning"]
        )
        + EAG.delta_3
        * gp.quicksum(
            y[events[0]] - float(events[0].scheduled_time)
            for events in EAG.events_with_different_node_tracks
            if not events[0].aggregated and events[0].node_type == "regular"
        )
        + EAG.delta_4 * gp.quicksum(delta[arc] for arc in train_waiting_activities)
        + EAG.delta_5
        * gp.quicksum(
            x[arc]
            for arc in EAG.grouped_activities["train running"]
            if arc.section_track_planned != arc.section_track and arc.origin.node_type == "regular"
        )
        + EAG.delta_6
        * gp.quicksum(
            x[arc]
            for arc in EAG.grouped_activities["train waiting"] + EAG.grouped_activities["pass-through"]
            if arc.origin.node_track_planned != arc.origin.node_track and arc.origin.node_type == "regular"
        )
    )


def construct_model(
    EAG: Any,
    skip_pass_graph: bool,
) -> Tuple[Any, ...]:
    """
    Build and return the Gurobi optimization model for railway rescheduling.
    Returns a tuple of model and variables.
    """
    # Model parameters
    end_time_window = EAG.end_time_window
    start_time_window = EAG.start_time_window
    time_extra = EAG.time_extra  # Allow events to be rescheduled slightly after the time window

    # Preprocess passenger paths

    paths = []
    all_paths = {}
    for group in EAG.passengers_groups:
        all_paths[group.id] = find_all_paths(group, EAG, 15, 2, 2)
        paths.extend(all_paths[group.id])
        print("Paths created for passenger group:", group.id)
    print(f"Total number of paths for all passenger groups: {len(paths)}")

    def is_in_path(arc, path):
        arc = next(a for acts in EAG.grouped_activities.values() for a in acts if a.id == arc.id)
        return 1 if arc in path else 0

    big_M = (
        (end_time_window - start_time_window)
        + time_extra
        + max(
            EAG.waiting_time,
            EAG.minimum_separation_time,
            EAG.minimum_headway_passenger_trains,
            EAG.minimum_headway_freight_trains,
        )
    )

    m = gp.Model()  # passenger cost optimization
    m.setParam("Method", 1)
    m.setParam("NodefileStart", 0.3)  # start writing when reaches half of the available memory
    m.setParam("NodefileDir", "/tmp")  # or any fast temp directory with enough space
    m.setParam("Threads", 2)  # Corrected to 'Threads' for thread count (not 'ThreadLimit')
    m.setParam("PreSparsify", 0)  # Pre-sparsification before optimization (keep as is)

    # variables
    x = m.addVars(EAG.categorized_activities["train"], vtype="B", name="x")
    for a in EAG.categorized_activities["train"]:
        x[a].VarName = f"x[{a.id}]"

    # Pairwise train separation variables
    q = {}
    for station in EAG.stations:
        for station_track in station.node_tracks:
            acts = EAG.A_waiting_pass_through_dict[(station, station_track)]
            pairs = [(a1, a2) for a1 in acts for a2 in acts if a1.origin.train != a2.origin.train]
            if pairs:
                q.update(m.addVars(pairs, vtype="B", name="q"))

    # Pairwise train headway variables
    q2 = {}
    for section_track in EAG.section_tracks:
        acts = EAG.train_running_dict[section_track]
        pairs = [(a1, a2) for a1 in acts for a2 in acts if a1.origin.train != a2.origin.train]
        if pairs:
            q2.update(m.addVars(pairs, vtype="B", name="q2"))

    for station in EAG.stations:
        for station_track in station.node_tracks:
            acts = EAG.A_waiting_pass_through_dict[(station, station_track)]
            pairs = [(a1, a2) for a1 in acts for a2 in acts if a1.origin.train != a2.origin.train]
            if pairs:
                q.update(m.addVars(pairs, vtype="B", name="q"))

    # Time variables
    y_lb = start_time_window
    y_ub = start_time_window + EAG.time_horizon + time_extra
    y = m.addVars(EAG.events, lb=y_lb, ub=y_ub, vtype="C")
    for e in EAG.events:
        y[e].VarName = f"y[{e.id}]"

    delta = m.addVars(EAG.grouped_activities["train waiting"], vtype="B", name="delta")
    z = m.addVars(EAG.regular_disaggregated_events, vtype="B", name="z")
    for e in EAG.regular_disaggregated_events:
        z[e].VarName = f"z[{e.id}]"

    # Passenger graph variables (if not skipped)
    if not skip_pass_graph:
        w = m.addVars(EAG.categorized_activities["group"], EAG.passengers_groups, vtype="B", name="w")
        v = m.addVars(
            EAG.grouped_activities["passenger running"] + EAG.grouped_activities["emergency bus"],
            EAG.passengers_groups,
            lb=0,
            ub=EAG.time_horizon,
            vtype="C",
            name="v",
        )
        v2 = m.addVars(
            EAG.grouped_activities["transferring"],
            EAG.passengers_groups,
            lb=0,
            ub=EAG.time_horizon + EAG.beta_2,
            vtype="C",
            name="v2",
        )
        v3 = m.addVars(
            EAG.grouped_activities["dwelling"], EAG.passengers_groups, lb=0, ub=EAG.time_horizon, vtype="C", name="v3"
        )
        v4 = m.addVars(EAG.grouped_activities["access"], lb=0, ub=EAG.time_horizon, vtype="C", name="v4")
        phi = m.addVars(EAG.grouped_activities["emergency bus"], vtype="B", name="phi")
        p = m.addVars(range(len(paths)), EAG.passengers_groups, vtype="B", name="p")  # new
        u = m.addVars(EAG.categorized_activities["group"], EAG.passengers_groups, vtype="B", name="u")  # new

        # Auxiliary variables for max() linearization
        z_before_pref_time = {}
        z_after_pref_time = {}
        for arc in EAG.grouped_activities["access"]:
            group = arc.passenger_group
            z_before_pref_time[arc] = m.addVar(lb=0, name=f"z_bef_{arc}", vtype="C")
            z_after_pref_time[arc] = m.addVar(lb=0, name=f"z_aft_{arc}", vtype="C")
            # Linearization constraints
            m.addConstr(z_before_pref_time[arc] >= group.time - y[arc.destination], name=f"z_up_{arc.id}")
            m.addConstr(z_after_pref_time[arc] >= y[arc.destination] - group.time, name=f"z_down_{arc.id}")
            m.addConstr(
                v4[arc]
                >= (EAG.beta_3 * z_before_pref_time[arc] + EAG.beta_4 * z_after_pref_time[arc])
                - (big_M * EAG.beta_3 + big_M * EAG.beta_4) * (1 - w[arc, group]),
                name=f"access_cost_{arc.id}",
            )
        for arc in EAG.grouped_activities["transferring"]:
            for group in EAG.passengers_groups:
                m.addConstr(
                    v2[arc, group]
                    >= (EAG.beta_1 * (y[arc.destination] - y[arc.origin]) + EAG.beta_2)
                    - ((EAG.beta_1 * big_M + EAG.beta_2) * (1 - w[arc, group])),
                    name=f"transfer_cost_{arc.id}_{group}",
                )
        for a in EAG.grouped_activities["emergency bus"]:
            m.addConstr(len(EAG.passengers_groups) * phi[a] >= gp.quicksum(w[a, g] for g in EAG.passengers_groups))
        for arc in EAG.grouped_activities["passenger running"] + EAG.grouped_activities["emergency bus"]:
            for group in EAG.passengers_groups:
                m.addConstr(
                    v[arc, group] >= y[arc.destination] - y[arc.origin] - big_M * (1 - w[arc, group]),
                    name=f"passenger_running_cost_{arc.id}_{group}",
                )
        for arc in EAG.grouped_activities["dwelling"]:
            for group in EAG.passengers_groups:
                m.addConstr(
                    v3[arc, group]
                    >= (EAG.beta_1 * (y[arc.destination] - y[arc.origin])) - (big_M * EAG.beta_1) * (1 - w[arc, group]),
                    name=f"dwelling_cost_{arc.id}_{group}",
                )

    # Constraints to set delta variables
    for t in EAG.trains:
        for s in EAG.get_stations_per_train(t):
            if not s.junction:
                waiting_arcs = [
                    activity
                    for track in s.node_tracks
                    for activity in EAG.A_waiting_pass_through_dict[(s, track)]
                    if activity in EAG.A_train[t] and activity in EAG.grouped_activities["train waiting"]
                ]
                if any(a.in_timetable for a in waiting_arcs):
                    m.addConstr(gp.quicksum(x[arc] for arc in waiting_arcs) + delta[waiting_arcs[0]] == 1)

    # Valid inequalities and time ordering
    for events in EAG.events_with_different_node_tracks:
        for event in events[1:]:
            m.addConstr(y[event] == y[events[0]], name=f"valid_ineq_{event.id}")

    for t in EAG.trains:
        m.addConstr(
            gp.quicksum(z[a.origin] for a in EAG.A_train[t] if a.origin in EAG.regular_disaggregated_events) <= 1,
            name=f"train_{t.id}_z_sum",
        )

    for arc in EAG.categorized_activities["train"]:
        if arc.activity_type not in ["starting", "ending"]:
            m.addConstr(
                y[arc.destination] >= y[arc.origin],
                name=f"time_order_arc_{arc.id}_train_{arc.origin.train.id}_from_{arc.origin.station.id}_to_{arc.destination.station.id}",
            )

    # Start and flow conservation constraints
    for train in EAG.trains:
        m.addConstr(
            gp.quicksum(x[arc] for arc in EAG.starting_activities_dict[train]) == 1,
            name=f"train_{train.id}_start_constraint",
        )

    for event in EAG.regular_rerouting_turning_events:
        if (not event.station.junction) and (event.node_type in ["regular", "rerouting"]):
            m.addConstr(
                gp.quicksum(x[arc] for arc in EAG.A_plus[event])
                == gp.quicksum(x[arc] for arc in EAG.A_minus[event]) + z[event],
                name=f"event_{event.id}_flow_conservation",
            )
        else:
            m.addConstr(
                gp.quicksum(x[arc] for arc in EAG.A_plus[event]) == gp.quicksum(x[arc] for arc in EAG.A_minus[event]),
                name=f"event_{event.id}_flow_conservation",
            )

    # No cancellation at shunting yard without capacity constraints
    for event in EAG.regular_disaggregated_events:
        if not event.station.shunting_yard_capacity:
            m.addConstr(z[event] == 0, name=f"event_{event.id}_shunting_capacity")

    # Minimum running time constraints
    for _, value in EAG.A_train_running_similar.items():
        track = value[0].section_track
        if str(track.id).startswith("a"):
            continue
        if value[0].origin.train in track.travel_time:
            min_duration = track.travel_time[value[0].origin.train]
            m.addConstr(
                y[value[0].destination]
                >= y[value[0].origin] + min_duration - min_duration * (1 - gp.quicksum(x[arc] for arc in value)),
                name=f"min_running_time_{value[0].origin.id}_{value[0].destination.id}_{value[0].origin.train.id}",
            )
        else:
            raise ValueError(
                f"track missing between station {value[0].origin.station.id} and station "
                f"{value[0].destination.station.id}"
            )

    # Activity-specific time constraints
    for arc in EAG.categorized_activities["train"]:
        if arc.activity_type == "train waiting":
            m.addConstr(
                y[arc.destination] >= y[arc.origin] + EAG.waiting_time - big_M * (1 - x[arc]),
                name=f"waiting_time_{arc.id}",
            )
        elif arc.activity_type == "pass-through":
            if not arc.origin.station.node_tracks:
                m.addConstr(y[arc.destination] >= y[arc.origin] - big_M * (1 - x[arc]), name=f"pass_through_{arc.id}_1")
                m.addConstr(y[arc.destination] <= y[arc.origin] + big_M * (1 - x[arc]), name=f"pass_through_{arc.id}_2")
            else:
                m.addConstr(y[arc.destination] >= y[arc.origin] - big_M * (1 - x[arc]), name=f"pass_through_{arc.id}_3")
        elif arc.activity_type == "short-turning":
            m.addConstr(
                y[arc.destination] >= y[arc.origin] + EAG.short_turning - big_M * (1 - x[arc]),
                name=f"short_turning_{arc.id}",
            )

    # Maximum delay constraints
    for event in EAG.events:
        if event.node_type not in ["passenger origin", "passenger destination"]:
            m.addConstr(y[event] >= float(event.scheduled_time), name=f"event_{event.id}_scheduled_time")
        if (
            event.node_type not in ["passenger origin", "passenger destination", "train origin", "train destination"]
            and event.node_type == "regular"
        ):
            if event.train.capacity > 0:
                m.addConstr(
                    y[event] - event.scheduled_time <= EAG.passenger_train_max_delay,
                    name=f"event_{event.id}_max_delay_passenger",
                )
            else:
                m.addConstr(
                    y[event] - event.scheduled_time <= EAG.freight_train_max_delay,
                    name=f"event_{event.id}_max_delay_freight",
                )

    # Separation time constraints (at station)
    minimum_separation_time_13 = EAG.minimum_separation_time
    minimum_separation_time_14 = EAG.minimum_separation_time
    for station in EAG.stations:
        for station_track in station.node_tracks:
            activities_at_station_track = EAG.A_waiting_pass_through_dict[(station, station_track)]

            # Group activities by train and their scheduled origin time
            grouped_activities: DefaultDict[Any, DefaultDict[Any, list]] = defaultdict(lambda: defaultdict(list))

            for act in activities_at_station_track:
                train = act.origin.train
                grouped_activities[train][act.origin.scheduled_time].append(act)

            for t1 in EAG.trains:
                for t2 in EAG.trains:
                    if t1 != t2:
                        for _, t1_act in grouped_activities[t1].items():
                            for _, t2_act in grouped_activities[t2].items():
                                if len(t1_act) > 0 and len(t2_act) > 0:

                                    m.addConstr(
                                        y[t1_act[0].origin]
                                        >= y[t2_act[0].destination]
                                        + minimum_separation_time_13
                                        - big_M * q[t1_act[0], t2_act[0]]
                                        - big_M
                                        * (
                                            2
                                            - gp.quicksum(x[a1] for a1 in t1_act)
                                            - gp.quicksum(x[a2] for a2 in t2_act)
                                        ),
                                        name=f"sectiontrack_{station_track.id}_train1_{t1.id}_train2_{t2.id}_constraint12",
                                    )

                                    m.addConstr(
                                        y[t2_act[0].origin]
                                        >= y[t1_act[0].destination]
                                        + minimum_separation_time_14
                                        - big_M
                                        * (
                                            3
                                            - q[t1_act[0], t2_act[0]]
                                            - gp.quicksum(x[a1] for a1 in t1_act)
                                            - gp.quicksum(x[a2] for a2 in t2_act)
                                        ),
                                        name=f"sectiontrack_{station_track.id}_train1_{t1.id}_train2_{t2.id}_constraint13",
                                    )

    # Crossing conflict constraints
    print("Adding crossing conflict constraints...")
    nb_crossing_constraints = 0
    q3 = {}
    q4 = {}
    q5 = {}
    q6 = {}
    i = 0
    for s in EAG.stations:
        for st1 in s.node_tracks:
            for st2 in s.node_tracks:
                # if st1.id == st2.id:
                #    continue
                activities_at_station_track_1 = EAG.A_waiting_pass_through_dict[(s, st1)]
                activities_at_station_track_2 = EAG.A_waiting_pass_through_dict[(s, st2)]

                # Group activities by train and their scheduled origin time
                grouped_activities_1: DefaultDict[Any, DefaultDict[Any, list]] = defaultdict(lambda: defaultdict(list))
                grouped_activities_2: DefaultDict[Any, DefaultDict[Any, list]] = defaultdict(lambda: defaultdict(list))

                for act in activities_at_station_track_1:
                    train = act.origin.train
                    grouped_activities_1[train][act.origin.scheduled_time].append(act)

                for act in activities_at_station_track_2:
                    train = act.origin.train
                    grouped_activities_2[train][act.origin.scheduled_time].append(act)

                for t1 in EAG.trains:
                    for t2 in EAG.trains:
                        if t1 != t2:
                            for _, t1_act in grouped_activities_1[t1].items():
                                for _, t2_act in grouped_activities_2[t2].items():
                                    if len(t1_act) > 0 and len(t2_act) > 0:
                                        for at1 in t1_act:
                                            for at2 in t2_act:
                                                key = (at1, at2)
                                                if key not in q3:
                                                    q3[key] = m.addVar(vtype="B", name=f"q3_{i}")
                                                    i += 1
                                                if key not in q4:
                                                    q4[key] = m.addVar(vtype="B", name=f"q4_{i}")
                                                    i += 1
                                                if key not in q5:
                                                    q5[key] = m.addVar(vtype="B", name=f"q5_{i}")
                                                    i += 1
                                                if key not in q6:
                                                    q6[key] = m.addVar(vtype="B", name=f"q6_{i}")
                                                    i += 1
                                                # check if the two activities have a potential crossing conflict
                                                t1_incomings = EAG.A_plus[at1.origin]
                                                t1_outgoings = EAG.A_minus[at1.destination]
                                                t2_incomings = EAG.A_plus[at2.origin]
                                                t2_outgoings = EAG.A_minus[at2.destination]

                                                if at1.activity_type == "train waiting":
                                                    ss1 = StopStatus.commercial_stop
                                                elif at1.activity_type == "pass-through":
                                                    ss1 = StopStatus.passing
                                                else:
                                                    raise ValueError("Invalid activity type for incoming activity a1")

                                                if at2.activity_type == "train waiting":
                                                    ss2 = StopStatus.commercial_stop
                                                elif at2.activity_type == "pass-through":
                                                    ss2 = StopStatus.passing
                                                else:
                                                    raise ValueError("Invalid activity type for incoming activity a2")

                                                for a1 in t1_incomings:
                                                    for a2 in t2_incomings:
                                                        if (a1.activity_type != "train running") or (
                                                            a2.activity_type != "train running"
                                                        ):
                                                            continue

                                                        if (
                                                            a1.origin.station == a2.origin.station
                                                            and a1.destination.station == a2.destination.station
                                                        ):
                                                            # Trains run in the same direction
                                                            sep_time_left = EAG.separation_times[(
                                                                a1.section_track,
                                                                a1.destination.node_track,
                                                                "incoming",
                                                                ss1,
                                                                a2.section_track,
                                                                a2.destination.node_track,
                                                                "incoming",
                                                                ss2,
                                                            )]
                                                            if sep_time_left > 0:

                                                                m.addConstr(
                                                                    y[at1.origin] - y[at2.origin]
                                                                    >= -big_M * (1 - q3[key])
                                                                )

                                                                m.addConstr(
                                                                    y[at2.origin] - y[at1.origin] >= -big_M * q3[key]
                                                                )

                                                                m.addConstr(
                                                                    y[at1.origin]
                                                                    >= y[at2.origin]
                                                                    + sep_time_left
                                                                    - big_M
                                                                    * (
                                                                        (1 - q3[key])
                                                                        + (4 - x[at1] - x[at2] - x[a1] - x[a2])
                                                                    ),
                                                                    name=f"{i}_conflicts",
                                                                )
                                                                nb_crossing_constraints += 1
                                                        else:
                                                            continue  # no possible crossing conflict

                                                for a3 in t1_outgoings:
                                                    for a4 in t2_outgoings:
                                                        if (a3.activity_type != "train running") or (
                                                            a4.activity_type != "train running"
                                                        ):
                                                            continue
                                                        if (
                                                            a3.origin.station == a4.origin.station
                                                            and a3.destination.station == a4.destination.station
                                                        ):
                                                            # Trains run in the same direction
                                                            sep_time_right = EAG.separation_times[(
                                                                a3.section_track,
                                                                a3.origin.node_track,
                                                                "outgoing",
                                                                ss1,
                                                                a4.section_track,
                                                                a4.origin.node_track,
                                                                "outgoing",
                                                                ss2,
                                                            )]
                                                            if sep_time_right > 0:

                                                                m.addConstr(
                                                                    y[at1.destination] - y[at2.destination]
                                                                    >= -big_M * (1 - q4[key])
                                                                )

                                                                m.addConstr(
                                                                    y[at2.destination] - y[at1.destination]
                                                                    >= -big_M * q4[key]
                                                                )

                                                                m.addConstr(
                                                                    y[at1.destination]
                                                                    >= y[at2.destination]
                                                                    + sep_time_right
                                                                    - big_M
                                                                    * (
                                                                        (1 - q4[key])
                                                                        + (4 - x[at1] - x[at2] - x[a3] - x[a4])
                                                                    ),
                                                                    name=f"{i}_conflicts",
                                                                )
                                                                nb_crossing_constraints += 1

                                                        else:
                                                            continue  # no possible crossing conflict

                                                for a5 in t1_incomings:
                                                    for a6 in t2_outgoings:
                                                        if (a5.activity_type != "train running") or (
                                                            a6.activity_type != "train running"
                                                        ):
                                                            continue
                                                        if a5.destination.station == a6.origin.station:
                                                            # Trains run in opposite directions
                                                            sep_time_1 = EAG.separation_times[(
                                                                a5.section_track,
                                                                a5.destination.node_track,
                                                                "incoming",
                                                                ss1,
                                                                a6.section_track,
                                                                a6.origin.node_track,
                                                                "outgoing",
                                                                ss2,
                                                            )]
                                                            if sep_time_1 > 0:

                                                                m.addConstr(
                                                                    y[at1.origin] - y[at2.destination]
                                                                    >= -big_M * (1 - q5[key])
                                                                )

                                                                m.addConstr(
                                                                    y[at2.destination] - y[at1.origin]
                                                                    >= -big_M * q5[key]
                                                                )

                                                                m.addConstr(
                                                                    y[at1.origin]
                                                                    >= y[at2.destination]
                                                                    + sep_time_1
                                                                    - big_M
                                                                    * (
                                                                        (1 - q5[key])
                                                                        + (4 - x[at1] - x[at2] - x[a5] - x[a6])
                                                                    ),
                                                                    name=f"{i}_conflicts",
                                                                )
                                                                nb_crossing_constraints += 1

                                                        else:
                                                            raise ValueError(
                                                                "Incoming and outgoing activities",
                                                                "at the same station track with ",
                                                                "the same destination station",
                                                            )

                                                for a7 in t1_outgoings:
                                                    for a8 in t2_incomings:
                                                        if (a7.activity_type != "train running") or (
                                                            a8.activity_type != "train running"
                                                        ):
                                                            continue
                                                        if a7.origin.station == a8.destination.station:
                                                            # Trains run in opposite directions
                                                            sep_time_2 = EAG.separation_times[(
                                                                a7.section_track,
                                                                a7.origin.node_track,
                                                                "outgoing",
                                                                ss1,
                                                                a8.section_track,
                                                                a8.destination.node_track,
                                                                "incoming",
                                                                ss2,
                                                            )]
                                                            if sep_time_2 > 0:

                                                                m.addConstr(
                                                                    y[at2.origin] - y[at1.destination]
                                                                    >= -big_M * (1 - q6[key])
                                                                )

                                                                m.addConstr(
                                                                    y[at1.destination] - y[at2.origin]
                                                                    >= -big_M * q6[key]
                                                                )

                                                                m.addConstr(
                                                                    y[at2.origin]
                                                                    >= y[at1.destination]
                                                                    + sep_time_2
                                                                    - big_M
                                                                    * (
                                                                        (1 - q6[key])
                                                                        + (4 - x[at1] - x[at2] - x[a7] - x[a8])
                                                                    ),
                                                                    name=f"{i}_conflicts",
                                                                )
                                                                nb_crossing_constraints += 1

                                                        else:
                                                            raise ValueError(
                                                                "Outgoing and incoming activities at the same station"
                                                                " track with the same origin station"
                                                            )

    print(f"Number of crossing conflict constraints added: {nb_crossing_constraints}")

    # Headway constraints (on section track)
    for section_track in EAG.section_tracks:
        activities_at_section_track = EAG.train_running_dict[section_track]

        # Group activities by train and their scheduled origin time
        grouped_activities = defaultdict(lambda: defaultdict(list))  # {train: {scheduled_time: [acts]}}

        for act in activities_at_section_track:
            train = act.origin.train
            grouped_activities[train][act.origin.scheduled_time].append(act)

        for t1 in EAG.trains:
            for t2 in EAG.trains:
                if t1 != t2:
                    for t1_time, t1_act in grouped_activities[t1].items():
                        for t2_time, t2_act in grouped_activities[t2].items():

                            if len(t1_act) > 0 and len(t2_act) > 0:
                                if (
                                    t1_act[0].origin.station == t2_act[0].origin.station
                                ):  # trains running in the same direction
                                    if t2_act[0].origin.train.capacity > 0:
                                        minimum_headway = EAG.minimum_headway_passenger_trains
                                    else:
                                        minimum_headway = EAG.minimum_headway_freight_trains

                                    big_M_term_14 = (big_M) * (
                                        q2[t1_act[0], t2_act[0]]
                                        + (
                                            2
                                            - gp.quicksum(x[a1] for a1 in t1_act)
                                            - gp.quicksum(x[a2] for a2 in t2_act)
                                        )
                                    )
                                    m.addConstr(
                                        y[t1_act[0].origin] >= y[t2_act[0].origin] + minimum_headway - big_M_term_14,
                                        name=f"arc1_{t1_act[0].id}_arc2_{t2_act[0].id}_constraint14",
                                    )  # (14)
                                    m.addConstr(
                                        y[t1_act[0].destination]
                                        >= y[t2_act[0].destination] + minimum_headway - big_M_term_14,
                                        name=f"arc1_{t1_act[0].id}_arc2_{t2_act[0].id}_constraint15",
                                    )  # (15)

                                    if t1_act[0].origin.train.capacity > 0:
                                        minimum_headway = EAG.minimum_headway_passenger_trains
                                    else:
                                        minimum_headway = EAG.minimum_headway_freight_trains

                                    big_M_term_16 = (big_M) * (
                                        3
                                        - q2[t1_act[0], t2_act[0]]
                                        - gp.quicksum(x[a1] for a1 in t1_act)
                                        - gp.quicksum(x[a2] for a2 in t2_act)
                                    )

                                    m.addConstr(
                                        y[t2_act[0].origin] >= y[t1_act[0].origin] + minimum_headway - big_M_term_16,
                                        name=f"arc1_{t1_act[0].id}_arc2_{t2_act[0].id}_constraint16",
                                    )  # (16)
                                    m.addConstr(
                                        y[t2_act[0].destination]
                                        >= y[t1_act[0].destination] + minimum_headway - big_M_term_16,
                                        name=f"arc1_{t1_act[0].id}_arc2_{t2_act[0].id}_constraint17",
                                    )  # (17)

                                elif (
                                    t1_act[0].origin.station == t2_act[0].destination.station
                                ):  # trains running in opposite directions
                                    if t1_act[0].origin.train.capacity > 0:
                                        minimum_headway = EAG.minimum_headway_passenger_trains
                                    else:
                                        minimum_headway = EAG.minimum_headway_freight_trains
                                    m.addConstr(
                                        y[t2_act[0].origin]
                                        >= y[t1_act[0].destination]
                                        + minimum_headway
                                        - big_M * q2[t1_act[0], t2_act[0]]
                                        - big_M
                                        * (
                                            2
                                            - gp.quicksum(x[a1] for a1 in t1_act)
                                            - gp.quicksum(x[a2] for a2 in t2_act)
                                        ),
                                        name=f"arc1_{t1_act[0].id}_arc2_{t2_act[0].id}_constraint18",
                                    )  # (18)
                                    if t2_act[0].origin.train.capacity > 0:
                                        minimum_headway = EAG.minimum_headway_passenger_trains
                                    else:
                                        minimum_headway = EAG.minimum_headway_freight_trains
                                    m.addConstr(
                                        y[t1_act[0].origin]
                                        >= y[t2_act[0].destination]
                                        + minimum_headway
                                        - big_M
                                        * (
                                            3
                                            - q2[t1_act[0], t2_act[0]]
                                            - gp.quicksum(x[a1] for a1 in t1_act)
                                            - gp.quicksum(x[a2] for a2 in t2_act)
                                        ),
                                        name=f"arc1_{t1_act[0].id}_arc2_{t2_act[0].id}_constraint19",
                                    )  # (19)

                                else:
                                    raise ValueError(
                                        f"One track {track.id} is associated with two pairs of stations"
                                        f"{t1_act[0].origin.station}, {t1_act[0].destination.station} and"
                                        f"{t2_act[0].origin.station}, {t2_act[0].destination.station}"
                                    )

    if not skip_pass_graph:

        for group in EAG.passengers_groups:

            m.addConstr(
                sum(w[arc, group] for arc in EAG.grouped_activities["access"] if arc.passenger_group == group)
                + sum(w[arc, group] for arc in EAG.grouped_activities["penalty"] if arc.passenger_group == group)
                == 1
            )  # (22)

            m.addConstr(
                sum(w[arc, group] for arc in EAG.grouped_activities["egress"] if arc.passenger_group == group)
                + sum(w[arc, group] for arc in EAG.grouped_activities["penalty"] if arc.passenger_group == group)
                == 1
            )  # (23)

            for event in EAG.events:
                if event.aggregated:
                    m.addConstr(
                        sum(
                            w[arc, group]
                            for arc in EAG.A_minus_agg[event]
                            if arc.activity_type not in ["access", "egress", "penalty"]
                        )
                        + sum(
                            w[arc, group]
                            for arc in EAG.grouped_activities["egress"]
                            if arc.passenger_group == group and arc.origin == event
                        )
                        == sum(
                            w[arc, group]
                            for arc in EAG.A_plus_agg[event]
                            if arc.activity_type not in ["access", "egress", "penalty"]
                        )
                        + sum(
                            w[arc, group]
                            for arc in EAG.grouped_activities["access"]
                            if arc.passenger_group == group and arc.destination == event
                        ),
                    )  # (24)

            for arc in EAG.grouped_activities["access"]:
                if arc.passenger_group == group:
                    m.addConstr(w[arc, group] <= sum(x[a] for a in EAG.A_waiting_minus[arc.destination]))  # (25)

            for arc in EAG.grouped_activities["egress"]:
                if arc.passenger_group == group:
                    m.addConstr(w[arc, group] <= sum(x[a] for a in EAG.A_waiting_plus[arc.origin]))  # (26)

            for arc in EAG.grouped_activities["transferring"]:
                if arc.origin.node_type == "emergency" and arc.destination.node_type == "emergency":
                    raise ValueError("Error in transferring activity")

                elif arc.origin.node_type == "emergency":
                    m.addConstr(w[arc, group] <= sum(x[arc_] for arc_ in EAG.A_waiting_minus[arc.destination]))  # (28)
                elif arc.destination.node_type == "emergency":
                    m.addConstr(w[arc, group] <= sum(x[a] for a in EAG.A_waiting_plus[arc.origin]))  # (29)

                else:
                    m.addConstr(
                        2 * w[arc, group]
                        <= sum(x[a] for a in EAG.A_waiting_plus[arc.origin])
                        + sum(x[arc_] for arc_ in EAG.A_waiting_minus[arc.destination])
                    )  # (27)

                m.addConstr(
                    y[arc.destination] - y[arc.origin] >= EAG.minimum_transfer_time - big_M * (1 - w[arc, group])
                )  # (30)
                m.addConstr(
                    y[arc.destination] - y[arc.origin] <= EAG.maximum_transfer_time + big_M * (1 - w[arc, group])
                )  # (31)

            # old constraints
            """
            for arc in EAG.grouped_activities["passenger running"]:
                m.addConstr(w[arc, group] <= sum(x[arc_] for arc_ in EAG.agg_to_disagg_activities[arc]))  # (32)

                #tmp_large_capacity = arc.origin.train.capacity * 1000
                m.addConstr(
                    sum(group.num_passengers * w[arc, group] for group in EAG.passengers_groups)
                    <= arc.origin.train.capacity * sum(x[arc_] for arc_ in EAG.agg_to_disagg_activities[arc])
                )  # (33)
            """

            # new constraints

            for arc in EAG.grouped_activities["passenger running"]:
                m.addConstr(u[arc, group] <= sum(x[arc_] for arc_ in EAG.agg_to_disagg_activities[arc]))  # (32)

            # one path chosen
            m.addConstr(gp.quicksum(p[path_id, group] for path_id, path in enumerate(all_paths[group.id])) == 1)

            prev_groups = [g for g in EAG.passengers_groups if g.priority < group.priority]

            for arc in EAG.grouped_activities["passenger running"]:

                # free arcs
                m.addConstr(
                    u[arc, group]
                    <= gp.quicksum(
                        is_in_path(arc, path) * p[path_id, group] for path_id, path in enumerate(all_paths[group.id])
                    )
                )

                # used arc must be free
                m.addConstr(w[arc, group] <= u[arc, group])

                used_capacity = gp.quicksum(g_.num_passengers * w[arc, g_] for g_ in prev_groups)

                # capacity constraint
                # arc.origin.train.capacity = 10**10
                # tmp_large_capacity = arc.origin.train.capacity * 10000
                m.addConstr(
                    used_capacity
                    <= (arc.origin.train.capacity - group.num_passengers) * u[arc, group]
                    + arc.origin.train.capacity * (1 - u[arc, group])
                )

            for path_id, path in enumerate(all_paths[group.id]):
                for arc in path:
                    arc = next(a for acts in EAG.grouped_activities.values() for a in acts if a.id == arc.id)

                    # used arcs only in used path
                    m.addConstr(w[arc, group] >= p[path_id, group])

                    if arc.activity_type == "passenger running":
                        # used path need trains
                        m.addConstr(p[path_id, group] <= sum(x[arc_] for arc_ in EAG.agg_to_disagg_activities[arc]))

            # end of new constraints

        for event_1 in EAG.regular_rerouting_turning_aggregated_events:
            m.addConstrs(y[event_1] == y[e] for e in EAG.agg_to_disagg_events[event_1])  # (35)

        for a in EAG.grouped_activities["emergency bus"]:
            min_duration = a.section_track.travel_time[a.origin.train]

            m.addConstr(
                y[a.destination] >= y[a.origin] + min_duration - min_duration * (1 - phi[a])
            )  # time precedence of emergency bus

            m.addConstr(
                sum(group.num_passengers * w[a, group] for group in EAG.passengers_groups)
                <= arc.origin.train.capacity * phi[a]
            )  # capacity constraints of emergency bus

    m.update()

    # Return model and variables
    if skip_pass_graph:
        return m, x, None, y, None, None, None, None, z, None, None, delta, None
    else:
        return (
            m,
            x,
            w,
            y,
            v,
            v2,
            v3,
            v4,
            z,
            z_before_pref_time,
            z_after_pref_time,
            delta,
            phi,
        )
