from collections import defaultdict
from typing import Any, DefaultDict, Tuple

import gurobipy as gp


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
    time_extra = 60  # Allow events to be rescheduled slightly after the time window

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

    q2 = {}
    for section_track in EAG.section_tracks:
        acts = EAG.train_running_dict[section_track]
        pairs = [(a1, a2) for a1 in acts for a2 in acts if a1.origin.train != a2.origin.train]
        if pairs:
            q2.update(m.addVars(pairs, vtype="B", name="q2"))

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

    # Constraints for train activities, time, and separation
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

    # Train path selection constraints
    for train in EAG.trains:
        m.addConstr(
            gp.quicksum(x[arc] for arc in EAG.starting_activities_dict[train]) == 1,
            name=f"train_{train.id}_start_constraint",
        )

    # Flow conservation and event constraints
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

    for event in EAG.regular_disaggregated_events:
        if not event.station.shunting_yard_capacity:
            m.addConstr(z[event] == 0, name=f"event_{event.id}_shunting_capacity")

    # Minimum running time constraints
    for key, value in EAG.A_train_running_similar.items():
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

    # Event time window constraints
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
                        for t1_time, t1_act in grouped_activities[t1].items():
                            for t2_time, t2_act in grouped_activities[t2].items():
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
                                    )  # updated (12)

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
                                    )  # updated (13)

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

            for arc in EAG.grouped_activities["passenger running"]:
                m.addConstr(w[arc, group] <= sum(x[arc_] for arc_ in EAG.agg_to_disagg_activities[arc]))  # (32)

                m.addConstr(
                    sum(group.num_passengers * w[arc, group] for group in EAG.passengers_groups)
                    <= arc.origin.train.capacity * sum(x[arc_] for arc_ in EAG.agg_to_disagg_activities[arc])
                )  # (33)

        for event_1 in EAG.regular_rerouting_turning_aggregated_events:
            m.addConstrs(y[event_1] == y[e] for e in EAG.agg_to_disagg_events[event_1])  # (35)

        for a in EAG.grouped_activities["emergency bus"]:
            min_duration = a.section_track.travel_time[a.origin.train]

            m.addConstr(
                y[a.destination] >= y[a.origin] + min_duration - min_duration * (1 - phi[a])
            )  # time precedence of emergency bus

            m.addConstr(
                sum(group.num_passengers * w[a, group] for group in EAG.passengers_groups)
                <= a.origin.train.capacity * phi[a]
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
