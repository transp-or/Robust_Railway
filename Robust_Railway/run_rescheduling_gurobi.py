import os
import pickle

import gurobipy as gp
import numpy as np

from Robust_Railway.passenger_assignment import passenger_assignment
from Robust_Railway.rescheduling_gurobi_model import z_d, z_o, z_p
from Robust_Railway.Viriato_plugin.write_back_to_viriato import write_back_to_viriato


def warm_start(EAG, m, x, y):
    # Set initial values for Gurobi variables based on the timetable
    for event in EAG.events:
        if (
            not event.aggregated
            and event.in_timetable
            and event.scheduled_time is not None
            and event.node_type == "regular"
        ):
            y[event].Start = float(event.scheduled_time)
    for activity in EAG.activities:
        if not activity.aggregated:
            x[activity].Start = 1 if activity.in_timetable else 0
    m.update()
    return m, x, y


def print_passenger_costs(EAG, w, y, z_before_pref_time, z_after_pref_time):
    # Print passenger costs per group and total
    running_costs = waiting_costs = transfer_cost = access_cost = penalty_cost_sol = nb_egress = 0
    cost_per_group = np.zeros(len(EAG.passengers_groups))
    for arc in EAG.categorized_activities["group"]:
        for group in EAG.passengers_groups:
            if arc.activity_type == "passenger running" or arc.activity_type == "emergency bus":
                cost = w[arc, group].x * (y[arc.destination].x - y[arc.origin].x) * group.num_passengers
                running_costs += cost
                cost_per_group[group.id] += cost
            elif arc.activity_type == "dwelling":
                cost = w[arc, group].x * EAG.beta_1 * (y[arc.destination].x - y[arc.origin].x) * group.num_passengers
                waiting_costs += cost
                cost_per_group[group.id] += cost
            elif arc.activity_type == "transferring":
                cost = w[arc, group].x * (EAG.beta_2 + (y[arc.destination].x - y[arc.origin].x)) * group.num_passengers
                transfer_cost += cost
                cost_per_group[group.id] += cost
            elif arc.activity_type == "access":
                group = arc.passenger_group
                cost = (
                    w[arc, group].x
                    * (EAG.beta_3 * z_before_pref_time[arc].x + EAG.beta_4 * z_after_pref_time[arc].x)
                    * group.num_passengers
                )
                access_cost += cost
                cost_per_group[group.id] += cost
            elif arc.activity_type == "penalty":
                group = arc.passenger_group
                cost = w[arc, group].x * EAG.penalty_cost * group.num_passengers
                penalty_cost_sol += cost
                cost_per_group[group.id] += cost
            elif arc.activity_type == "egress":
                group = arc.passenger_group
                nb_egress += w[arc, group].x * group.num_passengers
    total_cost = running_costs + waiting_costs + transfer_cost + access_cost + penalty_cost_sol
    print("Passenger cost per group:", cost_per_group)
    print("Passenger Running Cost (including emergency bus):", running_costs)
    print("Dwelling (Waiting) Cost:", waiting_costs)
    print("Transfer Cost:", transfer_cost)
    print("Access Cost:", access_cost)
    print("Penalty Cost:", penalty_cost_sol)
    print("Total Cost:", total_cost)
    print("Number of egress activities:", nb_egress)


def print_passenger_timetable(EAG, w, y, z_before_pref_time, z_after_pref_time):
    # Print passenger timetable details
    for arc in EAG.categorized_activities["group"]:
        for group in EAG.passengers_groups:
            if w[arc, group].x >= 0.5:
                if arc.activity_type in ["passenger running", "emergency bus"]:
                    cost = w[arc, group].x * group.num_passengers * (y[arc.destination].x - y[arc.origin].x)
                    print(
                        f"{y[arc.origin].X}-{y[arc.destination].X} : group {group.id} "
                        f"travels from station {arc.origin.station.id} to station {arc.destination.station.id} "
                        f"with {arc.activity_type} and train {arc.origin.train.id} --- passenger cost {cost}\n"
                    )
                elif arc.activity_type == "dwelling":
                    cost = (
                        w[arc, group].x * group.num_passengers * EAG.beta_1 * (y[arc.destination].x - y[arc.origin].x)
                    )
                    print(
                        f"{y[arc.origin].X}-{y[arc.destination].X} : group {group.id} "
                        f"travels from station {arc.origin.station.id} to station {arc.destination.station.id} "
                        f"with {arc.activity_type} and train {arc.origin.train.id} --- passenger cost {cost}\n"
                    )
                elif arc.activity_type == "access":
                    cost = (
                        w[arc, group].x
                        * group.num_passengers
                        * (EAG.beta_3 * z_before_pref_time[arc].x + EAG.beta_4 * z_after_pref_time[arc].x)
                    )
                    print(
                        f"{y[arc.origin].X}-{y[arc.destination].X} : group {group.id} "
                        f"travels to station {arc.destination.station.id} with "
                        f"{arc.activity_type} and train {arc.destination.train.id} --- passenger cost {cost}\n"
                    )
                elif arc.activity_type == "egress":
                    print(
                        f"{y[arc.origin].X}-{y[arc.destination].X} : group {group.id} "
                        f"travels from station {arc.origin.station.id} with "
                        f"{arc.activity_type} and train {arc.origin.train.id}\n"
                    )
                elif arc.activity_type == "penalty":
                    penal_or = arc.passenger_group.origin.id
                    penal_dest = arc.passenger_group.destination.id
                    print(f"Group {arc.passenger_group.id} takes penalty arc (from {penal_or} to {penal_dest})")
                elif arc.activity_type == "transferring":
                    cost = (
                        w[arc, group].x * group.num_passengers * (EAG.beta_2 + (y[arc.destination].x - y[arc.origin].x))
                    )
                    print(
                        f"{y[arc.origin].X}-{y[arc.destination].X} : "
                        f"group {group.id} travels from station {arc.origin.station.id} to station "
                        f"{arc.destination.station.id} with {arc.activity_type} --- passenger cost {cost} \n"
                    )


def print_train_timetable(EAG, x, y, z, phi):
    # Print train timetable and statistics
    tot_time_difference = nb_w_act = nb_pt_act = num_track_changed = num_platform_changed = 0
    for train in EAG.trains:
        t_act = EAG.A_train[train]
        for arc in t_act:
            if x[arc].X > 0.5:
                if arc.activity_type != "starting":
                    tot_time_difference += abs(y[arc.origin].X - arc.origin.scheduled_time)
                if (
                    arc.origin in EAG.regular_rerouting_turning_events
                    and arc.destination in EAG.regular_rerouting_turning_events
                ):
                    track_changed = False
                    first_node_track = arc.origin.node_track.id if arc.origin.node_track else None
                    second_node_track = arc.destination.node_track.id if arc.destination.node_track else None
                    section_track = None
                    if arc.activity_type == "train waiting":
                        nb_w_act += 1
                        track = arc.origin.node_track
                        if track != arc.origin.node_track_planned:
                            num_platform_changed += 1
                    elif arc.activity_type == "pass-through":
                        nb_pt_act += 1
                        track = arc.origin.node_track
                        if track != arc.origin.node_track_planned:
                            num_platform_changed += 1
                    elif arc.activity_type == "train running":
                        track = arc.section_track
                        if track != arc.section_track_planned:
                            num_track_changed += 1
                            track_changed = True
                        section_track = track
                        if y[arc.destination].X - arc.destination.scheduled_time > 1:
                            print(
                                f"Arc {EAG.print_activity_info(arc)} is delayed for more than 1 minute. Delay: "
                                f"{y[arc.destination].X - arc.destination.scheduled_time} minutes"
                            )
                    # Print timetable details
                    if (
                        y[arc.origin].X != arc.origin.scheduled_time
                        or y[arc.destination].X != arc.destination.scheduled_time
                    ):
                        if section_track:
                            print(
                                f"{y[arc.origin].X} - {y[arc.destination].X} : train {arc.origin.train.id} "
                                f"travels from station {arc.origin.station.id} node_track {first_node_track} "
                                f"to station {arc.destination.station.id} node_track {second_node_track} with "
                                f"section track {section_track.id} with {arc.activity_type} | vs timetable: "
                                f"In timetable?  {arc.in_timetable}, {arc.origin.scheduled_time} - "
                                f"{arc.destination.scheduled_time}, section track changed? {track_changed}, "
                                f"minimum travel time {section_track.travel_time[arc.origin.train]}\n"
                            )
                        else:
                            print(
                                f"{y[arc.origin].X} - {y[arc.destination].X} : train {arc.origin.train.id} "
                                f"travels from station {arc.origin.station.id} node_track {first_node_track} "
                                f"to station {arc.destination.station.id} node_track {second_node_track} "
                                f"with {arc.activity_type} | vs timetable:  In timetable? "
                                f"{arc.in_timetable}, {arc.origin.scheduled_time} - {arc.destination.scheduled_time}"
                                f", section track changed? {track_changed}\n"
                            )
                    else:
                        if section_track:
                            print(
                                f"{y[arc.origin].X} - {y[arc.destination].X} : train {arc.origin.train.id} "
                                f"travels from station {arc.origin.station.id} node_track {first_node_track} "
                                f"to station {arc.destination.station.id} node_track {second_node_track} with "
                                f"section track {section_track.id}  with {arc.activity_type} | vs timetable: "
                                f"In timetable? {arc.in_timetable}"
                                f", section track changed? {track_changed}\n"
                            )
                        else:
                            print(
                                f"{y[arc.origin].X} - {y[arc.destination].X} : train {arc.origin.train.id} "
                                f"travels from station {arc.origin.station.id} node_track {first_node_track} "
                                f"to station {arc.destination.station.id} node_track {second_node_track} "
                                f"with {arc.activity_type} | vs timetable: In timetable? "
                                f"{arc.in_timetable}"
                                f", section track changed? {track_changed}\n"
                            )
            # Print cancelled events
            if (
                arc.activity_type not in ["starting", "ending", "artificial", "short-turning"]
                and arc.origin.node_type == "regular"
            ):
                if z[arc.origin].X > 0.5:
                    if (
                        arc.origin in EAG.regular_rerouting_turning_events
                        and arc.destination in EAG.regular_rerouting_turning_events
                    ):
                        print(
                            f"{y[arc.origin].X} - {y[arc.destination].X} : train {arc.origin.train.id} "
                            f"cancelled at station {arc.origin.station.id} node_track {arc.origin.node_track.id}"
                        )
    print(f"Total time difference with initial timetable: {tot_time_difference} minutes")
    print(f"Total number of waiting arcs difference: {nb_w_act - EAG.nb_waiting_activities}")
    print(f"Total number of pass through arcs difference: {nb_pt_act - EAG.nb_pass_through_activities}")
    print(f"# section track changes: {num_track_changed}")
    print(f"# station track changes: {num_platform_changed}")
    nb_emergency_bus = sum(phi[a].X for a in EAG.grouped_activities["emergency bus"])
    for a in EAG.grouped_activities["emergency bus"]:
        if phi[a].X > 0.5:
            print(
                f"{y[a.origin].X} - {y[a.destination].X} : Using emergency train {a.origin.train.id} "
                f"from station {a.origin.station.id} to station {a.destination.station.id}"
            )
    print(f"Number of emergency bus used: {nb_emergency_bus}")


def analyze_optimization_solution(
    EAG,
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
    phi,
    skip_pass_graph,
):
    # Print passenger costs and timetable if not skipping passenger graph
    if not skip_pass_graph:
        print_passenger_costs(EAG, w, y, z_before_pref_time, z_after_pref_time)
        print_passenger_timetable(EAG, w, y, z_before_pref_time, z_after_pref_time)
    # Print cancelled events
    cancelled_events = sum(z[event].x for event in EAG.regular_disaggregated_events)
    for event in EAG.regular_disaggregated_events:
        if z[event].x > 0.5:
            print("cancelled event", EAG.print_event_info(event))
    print("Number of cancelled events:", cancelled_events)
    # Print train timetable
    print_train_timetable(EAG, x, y, z, phi)


def run_rescheduling_gurobi(
    api_url,
    EAG,
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
    save_timetable,
    skip_pass_graph,
    write_back_viriato,
    time_limit=None,
    gap=None,
):
    # Set up the model and warm start
    end_time_window = EAG.end_time_window
    grouped_activities = EAG.grouped_activities
    train_running_activities = grouped_activities["train running"]
    train_pass_through_activities = grouped_activities["pass-through"]
    train_waiting_activities = grouped_activities["train waiting"]
    train_waiting_pass_through_activities = train_waiting_activities + train_pass_through_activities
    big_M1 = end_time_window

    m, x, y = warm_start(EAG, m, x, y)

    # Add disruption scenario constraints
    if EAG.disruption_scenario:
        d_end_time = EAG.disruption_scenario.end_time
        for t in EAG.disruption_scenario.section_tracks:
            for arc in train_running_activities:
                if arc.section_track == t:
                    m.addConstr(y[arc.origin] >= d_end_time - big_M1 * (1 - x[arc]))
        for t in EAG.disruption_scenario.node_tracks:
            for arc in train_waiting_pass_through_activities:
                if arc.origin.node_track == t:
                    m.addConstr(y[arc.origin] >= d_end_time - big_M1 * (1 - x[arc]))
    m.update()

    # Set objectives based on scenario
    m.ModelSense = gp.GRB.MINIMIZE
    if not EAG.disruption_scenario and not skip_pass_graph:
        m.setObjectiveN(z_p(EAG, w, y, v, v2, v3, v4), 2, priority=3, reltol=0.00)
        m.setObjectiveN(z_d(EAG, x, z, y, delta), 1, priority=2, reltol=0.00)
        m.setObjectiveN(z_o(EAG, phi), 0, priority=1)
    elif EAG.disruption_scenario and skip_pass_graph:
        m.setObjectiveN(z_d(EAG, x, z, y, delta), 1, priority=2, reltol=0)
        m.setObjectiveN(z_o(EAG, phi), 0, priority=1)
    elif EAG.disruption_scenario and not skip_pass_graph:
        m.setObjectiveN(z_d(EAG, x, z, y, delta), 2, priority=3, reltol=0.00)
        m.setObjectiveN(z_p(EAG, w, y, v, v2, v3, v4), 1, priority=2, reltol=0.00)
        m.setObjectiveN(z_o(EAG, phi), 0, priority=1)
    elif not EAG.disruption_scenario and skip_pass_graph:
        m.setObjectiveN(z_d(EAG, x, z, y, delta), 1, priority=2, reltol=0)
        m.setObjectiveN(z_o(EAG, phi), 0, priority=1)
    else:
        raise ValueError("Scenario not implemented in terms of the objective function")
    m.update()
    if time_limit:
        m.setParam("TimeLimit", time_limit)
    if gap:
        m.setParam("MIPGap", gap)

    m.optimize()

    # Extract values from Gurobi variables
    x_values = {k.id: v.X for k, v in x.items()}
    y_values = {k.id: v.X for k, v in y.items()}
    z_values = {k.id: v.X for k, v in z.items()}
    phi_values = {k.id: v.X for k, v in phi.items()}

    if write_back_viriato:
        write_back_to_viriato(EAG, api_url, None, x_values, y_values, z_values)

    if save_timetable:
        # Print objective values
        if not EAG.disruption_scenario and not skip_pass_graph:
            print("z_d():", m.getObjective(2).getValue())
            print("z_p():", m.getObjective(1).getValue())
            print("z_o():", m.getObjective(0).getValue())
        elif EAG.disruption_scenario and skip_pass_graph:
            print("z_d():", m.getObjective(1).getValue())
            print("z_o():", m.getObjective(0).getValue())
        elif EAG.disruption_scenario and not skip_pass_graph:
            print("z_p():", m.getObjective(0).getValue())
            print("z_d():", m.getObjective(1).getValue())
            print("z_o():", m.getObjective(2).getValue())
        elif not EAG.disruption_scenario and skip_pass_graph:
            print("z_d():", m.getObjective(1).getValue())
            print("z_o():", m.getObjective(0).getValue())

        analyze_optimization_solution(
            EAG,
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
            phi,
            skip_pass_graph,
        )

        job_id = os.environ.get("SLURM_JOB_ID", "nojobid")

        # Compute waiting arc cost
        waiting_cost = sum(EAG.delta_4 * delta[arc] for arc in EAG.grouped_activities["train waiting"])
        print("waiting cost", waiting_cost)

        # Passenger equilibrium costs
        z_p_pass_equi_priority = passenger_assignment(EAG, x_values, y_values, phi_values, level_of_detail=0)
        print("Passenger equilibrium cost (priority):", z_p_pass_equi_priority)
        z_p_pass_equi_first_in = passenger_assignment(
            EAG, x_values, y_values, phi_values, use_presence_in_train=True, level_of_detail=0
        )
        print("Passenger equilibrium cost (First-in):", z_p_pass_equi_first_in)

        # Save timetable to pickle
        with open(f"results_event_activity/Viriato_network/timetable_solution_{job_id}.pkl", "wb") as f:
            pickle.dump({"x": x_values, "y": y_values, "z": z_values, "phi": phi_values}, f)

        # Update EAG if not disruption scenario
        if not EAG.disruption_scenario:
            section_tracks = {}
            station_tracks = {}
            for key, val in x.items():
                activity = key
                if val.X > 0.5:
                    activity.in_timetable = True
                    if activity.activity_type == "train running":
                        key_tuple = (
                            activity.origin.station,
                            activity.destination.station,
                            int(activity.origin.scheduled_time),
                            activity.origin.train,
                        )
                        section_tracks[key_tuple] = activity.section_track
                    if activity.activity_type in ["train waiting", "pass-through"]:
                        key_tuple = (
                            activity.origin.station,
                            int(activity.origin.scheduled_time),
                            activity.origin.train,
                        )
                        station_tracks[key_tuple] = activity.origin.node_track
                else:
                    activity.in_timetable = False

            # Update planned tracks
            for key, value in EAG.A_train_running_similar.items():
                for a in value:
                    a.section_track_planned = section_tracks.get(key, a.section_track_planned)
            for a in EAG.grouped_activities["train waiting"] + EAG.grouped_activities["pass-through"]:
                key_tuple = (a.origin.station, int(a.origin.scheduled_time), a.origin.train)
                a.origin.node_track_planned = station_tracks.get(key_tuple, a.origin.node_track_planned)

            # Update events based on y values
            for key, val in y.items():
                event = key
                if event:
                    event.scheduled_time = float(val.X)

        output_path = f"results_event_activity/Viriato_network/EAG_updated_{job_id}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(EAG, f)

        m.write(f"results_event_activity/Viriato_network/model_{job_id}.lp")
