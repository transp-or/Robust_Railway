import os
import pickle

from Robust_Railway.alns import ParetoClass, alns
from Robust_Railway.event_activity_graph_multitracks import Train
from Robust_Railway.operators.repair_operators_cancel import (
    cancel_partially_operator,
    cancel_train_completely,
    set_train_to_timetable,
)
from Robust_Railway.operators.repair_operators_delay import (
    cancel_if_too_delayed,
    delay_train_and_retrack,
)
from Robust_Railway.orderings.orderings import keep_order_at_disruption, regret_2step
from Robust_Railway.passenger_assignment import passenger_assignment
from Robust_Railway.rescheduling import DispositionTimetable, Rescheduling
from Robust_Railway.Viriato_plugin.write_back_to_viriato import write_back_to_viriato


def print_updated_timetable(EAG, Xplus, Yplus, Zplus, PHIplus):
    # Print trains timetable and statistics
    tot_time_difference = 0
    nb_w_act = 0
    nb_pt_act = 0
    num_track_changed = 0
    num_platform_changed = 0
    for train in EAG.trains:
        t_act = EAG.A_train[train]
        for arc in t_act:
            if Xplus[arc.id] > 0.5:
                if arc.activity_type != "starting":
                    tot_time_difference += abs(Yplus[arc.origin.id] - arc.origin.scheduled_time)
                if (
                    arc.origin in EAG.regular_rerouting_turning_events
                    and arc.destination in EAG.regular_rerouting_turning_events
                ):
                    track_changed = False
                    first_node_track = arc.origin.node_track.id if arc.origin.node_track else None
                    second_node_track = arc.destination.node_track.id if arc.destination.node_track else None
                    track = None
                    if arc.activity_type in ["train waiting", "pass-through"]:
                        if arc.origin.node_track != arc.origin.node_track_planned:
                            num_platform_changed += 1
                        if arc.activity_type == "train waiting":
                            nb_w_act += 1
                        else:
                            nb_pt_act += 1
                        track = arc.origin.node_track
                        if track:
                            track = track.id
                    elif arc.activity_type == "train running":
                        track = arc.section_track.id
                        if arc.section_track != arc.section_track_planned:
                            num_track_changed += 1
                            track_changed = True
                    if (
                        Yplus[arc.origin.id] != arc.origin.scheduled_time
                        or Yplus[arc.destination.id] != arc.destination.scheduled_time
                    ):
                        print(
                            f"{Yplus[arc.origin.id]} - {Yplus[arc.destination.id]} : train {arc.origin.train.id} "
                            f"travels from station {arc.origin.station.id} node_track {first_node_track} "
                            f"to station {arc.destination.station.id} node_track {second_node_track} with "
                            f"section track {track} with {arc.activity_type} | vs timetable:  In timetable? "
                            f"{arc.in_timetable}, {arc.origin.scheduled_time} - {arc.destination.scheduled_time}"
                            f", section track changed? {track_changed}\n"
                        )
                    else:
                        print(
                            f"{Yplus[arc.origin.id]} - {Yplus[arc.destination.id]} : train {arc.origin.train.id} "
                            f"travels from station {arc.origin.station.id} node_track {first_node_track} "
                            f"to station {arc.destination.station.id} node_track {second_node_track} with "
                            f"section track {track}  with {arc.activity_type} | vs timetable: In timetable? "
                            f"{arc.in_timetable}"
                            f", section track changed? {track_changed}\n"
                        )

            if arc.activity_type not in ["starting", "ending", "artificial"]:
                if Zplus[arc.origin.id] > 0.5:
                    if (
                        arc.origin in EAG.regular_rerouting_turning_events
                        and arc.destination in EAG.regular_rerouting_turning_events
                    ):
                        print(
                            f"{Yplus[arc.origin.id]} - {Yplus[arc.destination.id]} : train {arc.origin.train.id} "
                            f"cancelled at station {arc.origin.station.id} node_track {arc.origin.node_track.id}"
                        )

    print(f"Total time difference with initial timetable: {tot_time_difference} minutes")
    print(f"Total number of waiting arcs difference: {nb_w_act - EAG.nb_waiting_activities}")
    print(f"Total number of pass through arcs difference: {nb_pt_act - EAG.nb_pass_through_activities}")
    print(f"# section track changes: {num_track_changed}")
    print(f"# station track changes: {num_platform_changed}")
    nb_emergency_bus = 0
    for a in EAG.grouped_activities["emergency bus"]:
        nb_emergency_bus += PHIplus[a.id]
        if PHIplus[a.id] > 0.5:
            print(
                f"{Yplus[a.origin.id]} - {Yplus[a.destination.id]} : Using emergency train {a.origin.train.id} "
                f" between stations {a.origin.station.id} and {a.destination.station.id}"
            )
    print(f"Number of emergency bus used: {nb_emergency_bus}")


def set_initial_dict(EAG, train_activities, x, y):
    # Initialize decision dictionaries for timetable
    X, Y, Z, PHI = {}, {}, {}, {}
    for a in train_activities:
        if a.origin.node_type == "regular" and a.destination.node_type == "regular":
            Y[a.origin.id] = y[a.origin.id]
            Y[a.destination.id] = y[a.destination.id]
            X[a.id] = x[a.id]
        else:
            Y[a.origin.id] = a.origin.scheduled_time
            Y[a.destination.id] = a.destination.scheduled_time
            X[a.id] = 0
            if a.activity_type not in ["starting", "ending", "artificial"]:
                Z[a.origin.id] = 0
                Z[a.destination.id] = 0
    for a in EAG.grouped_activities["emergency bus"]:
        PHI[a.id] = 0
        Y[a.origin.id] = a.origin.scheduled_time
        Y[a.destination.id] = a.destination.scheduled_time
    return X, Y, Z, PHI


def generate_starting_solution(EAG, train_list, x, y, **kwargs):
    # Helper to generate a starting solution with given operators
    X, Y, Z, PHI = set_initial_dict(EAG, EAG.categorized_activities["train"], x, y)
    for t in EAG.trains:
        X, Y, Z, PHI = set_train_to_timetable(EAG, X, Y, Z, PHI, t)
    Xd, Yd, Zd, PHId = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    for t in train_list:
        Xd, Yd, Zd, PHId = cancel_train_completely(EAG, Xd, Yd, Zd, PHId, t)
    Xplus, Yplus, Zplus, PHIplus = Xd.copy(), Yd.copy(), Zd.copy(), PHId.copy()
    for t in train_list:
        Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplus, Yplus, Zplus, PHIplus, t)
        Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(EAG, Xplus, Yplus, Zplus, PHIplus, t, **kwargs)
        Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, t)
    return DispositionTimetable.from_decisions(Xplus, Yplus, Zplus, PHIplus)


def run_rescheduling_heuristic(
    EAG,
    initial_timetable_and_graph_ID,
    api_url,
    write_back_viriato,
    time_limit,
):
    # Load the pickle file
    with open(
        f"results_event_activity/Viriato_network/timetable_solution_{initial_timetable_and_graph_ID}.pkl", "rb"
    ) as f:
        solution = pickle.load(f)
    x, y, z, phi = solution["x"], solution["y"], solution["z"], solution["phi"]

    # Ensure emergency bus dictionary is initialized
    if not phi:
        for a in EAG.grouped_activities["emergency bus"]:
            phi[a.id] = 0

    # Check that there is no cancelled events or emergency trains used in initial timetable
    if any(value > 0.5 for value in z.values()):
        raise ValueError("Initial timetable contains cancelled trains")
    if any(value > 0.5 for value in phi.values()):
        raise ValueError("Initial timetable contains emergency vehicles")

    train_activities = EAG.categorized_activities["train"]
    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    FILE_NAME = f"results_event_activity/Viriato_network/ALNS/pareto_{job_id}"

    the_rescheduling = Rescheduling(EAG)
    DispositionTimetable.set_EAG(EAG)

    # Generate initial solutions
    if not EAG.disruption_scenario:
        raise ValueError("No disruption scenario specified")

    # First initial solution: delay and retrack trains at disrupted section
    trains_at_disrupted = EAG.get_ordered_trains_at_section_track(EAG.disruption_scenario.section_tracks)
    starting_solution_1 = generate_starting_solution(
        EAG, trains_at_disrupted, x, y, section_track_change=2, station_track_change=2
    )

    # Second initial solution: partial cancellation
    X, Y, Z, PHI = set_initial_dict(EAG, train_activities, x, y)
    for t in EAG.trains:
        X, Y, Z, PHI = set_train_to_timetable(EAG, X, Y, Z, PHI, t)
    Xd, Yd, Zd, PHId = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    for t in EAG.trains:
        Xd, Yd, Zd, PHId = cancel_train_completely(EAG, Xd, Yd, Zd, PHId, t)
    Xplus, Yplus, Zplus, PHIplus = Xd.copy(), Yd.copy(), Zd.copy(), PHId.copy()
    for t in EAG.trains:
        Xplus, Yplus, Zplus, PHIplus = cancel_partially_operator(EAG, Xplus, Yplus, Zplus, PHIplus, X, Y, Z, PHI, t)
    starting_solution_2 = DispositionTimetable.from_decisions(Xplus, Yplus, Zplus, PHIplus)

    # Third initial solution: delay and retrack with no track change
    starting_solution_3 = generate_starting_solution(
        EAG, trains_at_disrupted, x, y, section_track_change=0, station_track_change=0
    )

    # Fourth initial solution: keep order at disruption for eligible trains
    disrupted_sections = {(track.origin, track.destination) for track in EAG.disruption_scenario.section_tracks}
    tracks = {
        track
        for track in EAG.section_tracks
        if (track.origin, track.destination) in disrupted_sections
        or (track.destination, track.origin) in disrupted_sections
    }

    def t_through_disrupted_sections_period_plus10(t: Train):
        return any(
            (
                (EAG.disruption_scenario.end_time + 10 >= a.origin.scheduled_time)
                and (EAG.disruption_scenario.start_time < a.origin.scheduled_time)
                or (EAG.disruption_scenario.end_time + 10 >= a.destination.scheduled_time)
                and (EAG.disruption_scenario.start_time < a.destination.scheduled_time)
            )
            for a in EAG.A_train[t]
            if x[a.id] > 0.5 and a.section_track is not None and a.section_track in tracks
        )

    eligible_trains = [t for t in EAG.trains if t_through_disrupted_sections_period_plus10(t)]
    X, Y, Z, PHI = set_initial_dict(EAG, train_activities, x, y)
    for t in EAG.trains:
        X, Y, Z, PHI = set_train_to_timetable(EAG, X, Y, Z, PHI, t)
    Xd, Yd, Zd, PHId = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    for t in eligible_trains:
        Xd, Yd, Zd, PHId = cancel_train_completely(EAG, Xd, Yd, Zd, PHId, t)
    Xplus, Yplus, Zplus, PHIplus = Xd.copy(), Yd.copy(), Zd.copy(), PHId.copy()
    ordered_trains = keep_order_at_disruption(EAG, X, Y, Z, PHI, None, eligible_trains)

    starting_solution_4 = generate_starting_solution(
        EAG, ordered_trains, x, y, section_track_change=0, station_track_change=0
    )

    # Fifth initial solution: regret ordering for eligible trains
    def t_through_disrupted_sections_period(t: Train):
        return any(
            (
                (EAG.disruption_scenario.end_time >= a.origin.scheduled_time)
                and (EAG.disruption_scenario.start_time < a.origin.scheduled_time)
                or (EAG.disruption_scenario.end_time >= a.destination.scheduled_time)
                and (EAG.disruption_scenario.start_time < a.destination.scheduled_time)
            )
            for a in EAG.A_train[t]
            if x[a.id] > 0.5 and a.section_track is not None and a.section_track in tracks
        )

    eligible_trains_5 = [t for t in EAG.trains if t_through_disrupted_sections_period(t)]
    X, Y, Z, PHI = set_initial_dict(EAG, train_activities, x, y)
    for t in EAG.trains:
        X, Y, Z, PHI = set_train_to_timetable(EAG, X, Y, Z, PHI, t)
    Xd, Yd, Zd, PHId = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    for t in eligible_trains_5:
        Xd, Yd, Zd, PHId = cancel_train_completely(EAG, Xd, Yd, Zd, PHId, t)
    Xplus, Yplus, Zplus, PHIplus = Xd.copy(), Yd.copy(), Zd.copy(), PHId.copy()
    ordered_trains_5 = regret_2step(EAG, X, Y, Z, PHI, None, eligible_trains_5)
    starting_solution_5 = generate_starting_solution(
        EAG, ordered_trains_5, x, y, section_track_change=0, station_track_change=0
    )

    # Pareto optimization
    the_pareto = ParetoClass(max_neighborhood=int(len(EAG.trains)), pareto_file=FILE_NAME)
    the_pareto = alns(
        problem=the_rescheduling,
        first_solutions=[
            starting_solution_1.get_element(),
            starting_solution_2.get_element(),
            starting_solution_3.get_element(),
            starting_solution_4.get_element(),
            starting_solution_5.get_element(),
        ],
        pareto=the_pareto,
        time_limit_seconds=time_limit,
    )

    for i, sol in enumerate(list(the_pareto.pareto)):
        print_updated_timetable(EAG, sol.X, sol.Y, sol.Z, sol.PHI)
        passenger_assignment(EAG, x, y, phi, stopping_everywhere=False, verbose=0, level_of_detail=1)

    if write_back_viriato:
        write_back_to_viriato(EAG, api_url, the_pareto.pareto, None, None, None)
