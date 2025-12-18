import csv
import hashlib
import logging
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Tuple, cast

from Robust_Railway.event_activity_graph_multitracks import EARailwayNetwork, Station
from Robust_Railway.neighborhood import Neighborhood, OperatorOutput
from Robust_Railway.operators.destroy_operators import (
    cancelled,
    mean_less_used,
    mean_most_used,
    most_delayed,
    most_used_station,
    random_selection,
    select_section,
    select_station,
    select_station_close_time,
    through_disrupted_section,
    through_disrupted_section_period,
    through_disrupted_section_period_plus10,
    through_disrupted_section_period_plus10_close,
)
from Robust_Railway.operators.destroy_plus_repair import add_emergency_bus, add_emergency_bus_and_short_turn
from Robust_Railway.operators.repair_operators_cancel import cancel_partially_operator, cancel_train_completely
from Robust_Railway.operators.repair_operators_delay import (
    delay_disrupted_track_operator,
    delay_mix,
    delay_operator,
    delay_track_operator,
)
from Robust_Railway.operators.repair_operators_stops import add_one_stop, remove_one_stop
from Robust_Railway.operators.repair_rerouting import short_turn_train
from Robust_Railway.orderings.orderings import (
    group_two_by_two,
    keep_order_at_disruption,
    mean_more_used_first,
    random_order,
    regret_2step,
)
from Robust_Railway.pareto import SetElement
from Robust_Railway.passenger_assignment import passenger_assignment

logger = logging.getLogger(__name__)


class DispositionTimetable:
    """Representation of a disposition timetable solution."""

    SEPARATOR = "-"
    EAG: ClassVar[Optional[EARailwayNetwork]] = None  # shared EAG for all solutions

    def __init__(
        self,
        X: dict,
        Y: dict,
        Z: dict,
        PHI: dict,
        objectives: list | None = None,
        arc_usage: dict | None = None,
    ):
        """
        Creates a disposition timetable from the given decision variables.

        Args:
            X (dict): Binary decision variable for train running activities.
            Y (dict): Continuous decision variable for event times.
            Z (dict): Decision variable for group assignments.
            PHI (dict): Decision variable for the use of emergency buses (binary).
            objectives (tuple, optional): A tuple containing pre-calculated objectives (z_d, z_p, z_o).
            arc_usage (dict, optional): A dictionary representing the arc usage across the network.

        Initializes:
            - `z_o`: Cost related to emergency bus usage.
            - `z_d`: Deviation cost from initial timetable.
            - `z_p`: Passenger assignment cost.
            - `arc_usage`: Usage of arcs in the disposition timetable.

        """
        if self.EAG is None:
            raise ValueError("EAG must be initialized before creating a DispositionTimetable.")

        self.X = X
        self.Y = Y
        self.Z = Z
        self.PHI = PHI

        # If objectives and arc_usage are not provided, compute them
        if objectives is None and arc_usage is None:
            self.z_o = 0.0
            # Emergency bus cost
            for arc in self.EAG.grouped_activities["emergency bus"]:
                if arc.section_track is None:
                    raise ValueError(f"Emergency bus activity {arc.id} has no section track assigned.")
                self.z_o += (
                    float(self.PHI[arc.id]) * float(self.EAG.km_cost_emergency_bus) * float(arc.section_track.distance)
                )

            # Count changed waiting activities (platform changes)
            nb_changed_waiting_act = 0
            for t in self.EAG.trains:
                for s in self.EAG.get_stations_per_train(t):
                    waiting_arcs = [
                        activity
                        for track in s.node_tracks
                        for activity in self.EAG.A_waiting_pass_through_dict[(s, track)]
                        if activity in self.EAG.A_train[t] and activity in self.EAG.grouped_activities["train waiting"]
                    ]
                    if any(a.in_timetable for a in waiting_arcs) and sum(self.X[arc.id] for arc in waiting_arcs) < 0.5:
                        nb_changed_waiting_act += 1

            # z_d components
            z_d_arrivals = float(self.EAG.delta_1) * float(
                sum(
                    ((self.EAG.arrival_time_train_end[event.train] + 10) - event.scheduled_time) * self.Z[event.id]
                    for event in self.EAG.regular_disaggregated_events
                )
            )

            z_d_rerouting_km = float(self.EAG.delta_2) * float(
                sum(
                    self.X[a.id] * float(a.section_track.distance)
                    for a in self.EAG.grouped_activities["train running"]
                    if a.origin.node_type in ["rerouting", "short-turning"]
                    and a.destination.node_type in ["rerouting", "short-turning"]
                    and a.section_track
                )
            )

            z_d_delay = float(self.EAG.delta_3) * float(
                sum(
                    float(self.Y[event.id]) - float(event.scheduled_time)
                    for event in self.EAG.events
                    if not event.aggregated
                    and event.node_type == "regular"
                    and sum(self.X[a.id] for a in self.EAG.A_plus[event]) > 0.5
                )
            )

            z_d_platform_changes = float(self.EAG.delta_4) * float(nb_changed_waiting_act)

            z_d_section_track_changes = float(self.EAG.delta_5) * float(
                sum(
                    float(self.X[arc.id])
                    for arc in self.EAG.grouped_activities["train running"]
                    if arc.section_track_planned != arc.section_track and arc.origin.node_type == "regular"
                )
            )

            z_d_station_track_changes = float(self.EAG.delta_6) * float(
                sum(
                    float(self.X[arc.id])
                    for arc in self.EAG.grouped_activities["train waiting"]
                    + self.EAG.grouped_activities["pass-through"]
                    if arc.origin.node_track_planned != arc.origin.node_track and arc.origin.node_type == "regular"
                )
            )

            self.z_d = (
                z_d_arrivals
                + z_d_rerouting_km
                + z_d_delay
                + z_d_platform_changes
                + z_d_section_track_changes
                + z_d_station_track_changes
            )

            # Compute passenger assignment to get z_p and arc_usage
            _, arc_usage, _, zp_cost = passenger_assignment(
                self.EAG, X, Y, PHI, stopping_everywhere=False, level_of_detail=1
            )  # type: ignore[misc]
            self.z_p = zp_cost
            self.arc_usage = arc_usage

        elif objectives is not None and arc_usage is not None:
            # Use provided objectives and arc_usage
            self.z_d, self.z_p, self.z_o = objectives[0], objectives[1], objectives[2]
            self.arc_usage = arc_usage
        else:
            raise ValueError("Either both objectives and arc_usage must be provided, or neither.")

    @classmethod
    def from_decisions(cls, X: dict, Y: dict, Z: dict, PHI: dict):
        """Creates a disposition timetable from the actual decisions"""
        return cls(X, Y, Z, PHI)

    def get_element(self) -> SetElement:
        """Return a SetElement representation for pareto or neighborhood usage."""
        if self.EAG is None:
            raise ValueError("EAG must be initialized before creating a DispositionTimetable.")
        num_cancelled_events = sum(self.Z[e.id] for e in self.EAG.regular_disaggregated_events)
        num_short_turning = sum(self.X[a.id] for a in self.EAG.grouped_activities["short-turning"])
        num_emergency_bus = sum(self.PHI[a.id] for a in self.EAG.grouped_activities["emergency bus"])
        return SetElement(
            self.code_id(),
            [self.z_d, self.z_p, self.z_o],
            X=self.X,
            Y=self.Y,
            Z=self.Z,
            PHI=self.PHI,
            arc_usage=self.arc_usage,
            num_cancelled_events=num_cancelled_events,
            num_short_turning=num_short_turning,
            num_emergency_bus=num_emergency_bus,
        )

    def code_id(self) -> str:
        """Return a unique hashed identifier for the solution."""
        return self.code_decisions(self.X, self.Y, self.Z, self.PHI)

    @staticmethod
    def code_decisions(X, Y, Z, PHI) -> str:
        """Generate a compact SHA-256 hash of the decisions."""
        return hashlib.sha256(pickle.dumps((X, Y, Z, PHI))).hexdigest()

    @classmethod
    def set_EAG(cls, EAG: EARailwayNetwork) -> None:
        cls.EAG = EAG


def run_operators(destroy: Callable, repair: Callable, ordering: Callable, element: SetElement, size: int):
    """Run a combined destroy/repair/order operator and return new SetElement and number of changes."""
    logger.debug("Running repair %s", repair.__name__)
    solution = DispositionTimetable(element.X, element.Y, element.Z, element.PHI, element.objectives, element.arc_usage)
    EAG = solution.EAG
    if EAG is None:
        raise ValueError("EAG is not initialized")

    X, Y, Z, PHI = solution.X.copy(), solution.Y.copy(), solution.Z.copy(), solution.PHI.copy()
    arc_usage = solution.arc_usage.copy() if solution.arc_usage is not None else {}

    # Some repair operators expect fixed size semantics
    if repair.__name__ == "add_one_stop":
        size = 2

    # Certain combined operators use a different calling signature and always size 1
    combined_ops = {
        add_emergency_bus.__name__,
        add_emergency_bus_and_short_turn.__name__,
    }
    if destroy.__name__ in combined_ops:
        # Combined destroy/repair returns full decisions directly
        size = 1
        logger.debug("Running combined operator %s with size %d", destroy.__name__, size)
        Xplus, Yplus, Zplus, PHIplus = destroy(EAG, X, Y, Z, PHI, arc_usage, size)
        neighbor = DispositionTimetable.from_decisions(Xplus, Yplus, Zplus, PHIplus)
        return neighbor.get_element(), size

    trains = destroy(EAG, X, Y, Z, PHI, arc_usage, size)
    ordered_trains = ordering(EAG, X, Y, Z, PHI, arc_usage, trains)

    if not trains:
        logger.debug("Destroy operator %s returned no trains (size=%s).", destroy.__name__, size)
        return element, 0

    # Remove trains completely from solution before repair
    X_d, Y_d, Z_d, PHI_d = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    for train in trains:
        X_d, Y_d, Z_d, PHI_d = cancel_train_completely(EAG, X_d, Y_d, Z_d, PHI_d, train)

    # Different repair behaviors depending on repair operator
    if repair.__name__ == "cancel_train_completely":
        X_plus, Y_plus, Z_plus, PHI_plus = X_d.copy(), Y_d.copy(), Z_d.copy(), PHI_d.copy()

    elif repair.__name__ == "change_node_track":
        # Choose station with highest number of events among ordered_trains
        nb_events_at_station: defaultdict[Station, int] = defaultdict(int)
        for t in ordered_trains:
            for a in EAG.A_train[t]:
                if a.activity_type in ["train waiting", "pass-through"]:
                    nb_events_at_station[a.origin.station] += 1

        if not nb_events_at_station:
            neighbor = DispositionTimetable.from_decisions(X_d, Y_d, Z_d, PHI_d)
            return neighbor.get_element(), len(trains)

        stations = list(nb_events_at_station.keys())
        weights = [nb_events_at_station[st] for st in stations]
        selected_station = random.choices(stations, weights=weights, k=1)[0]

        # repair expects a specific signature (train + selected station)
        X_plus, Y_plus, Z_plus, PHI_plus = repair(EAG, X_d, Y_d, Z_d, PHI_d, X, Y, Z, PHI, train, selected_station)

    elif repair.__name__ == "remove_one_stop":
        # First remove stops from first train(s), then reschedule other trains
        X_plus, Y_plus, Z_plus, PHI_plus = X_d.copy(), Y_d.copy(), Z_d.copy(), PHI_d.copy()
        first_group = ordered_trains[:1]
        second_group = ordered_trains[1:]

        for train in first_group:
            logger.debug("Removing one stop from train %s", train.id)
            X_plus, Y_plus, Z_plus, PHI_plus = repair(EAG, X_plus, Y_plus, Z_plus, PHI_plus, X, Y, Z, PHI, train)

        for train in second_group:
            X_plus, Y_plus, Z_plus, PHI_plus = delay_disrupted_track_operator(
                EAG, X_plus, Y_plus, Z_plus, PHI_plus, X, Y, Z, PHI, train
            )

    elif repair.__name__ == "delay_mix":
        X_plus, Y_plus, Z_plus, PHI_plus = X_d.copy(), Y_d.copy(), Z_d.copy(), PHI_d.copy()
        # find number of trains to change tracks
        nb_disrupted_trains = 0
        for t in EAG.trains:
            for a in EAG.A_train[t]:
                if (
                    EAG.disruption_scenario
                    and isinstance(EAG.disruption_scenario.section_tracks, list)
                    and a.section_track in EAG.disruption_scenario.section_tracks
                    and a.in_timetable
                    and a.origin.scheduled_time >= EAG.disruption_scenario.start_time
                    and a.origin.scheduled_time <= EAG.disruption_scenario.end_time
                ):
                    nb_disrupted_trains += 1
                    break

        max_nb_track_changes = random.randint(1, nb_disrupted_trains)

        nb_change = 0

        for train in ordered_trains:
            disrupted = False
            for a in EAG.A_train[train]:
                if (
                    EAG.disruption_scenario
                    and isinstance(EAG.disruption_scenario.section_tracks, list)
                    and a.section_track in EAG.disruption_scenario.section_tracks
                    and a.in_timetable
                ):
                    disrupted = True
            if disrupted and nb_change < max_nb_track_changes:
                X_plus, Y_plus, Z_plus, PHI_plus = repair(EAG, X_plus, Y_plus, Z_plus, PHI_plus, X, Y, Z, PHI, train, 1)
                nb_change += 1
            else:
                X_plus, Y_plus, Z_plus, PHI_plus = repair(EAG, X_plus, Y_plus, Z_plus, PHI_plus, X, Y, Z, PHI, train, 2)
    else:
        # Generic repair: apply repair to ordered trains
        X_plus, Y_plus, Z_plus, PHI_plus = X_d.copy(), Y_d.copy(), Z_d.copy(), PHI_d.copy()
        for train in ordered_trains:
            X_plus, Y_plus, Z_plus, PHI_plus = repair(EAG, X_plus, Y_plus, Z_plus, PHI_plus, X, Y, Z, PHI, train)

    neighbor = DispositionTimetable.from_decisions(X_plus, Y_plus, Z_plus, PHI_plus)
    return neighbor.get_element(), len(trains)


class Rescheduling(Neighborhood):
    """Neighborhood class for the rescheduling problem (ALNS)."""

    def __init__(self, EAG: EARailwayNetwork):
        self.EAG = EAG
        self.operators: dict[str, tuple[Callable, Callable, Callable]] = {}
        base_dir = Path(__file__).parent
        forbidded_file = base_dir / "forbidded_operators.csv"

        forbidded_operators = set()
        if forbidded_file.exists():
            with forbidded_file.open(newline="", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        forbidded_operators.add(row[0].strip())

        # Build operator combinations while skipping forbidden ones
        destroy_ops = (
            random_selection,
            mean_less_used,
            mean_most_used,
            cancelled,
            select_section,
            select_station,
            through_disrupted_section,
            through_disrupted_section_period,
            through_disrupted_section_period_plus10,
            through_disrupted_section_period_plus10_close,
            most_used_station,
            select_station_close_time,
            most_delayed,
        )

        repair_ops = (
            delay_operator,
            delay_track_operator,
            delay_mix,
            delay_disrupted_track_operator,
            cancel_partially_operator,
            cancel_train_completely,
            add_one_stop,
            remove_one_stop,
            short_turn_train,
        )

        order_ops = (
            random_order,
            keep_order_at_disruption,
            regret_2step,
            mean_more_used_first,
            group_two_by_two,
        )

        for d in destroy_ops:
            for r in repair_ops:
                for o in order_ops:
                    key = f"{d.__name__}-{r.__name__}-{o.__name__}"
                    if key not in forbidded_operators:
                        self.operators[key] = cast(
                            Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]], (d, r, o)
                        )

        # If multiple disrupted section tracks, enable emergency bus operators
        if (
            EAG.disruption_scenario
            and isinstance(EAG.disruption_scenario.section_tracks, list)
            and len(EAG.disruption_scenario.section_tracks) > 1
        ):
            self.operators[add_emergency_bus.__name__] = (add_emergency_bus, add_emergency_bus, add_emergency_bus)
            self.operators[add_emergency_bus_and_short_turn.__name__] = (
                add_emergency_bus_and_short_turn,
                add_emergency_bus_and_short_turn,
                add_emergency_bus_and_short_turn,
            )

        self.last_operator = None

        self.jobid = os.environ.get("SLURM_JOB_ID", "nojobid")
        self.op_perform_file = f"results_event_activity/Viriato_network/ALNS/operator_performance_{self.jobid}.csv"

        # Initialize operator performance CSV with header
        header = [
            "Operators",
            "Neighborhood Size",
            "Win",
            "Attempt",
            "Number of changes",
            "Time",
            "Previous Z_d",
            "Previous Z_o",
            "Previous Z_p",
            "New Z_d",
            "New Z_o",
            "New Z_p",
            "Accepted",
        ]
        Path(self.op_perform_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.op_perform_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

        # Choose initial scores file depending on disruption complexity
        if (
            EAG.disruption_scenario
            and isinstance(EAG.disruption_scenario.section_tracks, list)
            and len(EAG.disruption_scenario.section_tracks) == 1
        ):
            init_scores_file = "operators_init_scores_partial.csv"
        else:
            init_scores_file = "operators_init_scores_complete.csv"

        super().__init__(self.operators, init_scores_file)

        self.op_score_file = f"results_event_activity/Viriato_network/ALNS/operator_score_{self.jobid}.csv"
        # Write current operator scores (keys order consistent)
        Path(self.op_score_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.op_score_file, "w", newline="") as f:
            writer = csv.writer(f)
            # write header of operator names then on next lines scores will be appended in generate_neighbor
            writer.writerow(list(self.operators.keys()))

    def last_neighbor_rejected(self) -> None:
        with open(self.op_perform_file, "r+") as f:
            lines = f.readlines()
            if lines:
                lines[-1] = lines[-1].strip() + ", Rejected\n"
                f.seek(0)
                f.writelines(lines)
        return super().last_neighbor_rejected()

    def last_neighbor_accepted(self) -> None:
        with open(self.op_perform_file, "r+") as f:
            lines = f.readlines()
            if lines:
                lines[-1] = lines[-1].strip() + ", Accepted\n"
                f.seek(0)
                f.writelines(lines)
        return super().last_neighbor_accepted()

    def last_neighbor_added_to_pareto(self, obj_improvement) -> None:
        with open(self.op_perform_file, "r+") as f:
            lines = f.readlines()
            if lines:
                lines[-1] = lines[-1].strip() + ", Pareto\n"
                f.seek(0)
                f.writelines(lines)
        return super().last_neighbor_added_to_pareto(obj_improvement)

    def reset_scores(self):
        return super().reset_scores()

    def generate_neighbor(self, element: SetElement, neighborhood_size: int, attempts: int = 5) -> OperatorOutput:
        """
        Try to generate a neighbor using ALNS operators. Record operator scores each try.

        Returns the neighbor SetElement and number_of_changes (0 if nothing changed).
        """
        # Append current operator scores (values in consistent order) to score file
        with open(self.op_score_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.operators_management.scores.get(k, 0) for k in self.operators.keys()])

        for attempt in range(attempts):
            destroy_op, repair_op, order_op = self.operators_management.select_operator()
            start_time = time.time()
            neighbor, number_of_changes = run_operators(destroy_op, repair_op, order_op, element, neighborhood_size)
            compu_time = time.time() - start_time

            with open(self.op_perform_file, "a", newline="") as f:
                writer = csv.writer(f)
                if number_of_changes > 0:
                    values = (
                        [
                            self.operators_management.last_operator_name,
                            neighborhood_size,
                            True,
                            attempt,
                            number_of_changes,
                            compu_time,
                        ]
                        + list(element.objectives)
                        + list(neighbor.objectives)
                    )
                    writer.writerow(values)
                    return neighbor, number_of_changes
                else:
                    values = (
                        [
                            self.operators_management.last_operator_name,
                            neighborhood_size,
                            False,
                            attempts,
                            0,
                            compu_time,
                        ]
                        + list(element.objectives)
                        + list(element.objectives)
                    )
                    writer.writerow(values)

        return element, 0

    def is_valid(self, element: SetElement):
        """
        Validate timetable solution against operational constraints.
        Returns (True, None) if valid, else (False, reason).
        """
        solution = DispositionTimetable(element.X, element.Y, element.Z, element.PHI)
        EAG = solution.EAG
        if EAG is None:
            raise ValueError("EAG is not initialized")

        train_activities = EAG.categorized_activities["train"]

        # Flow conservation: exactly one starting activity per train
        for train in EAG.trains:
            start_count = sum(solution.X[arc.id] for arc in EAG.starting_activities_dict[train])
            if start_count != 1:
                return False, "Flow conservation constraints violated: starting activities"

        # Flow conservation at rerouting/turning events
        for event in EAG.regular_rerouting_turning_events:
            inflow = sum(solution.X[a.id] for a in EAG.A_plus[event])
            outflow = sum(solution.X[a.id] for a in EAG.A_minus[event])
            if not event.station.junction and event.node_type == "regular":
                outflow += solution.Z[event.id]
            if inflow != outflow:
                return False, "Flow conservation constraints violated at event"

        # Cancellation feasibility
        for event in EAG.regular_disaggregated_events:
            if solution.Z[event.id] == 1 and not event.station.shunting_yard_capacity:
                return False, "Cancellation constraints violated"

        # Time precedence constraints
        EPS = 1e-6
        for arc in train_activities:
            if solution.X[arc.id] > 0.5:
                if arc.activity_type == "train waiting":
                    if arc.origin.station.junction:
                        return False, "Waiting activity at a junction not allowed"
                    if solution.Y[arc.destination.id] < solution.Y[arc.origin.id] + EAG.waiting_time - EPS:
                        return False, "Minimum waiting time violated"
                elif arc.activity_type == "train running":
                    if not arc.section_track:
                        return False, "Train running activity without section track"
                    minimum_travel_time = arc.section_track.travel_time[arc.origin.train]
                    if solution.Y[arc.destination.id] < solution.Y[arc.origin.id] + minimum_travel_time - EPS:
                        return False, "Minimum travel time violated"
                elif arc.activity_type == "short-turning":
                    if solution.Y[arc.destination.id] < solution.Y[arc.origin.id] + EAG.short_turning - EPS:
                        return False, "Minimum short-turning time violated"
                elif arc.activity_type not in ["starting", "ending"]:
                    if solution.Y[arc.destination.id] < solution.Y[arc.origin.id] - EPS:
                        return False, "Negative time precedence detected"
                    if arc.activity_type == "pass-through" and arc.origin.station.node_tracks == [None]:
                        if abs(solution.Y[arc.destination.id] - solution.Y[arc.origin.id]) > EPS:
                            return False, "Pass-through at junction with non-zero waiting time"

        # Events cannot be scheduled before their original scheduled time (except passenger origins/destinations)
        for event in EAG.events:
            if event.node_type not in ["passenger origin", "passenger destination"]:
                if event.scheduled_time - solution.Y[event.id] > EPS:
                    return False, "Event scheduled before planned time"
            # Maximum allowed delay per event depending on train type
            if (
                event.node_type
                not in ["passenger origin", "passenger destination", "train origin", "train destination"]
                and event.node_type == "regular"
            ):
                max_delay = EAG.passenger_train_max_delay if event.train.capacity > 0 else EAG.freight_train_max_delay
                if solution.Y[event.id] > event.scheduled_time + max_delay:
                    return False, "Event delayed beyond maximum allowed"

        # Station track occupation constraints
        for station in EAG.stations:
            for station_track in station.node_tracks:
                activities_at_station_track = EAG.A_waiting_pass_through_dict[(station, station_track)]
                for arc_1 in activities_at_station_track:
                    for arc_2 in activities_at_station_track:
                        if arc_1.origin.train != arc_2.origin.train:
                            mst1 = EAG.minimum_separation_time
                            mst2 = EAG.minimum_separation_time
                            if (
                                solution.Y[arc_1.origin.id] < solution.Y[arc_2.destination.id] + mst1 - EPS
                                and solution.Y[arc_2.origin.id] < solution.Y[arc_1.destination.id] + mst2 - EPS
                                and solution.X[arc_1.id] > 0.5
                                and solution.X[arc_2.id] > 0.5
                            ):
                                # return False, "Station track occupation constraint violated"
                                msg = (
                                    f"Station track occupation constraint violated ARC1: "
                                    f"{EAG.print_activity_info(arc_1)}, ARC2: {EAG.print_activity_info(arc_2)}, "
                                    f"{solution.Y[arc_1.origin.id]}, {solution.Y[arc_1.destination.id]}, "
                                    f"{solution.Y[arc_2.origin.id]}, {solution.Y[arc_2.destination.id]}"
                                )
                                return (False, msg)

        # Section track occupation constraints
        for section_track in EAG.section_tracks:
            activities_at_section_track = EAG.train_running_dict[section_track]
            for arc_1 in activities_at_section_track:
                for arc_2 in activities_at_section_track:
                    if arc_1.origin.train == arc_2.origin.train:
                        continue

                    # Same direction case
                    if arc_1.origin.station == arc_2.origin.station:
                        condition_not_satisfied = False
                        if solution.Y[arc_1.origin.id] >= solution.Y[arc_2.origin.id]:
                            minimum_headway = (
                                EAG.minimum_headway_passenger_trains
                                if arc_2.origin.train.capacity > 0
                                else EAG.minimum_headway_freight_trains
                            )
                            travel_time_diff = max(
                                0,
                                (solution.Y[arc_2.destination.id] - solution.Y[arc_2.origin.id])
                                - (solution.Y[arc_1.destination.id] - solution.Y[arc_1.origin.id]),
                            )
                            if (
                                solution.X[arc_1.id] > 0.5
                                and solution.X[arc_2.id] > 0.5
                                and solution.Y[arc_1.origin.id]
                                < solution.Y[arc_2.origin.id] + travel_time_diff + minimum_headway - EPS
                            ):
                                condition_not_satisfied = True
                        else:
                            minimum_headway = (
                                EAG.minimum_headway_passenger_trains
                                if arc_1.origin.train.capacity > 0
                                else EAG.minimum_headway_freight_trains
                            )
                            travel_time_diff2 = max(
                                0,
                                (solution.Y[arc_1.destination.id] - solution.Y[arc_1.origin.id])
                                - (solution.Y[arc_2.destination.id] - solution.Y[arc_2.origin.id]),
                            )
                            if (
                                solution.X[arc_1.id] > 0.5
                                and solution.X[arc_2.id] > 0.5
                                and solution.Y[arc_2.origin.id]
                                < solution.Y[arc_1.origin.id] + travel_time_diff2 + minimum_headway - EPS
                            ):
                                condition_not_satisfied = True

                        if condition_not_satisfied:
                            return False, "Section track occupation violated for same-direction trains"

                    # Opposite directions
                    elif arc_1.origin.station == arc_2.destination.station:
                        condition_not_satisfied = False
                        if solution.Y[arc_2.origin.id] >= solution.Y[arc_1.destination.id]:
                            minimum_headway = (
                                EAG.minimum_headway_passenger_trains
                                if arc_1.origin.train.capacity > 0
                                else EAG.minimum_headway_freight_trains
                            )
                            if (
                                solution.X[arc_1.id] > 0.5
                                and solution.X[arc_2.id] > 0.5
                                and solution.Y[arc_2.origin.id]
                                < solution.Y[arc_1.destination.id] + minimum_headway - EPS
                            ):
                                condition_not_satisfied = True
                        else:
                            minimum_headway = (
                                EAG.minimum_headway_passenger_trains
                                if arc_2.origin.train.capacity > 0
                                else EAG.minimum_headway_freight_trains
                            )
                            if (
                                solution.X[arc_1.id] > 0.5
                                and solution.X[arc_2.id] > 0.5
                                and solution.Y[arc_1.origin.id]
                                < solution.Y[arc_2.destination.id] + minimum_headway - EPS
                            ):
                                condition_not_satisfied = True

                        if condition_not_satisfied:
                            return False, "Section track occupation violated for opposite-direction trains"

        # Disruption constraints: no train should use disrupted tracks before disruption end
        if EAG.disruption_scenario:
            d_end_time = EAG.disruption_scenario.end_time
            if isinstance(EAG.disruption_scenario.section_tracks, list):
                for t in EAG.disruption_scenario.section_tracks:
                    for arc in EAG.grouped_activities["train running"]:
                        if (
                            arc.section_track == t
                            and solution.X[arc.id] > 0.5
                            and solution.Y[arc.origin.id] < d_end_time - EPS
                        ):
                            # return False, "Train running on disrupted section track before end time"
                            msg = (
                                f"Train {arc.origin.train.id} running on a disrupted section track from , "
                                f"{arc.origin.station.id} to {arc.destination.station.id}, end time  "
                                f"{solution.Y[arc.origin.id]} - {solution.Y[arc.destination.id]} on track {t.id}"
                            )
                            return (False, msg)
            if isinstance(EAG.disruption_scenario.node_tracks, list):
                for t2 in EAG.disruption_scenario.node_tracks:
                    for arc in EAG.grouped_activities["train waiting"] + EAG.grouped_activities["pass-through"]:
                        if (
                            arc.origin.node_track == t2
                            and solution.X[arc.id] > 0.5
                            and solution.Y[arc.origin.id] < d_end_time - EPS
                        ):
                            return False, "Train using disrupted node track before end time"

        return True, None
