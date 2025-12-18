from __future__ import annotations

import logging
import math
import random
from datetime import datetime

import tomlkit as tk
from matplotlib.axes import Axes

from Robust_Railway.exceptions import OptimizationError

try:
    import matplotlib.pyplot as plt

    CAN_PLOT = True
except ModuleNotFoundError:
    CAN_PLOT = False

logger = logging.getLogger(__name__)

DATE_TIME_STRING = "__DATETIME__"


def replace_date_time(a_string: str) -> str:
    """Replace DATE_TIME_STRING in a_string with current date and time."""
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%B %d, %Y. %H:%M:%S")
    return a_string.replace(DATE_TIME_STRING, formatted_datetime)


class SetElement:
    """
    Element of the Pareto set (all objectives are minimized).

    Args:
        element_id [str]: Identifier.
        objectives [list[float]]: Objective values.
        X, Y, Z, PHI: Solution variables (dict).
        arc_usage: Arc usage data.
        num_cancelled_events [int]: Cancelled events.
        num_short_turning [int]: Short turnings.
        num_emergency_bus [int]: Emergency bus usages.
    """

    def __init__(
        self,
        element_id: str,
        objectives: list[float],
        X: dict,
        Y: dict,
        Z: dict,
        PHI: dict,
        arc_usage,
        num_cancelled_events: int,
        num_short_turning: int,
        num_emergency_bus: int,
    ) -> None:
        self.element_id = element_id
        self.objectives = objectives
        self.X = X
        self.Y = Y
        self.Z = Z
        self.PHI = PHI
        self.arc_usage = arc_usage
        self.num_cancelled_events = num_cancelled_events
        self.num_short_turning = num_short_turning
        self.num_emergency_bus = num_emergency_bus

        if any(obj is None for obj in objectives):
            raise OptimizationError(f"All objectives must be defined: {objectives}")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SetElement):
            if self.element_id == other.element_id:
                if self.objectives != other.objectives:
                    raise OptimizationError(
                        f"Two elements named {self.element_id} have different objective values: "
                        f"{self.objectives} and {other.objectives}"
                    )
                return True
            return False
        return False

    def __hash__(self) -> int:
        return hash((self.element_id, tuple(self.objectives)))

    def __str__(self) -> str:
        return f"{self.element_id} {self.objectives}"

    def __repr__(self) -> str:
        return self.element_id

    def dominates(self, other: SetElement) -> bool:
        """Check if self dominates other."""
        if len(self.objectives) != len(other.objectives):
            raise OptimizationError(f"Incompatible sizes: {len(self.objectives)} and {len(other.objectives)}")
        # Domination: all objectives <=, at least one <
        return all(my_f <= her_f for my_f, her_f in zip(self.objectives, other.objectives)) and any(
            my_f < her_f for my_f, her_f in zip(self.objectives, other.objectives)
        )

    def is_equivalent(self, other: SetElement, tol: float = 1e-4) -> bool:
        """Check if self is equivalent to other within tolerance."""

        def dicts_close(d1, d2):
            if set(d1.keys()) != set(d2.keys()):
                return False
            for k in d1:
                if abs(d1[k] - d2[k]) > tol:
                    return False
            return True

        return (
            dicts_close(self.X, other.X)
            and dicts_close(self.Y, other.Y)
            and dicts_close(self.Z, other.Z)
            and dicts_close(self.PHI, other.PHI)
        )


class Pareto:
    """
    Manages a Pareto set for minimization objectives.

    Args:
        filename [str|None]: Output file name.
    """

    def __init__(self, filename: str | None = None) -> None:
        self.size_init_pareto = 0
        self.size_init_considered = 0
        self.size_init_invalid = 0
        self.filename = filename
        self.comments = [
            f"File automatically created on {DATE_TIME_STRING}",
            "Information displayed: [Z_d, Z_p, Z_o, num_cancelled_events, num_short_turning, num_emergency_bus]",
        ]
        self.pareto: set[SetElement] = set()
        self.accepted: set[SetElement] = set()
        self.removed: set[SetElement] = set()
        self.considered: set[SetElement] = set()
        self.invalid: set[SetElement] = set()
        self.max_accepted_size = 10

    def __str__(self) -> str:
        return (
            f"Pareto: {self.pareto} Removed: {self.removed} " f"Considered: {self.considered} Invalid: {self.invalid}"
        )

    def dump(self) -> None:
        """
        Dump all sets to file.

        Raises:
            OptimizationError: If dumping fails.
        """
        if self.filename is None:
            logger.warning("No Pareto file has been provided")
            return
        doc = tk.document()
        final_comments = [replace_date_time(comment) for comment in self.comments]
        for comment in final_comments:
            doc.add(tk.comment(comment))

        def add_table(doc, name, elements):
            table = tk.table()
            for elem in elements:
                table[elem.element_id] = [float(obj) for obj in elem.objectives] + [
                    elem.num_cancelled_events,
                    elem.num_short_turning,
                    elem.num_emergency_bus,
                ]
            doc[name] = table

        add_table(doc, "Pareto", self.pareto)
        add_table(doc, "Accepted", self.accepted)
        add_table(doc, "Considered", self.considered)
        add_table(doc, "Removed", self.removed)
        add_table(doc, "Invalid", self.invalid)

        with open(self.filename, "w", encoding="utf-8") as f:
            print(tk.dumps(doc), file=f)

    def get_element_from_id(self, the_id: str) -> SetElement | None:
        """Get element by ID."""
        found = {elem for elem in self.considered if elem.element_id == the_id}
        if not found:
            return None
        if len(found) > 1:
            raise OptimizationError(f"There are {len(found)} elements with ID {the_id}")
        return next(iter(found))

    def length(self) -> int:
        """Get Pareto set length."""
        return len(self.pareto)

    def length_of_all_sets(self) -> tuple[int, int, int, int]:
        """Get lengths of all sets."""
        return (
            len(self.pareto),
            len(self.considered),
            len(self.removed),
            len(self.invalid),
        )

    def add_invalid(self, element: SetElement) -> bool:
        """Add invalid element."""
        if element in self.invalid:
            logger.debug(f"Invalid element {element.element_id} already inserted")
            return False
        self.invalid.add(element)
        return True

    def add_with_sa(self, element: SetElement, temperature: float) -> tuple[bool, bool, tuple[float, float] | None]:
        """Add solution using simulated annealing."""
        if element in self.considered:
            logger.debug(f"Element {element.element_id} already considered")
            return False, False, None
        self.considered.add(element)

        if not self.accepted:
            self.accepted.add(element)
            self.pareto.add(element)
            return True, True, (0.0, 0.0)

        dominated_by_element = set()
        dominates_element = set()
        for candidate in self.accepted:
            if element.dominates(candidate):
                dominated_by_element.add(candidate)
            if candidate.dominates(element):
                dominates_element.add(candidate)

        if not dominates_element:
            if not any(element.is_equivalent(candidate) for candidate in self.accepted):
                self.pareto.add(element)
                self.pareto -= dominated_by_element
                self.removed |= dominated_by_element
                self.accepted.add(element)
                self._prune_accepted_pool()
                # Compute proportional improvement
                improvements = []
                for obj_index in range(len(element.objectives)):
                    dominated_values = [k.objectives[obj_index] for k in dominated_by_element]
                    if dominated_values:
                        min_dominated = min(dominated_values)
                        improvement = (
                            (min_dominated - element.objectives[obj_index]) / abs(min_dominated)
                            if min_dominated != 0
                            else 0.0
                        )
                    else:
                        improvement = 0.0
                    improvements.append(improvement)
                return True, True, (improvements[0], improvements[1])
            else:
                return False, False, None

        # Element is dominated → accept with probability
        acceptance_prob = math.exp(-len(dominates_element) / temperature)
        if random.random() < acceptance_prob:
            if not any(element.is_equivalent(candidate) for candidate in self.accepted):
                self.accepted.add(element)
                self._prune_accepted_pool()
                return True, False, None
            else:
                return False, False, None

        return False, False, None

    def _prune_accepted_pool(self) -> None:
        """
        Prune the accepted pool to max_accepted_size.
        Prefer to remove dominated solutions, then arbitrary ones if needed.
        """
        if len(self.accepted) <= self.max_accepted_size:
            return

        accepted_list = list(self.accepted)
        dominated = set()
        for i, elem in enumerate(accepted_list):
            for j, other in enumerate(accepted_list):
                if i != j and other.dominates(elem):
                    dominated.add(elem)
                    break

        excess = len(self.accepted) - self.max_accepted_size
        dominated_list = list(dominated)
        random.shuffle(dominated_list)
        dominated_to_remove = dominated_list[:excess]
        self.accepted -= set(dominated_to_remove)

        still_excess = len(self.accepted) - self.max_accepted_size
        if still_excess > 0:
            for _ in range(still_excess):
                self.accepted.pop()

    def statistics(self) -> tuple[str, str, str]:
        """Get Pareto statistics."""
        if not self.pareto:
            return "", "", ""
        return (
            f"Pareto: {len(self.pareto)} ",
            f"Considered: {len(self.considered)} ",
            f"Removed: {len(self.removed)}",
        )

    def plot(
        self,
        objective_x: int = 0,
        objective_y: int = 1,
        label_x: str | None = None,
        label_y: str | None = None,
        margin_x: int = 5,
        margin_y: int = 5,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot Pareto set for two objectives."""
        if not CAN_PLOT:
            raise OptimizationError("Install matplotlib.")
        ax = ax or plt.gca()

        if self.length() == 0:
            raise OptimizationError("Cannot plot an empty Pareto set")

        first_elem = next(iter(self.pareto))
        number_of_objectives = len(first_elem.objectives)
        if number_of_objectives < 2:
            raise OptimizationError("At least two objectives functions are required for the plot of the Pareto set.")

        if objective_x >= number_of_objectives:
            raise OptimizationError(
                f"Index of objective x is {objective_x}, but there are only {number_of_objectives}. "
                f"Give a number between 0 and {number_of_objectives-1}."
            )

        def get_xy(elements):
            return [elem.objectives[objective_x] for elem in elements], [
                elem.objectives[objective_y] for elem in elements
            ]

        par_x, par_y = get_xy(self.pareto)
        con_x, con_y = get_xy(self.considered)
        rem_x, rem_y = get_xy(self.removed)
        inv_x, inv_y = get_xy(self.invalid)

        ax.axis([
            min(par_x) - margin_x,
            max(par_x) + margin_x,
            min(par_y) - margin_y,
            max(par_y) + margin_y,
        ])
        ax.plot(par_x, par_y, "o", label="Pareto")
        ax.plot(rem_x, rem_y, "x", label="Removed")
        ax.plot(con_x, con_y, ",", label="Considered")
        if self.invalid:
            ax.plot(inv_x, inv_y, "*", label="Invalid")
        ax.set_xlabel(label_x or f"Objective {objective_x}")
        ax.set_ylabel(label_y or f"Objective {objective_y}")
        ax.legend()
        return ax
