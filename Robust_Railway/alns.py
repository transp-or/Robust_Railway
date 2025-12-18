import csv
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from Robust_Railway.exceptions import OptimizationError
from Robust_Railway.pareto import Pareto, SetElement
from Robust_Railway.rescheduling import Rescheduling

# Constants
DEFAULT_LOG_FILE = "results_event_activity/alns_log.log"
DEFAULT_RESULTS_DIR = "results_event_activity/Viriato_network/ALNS/"
SA_INITIAL_TEMPERATURE = 100
SA_ALPHA = 0.99
SA_MIN_TEMPERATURE = 1e-4
SELECT_BEST_PROB = 0.5

logger = logging.getLogger(__name__)

os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(DEFAULT_RESULTS_DIR, "alns_log.log"))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ParetoClass(Pareto):
    """Class managing the solutions"""

    def __init__(self, max_neighborhood: int, pareto_file: Optional[str] = None) -> None:
        """
        :param max_neighborhood: the maximum size of the neighborhood
            that must be considered
        :type max_neighborhood: int

        :param pareto_file: name of a  file containing sets from previous runs
        :type pareto_file: str

        """
        super().__init__(pareto_file)

        # Maximum neighborhood size for exploration
        self.max_neighborhood = max_neighborhood
        self.neighborhood_size: defaultdict[int, int] = defaultdict(int)

        # Simulated annealing parameters
        self.temperature: float = SA_INITIAL_TEMPERATURE
        self.alpha = SA_ALPHA
        self.min_temperature = SA_MIN_TEMPERATURE

    def add(self, element: SetElement) -> Tuple[bool, bool, Optional[Tuple[float, float]]]:
        """
        Add an element
        :param element: element to be considered for inclusion in the Pareto set.
        :type element: SetElement

        :return: A tuple indicating whether the solution was accepted, added to the Pareto
        set, and any objective improvements.
        :rtype: tuple[bool, bool, tuple[float, float] | None]
        """
        accepted, added_to_pareto, obj_improvements = super().add_with_sa(element, self.temperature)

        # Update temperature after every iteration
        self.temperature = max(self.temperature * self.alpha, self.min_temperature)

        return accepted, added_to_pareto, obj_improvements

    def select(self) -> Optional[SetElement]:
        """
        Select a candidate to be modified during the next iteration.
        Occasionally favor solutions that are best in one objective.

        :return: A candidate element from the Pareto or accepted set.
        """
        select_from = list(set(list(self.accepted) + list(self.pareto)))
        if not select_from:
            return None
        chosen = random.choice(select_from)
        return chosen

    def get_best_values(self) -> List[float]:
        """
        Get the minimum value for each objective across the current Pareto set.

        :return: A list of the minimum objective values.
        :rtype: list
        """

        # Returns minimum of each objective across current Pareto set
        return list(map(min, zip(*[s.objectives for s in self.considered])))


def alns(
    problem: Rescheduling,
    first_solutions: List[SetElement],
    pareto: ParetoClass,
    time_limit_seconds: float,
    maximum_attempts: int = 100000,
) -> ParetoClass:
    """
    Multi-objective Adaptive Large Neighborhood Search(ALNS) algorithm

    Args:
        problem (Rescheduling): Instance of the problem to optimize.
        first_solutions (List[SetElement]): Initial solutions to validate and add to Pareto.
        pareto (ParetoClass): Pareto set to store non-dominated solutions.
        time_limit_seconds (float): Time limit in seconds for the algorithm.
        maximum_attempts (int, optional): Maximum number of iterations. Default: 1_000_000.

    Returns:
        ParetoClass: Updated Pareto set.
    """
    if first_solutions is not None:
        for solution in first_solutions:
            valid, why = problem.is_valid(solution)
            if valid:
                pareto.add(solution)
                logger.info(solution)
            else:
                logger.debug("invalid solution found", why)
                pareto.add_invalid(solution)
                logger.warning(solution)
                logger.warning(f"Default specification is invalid: {why}")

    nb_objectives = len(first_solutions[0].objectives)

    if pareto.length() == 0:
        raise OptimizationError("Cannot start the algorithm with an empty Pareto set.")

    logger.debug(f"Initial pareto: {pareto.length()}")

    start_time = time.time()
    convergence_data: Dict[str, List[float]] = {
        "iteration": [],
        "time": [],
    }
    for i in range(nb_objectives):
        convergence_data[f"objective_{i}"] = []

    score_reseted = False

    for attempt in range(maximum_attempts):
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit_seconds:
            logger.info(f"Time limit of {time_limit_seconds} seconds reached. Stopping VNS.")
            break

        if elapsed_time > time_limit_seconds / 2 and not score_reseted:
            logger.info("Update the scores")
            score_reseted = True
            problem.reset_scores()

        solution_to_improve = pareto.select()
        # neighborhood_size = random.randint(1, pareto.max_neighborhood)
        neighborhood_size = int(random.triangular(1, pareto.max_neighborhood, pareto.max_neighborhood))

        if solution_to_improve is None:
            continue

        logger.debug(f"Attempt {attempt}/{maximum_attempts}")
        logger.debug(f"----> Current solution: {solution_to_improve}")
        logger.debug(f"----> Neighbor of size {neighborhood_size}")

        a_neighbor, number_of_changes = problem.generate_neighbor(solution_to_improve, neighborhood_size)
        logger.debug(f"----> Neighbor: {a_neighbor} Nbr of changes {number_of_changes}")

        if number_of_changes == 0:
            problem.last_neighbor_rejected()
            logger.info(f"No neighbor found with neighborhood size {neighborhood_size}.")
            continue

        if a_neighbor in pareto.considered:
            problem.last_neighbor_rejected()
            logger.debug(f"*** Neighbor of size {neighborhood_size} already considered ***")
            continue

        valid, why = problem.is_valid(a_neighbor)
        if valid:
            accepted, added_to_pareto, obj_improvements = pareto.add(a_neighbor)
            if accepted:
                logger.debug(f"*** New accepted solution: {a_neighbor}")
                pareto.dump()
                if added_to_pareto:
                    logger.debug(f"*** New pareto solution: {a_neighbor}")
                    problem.last_neighbor_added_to_pareto(obj_improvements)
                else:
                    problem.last_neighbor_accepted()
            else:
                logger.debug(f"*** Neighbor of size {neighborhood_size}: dominated ***")
                problem.last_neighbor_rejected()
        else:
            pareto.add_invalid(a_neighbor)
            logger.debug(f"*** Neighbor of size {neighborhood_size} invalid: {why} ***")
            problem.last_neighbor_rejected()

        # Track convergence at each iteration
        current_time = time.time() - start_time
        convergence_data["iteration"].append(attempt)
        convergence_data["time"].append(current_time)
        best_values = pareto.get_best_values()
        for i, val in enumerate(best_values):
            convergence_data[f"objective_{i}"].append(val)

    pareto.dump()

    # Save CSV
    job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
    filename = os.path.join(DEFAULT_RESULTS_DIR, f"convergence_data_{job_id}.csv")
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        headers = list(convergence_data.keys())
        writer.writerow(headers)
        num_rows = len(convergence_data["iteration"])
        for i in range(num_rows):
            row = [convergence_data[key][i] for key in headers]
            writer.writerow(row)

    # Plot graphs
    obj_name = {0: "z_d", 1: "z_p", 2: "z_o"}

    def plot_convergence(data: Dict[str, List[float]], nb_obj: int, job_id: str):
        for i in range(nb_obj):
            plt.figure()
            plt.plot(data["iteration"], data[f"objective_{i}"], label="By iteration")
            plt.xlabel("Iteration")
            plt.ylabel(f"Objective {obj_name.get(i, str(i))} value")
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    DEFAULT_RESULTS_DIR, f"convergence_objective_{obj_name.get(i, str(i))}_iterations_{job_id}.png"
                )
            )

            plt.figure()
            plt.plot(data["time"], data[f"objective_{i}"], label="By time")
            plt.xlabel("Time (s)")
            plt.ylabel(f"Objective {obj_name.get(i, str(i))} value")
            plt.grid(True)
            plt.savefig(
                os.path.join(DEFAULT_RESULTS_DIR, f"convergence_objective_{obj_name.get(i, str(i))}_time_{job_id}.png")
            )

    plot_convergence(convergence_data, nb_objectives, job_id)

    return pareto
