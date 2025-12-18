import abc
import csv
import logging
import os
from typing import Callable, Optional, Tuple

import numpy as np

from Robust_Railway.exceptions import OptimizationError
from Robust_Railway.pareto import SetElement

logger = logging.getLogger(__name__)

OperatorOutput = Tuple[SetElement, int]
Operator = Callable[[SetElement, int], OperatorOutput]


class OperatorsManagement:
    """
    Manages selection and performance analysis of operators.

    Args:
        operators [dict[str, tuple[Operator]]]: Operator functions.
        init_file_name [str]: Initialization file name.
    """

    def __init__(self, operators: dict[str, tuple[Operator, Operator, Operator]], init_file_name: str):
        self.operators: dict[str, tuple[Operator, Operator, Operator]] = operators
        # self.scores: dict[str, float] = {k: 0.01 for k in operators}
        base_dir = os.path.dirname(__file__)
        filename = os.path.join(base_dir, "operators_init_scores_partial.csv")

        with open(filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.init_scores = {row["operator"]: float(row["score"]) for row in reader}
        self.scores: dict[str, float] = {}
        for k in operators:
            if k in self.scores:
                self.scores[k] = self.init_scores[k]
            else:
                self.scores[k] = 0.01

        self.names: list[str] = list(operators.keys())
        self.available: dict[str, bool] = {k: True for k in operators}
        self.last_operator_name: Optional[str] = None
        self.scale: float = 1.0 / np.sqrt(len(self.scores))

    def increase_score_last_operator(self, obj_improvements, pareto: bool) -> None:
        """Increase score of last operator."""
        if self.last_operator_name not in self.scores:
            raise OptimizationError(f"Unknown operator: {self.last_operator_name}")
        # if pareto and obj_improvements is not None:
        #    self.scores[self.last_operator_name] += max(sum(obj_improvements), 0.01)
        # else:
        #    self.scores[self.last_operator_name] += 0.01

    def decrease_score_last_operator(self) -> None:
        """Decrease score of last operator."""
        if self.last_operator_name not in self.scores:
            raise OptimizationError(f"Unknown operator: {self.last_operator_name}")
        # No decrease implemented (kept for compatibility)
        pass

    def reset_scores(self) -> None:
        """Reset all operator scores."""
        for k in self.operators:
            if k in self.scores:
                self.scores[k] = self.init_scores[k]
            else:
                self.scores[k] = 0.01

    def probability_from_scores(self) -> Optional[np.ndarray]:
        """Calculate selection probabilities from scores."""
        scores = list(self.scores.values())
        if not scores:
            return None
        total_score = sum(scores)
        prob = np.array([float(s) / float(total_score) for s in scores])
        return prob

    def select_operator(self) -> tuple[Operator, Operator, Operator]:
        """
        Select an operator based on scores.

        Returns:
            tuple[Operator, Operator, Operator]: Selected operator.
        """
        prob = self.probability_from_scores()
        self.last_operator_name = np.random.choice(self.names, 1, p=prob)[0]
        assert self.last_operator_name is not None
        return self.operators[self.last_operator_name]


class Neighborhood(metaclass=abc.ABCMeta):
    """Abstract class for neighborhood structure."""

    def __init__(self, operators: dict[str, tuple[Operator, Operator, Operator]], init_file_name: str):
        self.operators_management = OperatorsManagement(operators, init_file_name)

    @abc.abstractmethod
    def last_neighbor_rejected(self) -> None:
        """Notify that last neighbor was rejected."""
        self.operators_management.decrease_score_last_operator()

    def last_neighbor_accepted(self) -> None:
        """Notify that last neighbor was accepted."""
        self.operators_management.increase_score_last_operator(obj_improvements=None, pareto=False)

    def reset_scores(self) -> None:
        """Reset operator scores."""
        self.operators_management.reset_scores()

    def last_neighbor_added_to_pareto(self, obj_improvements) -> None:
        """Notify that last neighbor was added to Pareto set."""
        self.operators_management.increase_score_last_operator(obj_improvements=obj_improvements, pareto=True)
