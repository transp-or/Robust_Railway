
[![Project Status](https://img.shields.io/badge/status-under%20development-yellow)](https://github.com/transp-or/Robust_Railway)


# Robust Railway Timetable Rescheduling Tool

This repository provides a framework for **railway timetable rescheduling** under disruptions.
It integrates exact and heuristic optimization methods to generate feasible, high-quality rescheduling plans, aiming to minimize passenger inconvenience, deviations from the reference timetable, and operational costs.

---

## 🚄 Overview

Railway operations are highly sensitive to disruptions (e.g., infrastructure failures).
This project provides algorithms and tools to **repair** an existing timetable after such disruptions using a **multi-objective optimization** framework.

The project implements:
- A **Gurobi-based exact approach** for small instances.
- A **heuristic** approach (Adaptive Large Neighborhood Search).
- Integration with the industrial tool **Viriato** for data gathering, visualization, and feasibility validation.

---

## 🧩 Key Components

| Module | Description |
|--------|--------------|
| `main.py` | Entry point for running experiments. |
| `rescheduling.py` | Representation of a disposition timetable solution and feasible updates. |
| `alns.py` | Implementation of the Adaptive Large Neighborhood Search heuristic. |
| `pareto.py` | Construction and management of Pareto-optimal solution sets. |
| `neighborhood.py` | Defines neighborhood structures. |
| `rescheduling_gurobi_model.py` | MILP formulation for exact rescheduling using Gurobi. |
| `run_rescheduling_gurobi.py` | Script to run timetable rescheduling using the exact Gurobi-based approach. |
| `run_rescheduling_heuristic.py` | Script to run timetable rescheduling using the heuristic approach. |
| `event_activity_graph_multitracks.py` | Event–activity graph for multi-track railway networks. |
| `passenger_assignment.py` | Passenger flow and demand assignment module. |
| `operators/` | Set of destroy and repair operators for heuristic search. |
| `orderings/` | Rules for ordering trains during rescheduling. |

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-org>/Robust_Railway.git
   cd Robust_Railway
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```
   or, if you use Poetry:
   ```bash
   poetry install
   ```

> 💡 A valid **Gurobi** license is required to run optimization models that rely on it.

---

## ▶️ Usage

You can launch the main program directly from the command line:

```bash
python ../Robust_Railway/main.py [OPTIONS]
```

### Available command-line arguments

| Argument | Description |
|-----------|--------------|
| `--api_url` | (Required) API URL to connect with **Viriato**. |
| `--heuristic` | Enable solving the problem **heuristically** *(boolean flag)*. |
| `--initial_timetable_and_graph_ID` | Job ID of the initial timetable and event–activity graph to load. |
| `--write_back_viriato` | Enable writing back the results to **Viriato**. |
| `--save_timetable` | Enable saving the solution as a pickle file *(boolean flag)*. |
| `--time_limit` | Time limit for the exact or heuristic method (in seconds). |

### Example commands

**Run the heuristic mode:**
```bash
python ../Robust_Railway/main.py --api_url http://localhost:8080 --heuristic --time_limit 600
```

**Run with Viriato integration:**
```bash
python ../Robust_Railway/main.py --api_url http://localhost:8080 --write_back_viriato
```

---

## 🧪 Testing

Run the full test suite:
```bash
pytest
```

Pre-commit hooks, coverage, and linting (via `ruff`) are already configured in the repository.

---

## 🧾 Citation

If you use this tool in your research, please cite it via the `CITATION.cff` file.

---

## 👥 Contributing

Contributions are welcome!
Please review [CONTRIBUTING.md](CONTRIBUTING.md) and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📜 License

This project is released under the [Apache License](LICENSE).
---

## 🧠 Maintainers

Developed as part of a research project on **rail disruption management and timetable optimization** at EPFL, in close collaboration with our **industry partners** SBB and SMA.
For collaboration or integration inquiries, please contact the maintainers.
