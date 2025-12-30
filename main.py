import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from py_client.algorithm_interface import algorithm_interface_factory

from Robust_Railway.rescheduling_gurobi_model import construct_model
from Robust_Railway.run_rescheduling_gurobi import run_rescheduling_gurobi
from Robust_Railway.run_rescheduling_heuristic import run_rescheduling_heuristic
from Robust_Railway.Viriato_plugin.build_event_activity_graph_viriato import build_graph_from_viriato


# --- Path management ---
def get_instance_paths(base_path):
    base_path = Path(base_path)

    paths = {
        "junctions": base_path / "junctions.csv",
        "links": base_path / "links.csv",
        "stations": base_path / "stations.csv",
        "bus": base_path / "bus.csv",
        "ODs": base_path / "ODs.csv",
        "manual_corrections": base_path / "manual_link_corrections.csv",
        "trains_ignore": base_path / "trains_to_ignore.csv",
        "intersections": base_path / "intersections.csv",
    }

    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) in {base_path}:\n" + "\n".join(f"- {name}: {paths[name]}" for name in missing)
        )

    # Convert back to strings if the rest of your code expects strings
    return {k: str(v) for k, v in paths.items()}


# --- Utility functions ---
def load_csv_column(filepath, column):
    try:
        df = pd.read_csv(filepath)
        return df[column].tolist()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def load_manual_link_corrections(filepath):
    try:
        df = pd.read_csv(filepath)
        corrections_dict = {}
        for _, row in df.iterrows():
            key = (row["from"], row["to"])
            corrections_dict.setdefault(key, []).append(row["correction"])
        return corrections_dict
    except Exception as e:
        print(f"Error loading corrections: {e}")
        return {}


# --- Main function ---
def main():
    parser = argparse.ArgumentParser(description="Seeks an optimal disposition timetable given a disruption scenario")

    # Optional arguments
    parser.add_argument("--api_url", type=str, required=False, help="API URL to connect with Viriato")
    parser.add_argument(
        "--skip_pass_graph",
        action="store_true",
        required=False,
        help="Enable skipping building the passengers graph (boolean flag)",
    )
    parser.add_argument(
        "--save_timetable",
        action="store_true",
        required=False,
        help="Enable saving the solution in a pickle file (boolean flag)",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        required=False,
        help="Enable solving the problem heuristically (boolean flag)",
    )
    parser.add_argument(
        "--initial_timetable_and_graph_ID",
        type=int,
        required=False,
        help="JobID of the initial timetable and graph to load",
    )
    parser.add_argument(
        "--write_back_viriato", action="store_true", required=False, help="Enable writing back the results in Viriato"
    )

    parser.add_argument(
        "--time_limit",
        type=int,
        default=300,
        required=False,
        help="Time limit in seconds for the exact or heuristic method (default 300)",
    )

    parser.add_argument(
        "--instance_name",
        required=True,
        help="Name of the instance to solve (folder name under Robust_Railway_test/Instances/)",
    )

    args = parser.parse_args()

    # Debug: Print received arguments
    print(f"API URL: {args.api_url}")
    print(f"Skip building passengers graph: {args.skip_pass_graph}")
    print(f"Heuristic: {args.heuristic}")

    api_url = args.api_url
    skip_pass_graph = args.skip_pass_graph
    save_timetable = args.save_timetable
    heuristic = args.heuristic
    initial_timetable_and_graph_ID = args.initial_timetable_and_graph_ID
    write_back_viriato = args.write_back_viriato
    time_limit = args.time_limit
    instance_name = args.instance_name

    INSTANCE_PATHS = get_instance_paths("Robust_Railway_test/instances/" + instance_name + "/")
    PARAMS_FILE = Path(__file__).parent / "Robust_Railway" / "params.in"

    # --- Data loading ---
    MANUAL_LINK_CORRECTIONS = load_manual_link_corrections(INSTANCE_PATHS["manual_corrections"])
    TRAINS_TO_IGNORE = load_csv_column(INSTANCE_PATHS["trains_ignore"], "train_id")
    INTERSECTIONS = load_csv_column(INSTANCE_PATHS["intersections"], "station_code")

    # General problem parameters
    with open(PARAMS_FILE) as file:
        params = json.load(file)

    if "base_date" in params:
        params["base_date"] = datetime.strptime(params["base_date"], "%Y-%m-%d")

    print("Loaded parameters:")
    print(json.dumps({k: str(v) for k, v in params.items()}, indent=4))

    """

    Construct event-activity network

    """

    df = pd.read_csv(INSTANCE_PATHS["ODs"])

    # Create a tuple identifier for each passenger group
    df["group"] = list(zip(df["from_code"], df["to_code"]))

    # Normalize probabilities (in case they don’t sum to 1 exactly)
    probabilities = df["probability_having_group"].values
    probabilities = probabilities / probabilities.sum()

    def build_graph(solve_init_timetable, initial_timetable_and_graph_ID):

        # If solve_init_timetable = True, do not apply disruption scenario yet

        event_activity_graph = build_graph_from_viriato(
            api_url,
            INSTANCE_PATHS["junctions"],
            INSTANCE_PATHS["links"],
            INSTANCE_PATHS["stations"],
            INSTANCE_PATHS["bus"],
            df["group"],
            probabilities,
            initial_timetable_and_graph_ID,
            solve_init_timetable,
            MANUAL_LINK_CORRECTIONS,
            TRAINS_TO_IGNORE,
            INTERSECTIONS,
            **params,
        )

        tot_individuals = 0
        for group in event_activity_graph.passengers_groups:
            tot_individuals += group.num_passengers

        print("number of individuals", tot_individuals)

        print("Network built")
        print("Number of nodes: ", len(event_activity_graph.events))
        print("Number of edges: ", len(event_activity_graph.activities))

        return event_activity_graph

    with algorithm_interface_factory.create(api_url) as api:
        time_window = api.get_time_window_algorithm_parameter("timeWindowParameterMandatory")
        disruption_scenario = bool(api.get_section_track_closures(time_window)) or bool(
            api.get_node_track_closures(time_window)
        )

    if heuristic and not disruption_scenario:
        raise ValueError("Cannot run the ALNS heuristic without a disruption scenario")

    if initial_timetable_and_graph_ID is None and disruption_scenario:
        # Need to presolve to get a feasible initial timetable
        event_activity_graph = build_graph(True, None)
        (
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
        ) = construct_model(
            event_activity_graph,
            skip_pass_graph,
        )
        run_rescheduling_gurobi(
            api_url,
            event_activity_graph,
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
            True,  # Save timetable
            skip_pass_graph,
            False,
            None,  # time limit
            0.10,  # MIP gap
        )
        initial_timetable_and_graph_ID = os.environ.get(
            "SLURM_JOB_ID", "nojobid"
        )  # Get the SLURM job ID from the environment

    event_activity_graph = build_graph(False, initial_timetable_and_graph_ID)

    if not heuristic:
        (
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
        ) = construct_model(
            event_activity_graph,
            skip_pass_graph,
        )

        print("MILP constructed")

        run_rescheduling_gurobi(
            api_url,
            event_activity_graph,
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
            time_limit,
            0.10,
        )

    else:

        run_rescheduling_heuristic(
            event_activity_graph, initial_timetable_and_graph_ID, api_url, write_back_viriato, time_limit
        )


if __name__ == "__main__":
    main()
