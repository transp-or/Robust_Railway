import logging
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

from Robust_Railway.event_activity_graph_multitracks import Activity, Bus, EARailwayNetwork, Train

logger = logging.getLogger(__name__)


def run_shortest_path_over_ODs(ODs: np.ndarray, arcs_array: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute shortest paths for multiple OD pairs using Dijkstra’s algorithm.

    Args:
        ODs [np.ndarray]: OD pairs (origin, destination)
        arcs_array [np.ndarray]: Arcs as [origin, destination, cost, train, capacity, activity type id]
        n_nodes [int]: Number of nodes in the graph

    Returns:
        costs_per_group [np.ndarray]: Shortest path costs per OD pair
        path_per_group [np.ndarray]: Boolean array, arcs used per OD pair
    """
    i, j, c = arcs_array[:, :3].T
    i, j = i.astype(np.int32), j.astype(np.int32)
    c = np.where(c < 0, 0.0, c)
    A = csr_array((c, (j, i)), shape=(n_nodes, n_nodes))  # Run backward for efficiency

    dests = np.unique(ODs[:, 1]).astype(np.int32)
    dests_index = (ODs[:, 1].reshape(-1, 1) == dests.reshape(1, -1)).argmax(1)
    labels, successors = dijkstra(csgraph=A, directed=True, return_predecessors=True, indices=dests)

    costs_per_group = labels[dests_index, ODs[:, 0]]
    n_arcs = arcs_array.shape[0]
    n_groups = len(ODs)
    path_per_group = np.full((n_groups, n_arcs), False)

    for k, o in enumerate(ODs[:, 0]):
        arcs_used = []
        j = successors[dests_index[k], o]
        i = o
        while j >= 0:
            arcs_used.append([i, j])
            i = j
            j = successors[dests_index[k], i]
        arcs_used = np.array(arcs_used)
        if len(arcs_used) == 0:
            arcs_used = np.empty((0, 2))
        arcs_used_mask = (arcs_array[:, None, :2] == arcs_used).all(2).any(1)
        path_per_group[k] = arcs_used_mask

    return costs_per_group, path_per_group


def run_shortest_path_w_banned_arcs(
    ODs: np.ndarray,
    banned_links_per_group: np.ndarray,
    arcs_array: np.ndarray,
    n_nodes: int,
    costs_per_group: np.ndarray,
    path_per_group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute shortest paths for OD pairs considering banned arcs per group.

    Args:
        ODs [np.ndarray]: OD pairs
        banned_links_per_group [np.ndarray]: Boolean array, banned arcs per group
        arcs_array [np.ndarray]: Arcs as [origin, destination, cost, train, capacity, activity type id]
        n_nodes [int]: Number of nodes
        costs_per_group [np.ndarray]: Shortest path costs per OD pair
        path_per_group [np.ndarray]: Boolean array, arcs used per OD pair

    Returns:
        costs_per_group [np.ndarray]: Updated costs per group
        path_per_group [np.ndarray]: Updated paths per group
    """
    # Group the ODs by banned links
    unique_bans, inverse = np.unique(banned_links_per_group[ODs[:, 2]], return_inverse=True, axis=0)
    for k in range(inverse.max() + 1):
        banned_links = unique_bans[k]
        these_ODs = ODs[inverse == k]
        groups = these_ODs[:, 2]
        arcs_array_mod = arcs_array[~banned_links]
        costs, paths = run_shortest_path_over_ODs(these_ODs, arcs_array_mod, n_nodes)
        costs_per_group[groups] = costs
        path_per_group[np.ix_(groups, banned_links)] = False
        path_per_group[np.ix_(groups, ~banned_links)] = paths
    return costs_per_group, path_per_group


def get_groups_idx_to_ban_presence_in_train(
    path_per_group: np.ndarray,
    ODs: np.ndarray,
    arcs_array: np.ndarray,
    train_capacity: float,
    arc_idx: int,
) -> np.ndarray:
    """
    Find groups to ban from an arc due to train capacity.

    Args:
        path_per_group [np.ndarray]: Boolean array, arcs used per OD pair
        ODs [np.ndarray]: OD pairs
        arcs_array [np.ndarray]: Arcs as [origin, destination, cost, train, capacity, activity type id]
        train_capacity [float]: Train capacity
        arc_idx [int]: Arc index

    Returns:
        banned_groups_idx [np.ndarray]: Indices of banned groups
    """
    groups_idx_concerned = np.nonzero(path_per_group[:, arc_idx])[0]
    groups_info = ODs[groups_idx_concerned]
    i = arcs_array[arc_idx, 1]
    train = arcs_array[arc_idx, 3]
    arcs_idx_this_train = []
    while (mask := (arcs_array[:, [1, 3]] == [i, train]).all(1)).any():
        i = arcs_array[mask, 0][0]
        arcs_idx_this_train.append(mask.argmax())
    arcs_idx_this_train.reverse()
    time_in_train = path_per_group[np.ix_(groups_idx_concerned, arcs_idx_this_train)].cumsum(axis=1)[:, -1]
    order = np.lexsort((groups_info[:, 4], -time_in_train))
    groups_info = groups_info[order]
    banned_groups_idx = groups_idx_concerned[order][groups_info[:, 3].cumsum() > train_capacity]
    return banned_groups_idx


def passenger_assignment(
    EAG: EARailwayNetwork,
    x: dict,
    y: dict,
    phi: dict,
    stopping_everywhere: bool = False,
    verbose: int = 0,
    use_presence_in_train: bool = False,
    level_of_detail: int = 1,
) -> float | tuple[list, dict, dict, float]:
    """
    Assign passenger groups to shortest paths, enforcing capacity constraints.

    Args:
        EAG [EARailwayNetwork]: Event-Activity Graph
        x [dict]: Activity decision variables
        y [dict]: Event time variables
        phi [dict]: Emergency bus variables
        stopping_everywhere [bool]: Allow stops everywhere
        verbose [int]: Verbosity level
        use_presence_in_train [bool]: Use train presence for banning
        level_of_detail [int]: Output detail level

    Returns:
        float: Total cost
        OR
        tuple: (costs_per_group [list], activity occupation [dict], train occupation [dict], total cost [float])
    """
    start_time = time.time()
    activity_type_dict = {
        "access": 0,
        "egress": 1,
        "transferring": 2,
        "passenger running": 3,
        "dwelling": 4,
        "penalty": 5,
        "emergency bus": 6,
    }

    # Utility functions
    def is_activated(activity: Activity) -> bool:
        if activity.activity_type == "emergency bus":
            return phi[activity.id] > 0.5
        return any(x[arc.id] > 0.5 for arc in EAG.agg_to_disagg_activities[activity])

    def is_activated_and_stopping(
        activity: Activity, stopping_everywhere: bool, do_origin: bool = True, do_destination: bool = True
    ) -> bool:
        answer = True
        can_end_at = (
            ["train waiting", "pass-through", "emergency bus"]
            if stopping_everywhere
            else ["train waiting", "emergency bus"]
        )
        can_start_at = (
            ["train waiting", "pass-through", "emergency bus"]
            if stopping_everywhere
            else ["train waiting", "emergency bus"]
        )
        if do_origin:
            if isinstance(activity.origin.train, Train):
                sum_ = sum(
                    x[arc.id]
                    for event in EAG.agg_to_disagg_events[activity.origin]
                    for arc in EAG.A_plus[event]
                    if arc.activity_type in can_end_at
                )
                if sum_ < 0.5:
                    answer = False
            elif isinstance(activity.origin.train, Bus):
                bus_activity = EAG.A_plus_agg[activity.origin]
                if len(bus_activity) > 1:
                    raise ValueError("Incorrect specification of A_plus_agg")
                if phi[bus_activity[0].id] < 0.5:
                    answer = False
            else:
                raise ValueError("Error in definition of event")
        if do_destination:
            if isinstance(activity.destination.train, Train):
                sum_ = sum(
                    x[arc.id]
                    for event in EAG.agg_to_disagg_events[activity.destination]
                    for arc in EAG.A_minus[event]
                    if arc.activity_type in can_start_at
                )
                if sum_ < 0.5:
                    answer = False
            elif isinstance(activity.destination.train, Bus):
                bus_activity = EAG.A_minus_agg[activity.destination]
                if len(bus_activity) > 1:
                    raise ValueError("Incorrect specification of A_minus_agg")
                if phi[bus_activity[0].id] < 0.5:
                    answer = False
            else:
                raise ValueError("Error in definition of event")
        return answer

    # Convert Activities to graph for shortest path
    events_ids: list[int] = []
    arcs_idx_to_obj = {}

    def node_id(event_id):
        if event_id not in events_ids:
            events_ids.append(event_id)
            return len(events_ids) - 1
        return events_ids.index(event_id)

    arcs_array = []
    activities_objs = []
    train_start = []
    end_nodes: dict = {}
    idx = 0
    for activity in EAG.categorized_activities["group"]:
        include_this_activity = False
        dest = None
        e1, e2 = None, None
        if activity.activity_type == "access":
            potential_e1 = EAG.agg_to_disagg_events[activity.destination]
            for e_ in potential_e1:
                for a_ in EAG.A_minus[e_]:
                    if x[a_.id] > 0.5:
                        e1 = e_
                        e2 = activity.origin
            if not e1:
                if verbose > 0:
                    logger.debug("Disaggregated access event missing")
                continue
        elif activity.activity_type == "egress":
            potential_e2 = EAG.agg_to_disagg_events[activity.origin]
            for e_ in potential_e2:
                for a_ in EAG.A_plus[e_]:
                    if x[a_.id] > 0.5:
                        e2 = e_
                        e1 = activity.destination
            if not e2:
                if verbose > 0:
                    logger.debug("Disaggregated egress event missing")
                continue
        elif activity.activity_type == "penalty":
            e2 = activity.origin
            e1 = activity.destination
        elif activity.activity_type == "emergency bus":
            e2 = activity.origin
            e1 = activity.destination
        elif activity.activity_type == "transferring" and (
            activity.origin.node_type == "emergency" or activity.destination.node_type == "emergency"
        ):
            if activity.origin.node_type == "emergency":
                emergency_bus_activity = EAG.A_plus_agg[activity.origin]
                if len(emergency_bus_activity) != 1:
                    raise ValueError("Number of emergency bus activities from this event should be equal to 1")
                if phi[emergency_bus_activity[0].id] > 0.5:
                    e2 = activity.origin
                    potential_e1 = EAG.agg_to_disagg_events[activity.destination]
                    for e_ in potential_e1:
                        for a_ in EAG.A_minus[e_]:
                            if x[a_.id] > 0.5:
                                e1 = e_
                else:
                    continue
            elif activity.destination.node_type == "emergency":
                emergency_bus_activity = EAG.A_minus_agg[activity.destination]
                if len(emergency_bus_activity) != 1:
                    raise ValueError("Number of emergency bus activities from this event should be equal to 1")
                if phi[emergency_bus_activity[0].id] > 0.5:
                    e1 = activity.destination
                    potential_e2 = EAG.agg_to_disagg_events[activity.origin]
                    for e_ in potential_e2:
                        for a_ in EAG.A_plus[e_]:
                            if x[a_.id] > 0.5:
                                e2 = e_
                else:
                    continue
            else:
                raise ValueError("There should not be a transferring activity between two mergency events")
        else:
            potential_e1 = EAG.agg_to_disagg_events[activity.destination]
            for e_ in potential_e1:
                for a_ in EAG.A_minus[e_]:
                    if x[a_.id] > 0.5:
                        e1 = e_
            if not e1:
                if verbose > 0:
                    logger.debug("Disaggregated passenger event missing")
                continue
            potential_e2 = EAG.agg_to_disagg_events[activity.origin]
            for e_ in potential_e2:
                for a_ in EAG.A_plus[e_]:
                    if x[a_.id] > 0.5:
                        e2 = e_
            if not e2:
                if verbose > 0:
                    logger.debug("Disaggregated egress event missing")
                continue

        # Passenger activities
        if activity.activity_type == "passenger running":
            if is_activated(activity) and e1 and e2 and activity.origin.train:
                cost = y[e1.id] - y[e2.id]
                train = activity.origin.train.id
                capacity = activity.origin.train.capacity
                if capacity > 0 or capacity == float("inf"):
                    include_this_activity = True
        elif activity.activity_type == "dwelling":
            if is_activated(activity) and e1 and e2 and activity.origin.train:
                cost = EAG.beta_1 * (y[e1.id] - y[e2.id])
                train = activity.origin.train.id
                capacity = np.inf
                include_this_activity = True
        elif activity.activity_type == "access" and activity.passenger_group:
            if is_activated_and_stopping(activity, stopping_everywhere, do_origin=False) and e1:
                cost = EAG.beta_3 * max(0, activity.passenger_group.time - y[e1.id]) + EAG.beta_4 * (
                    max(0, y[e1.id] - activity.passenger_group.time)
                )
                train = np.nan
                capacity = np.inf
                include_this_activity = True
        elif activity.activity_type == "egress":
            if is_activated_and_stopping(activity, stopping_everywhere, do_destination=False):
                cost = 0
                train = np.nan
                capacity = np.inf
                include_this_activity = True
                if activity.destination.station.id not in end_nodes:
                    end_nodes[activity.destination.station.id] = -1 - len(end_nodes)
                dest = end_nodes[activity.destination.station.id]
        elif activity.activity_type == "transferring":
            if is_activated_and_stopping(activity, stopping_everywhere) and e1 and e2:
                time_ = y[e1.id] - y[e2.id]
                if time_ <= EAG.maximum_transfer_time and time_ >= EAG.minimum_transfer_time:
                    cost = EAG.beta_2 + EAG.beta_1 * time_
                    train = np.nan
                    capacity = np.inf
                    include_this_activity = True

        elif activity.activity_type == "penalty":
            cost = EAG.penalty_cost
            train = np.nan
            capacity = np.inf
            include_this_activity = True
            if activity.destination.station.id not in end_nodes:
                end_nodes[activity.destination.station.id] = -1 - len(end_nodes)
            dest = end_nodes[activity.destination.station.id]
        elif activity.activity_type == "emergency bus" and activity.origin.train:
            if is_activated(activity) and e1 and e2:
                cost = y[e1.id] - y[e2.id]
                train = activity.origin.train.id
                capacity = activity.origin.train.capacity
                if phi[activity.id] > 0.5:
                    include_this_activity = True

        if include_this_activity:
            arc_data = [
                node_id(activity.origin.id),
                node_id(dest) if dest is not None else node_id(activity.destination.id),
                cost,
                train,
                capacity,
                activity_type_dict[activity.activity_type],
            ]
            if arc_data not in arcs_array:
                arcs_array.append(arc_data)
                arcs_idx_to_obj[idx] = activity
                idx += 1
                activities_objs.append(activity)
                if activity.activity_type not in ["penalty", "egress"] and e1:
                    train_start.append(y[e1.id])

    valid_nodes = set()
    for arc in arcs_array:
        valid_nodes.add(arc[0])
        valid_nodes.add(arc[1])

    ODs = np.empty((len(EAG.passengers_groups), 5), dtype="int")
    for i, group in enumerate(EAG.passengers_groups):
        o = node_id(EAG.A_access[group][0].origin.id)
        d = node_id(end_nodes[EAG.A_egress[group][0].destination.station.id])
        if o not in valid_nodes or d not in valid_nodes:
            raise ValueError("origin or destination not in valid nodes")
        ODs[i] = [o, d, group.id, group.num_passengers, group.priority]

    arcs_array = np.array(arcs_array)
    events_ids = np.array(events_ids)
    activities_objs = np.array(activities_objs)
    train_start = np.array(train_start)

    if not use_presence_in_train:
        ODs = ODs[ODs[:, -1].argsort()]

    # ---------------------------
    # Initial assignment without capacity constraints
    # ---------------------------

    n_nodes = len(events_ids)
    costs_per_group, path_per_group = run_shortest_path_over_ODs(ODs, arcs_array, n_nodes)

    # ---------------------------
    # Enforce capacity constraint
    # ---------------------------

    if use_presence_in_train:
        arcs_usage = ODs[:, 3] @ path_per_group
        arcs_overloaded = arcs_usage > np.array(arcs_array)[:, 4]
        banned_links_per_group = np.full_like(path_per_group, False)
        while arcs_overloaded.sum() > 0:
            overloaded_indices = np.where(arcs_overloaded)[0]
            sorted_overloaded_indices = overloaded_indices[np.argsort(train_start[overloaded_indices])]
            arc_idx = sorted_overloaded_indices[0]
            banned_groups_idx = get_groups_idx_to_ban_presence_in_train(
                path_per_group, ODs, arcs_array, np.array(arcs_array)[arc_idx, 4], arc_idx
            )
            banned_links_per_group[banned_groups_idx, arc_idx] = True
            if verbose > 0:
                logger.debug(
                    f"Banning arc {arc_idx} ({activities_objs[arc_idx]}: {arcs_array[arc_idx]}, "
                    f"dep {train_start[arc_idx]}) for group(s) {', '.join((banned_groups_idx + 1).astype('str'))}"
                )
            ban_ODs = ODs[banned_groups_idx]
            costs_per_group, path_per_group = run_shortest_path_w_banned_arcs(
                ban_ODs, banned_links_per_group, arcs_array, n_nodes, costs_per_group, path_per_group
            )
            arcs_usage = ODs[:, 3] @ path_per_group
            potential_arcs_usage = ODs[:, 3] @ banned_links_per_group + arcs_usage
            group_arc_to_unban = (potential_arcs_usage <= np.array(arcs_array)[:, 4]) & banned_links_per_group
            if group_arc_to_unban.any():
                groups_to_unban, arcs_to_unban = np.nonzero(group_arc_to_unban)
                if verbose > 0:
                    for g, a in zip(groups_to_unban, arcs_to_unban):
                        logger.debug(f"Unbanning arc {a} for group {g+1}")
                banned_links_per_group[group_arc_to_unban] = False
                unban_ODs = ODs[groups_to_unban]
                costs_per_group, path_per_group = run_shortest_path_w_banned_arcs(
                    unban_ODs, banned_links_per_group, arcs_array, n_nodes, costs_per_group, path_per_group
                )
                arcs_usage = ODs[:, 3] @ path_per_group
            arcs_overloaded = arcs_usage > np.array(arcs_array)[:, 4]
    else:
        arcs_usage = (ODs[:, [3]] * path_per_group).cumsum(axis=0)
        arcs_overloaded = arcs_usage > np.array(arcs_array)[:, 4]
        banned_links_per_group = np.full_like(path_per_group, False)
        while arcs_overloaded.any():
            groups, arcs = arcs_overloaded.nonzero()
            group_id, arc_idx = int(groups[0]), arcs[0]
            banned_links_per_group[group_id:, arc_idx] = True
            if verbose > 0:
                logger.debug(
                    f"Banning arc {arc_idx} ({activities_objs[arc_idx]}: {arcs_array[arc_idx]}) for group(s) >= "
                    f"{group_id + 1} {1+group_id + np.nonzero(path_per_group[group_id:, arc_idx])[0]}"
                )
            ban_ODs = ODs[group_id:][path_per_group[group_id:, arc_idx]]
            costs_per_group, path_per_group = run_shortest_path_w_banned_arcs(
                ban_ODs, banned_links_per_group, arcs_array, n_nodes, costs_per_group, path_per_group
            )
            arcs_usage = (ODs[:, [3]] * path_per_group).cumsum(axis=0)
            arcs_overloaded = arcs_usage > np.array(arcs_array)[:, 4]

    if verbose > 0:
        logger.debug(f"Execution time passenger assignment: {time.time() - start_time:.6f} seconds")

    # Return results
    if level_of_detail == 1:
        dict_occu: defaultdict[Activity, int] = defaultdict(int)
        used_trains = defaultdict(set)
        for i, path_row in enumerate(path_per_group):
            for j, arc_used in enumerate(path_row):
                if arc_used:
                    act_obj = arcs_idx_to_obj[j]
                    if act_obj.activity_type == "passenger running":
                        used_trains[EAG.get_group_by_id(i)].add(act_obj.destination.train)
                    dict_occu[act_obj] += ODs[i, 3]
        return costs_per_group, dict_occu, used_trains, costs_per_group @ ODs[:, 3]
    return costs_per_group @ ODs[:, 3]
