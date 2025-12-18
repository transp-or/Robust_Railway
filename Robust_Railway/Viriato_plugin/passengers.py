import numpy as np

from Robust_Railway.event_activity_graph_multitracks import (
    PassengerGroup,
)


def find_all_simple_paths(
    EAG,
    links,
    tt_links,
    specific_group_origin=None,
    specific_group_destination=None,
):
    """
    Finds all simple paths (paths without cycles) between each pair of stations in the railway network,
    or only for a specific group if provided.

    :param EAG: Graph structure containing passenger groups
    :param links: List of tuples (station_code1, station_code2)
    :param tt_links: Dict with keys as (code1, code2) and values as travel times
    :param specific_group_origin: Optional, origin station id
    :param specific_group_destination: Optional, destination station id
    :return:
        - all_paths_dict: {(src, dest): [path, ...]}
        - all_tts: {(src, dest): [travel_time, ...]}
    """

    def dfs(current, target, visited, path, all_paths, tt, tts):
        if current == target:
            all_paths.append(path[:])
            tts.append(tt)
            return

        for link in links:
            key = (EAG.code_to_id[link[0]], EAG.code_to_id[link[1]])
            min_tt = tt_links[key]
            # Forward direction
            if EAG.code_to_id[link[0]] == current and EAG.code_to_id[link[1]] not in visited:
                visited.add(EAG.code_to_id[link[1]])
                path.append(EAG.code_to_id[link[1]])
                tt += min_tt
                dfs(EAG.code_to_id[link[1]], target, visited, path, all_paths, tt, tts)
                path.pop()
                visited.remove(EAG.code_to_id[link[1]])
            # Backward direction
            elif EAG.code_to_id[link[1]] == current and EAG.code_to_id[link[0]] not in visited:
                visited.add(EAG.code_to_id[link[0]])
                path.append(EAG.code_to_id[link[0]])
                tt += min_tt
                dfs(EAG.code_to_id[link[0]], target, visited, path, all_paths, tt, tts)
                path.pop()
                visited.remove(EAG.code_to_id[link[0]])

    all_paths_dict = {}
    all_tts = {}

    # Determine origin-destination pairs
    ODs = (
        [(specific_group_origin, specific_group_destination)]
        if specific_group_origin and specific_group_destination
        else [(group.origin.id, group.destination.id) for group in EAG.passengers_groups]
    )

    for src, dest in ODs:
        all_paths = []
        tts = []
        dfs(src, dest, {src}, [src], all_paths, 0, tts)
        all_paths_dict[(src, dest)] = all_paths
        all_tts[(src, dest)] = tts

    return all_paths_dict, all_tts


def create_passenger_groups(
    EAG,
    df,
    probabilities,
    links,
    tt_links,
    number_of_groups=25,
    group_size_increment=20,
    interval=5,
    end_time_buffer=0.5,
):
    """
    Creates passenger groups in the event-activity graph with random departure times.

    :param EAG: Event-Activity Graph object
    :param df: DataFrame or list of (origin_code, destination_code) tuples
    :param probabilities: Probability distribution for group selection
    :param links: List of tuples (station_code1, station_code2)
    :param tt_links: Dict with keys as (code1, code2) and values as travel times
    :param number_of_groups: Number of passenger groups to create
    :param group_size_increment: Increment for group size
    :param interval: Time interval between possible departures
    :param end_time_buffer: Buffer for latest possible arrival time
    :return: Updated EAG object
    """

    groups = df.tolist()
    time_intervals = list(range(EAG.start_time_window, EAG.end_time_window + 1, interval))

    group_id_counter = 0
    group_sizes = {}
    group_times = {}

    # Create a first group with fixed origin/destination
    origin_id = EAG.code_to_id["0085ROL"]
    destination_id = EAG.code_to_id["0085NY"]
    origin_obj = EAG.get_station_by_id(origin_id)
    destination_obj = EAG.get_station_by_id(destination_id)
    departure = 430
    passenger_group = PassengerGroup(
        group_id=group_id_counter,
        origin=origin_obj,
        destination=destination_obj,
        time=departure,
        num_passengers=20,
        priority=group_id_counter,
    )
    EAG.add_passengers_group(passenger_group)
    group_id_counter = 1

    while group_id_counter < number_of_groups:
        selected = np.random.choice(len(groups), p=probabilities)
        group = groups[selected]
        origin_code, destination_code = group
        origin_id = EAG.code_to_id[origin_code]
        destination_id = EAG.code_to_id[destination_code]
        departure = np.random.choice(time_intervals)

        origin_obj = EAG.get_station_by_id(origin_id)
        destination_obj = EAG.get_station_by_id(destination_id)

        all_paths_dict, all_tts = find_all_simple_paths(
            EAG,
            links,
            tt_links,
            specific_group_origin=origin_id,
            specific_group_destination=destination_id,
        )
        key = (origin_id, destination_id)
        min_tt = min(all_tts[key])
        arrival_time = departure + min_tt

        valid_group = False
        max_arrival = EAG.end_time_window - (end_time_buffer * (EAG.end_time_window - EAG.start_time_window))
        if arrival_time <= max_arrival:
            valid_group = True
        else:
            # Try to truncate the path
            min_tt_index = all_tts[key].index(min_tt)
            min_path = all_paths_dict[key][min_tt_index]
            previous_node = min_path[0]
            cumul_time = departure
            end_node_truncated = previous_node

            for node in min_path[1:]:
                link = (previous_node, node)
                cumul_time += tt_links[link]
                if cumul_time > max_arrival:
                    break
                end_node_truncated = node
                previous_node = node

            if end_node_truncated != min_path[0]:
                valid_group = True
                destination_obj = EAG.get_station_by_id(end_node_truncated)

        if valid_group:
            size = group_sizes.get(group, 0) + group_size_increment
            group_sizes[group] = size
            group_times[group] = departure

            passenger_group = PassengerGroup(
                group_id=group_id_counter,
                origin=origin_obj,
                destination=destination_obj,
                time=departure,
                num_passengers=size,
                priority=group_id_counter,
            )
            EAG.add_passengers_group(passenger_group)
            group_id_counter += 1

    return EAG
