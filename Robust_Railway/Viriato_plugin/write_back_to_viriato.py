import logging
from datetime import timedelta

from py_client.aidm import StopStatus, UpdateTimesTrainPathNode
from py_client.algorithm_interface import algorithm_interface_factory

logger = logging.getLogger(__name__)


def push_sol(api, api_trains, EAG, X, Y, Z, i):
    api.reset_trains()
    trains_updated = []

    for tapi in api_trains:
        trains = EAG.get_trains_by_code(tapi.code)
        train = None
        train_path_nodes = sorted([node for node in tapi.train_path_nodes], key=lambda node: node.arrival_time)
        # Find the corresponding train by matching arrival times
        for t in trains:
            if train_path_nodes and train_path_nodes[0].arrival_time == t.train_path_nodes[0].arrival_time:
                train = t
                break

        if train:
            # Check if train is short-turned
            short_turned = any(a.activity_type == "short-turning" and X[a.id] > 0.5 for a in EAG.A_train[train])
            if short_turned:
                node_id_lst = []
                idx_short_turn = 0
                for a in EAG.get_ordered_disagg_activities_train(train):
                    if a.activity_type != "starting":
                        if X[a.id] > 0.5 and (not node_id_lst or a.origin.station.id != node_id_lst[-1]):
                            node_id_lst.append(a.origin.station.id)
                            if a.origin.node_type != "short-turning":
                                idx_short_turn += 1
                logger.debug(f"Node id list for train {train.id}: {node_id_lst}")
                updated_tapi = api.copy_train_and_replace_route(tapi.id, node_id_lst)
                train_path_nodes = sorted([node for node in updated_tapi], key=lambda node: node.arrival_time)
                api.cancel_train(tapi.id)
                train_begin_at = 0
                train_end_at = len(train_path_nodes) - 1
            else:
                idx_short_turn = len(train_path_nodes)
                train_begin_at = train.begin_at
                train_end_at = train.end_at

            trains_updated.append(train)
            updated_train_path_nodes = []
            logger.debug(
                f"Found corresponding Viriato train with code {tapi.code} and id {train.id} "
                f"- starting and ending at {train_begin_at} -- {train_end_at}"
            )
            for n in train_path_nodes:
                logger.debug(
                    "Node id",
                    n.id,
                    "station",
                    n.node_id,
                    "arrival_time",
                    n.arrival_time,
                    "departure_time",
                    n.departure_time,
                )

            stop_status = StopStatus.operational_stop
            # Cancel train path nodes before the time window
            updated_train_path_node = UpdateTimesTrainPathNode(
                train_path_nodes[train_begin_at].id,
                train_path_nodes[train_begin_at].arrival_time,
                train_path_nodes[train_begin_at].departure_time,
                None,
                None,
                stop_status,
            )
            api.update_train_times(tapi.id, [updated_train_path_node])
            api.cancel_train_before(tapi.id, train_path_nodes[train_begin_at].id)

            # Cancel train path nodes after the time window
            updated_train_path_node = UpdateTimesTrainPathNode(
                train_path_nodes[train_end_at].id,
                train_path_nodes[train_end_at].arrival_time,
                train_path_nodes[train_end_at].departure_time,
                None,
                None,
                stop_status,
            )
            api.update_train_times(tapi.id, [updated_train_path_node])
            api.cancel_train_after(tapi.id, train_path_nodes[train_end_at].id)

            # Update nodes within the time window
            stations_visited = []
            train_cancelled = False
            prev_updated_departure_time = EAG.start_time_window
            for idx, train_path_node in enumerate(train_path_nodes[train_begin_at : train_end_at + 1]):
                found_start_time, found_end_time, cancelled = False, False, False
                updated_section_track = None
                updated_node_track = None
                node_type_expected = (
                    "short-turning"
                    if train_path_node.node_id in stations_visited and idx >= idx_short_turn
                    else "regular"
                )

                for event in EAG.regular_disaggregated_events:
                    if sum(X[a.id] for a in EAG.A_plus[event]) + Z[event.id] < 0.5:
                        continue
                    previous_station_event = None
                    if event.event_type == "arrival":
                        previous_station_event = EAG.A_plus[event][0].origin.station.id
                    else:
                        previous_event = EAG.A_plus[event][0].origin
                        previous_station_event = EAG.A_plus[previous_event][0].origin.station.id
                    previous_station_node = (
                        train_path_nodes[idx + train_begin_at - 1].node_id if idx > 0 else train_path_node.node_id
                    )

                    # Arrival event
                    if (
                        event.station.id == train_path_node.node_id
                        and event.train.id == train.id
                        and event.event_type == "arrival"
                        and event.node_type == node_type_expected
                        and previous_station_event == previous_station_node
                    ):
                        updated_arrival_time = EAG.base_date + timedelta(minutes=Y[event.id])
                        updated_arrival_time_minutes = Y[event.id]
                        updated_node_track = event.node_track.id
                        found_start_time = True
                        for a in EAG.A_plus[event]:
                            if X[a.id] > 0.5:
                                updated_section_track = a.section_track
                            if Z[event.id] > 0.5:
                                cancelled = True

                    # Departure event
                    if (
                        event.station.id == train_path_node.node_id
                        and event.train.id == train.id
                        and event.event_type == "departure"
                        and event.node_type == node_type_expected
                        and previous_station_event == previous_station_node
                    ):
                        updated_departure_time = EAG.base_date + timedelta(minutes=Y[event.id])
                        updated_departure_time_minutes = Y[event.id]
                        found_end_time = True
                        if Z[event.id] > 0.5:
                            cancelled = True

                if found_start_time and found_end_time and updated_section_track:
                    stations_visited.append(train_path_node.node_id)
                elif (
                    found_start_time
                    and found_end_time
                    and not updated_section_track
                    and train_path_node.sequence_number == train_begin_at
                ):
                    stations_visited.append(train_path_node.node_id)
                elif cancelled:
                    logger.debug(f"Cancelling train {train.id} at station {train_path_node.node_id}")
                    api.cancel_train(tapi.id)
                    train_cancelled = True
                    break
                else:
                    raise ValueError(
                        f"Missing train path node {train_path_node.id} at station "
                        f"{train_path_node.node_id} in train {train.id}"
                    )

                # Calculate minimum stop/run times and status
                updated_min_stop_time = None
                if idx == 0 or idx == train_end_at - train_begin_at:
                    stop_status = None
                    if updated_departure_time - updated_arrival_time < train_path_node.minimum_stop_time:
                        updated_min_stop_time = updated_departure_time - updated_arrival_time
                elif updated_departure_time_minutes - updated_arrival_time_minutes == 0:
                    if train_path_node.minimum_stop_time > timedelta(0):
                        updated_min_stop_time = timedelta(0)
                    stop_status = "Passing"
                else:
                    if updated_departure_time - updated_arrival_time < train_path_node.minimum_stop_time:
                        updated_min_stop_time = updated_departure_time - updated_arrival_time
                    stop_status = "OperationalStop"

                logger.debug(
                    f"update train {train.id}, station {train_path_node.node_id}, from "
                    f"{updated_arrival_time} to {updated_departure_time}"
                )

                logger.debug(
                    updated_arrival_time,
                    prev_updated_departure_time,
                    train_path_node.minimum_run_time,
                    idx,
                )
                if idx > 0:
                    if updated_arrival_time - prev_updated_departure_time < train_path_node.minimum_run_time:
                        updated_minimum_run_time = updated_arrival_time - prev_updated_departure_time
                    else:
                        updated_minimum_run_time = None
                    updated_train_path_node = UpdateTimesTrainPathNode(
                        train_path_node.id,
                        updated_arrival_time,
                        updated_departure_time,
                        updated_minimum_run_time,
                        updated_min_stop_time,
                        stop_status,
                    )
                else:
                    updated_train_path_node = UpdateTimesTrainPathNode(
                        train_path_node.id,
                        updated_arrival_time,
                        updated_departure_time,
                        None,
                        updated_min_stop_time,
                        stop_status,
                    )

                updated_train_path_nodes.append(updated_train_path_node)
                if not train_cancelled:
                    if updated_node_track:
                        api.update_node_track(tapi.id, train_path_node.id, updated_node_track)
                    if updated_section_track:
                        api.update_section_track(tapi.id, train_path_node.id, updated_section_track.id)

                prev_updated_departure_time = updated_departure_time
                print("updating previous departure time to", prev_updated_departure_time)

            # Update train path node times
            if not train_cancelled:
                api.update_train_times(tapi.id, updated_train_path_nodes)

    # Check that all trains have been updated
    for train in EAG.trains:
        if train not in trains_updated:
            raise ValueError(f"Train {train.code} not found")

    api.persist_trains(f"solution {i}")
    logger.debug(f"persisting trains of solution {i}")


def write_back_to_viriato(EAG, api_url, pareto, x_values, y_values, z_values):
    with algorithm_interface_factory.create(api_url) as api:
        time_window = api.get_time_window_algorithm_parameter("timeWindowParameterMandatory")
        api_trains = api.get_trains(time_window)
        max_nb_sols = 10

        if pareto:
            for i, sol in enumerate(list(pareto)[:max_nb_sols]):
                push_sol(api, api_trains, EAG, sol.X, sol.Y, sol.Z, i)
        elif x_values and y_values and z_values:
            push_sol(api, api_trains, EAG, x_values, y_values, z_values, 1)
