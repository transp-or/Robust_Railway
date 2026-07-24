from collections import defaultdict
from typing import DefaultDict

from Robust_Railway.event_activity_graph_multitracks import EARailwayNetwork, Event, PassengerGroup


# function for sets of paths
def find_all_paths(
    passenger_group: PassengerGroup,
    EAG: EARailwayNetwork,
    max_station_count=15,
    max_transfer_count=2,
    max_visits_per_station=2,
):

    all_paths = []
    visit_count: DefaultDict[int, int] = defaultdict(int)
    transfer_per_station: DefaultDict[int, int] = defaultdict(int)

    def dfs(current: Event, path, station_count: int, transfer_count: int):

        if station_count > max_station_count or transfer_count > max_transfer_count:
            return

        if current.station == passenger_group.destination:
            all_paths.append(path.copy())
            return

        for activity in EAG.A_minus_agg[current]:
            next_event = activity.destination

            inc_station = 1 if current.event_type == "arrival" else 0
            inc_transfer = 1 if activity.activity_type == "transferring" else 0
            station_id = next_event.station.id

            if activity.activity_type == "transferring":
                if transfer_per_station[station_id] >= 1:
                    continue

            if visit_count[station_id] + inc_station < max_visits_per_station:

                station_count += inc_station
                transfer_count += inc_transfer

                if next_event.event_type == "arrival":
                    visit_count[station_id] += inc_station

                if activity.activity_type == "transferring":
                    transfer_per_station[station_id] += 1

                path.append(activity)
                dfs(next_event, path, station_count, transfer_count)
                path.pop()

                station_count -= inc_station
                transfer_count -= inc_transfer

                if next_event.event_type == "arrival":
                    visit_count[station_id] -= inc_station

                if activity.activity_type == "transferring":
                    transfer_per_station[station_id] -= 1

    for activity in EAG.grouped_activities["passenger running"]:
        if activity.origin.station == passenger_group.origin:
            dfs(activity.destination, [activity], 0, 0)

    for penalty_arc in EAG.grouped_activities["penalty"]:
        if penalty_arc.passenger_group == passenger_group:
            all_paths.append([penalty_arc])

    return all_paths
