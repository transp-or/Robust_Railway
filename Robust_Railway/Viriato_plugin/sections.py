import logging
from collections import defaultdict
from typing import Any, Optional

from Robust_Railway.event_activity_graph_multitracks import (
    EARailwayNetwork,
    SectionTrack,
    Station,
    Train,
)

logger = logging.getLogger(__name__)


def add_section_tracks(
    event_graph: EARailwayNetwork,
    api: Any,
    section_tracks: list[int],
    origin: Station,
    destination: Station,
    travel_time_dict: dict[tuple[int, int], dict[Train, float]],
) -> None:
    """
    Add section tracks between two stations.

    Args:
        event_graph [EARrailwayNetwork]: The railway network graph.
        api [Any]: API client.
        section_tracks [list[int]]: List of section track IDs.
        origin [Station]: Origin station.
        destination [Station]: Destination station.
        travel_time_dict [dict]: Travel times per (origin, destination).

    Returns:
        None
    """
    for track_id in section_tracks:
        track_obj = api.get_section_track(track_id)
        # Merge travel times for both directions if available
        travel_times = travel_time_dict.get((origin.id, destination.id), {})
        travel_times_rev = travel_time_dict.get((destination.id, origin.id), {})
        merged_travel_times = {**travel_times, **travel_times_rev}
        new_section = SectionTrack(
            track_id=str(track_id),
            origin=origin,
            destination=destination,
            distance=track_obj.distance_units,
            travel_time=merged_travel_times,
        )
        event_graph.add_section_track(new_section)


def create_section_tracks(
    api,
    EAG,
    links,
    junctions,
    station_tracks,
    incoming_tracks,
    outgoing_tracks,
    incoming_junction_tracks,
    outgoing_junction_tracks,
    MANUAL_LINK_CORRECTIONS,
    travel_times,
    stations_visited,
):
    """
    Creates section tracks in the event-activity graph, handling direct links, links with
    junction(s), and manually corrected links.
    """
    intermediate_stations = defaultdict(list)
    tt_links = {}
    for origin_code, dest_code in links:
        connected_stations = False
        origin_id, dest_id = EAG.code_to_id.get(origin_code), EAG.code_to_id.get(dest_code)
        if origin_id is None or dest_id is None:
            logger.debug(f"Warning: Missing station {origin_code} or {dest_code} in the mapping.")
            continue

        origin_station = EAG.get_station_by_id(origin_id)
        dest_station = EAG.get_station_by_id(dest_id)

        # Check direct section track connection
        origin_tracks = {
            track
            for node_track in station_tracks[origin_id]
            for track in outgoing_tracks.get((origin_id, node_track), [])
        }
        dest_tracks = {
            track for node_track in station_tracks[dest_id] for track in incoming_tracks.get((dest_id, node_track), [])
        }
        direct_tracks = origin_tracks & dest_tracks
        if direct_tracks:
            connected_stations = True
            for track in direct_tracks:
                section_obj = api.get_section_track(track)
                EAG.add_section_track(
                    SectionTrack(
                        track_id=track,
                        origin=origin_station,
                        destination=dest_station,
                        distance=section_obj.distance_units,
                        travel_time=travel_times.get((origin_id, dest_id), 0)
                        | travel_times.get((dest_id, origin_id), 0),
                    )
                )
            tt_links[(origin_id, dest_id)] = min(
                (travel_times.get((origin_id, dest_id), 0) | travel_times.get((dest_id, origin_id), 0)).values()
            )
            tt_links[(dest_id, origin_id)] = min(
                (travel_times.get((origin_id, dest_id), 0) | travel_times.get((dest_id, origin_id), 0)).values()
            )
            continue  # Skip to next link if direct connection exists

        # Check if there's a single junction between the two stations
        for junction_id in junctions:
            junction_station = EAG.get_station_by_id(junction_id)
            first_segment = origin_tracks & {track.id for track in incoming_junction_tracks.get(junction_id, [])}
            second_segment = {track.id for track in outgoing_junction_tracks.get(junction_id, [])} & dest_tracks

            if first_segment and second_segment:
                if connected_stations:
                    raise ValueError(f"Multiple junctions found between {origin_code} and {dest_code}")

                connected_stations = True

                # Add section tracks for both segments
                for section_track in first_segment:
                    section_obj = api.get_section_track(section_track)
                    EAG.add_section_track(
                        SectionTrack(
                            track_id=section_track,
                            origin=origin_station,
                            destination=junction_station,
                            distance=section_obj.distance_units,
                            travel_time=travel_times.get((origin_id, junction_id), 0)
                            | travel_times.get((junction_id, origin_id), 0),
                        )
                    )

                tt_links[(origin_id, dest_id)] = min(
                    (
                        travel_times.get((origin_id, junction_id), 0) | travel_times.get((junction_id, origin_id), 0)
                    ).values()
                )
                tt_links[(dest_id, origin_id)] = min(
                    (
                        travel_times.get((origin_id, junction_id), 0) | travel_times.get((junction_id, origin_id), 0)
                    ).values()
                )

                for section_track in second_segment:
                    section_obj = api.get_section_track(section_track)
                    EAG.add_section_track(
                        SectionTrack(
                            track_id=section_track,
                            origin=junction_station,
                            destination=dest_station,
                            distance=section_obj.distance_units,
                            travel_time=travel_times.get((junction_id, dest_id), 0)
                            | travel_times.get((dest_id, junction_id), 0),
                        )
                    )
                tt_links[(origin_id, dest_id)] = min(
                    (travel_times.get((junction_id, dest_id), 0) | travel_times.get((dest_id, junction_id), 0)).values()
                )
                tt_links[(dest_id, origin_id)] = min(
                    (travel_times.get((junction_id, dest_id), 0) | travel_times.get((dest_id, junction_id), 0)).values()
                )

                for train in EAG.trains:
                    if origin_station in stations_visited[train] and dest_station in stations_visited[train]:
                        intermediate_stations[(origin_station.id, dest_station.id, train.id)].append(
                            junction_station.id
                        )
                        intermediate_stations[(dest_station.id, origin_station.id, train.id)].append(
                            junction_station.id
                        )

        if connected_stations:
            continue  # Junction-based connection found, move to the next link

        # Handle manual corrections if no connection was found
        corrected_stations_lst = MANUAL_LINK_CORRECTIONS.get((origin_code, dest_code))
        if corrected_stations_lst:
            for corrected_stations in corrected_stations_lst:
                intermediate_ids = [EAG.code_to_id.get(code) for code in corrected_stations]
                intermediate_ids = [s_id for s_id in intermediate_ids if s_id is not None]

                if len(intermediate_ids) < len(corrected_stations):
                    logger.debug(
                        f"Warning: Some manual correction stations are missing for {origin_code} → {dest_code}."
                    )
                    continue

                EAG, tt_links = process_manual_link_correction(
                    EAG,
                    api,
                    intermediate_ids,
                    origin_station,
                    dest_station,
                    origin_code,
                    dest_code,
                    incoming_tracks,
                    outgoing_tracks,
                    incoming_junction_tracks,
                    outgoing_junction_tracks,
                    origin_tracks,
                    dest_tracks,
                    travel_times,
                    tt_links,
                )
                for train in EAG.trains:
                    if origin_station in stations_visited[train] and dest_station in stations_visited[train]:
                        intermediate_stations[(origin_station.id, dest_station.id, train.id)].extend(intermediate_ids)
                        intermediate_stations[(dest_station.id, origin_station.id, train.id)].extend(intermediate_ids)

        else:
            print(f"No valid connection found for {origin_code} → {dest_code}. Consider updating manual corrections.")

    return EAG, intermediate_stations, tt_links


def connect_junctions(
    event_graph: EARailwayNetwork,
    api: Any,
    junction_a: Station | None,
    junction_b: Station | None,
    travel_time_dict: dict,
    junctions_incoming: dict,
    junctions_outgoing: dict,
    tt_links: dict,
    origin: int,
    dest: int,
) -> dict:
    """
    Connects two junctions by adding section tracks.

    Args:
        event_graph [EARailwayNetwork]: The railway network graph.
        api [Any]: API client.
        junction_a [Station | None]: First junction.
        junction_b [Station | None]: Second junction.
        travel_time_dict [dict]: Travel times.
        junctions_incoming [dict]: Incoming tracks per junction.
        junctions_outgoing [dict]: Outgoing tracks per junction.
        tt_links [dict]: Travel time links.
        origin [int]: Origin station id.
        dest [int]: Destination station id.

    Returns:
        dict: Updated travel time links.
    """
    if not junction_a or not junction_b:
        raise ValueError("One of the junctions is None.")
    common_tracks = list(set(junctions_outgoing[junction_a.id]) & set(junctions_incoming[junction_b.id]))
    if not common_tracks:
        raise ValueError(f"Issue with manual junctions between {junction_a.id} and {junction_b.id}")

    add_section_tracks(event_graph, api, common_tracks, junction_a, junction_b, travel_time_dict)
    tt_links[(origin, dest)] += min(
        (
            travel_time_dict.get((junction_a.id, junction_b.id), 0)
            | travel_time_dict.get((junction_b.id, junction_a.id), 0)
        ).values()
    )
    tt_links[(dest, origin)] += min(
        (
            travel_time_dict.get((junction_a.id, junction_b.id), 0)
            | travel_time_dict.get((junction_b.id, junction_a.id), 0)
        ).values()
    )
    return tt_links


def process_manual_link_correction(
    EAG: EARailwayNetwork,
    api: Any,
    junctions: list[int],
    origin_station: Station,
    dest_station: Station,
    origin_code: str,
    dest_code: str,
    incoming_tracks: dict,
    outgoing_tracks: dict,
    incoming_junction_tracks: dict,
    outgoing_junction_tracks: dict,
    origin_tracks: set,
    dest_tracks: set,
    travel_times: dict,
    tt_links: dict,
) -> tuple[EARailwayNetwork, dict]:
    """
    Processes manual corrections for links with multiple junctions between stations.

    Args:
        EAG [EARailwayNetwork]: Railway network graph.
        api [Any]: API client.
        junctions [list[int]]: Junction ids.
        origin_station [Station]: Origin station.
        dest_station [Station]: Destination station.
        origin_code [str]: Origin code.
        dest_code [str]: Destination code.
        incoming_tracks [dict]: Incoming tracks.
        outgoing_tracks [dict]: Outgoing tracks.
        incoming_junction_tracks [dict]: Incoming junction tracks.
        outgoing_junction_tracks [dict]: Outgoing junction tracks.
        origin_tracks [set]: Origin tracks.
        dest_tracks [set]: Destination tracks.
        travel_times [dict]: Travel times.
        tt_links [dict]: Travel time links.

    Returns:
        tuple: (EAG, tt_links)
    """
    nb_junctions = len(junctions)
    if nb_junctions > 5:
        raise ValueError(
            f"Manual link correction not implemented for more than 5 junctions between {origin_code} and {dest_code}"
        )

    origin_connected = destination_connected = False
    first_junction = second_junction = third_junction = forth_junction = fifth_junction = None
    second_or_third_junction_ids = []
    second_or_third__or_fourth_junction_ids = []

    junctions_incoming = {j: [t.id for t in incoming_junction_tracks[j]] for j in junctions}
    junctions_outgoing = {j: [t.id for t in outgoing_junction_tracks[j]] for j in junctions}
    tt_links[(origin_station.id, dest_station.id)] = 0
    tt_links[(dest_station.id, origin_station.id)] = 0
    for junction_id in junctions:
        junction_station = EAG.get_station_by_id(junction_id)

        incoming_tracks_set = set(junctions_incoming[junction_id])
        outgoing_tracks_set = set(junctions_outgoing[junction_id])

        common_origin = list(origin_tracks & incoming_tracks_set)
        common_dest = list(outgoing_tracks_set & dest_tracks)

        if common_origin and not origin_connected:
            origin_connected = True
            first_junction = junction_station
            add_section_tracks(EAG, api, common_origin, origin_station, first_junction, travel_times)
            tt_links[(origin_station.id, dest_station.id)] += min(
                (
                    travel_times.get((origin_station.id, first_junction.id), 0)
                    | travel_times.get((first_junction.id, origin_station.id), 0)
                ).values()
            )
            tt_links[(dest_station.id, origin_station.id)] += min(
                (
                    travel_times.get((origin_station.id, first_junction.id), 0)
                    | travel_times.get((first_junction.id, origin_station.id), 0)
                ).values()
            )
        elif common_dest and not destination_connected:
            destination_connected = True
            if nb_junctions == 2:
                second_junction = junction_station
            elif nb_junctions == 3:
                third_junction = junction_station
            elif nb_junctions == 4:
                forth_junction = junction_station
            elif nb_junctions == 5:
                fifth_junction = junction_station
            add_section_tracks(EAG, api, common_dest, junction_station, dest_station, travel_times)
            tt_links[(origin_station.id, dest_station.id)] += min(
                (
                    travel_times.get((junction_station.id, dest_station.id), 0)
                    | travel_times.get((dest_station.id, junction_station.id), 0)
                ).values()
            )
            tt_links[(dest_station.id, origin_station.id)] += min(
                (
                    travel_times.get((junction_station.id, dest_station.id), 0)
                    | travel_times.get((dest_station.id, junction_station.id), 0)
                ).values()
            )

        else:
            if nb_junctions < 3:
                raise ValueError(f"Manual correction issue between {origin_code} and {dest_code}")
            if nb_junctions == 4:
                second_or_third_junction_ids.append(junction_station)
            else:
                second_or_third__or_fourth_junction_ids.append(junction_station)

    if nb_junctions == 4:
        second_junction, third_junction, _ = identify_middle_junctions(
            first_junction, forth_junction, second_or_third_junction_ids, junctions_outgoing, junctions_incoming
        )
    elif nb_junctions == 5:
        second_junction, third_junction, forth_junction = identify_middle_junctions(
            first_junction,
            fifth_junction,
            second_or_third__or_fourth_junction_ids,
            junctions_outgoing,
            junctions_incoming,
        )

    if nb_junctions >= 3:
        tt_links = connect_junctions(
            EAG,
            api,
            first_junction,
            second_junction,
            travel_times,
            junctions_incoming,
            junctions_outgoing,
            tt_links,
            origin_station.id,
            dest_station.id,
        )
        tt_links = connect_junctions(
            EAG,
            api,
            second_junction,
            third_junction,
            travel_times,
            junctions_incoming,
            junctions_outgoing,
            tt_links,
            origin_station.id,
            dest_station.id,
        )

        if nb_junctions >= 4:
            tt_links = connect_junctions(
                EAG,
                api,
                third_junction,
                forth_junction,
                travel_times,
                junctions_incoming,
                junctions_outgoing,
                tt_links,
                origin_station.id,
                dest_station.id,
            )
        if nb_junctions == 5:
            tt_links = connect_junctions(
                EAG,
                api,
                forth_junction,
                fifth_junction,
                travel_times,
                junctions_incoming,
                junctions_outgoing,
                tt_links,
                origin_station.id,
                dest_station.id,
            )

    if nb_junctions == 2:
        tt_links = connect_junctions(
            EAG,
            api,
            first_junction,
            second_junction,
            travel_times,
            junctions_incoming,
            junctions_outgoing,
            tt_links,
            origin_station.id,
            dest_station.id,
        )

    if (
        not origin_connected
        or not destination_connected
        or not first_junction
        or (not second_junction and nb_junctions >= 2)
        or (not third_junction and nb_junctions >= 3)
        or (not forth_junction and nb_junctions >= 4)
        or (not fifth_junction and nb_junctions == 5)
    ):
        raise ValueError(f"Manual link correction issue between {origin_code} and {dest_code}")

    return EAG, tt_links


def identify_middle_junctions(
    first_junction: Station | None,
    last_junction: Station | None,
    candidates: list[Station],
    outgoing: dict[int, list[int]],
    incoming: dict[int, list[int]],
) -> tuple[Optional[Station], Optional[Station], Optional[Station]]:
    """
    Determines which of the candidates stations are second, third, and fourth junctions.

    Args:
        first_junction [Station | None]: First junction.
        last_junction [Station | None]: Last junction.
        candidates [list[Station]]: Candidate junctions.
        outgoing [dict[int, list[int]]]: Outgoing tracks per junction.
        incoming [dict[int, list[int]]]: Incoming tracks per junction.

    Returns:
        tuple: (second_junction, third_junction, forth_junction)
    """
    if not first_junction or not last_junction:
        raise ValueError("First or last junction is None.")
    second_junction = third_junction = forth_junction = None
    for junction in candidates:
        if set(outgoing[first_junction.id]) & set(incoming[junction.id]):
            second_junction = junction
        elif set(outgoing[junction.id]) & set(incoming[last_junction.id]):
            if len(candidates) == 2:
                third_junction = junction
            elif len(candidates) == 3:
                forth_junction = junction
            else:
                raise ValueError("Unexpected number of candidate junctions.")

    if len(candidates) == 3:
        for junction in candidates:
            if junction != second_junction and junction != forth_junction:
                third_junction = junction

    if second_junction is None or third_junction is None:
        raise ValueError("Error identifying second and third junctions.")
    if len(candidates) == 3 and forth_junction is None:
        raise ValueError("Error identifying forth junction.")
    return second_junction, third_junction, forth_junction
