import logging
import random
from typing import Any, Union

from ..event_activity_graph_multitracks import Activity, EARailwayNetwork, NodeTrack, SectionTrack, Train
from .repair_operators_cancel import cancel_if_too_delayed, cancel_train_completely
from .track_occupancy import determine_direction, get_junction_usage, get_minimum_headway, get_track_usage

logger = logging.getLogger(__name__)

# ------------------------
# Tools
# ------------------------


def set_train_to_timetable(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train):
    """Reset the given train to its planned timetable"""
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    for activity in EAG.get_activities_per_train(train):
        Xplus[activity.id] = int(activity.in_timetable)
    for event in EAG.get_events_per_train(train):
        Yplus[event.id] = event.scheduled_time
        Zplus[event.id] = 0
    return Xplus, Yplus, Zplus, PHIplus


def get_earliest_start_based_on_track_occupancy(
    EAG: EARailwayNetwork,
    activity: Activity,
    Y_origin: float,
    Y_destination: float,
    track_usage: dict,
    junction_usage: dict,
    backward: bool,
    min_stop_time: float,
    previous_activity_end_time: float,
    earliest_next_activity_time: float | None = None,
    verbose: int = 0,
):
    """
    Compute the earliest feasible start time for an activity based on current track and junction occupancy.

    Args:
        EAG (EARailwayNetwork): Event-activity graph of the railway network.
        activity (Activity): Activity for which the start and end times are to be determined.
        Y_origin (float): Time at origin event.
        Y_destination (float): Time at destination event.
        track_usage (dict): Information on track occupancy.
        junction_usage (dict): Information on junction occupancy.
        backward (bool): If True, process activities backward in time.
        min_stop_time (float): Minimum required stop time.
        previous_activity_end_time (float): End time of previous activity.
        earliest_next_activity_time (float): Earliest time for next activity.
        verbose (int): Verbosity level.

    Returns:
        (float, float, float): Earliest possible start time, minimum feasible end time, maximum feasible end time.
    """
    if verbose > 0:
        logger.debug("activity checking for track occupancy", EAG.print_activity_info(activity))
        logger.debug("earliest start time", earliest_next_activity_time)
        logger.debug("going back and checking track occupancy", backward)

    track: NodeTrack | SectionTrack | None
    if activity.section_track:
        track = activity.section_track
        at_station = False
        min_tt = track.travel_time[activity.origin.train]
    else:
        track = activity.origin.node_track
        at_station = True
        min_tt = 0
    min_activity_duration = min_stop_time + min_tt
    scheduled_start_time = activity.origin.scheduled_time
    scheduled_end_time = activity.destination.scheduled_time

    if not track:
        usage_info = junction_usage[activity.origin.station]
    else:
        usage_info = track_usage[track]

    if len(usage_info) == 0:
        return scheduled_start_time, scheduled_end_time, 1e16

    for i in range(len(usage_info)):
        last_end_time = 1e16
        occ_start, occ_end, occ_dir, occ_train, occ_headway = usage_info[i]
        correction = 0
        if occ_train == activity.origin.train.id:
            continue

        if occ_train not in ["D", "I"]:
            if not at_station:
                if occ_dir == determine_direction(activity, track):
                    correction = get_minimum_headway(EAG, EAG.get_train_by_id(occ_train))
                    should_end_after = max(occ_end + correction, scheduled_end_time)
                else:
                    correction = get_minimum_headway(EAG, EAG.get_train_by_id(occ_train)) + (occ_end - occ_start)
                    should_end_after = scheduled_end_time
            else:
                if track:
                    correction = EAG.minimum_separation_time + (occ_end - occ_start)
                else:
                    correction = occ_end - occ_start
                should_end_after = scheduled_end_time
        else:
            correction = occ_end - occ_start
            should_end_after = scheduled_end_time

        if backward:
            when_can_I_start = max(occ_start + correction, scheduled_start_time, previous_activity_end_time)
            should_end_after = max(
                should_end_after, earliest_next_activity_time, when_can_I_start + min_activity_duration
            )
        else:
            when_can_I_start = max(occ_start + correction, scheduled_start_time, previous_activity_end_time)
            should_end_after = max(should_end_after, when_can_I_start + min_activity_duration)

        if len(usage_info) > i + 1:
            next_occ_start, next_occ_end, next_occ_dir, next_occ_train, _ = usage_info[i + 1]
            if next_occ_train == activity.origin.train.id:
                if len(usage_info) == i + 2:
                    return when_can_I_start, should_end_after, last_end_time
                else:
                    next_occ_start, next_occ_end, next_occ_dir, next_occ_train, _ = usage_info[i + 2]
                    if next_occ_train == activity.origin.train.id:
                        if len(usage_info) == i + 3:
                            return when_can_I_start, should_end_after, last_end_time
                        else:
                            next_occ_start, next_occ_end, next_occ_dir, next_occ_train, _ = usage_info[i + 3]
                            if next_occ_train == activity.origin.train.id:
                                if len(usage_info) == i + 4:
                                    return when_can_I_start, should_end_after, last_end_time
                                else:
                                    next_occ_start, next_occ_end, next_occ_dir, next_occ_train, _ = usage_info[i + 4]
                            else:
                                next_occ_start, next_occ_end, next_occ_dir, next_occ_train, _ = usage_info[i + 3]
                    else:
                        next_occ_start, next_occ_end, next_occ_dir, next_occ_train, _ = usage_info[i + 2]

            if not next_occ_train == "D":
                if not at_station:
                    if next_occ_dir == determine_direction(activity, track):
                        time_to_compare_next_occ_start = when_can_I_start + get_minimum_headway(
                            EAG, activity.origin.train
                        )
                    else:
                        time_to_compare_next_occ_start = max(
                            when_can_I_start + min_activity_duration, should_end_after
                        ) + get_minimum_headway(EAG, activity.origin.train)
                else:
                    if track:
                        time_to_compare_next_occ_start = (
                            max(when_can_I_start + min_activity_duration, should_end_after)
                            + EAG.minimum_separation_time
                        )
                    else:
                        time_to_compare_next_occ_start = max(when_can_I_start + min_activity_duration, should_end_after)
            else:
                time_to_compare_next_occ_start = max(when_can_I_start + min_activity_duration, should_end_after)

            if not backward and time_to_compare_next_occ_start <= next_occ_start:
                if not at_station:
                    if next_occ_dir == determine_direction(activity, track):
                        last_end_time = next_occ_end - get_minimum_headway(EAG, activity.origin.train)
                    else:
                        last_end_time = next_occ_start - get_minimum_headway(EAG, activity.origin.train)
                else:
                    if track:
                        last_end_time = next_occ_start - EAG.minimum_separation_time
                    else:
                        last_end_time = next_occ_start

                if last_end_time >= should_end_after:
                    return when_can_I_start, should_end_after, last_end_time
                else:
                    continue
            else:
                continue
        else:
            return when_can_I_start, should_end_after, last_end_time

    raise ValueError("Did not found a new earliest start time on a resource")


def delay_train_and_retrack(
    EAG: EARailwayNetwork,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    section_track_change: int,
    station_track_change: int,
    verbose=0,
):
    """
    Reschedule a train to the earliest slot available.

    Args:
        EAG (EARailwayNetwork): Event-activity graph.
        X (dict): X variables for each activity.
        Y (dict): Y variables for each event.
        Z (dict): Z variables for each event.
        PHI (dict): PHI variables.
        train (Train): Train to reschedule.
        section_track_change (int): 0=all section tracks can be changed, 1=only at disruption, 2=no change
        station_track_change (int): 0=all station tracks can be changed, 1=only at disruption, 2=no change
        verbose (int): Verbosity level.

    Returns:
        (dict, dict, dict, dict): Updated X, Y, Z, PHI.
    """

    previous_end_time = None
    previous_chosen_activity = None
    previous_activity = None

    disrupted_sections = set()
    if EAG.disruption_scenario and EAG.disruption_scenario.section_tracks:
        for track in EAG.disruption_scenario.section_tracks:
            disrupted_sections.add((track.origin, track.destination))

    disrupted_tracks = set()
    disrupted_stations = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            disrupted_tracks.add(track)
            disrupted_stations.add(track.origin)
            disrupted_stations.add(track.destination)

    earliest_next_activity_time = 0.0
    track_usage = get_track_usage(EAG, X, Y, Z, train)
    junction_usage = get_junction_usage(EAG, X, Y, Z, train)

    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    this_train_activities = [
        a
        for a in EAG.get_ordered_activities_train(train)
        if a.activity_type in ("train running", "train waiting", "pass-through", "short-turning")
    ]
    this_train_activities = [a for a in this_train_activities if (a.id in X) and (X[a.id] >= 0.5)]

    start_time = Y[this_train_activities[0].origin.id]

    def get_similar_activities(activity):
        if activity.activity_type != "short-turning":
            agg_act = EAG.disagg_to_agg_activities[activity]
            similar_activities = [
                a
                for a in EAG.agg_to_disagg_activities[agg_act]
                if a.origin.station.id == activity.origin.station.id
                and a.destination.station.id == activity.destination.station.id
            ]
            if activity not in similar_activities:
                raise ValueError("activity missing")
            return similar_activities
        else:
            for act_list in EAG.similar_short_turning:
                if activity in act_list:
                    return act_list

    i = 0
    going_back = False
    last_end_times_steps = [1e16 for a in this_train_activities]
    chosen_tracks: list[NodeTrack | SectionTrack | None] = [None for _ in this_train_activities]
    chosen_start_times: dict[Any, float] = {}
    chosen_end_times: dict[int, float] = {}

    max_iter = 200
    nb_iter = 0

    while i < len(this_train_activities) and nb_iter < max_iter:
        short_turning = False
        nb_iter += 1
        activity = this_train_activities[i]
        if i > 0:
            previous_activity = this_train_activities[i - 1]
        if i + 2 < len(this_train_activities):
            next_activity = this_train_activities[i + 1]
            next_next_activity = this_train_activities[i + 2]
        else:
            next_activity = None
            next_next_activity = None

        if next_activity and next_activity.activity_type == "short-turning":
            short_turning = True
        if activity.activity_type == "short-turning":
            chosen_tracks[i] = chosen_tracks[i - 1]
            chosen_end_times[i] = chosen_end_times[i - 1]
            chosen_start_times[activity] = chosen_start_times[this_train_activities[i - 1]]
            last_end_times_steps[i] = last_end_times_steps[i - 1]
            i += 1
            continue
        if i - 1 >= 0 and this_train_activities[i - 1].activity_type == "short-turning":
            chosen_tracks[i] = chosen_tracks[i - 2]
            chosen_start_times[activity] = chosen_start_times[this_train_activities[i - 2]]
            chosen_end_times[i] = chosen_end_times[i - 2]
            last_end_times_steps[i] = last_end_times_steps[i - 2]
            i += 1
            continue

        start_times: dict[Union[None, NodeTrack, SectionTrack], float] = {}
        end_times: dict[Union[None, NodeTrack, SectionTrack], float] = {}
        last_end_times: dict[Union[None, NodeTrack, SectionTrack], float] = {}
        current_track = None
        at_station = activity.origin.station == activity.destination.station

        if verbose > 0:
            start_time = (
                earliest_next_activity_time if going_back else start_time if i == 0 else chosen_end_times[i - 1]
            )
            if at_station:
                logger.debug(
                    f"\n#{i}# At station {activity.origin.station.id} ({activity.activity_type})(start : {start_time}"
                    f"{' [BACK]' if going_back else ''})"
                )
            else:
                logger.debug(
                    f"\n#{i}# Between stations {activity.origin.station.id} and {activity.destination.station.id}#"
                    f" (start : {start_time}{' [BACK]' if going_back else ''})"
                )

        similar_activities = get_similar_activities(activity)
        for disagg_activity in similar_activities:
            Xplus[disagg_activity.id] = 0
            Zplus[disagg_activity.origin.id] = 0
            Zplus[disagg_activity.destination.id] = 0

            if at_station:
                track = disagg_activity.origin.node_track
                if (
                    station_track_change == 1 and activity.origin.station not in disrupted_stations
                ) or station_track_change == 2:
                    if track != activity.origin.node_track:
                        continue
                if (
                    disagg_activity.origin.node_track
                    and i > 0
                    and chosen_tracks[i - 1]
                    not in EAG.incoming_tracks[
                        (disagg_activity.origin.station.id, disagg_activity.origin.node_track.id)
                    ]
                ):
                    continue
                if (
                    disagg_activity.origin.station.id,
                    disagg_activity.origin.node_track.id,
                ) not in EAG.outgoing_tracks.keys():
                    continue
                if (
                    station_track_change == 1
                    and activity.origin.station in disrupted_stations
                    and next_activity
                    and next_activity.section_track not in disrupted_tracks
                ):
                    if (
                        next_activity.section_track_planned.id
                        not in EAG.outgoing_tracks[
                            (disagg_activity.origin.station.id, disagg_activity.origin.node_track.id)
                        ]
                    ):
                        continue
            else:
                track = disagg_activity.section_track
                if (
                    section_track_change == 1 and activity.section_track not in disrupted_tracks
                ) or section_track_change == 2:
                    if track != activity.section_track:
                        continue
                if (
                    i > 0
                    and previous_activity
                    and chosen_tracks[i - 1] is not None
                    and track.id
                    not in EAG.outgoing_tracks[(int(previous_activity.origin.station.id), int(chosen_tracks[i - 1].id))]  # type: ignore[union-attr]
                ):
                    continue
                if (
                    section_track_change == 1
                    and activity.section_track not in disrupted_tracks
                    and next_activity
                    and next_activity.origin.station not in disrupted_stations
                ):
                    if (
                        track.id
                        not in EAG.incoming_tracks[
                            (next_activity.origin.station.id, next_activity.origin.node_track_planned.id)
                        ]
                    ):
                        continue

            if at_station and disagg_activity.activity_type != activity.activity_type:
                continue

            if X[disagg_activity.id] > 0.5:
                if at_station:
                    current_track = disagg_activity.origin.node_track
                else:
                    current_track = disagg_activity.section_track

            if track in start_times:
                # We already looked at possibilities in this track
                continue

            min_stop_time = 0.0
            if short_turning:
                min_stop_time += EAG.short_turning
                if disagg_activity.activity_type == "train waiting":
                    min_stop_time += EAG.waiting_time
                if next_next_activity and next_next_activity.activity_type == "train waiting":
                    min_stop_time += EAG.waiting_time
                    min_stop_time = max(
                        min_stop_time,
                        disagg_activity.destination.scheduled_time
                        - disagg_activity.origin.scheduled_time
                        + EAG.short_turning
                        + EAG.waiting_time,
                    )
                else:
                    min_stop_time = max(
                        min_stop_time,
                        disagg_activity.destination.scheduled_time
                        - disagg_activity.origin.scheduled_time
                        + EAG.short_turning,
                    )
            else:
                if disagg_activity.activity_type == "train waiting":
                    min_stop_time += EAG.waiting_time

            # Compute the earliest slot on this track
            if i > 0:
                previous_activity_end_time = chosen_end_times[i - 1]
            else:
                previous_activity_end_time = EAG.start_time_window
            start_times[track], end_times[track], last_end_times[track] = get_earliest_start_based_on_track_occupancy(
                EAG,
                disagg_activity,
                Y[disagg_activity.origin.id],
                Y[disagg_activity.destination.id],
                track_usage,
                junction_usage,
                going_back,
                min_stop_time,
                previous_activity_end_time,
                earliest_next_activity_time,
                verbose - 1,
            )

        can_change_track = False
        if not at_station:
            if (section_track_change == 1 and activity.section_track in disrupted_tracks) or section_track_change == 0:
                can_change_track = True
        else:
            if (
                station_track_change == 1 and activity.origin.station in disrupted_stations
            ) or station_track_change == 0:
                can_change_track = True

        # Check if next activity is outside of the disruption
        restriced_tracks = []
        if (
            next_activity
            and section_track_change == 1
            and can_change_track
            and next_activity.origin.station not in disrupted_stations
        ):
            for t in end_times:
                if (
                    next_activity.section_track
                    in EAG.outgoing_tracks[(activity.origin.station.id, activity.origin.node_track.id)]
                ):
                    restriced_tracks.append(t)
        else:
            restriced_tracks = list(end_times.keys())

        chosen_track: NodeTrack | SectionTrack | None = None
        if can_change_track:
            if current_track:
                best_end_times = min([end_times[t] for t in restriced_tracks])
                bests_tracks = [t for t in restriced_tracks if abs(end_times[t] - best_end_times) < 1e-6]
                if any(t is None for t in bests_tracks):
                    raise ValueError("None detected in bests_tracks")
                if len(bests_tracks) == 1:
                    chosen_track = bests_tracks[0]
                else:
                    best_start_time = min(start_times[t] for t in bests_tracks)
                    bests_tracks_prime = [t for t in bests_tracks if abs(start_times[t] - best_start_time) < 1e-6]
                    if len(bests_tracks_prime) == 0:
                        raise ValueError(f"No tracks specified for activity {EAG.print_activity_info(activity)}")
                    else:
                        bests_tracks = bests_tracks_prime
                    if len(bests_tracks) == 1:
                        chosen_track = bests_tracks[0]
                    elif current_track in bests_tracks:
                        # check whether other tracks than current tracks can access other infrastructure
                        if at_station:
                            inaccessible_tracks_from_current_track = False
                            current_tracks = EAG.outgoing_tracks[
                                (disagg_activity.origin.station.id, disagg_activity.origin.node_track.id)
                            ]
                            for t in bests_tracks:
                                if t is None:
                                    raise ValueError("None detected in bests_tracks")
                                else:
                                    possible_tracks = EAG.outgoing_tracks[
                                        (int(disagg_activity.origin.station.id), int(t.id))
                                    ]
                                    if any(trk not in current_tracks for trk in possible_tracks):
                                        inaccessible_tracks_from_current_track = True
                                        break
                            if inaccessible_tracks_from_current_track:
                                # Select track with probability proportional to accessible tracks
                                weights = [
                                    len(EAG.outgoing_tracks[(int(disagg_activity.origin.station.id), int(t.id))])
                                    for t in bests_tracks
                                    if t is not None
                                ]
                                total = sum(weights)
                                probs = [w / total for w in weights]
                                chosen_track = random.choices(bests_tracks, weights=probs, k=1)[0]
                            else:
                                chosen_track = current_track
                        else:
                            chosen_track = current_track
                    else:
                        # Select track with probability proportional to accessible tracks
                        weights = [
                            len(EAG.outgoing_tracks[(int(disagg_activity.origin.station.id), int(t.id))])
                            for t in bests_tracks
                            if t is not None
                        ]
                        total = sum(weights)
                        probs = [w / total for w in weights]
                        chosen_track = random.choices(bests_tracks, weights=probs, k=1)[0]
            else:
                best_end_times = min([end_times[t] for t in restriced_tracks])
                bests_tracks = [t for t in restriced_tracks if abs(end_times[t] - best_end_times) < 1e-6]
                if len(bests_tracks) == 1:
                    chosen_track = bests_tracks[0]
                else:
                    best_start_time = min(start_times[t] for t in bests_tracks)
                    bests_tracks_prime = [t for t in bests_tracks if abs(start_times[t] - best_start_time) < 1e-6]
                    if len(bests_tracks_prime) == 0:
                        raise ValueError(f"No tracks specified for activity {EAG.print_activity_info(activity)}")
                    else:
                        bests_tracks = bests_tracks_prime
                    if len(bests_tracks) == 1:
                        chosen_track = bests_tracks[0]
                    else:
                        # Select track with probability proportional to accessible tracks
                        weights = [
                            len(EAG.outgoing_tracks[(int(disagg_activity.origin.station.id), int(t.id))])
                            for t in bests_tracks
                            if t is not None
                        ]
                        total = sum(weights)
                        probs = [w / total for w in weights]
                        chosen_track = random.choices(bests_tracks, weights=probs, k=1)[0]
        else:
            chosen_track = current_track

        if activity.origin.train.capacity > 0:
            max_delay = EAG.passenger_train_max_delay
        else:
            max_delay = EAG.freight_train_max_delay
        if (
            start_times[chosen_track] - activity.origin.scheduled_time > max_delay
            and activity.origin.node_type == "regular"
        ):
            return cancel_train_completely(EAG, X, Y, Z, PHI, train)

        # Check that we do not have a problem with the previous resource:
        if i > 0 and start_times[chosen_track] > last_end_times_steps[i - 1]:
            if verbose > 0:
                logger.debug(
                    f"-> Want to use {chosen_track} from {start_times[chosen_track]} to "
                    f"{end_times[chosen_track]}, but does not work with previous activity"
                )
                logger.debug(f"=> GOING BACK, {start_times[chosen_track]} > {last_end_times_steps[i-1]}")
            # We have problem with previous resource
            if this_train_activities[i - 2].activity_type == "short-turning":
                i -= 3
            elif (
                this_train_activities[i - 1].activity_type == "pass-through"
                and len(this_train_activities[i - 1].origin.station.node_tracks) == 0
            ):  # Need to go back two activities before to skip the junction without node track
                i -= 2
            else:
                i -= 1
            going_back = True
            earliest_next_activity_time = start_times[chosen_track]
            continue

        chosen_tracks[i] = chosen_track
        chosen_start_times[activity] = start_times[chosen_track]
        chosen_end_times[i] = end_times[chosen_track]
        last_end_times_steps[i] = last_end_times[chosen_track]
        if chosen_track:
            track_id = chosen_track.id
        else:
            track_id = None
        if verbose > 0:
            logger.debug(
                f"Iter completed: chosen track: {track_id}, start time: {chosen_start_times[activity]}, "
                f"end time: {chosen_end_times[i]}, last end time: {last_end_times_steps[i]}"
            )
        going_back = False
        i += 1

    if nb_iter == max_iter:
        logger.debug(f"Avoiding possible infinite loop in train {train.id}")
        return Xplus, Yplus, Zplus, PHIplus

    # Select the correct activities and update Xplus and Yplus
    for i, activity in enumerate(this_train_activities):
        similar_activities = get_similar_activities(activity)
        at_station = activity.origin.station == activity.destination.station
        if at_station:
            if activity.activity_type == "short-turning":
                chosen_track = chosen_tracks[i - 1]
            elif this_train_activities[i - 1].activity_type == "short-turning":
                chosen_track = chosen_tracks[i - 2]
            else:
                chosen_track = chosen_tracks[i]
            chosen_activity = next(
                disagg_activity
                for disagg_activity in similar_activities
                if (disagg_activity.origin.node_track == chosen_track)
                and (disagg_activity.activity_type == activity.activity_type)
            )
        else:
            found_activity = False
            for disagg_activity in similar_activities:
                if (
                    (disagg_activity.origin.node_track == chosen_tracks[i - 1])
                    and (disagg_activity.destination.node_track == chosen_tracks[i + 1])
                    and (disagg_activity.section_track == chosen_tracks[i])
                ):
                    found_activity = True
            if not found_activity:
                if chosen_tracks[i] is not None:
                    chose_tracks_i = chosen_tracks[i].id  # type: ignore[union-attr]
                else:
                    chose_tracks_i = None
                if chosen_tracks[i - 1] is not None:
                    chose_tracks_i_1 = chosen_tracks[i - 1].id  # type: ignore[union-attr]
                else:
                    chose_tracks_i_1 = None
                if chosen_tracks[i + 1] is not None:
                    chose_tracks_i1 = chosen_tracks[i + 1].id  # type: ignore[union-attr]
                else:
                    chose_tracks_i1 = None

                logger.debug(
                    f"Did not find disaggregated activity for activity {EAG.print_activity_info(activity)}, "
                    f"chosen tracks : {chose_tracks_i_1}, {chose_tracks_i}, {chose_tracks_i1}"
                )
                return X, Y, Z, PHI  # Return timetable without change
            chosen_activity = next(
                disagg_activity
                for disagg_activity in similar_activities
                if (disagg_activity.origin.node_track == chosen_tracks[i - 1])
                and (disagg_activity.destination.node_track == chosen_tracks[i + 1])
                and (disagg_activity.section_track == chosen_tracks[i])
            )

        for disagg_activity in similar_activities:
            if disagg_activity.activity_type == "short-turning":
                if this_train_activities[i - 1].activity_type == "train waiting":
                    start_time = max(chosen_start_times[activity] + EAG.waiting_time, activity.origin.scheduled_time)
                else:
                    start_time = max(chosen_start_times[activity], activity.origin.scheduled_time)
                end_time = max(start_time + EAG.short_turning, activity.destination.scheduled_time)
            elif i - 1 >= 0 and this_train_activities[i - 1].activity_type == "short-turning":
                if this_train_activities[i - 2].activity_type == "train waiting":
                    start_time_previous = max(
                        chosen_start_times[activity] + EAG.waiting_time,
                        this_train_activities[i - 1].origin.scheduled_time,
                    )
                else:
                    start_time_previous = max(
                        chosen_start_times[activity], this_train_activities[i - 1].origin.scheduled_time
                    )
                previous_end_time = max(
                    start_time_previous + EAG.short_turning, this_train_activities[i - 1].destination.scheduled_time
                )
                start_time = previous_end_time
                end_time = max(chosen_end_times[i], activity.destination.scheduled_time)

                EPS: float = 1e-6
                if end_time - start_time < EAG.waiting_time - EPS and activity.activity_type == "train waiting":
                    raise ValueError("Not enough time allocated to short-turn")

            elif i + 1 < len(this_train_activities) and this_train_activities[i + 1].activity_type == "short-turning":
                start_time = chosen_start_times[activity]
                end_time = max(chosen_start_times[activity], activity.destination.scheduled_time)
            else:
                start_time = chosen_start_times[activity]
                end_time = chosen_end_times[i]

            if disagg_activity == chosen_activity:
                Xplus[disagg_activity.id] = 1
                Yplus[disagg_activity.origin.id] = start_time
                Yplus[disagg_activity.destination.id] = end_time
            else:
                Xplus[disagg_activity.id] = 0
                Yplus[disagg_activity.origin.id] = start_time
                Yplus[disagg_activity.destination.id] = end_time

        if verbose > 0:
            logger.debug(
                f"Set new time ({chosen_activity.origin.station.id}-{chosen_activity.destination.station.id}) at "
                f"track {chosen_activity.section_track.id if chosen_activity.section_track else None}",
                chosen_activity.activity_type,
                chosen_activity.origin.train.id,
                Yplus[chosen_activity.origin.id],
                Yplus[chosen_activity.destination.id],
            )

        if i > 0:
            if Yplus[chosen_activity.origin.id] != previous_end_time:
                # Update previous end time
                if Yplus[chosen_activity.origin.id] <= last_end_times_steps[i] and previous_chosen_activity:
                    Yplus[previous_chosen_activity.destination.id] = Yplus[chosen_activity.origin.id]  # type: ignore[unreachable]
                    if verbose > 0:
                        logger.debug(
                            f"Reset new time ({previous_activity.origin.station.id}-"
                            f"{previous_activity.destination.station.id}) at "
                            f"track {previous_activity.section_track.id if previous_activity.section_track else None}",
                            previous_activity.activity_type,
                            previous_activity.origin.train.id,
                            Yplus[previous_chosen_activity.origin.id],
                            Yplus[previous_chosen_activity.destination.id],
                        )
                    if Yplus[previous_chosen_activity.destination.id] < Yplus[previous_chosen_activity.origin.id]:
                        raise ValueError(
                            "Invalid activity time",
                            Yplus[previous_chosen_activity.destination.id],
                            Yplus[previous_chosen_activity.origin.id],
                        )
                else:
                    raise ValueError("Problem with delay propagation")

        previous_end_time = Yplus[chosen_activity.destination.id]
        previous_chosen_activity = chosen_activity
        previous_activity = activity

    # Also update starting and ending activities
    for activity in EAG.A_train[train]:
        if activity.activity_type == "starting":
            Xplus[activity.id] = int(activity.destination.node_track == chosen_tracks[0])
        elif activity.activity_type == "ending":
            if not previous_chosen_activity:
                raise ValueError("No previous activity to ending activity")
            Xplus[activity.id] = int(activity.origin.node_track == chosen_tracks[-1])

    return Xplus, Yplus, Zplus, PHIplus


# --------------------------------------------------------------------------------
#                               Operators
# --------------------------------------------------------------------------------


def delay_operator(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    verbose: int = 0,
):
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=2, station_track_change=2, verbose=verbose
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus


def delay_track_operator(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    verbose: int = 0,
):
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)
    this_train_activities = [
        a
        for a in EAG.get_ordered_activities_train(train)
        if a.activity_type in ("train running", "train waiting", "pass-through")
    ]
    # Reduce this_train_activities to keep only activities in timetable
    this_train_activities = [a for a in this_train_activities if (a.id in Xplus) and (Xplus[a.id] >= 0.5)]
    if len(this_train_activities) == 0:
        return Xplus, Yplus, Zplus, PHIplus
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=0, station_track_change=0, verbose=verbose
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus


def delay_station_track_changes(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    verbose: int = 0,
):
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)
    # Reduce this_train_activities to keep only activities in timetable
    this_train_activities = [
        a
        for a in EAG.get_ordered_activities_train(train)
        if a.activity_type in ("train running", "train waiting", "pass-through")
    ]
    this_train_activities = [a for a in this_train_activities if (a.id in Xplus) and (Xplus[a.id] >= 0.5)]
    if len(this_train_activities) == 0:
        return Xplus, Yplus, Zplus, PHIplus
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=2, station_track_change=0, verbose=verbose
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus


def delay_disrupted_track_operator(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    verbose: int = 0,
):
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)
    this_train_activities = [
        a
        for a in EAG.get_ordered_activities_train(train)
        if a.activity_type in ("train running", "train waiting", "pass-through")
    ]
    # Reduce this_train_activities to keep only activities in timetable
    this_train_activities = [a for a in this_train_activities if (a.id in Xplus) and (Xplus[a.id] >= 0.5)]
    if len(this_train_activities) == 0:
        return Xplus, Yplus, Zplus, PHIplus
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG, Xplus, Yplus, Zplus, PHIplus, train, section_track_change=1, station_track_change=1, verbose=verbose
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus


def delay_mix_tmp(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    verbose: int = 0,
):
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)

    disrupted_sections = set()
    if EAG.disruption_scenario and EAG.disruption_scenario.section_tracks:
        for track in EAG.disruption_scenario.section_tracks:
            disrupted_sections.add((track.origin, track.destination))

    tracks = set()
    for track in EAG.section_tracks:
        if (track.origin, track.destination) in disrupted_sections or (
            track.destination,
            track.origin,
        ) in disrupted_sections:
            tracks.add(track)
    t_disruption = EAG.end_time_window
    for a in EAG.A_train[train]:
        if a.section_track_planned and a.section_track_planned in tracks:
            t_disruption = Y[a.origin.id]
            break

    pos = (t_disruption - EAG.start_time_window) / (EAG.end_time_window - EAG.start_time_window)
    epsilon = 0.1
    p1 = max(epsilon, 1 - pos)
    p2 = max(epsilon, pos)
    track_change = random.choices([1, 2], weights=[p1, p2], k=1)[0]

    logger.debug(train.id, "track change", track_change, t_disruption, p1, p2)
    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG,
        Xplus,
        Yplus,
        Zplus,
        PHIplus,
        train,
        section_track_change=track_change,
        station_track_change=track_change,
        verbose=verbose,
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus


def delay_mix(
    EAG: EARailwayNetwork,
    Xplusd: dict,
    Yplusd: dict,
    Zplusd: dict,
    PHIplusd: dict,
    X: dict,
    Y: dict,
    Z: dict,
    PHI: dict,
    train: Train,
    track_change: int,
    verbose: int = 0,
):
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)

    Xplus, Yplus, Zplus, PHIplus = delay_train_and_retrack(
        EAG,
        Xplus,
        Yplus,
        Zplus,
        PHIplus,
        train,
        section_track_change=track_change,
        station_track_change=track_change,
        verbose=verbose,
    )
    Xplus, Yplus, Zplus, PHIplus = cancel_if_too_delayed(EAG, Xplus, Yplus, Zplus, PHIplus, train, verbose=verbose)
    return Xplus, Yplus, Zplus, PHIplus
