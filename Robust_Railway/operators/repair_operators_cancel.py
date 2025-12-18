import logging

from ..event_activity_graph_multitracks import EARailwayNetwork, Event, Train
from .track_occupancy import determine_direction, get_minimum_headway, get_track_usage

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
#                               Tools
# -----------------------------------------------------------------------


def set_train_to_timetable(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train):
    """Reset the given train to its planned timetable"""
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    for activity in EAG.get_activities_per_train(train):
        Xplus[activity.id] = int(activity.in_timetable)
    for event in EAG.get_events_per_train(train):
        Yplus[event.id] = event.scheduled_time
        Zplus[event.id] = 0

    return Xplus, Yplus, Zplus, PHIplus


def cancel_train_at(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train, event: Event, verbose=0):
    """Cancel `train` at the last station before `event` with a shunting yard"""
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()

    train_events = EAG.get_ordered_events_train(train)
    train_events = [e for e in train_events if not e.aggregated and e.node_type == "regular"]
    reverse_train_events = train_events[::-1]  # reverse order

    # Find the last station with a shunting yard before the event
    ordered_stations = []
    for e in train_events:
        if e.station not in ordered_stations:
            ordered_stations.append(e.station)
    passed_cancel_event = False
    cancel_at = None
    for s in ordered_stations:
        if s.shunting_yard_capacity and not passed_cancel_event:
            cancel_at = s
        if s == event.station:
            passed_cancel_event = True
    if not cancel_at:
        logger.debug(f"Cancelling train {train.id} completely")
        return cancel_train_completely(EAG, X, Y, Z, PHI, train)

    reached_event = False
    found_shunting_yard = False
    for e in reverse_train_events:
        reached_event |= e == event
        if reached_event:
            found_shunting_yard |= e.station.shunting_yard_capacity

        # Remove all activities after current event
        for a in EAG.A_minus[e]:
            Xplus[a.id] = 0
            Yplus[a.destination.id] = a.destination.scheduled_time
        if found_shunting_yard and any(X[a.id] for a in EAG.A_plus[e]):
            if verbose > 0:
                logger.debug(f"[Train cancelled at {EAG.print_event_info(e)}]")
            Zplus[e.id] = 1
            break

    # Cancel all non-regular activities (rerouting, short-turning)
    for a in EAG.A_train[train]:
        if a.origin.node_type in ["rerouting", "short-turning"] or a.destination.node_type in [
            "rerouting",
            "short-turning",
        ]:
            Xplus[a.id] = 0

    return Xplus, Yplus, Zplus, PHIplus


# -----------------------------------------------------------------------
#                               Operators
# -----------------------------------------------------------------------


def cancel_train_completely(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train, verbose: int = 0
):
    """Cancel the train entirely and reset all activities/events"""
    Xplus, Yplus, Zplus, PHIplus = X.copy(), Y.copy(), Z.copy(), PHI.copy()
    started = False

    for activity in EAG.A_train[train]:
        Yplus[activity.origin.id] = activity.origin.scheduled_time
        Yplus[activity.destination.id] = activity.destination.scheduled_time
        if activity.activity_type == "starting" and not started:
            Xplus[activity.id] = 1
            Zplus[activity.destination.id] = 1
            started = True
        else:
            Xplus[activity.id] = 0
            Zplus[activity.destination.id] = 0

    return Xplus, Yplus, Zplus, PHIplus


def cancel_train_at_first_conflict(
    EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train, verbose=0
):
    """Cancel train at the first detected conflict (track occupancy or disruption)"""
    track_usage = get_track_usage(EAG, X, Y, Z, train)
    event = None

    for activity in EAG.get_ordered_activities_train(train):
        if activity.aggregated or X[activity.id] < 0.5:
            continue
        if activity.activity_type not in ("train running", "train waiting", "pass-through"):
            continue

        at_station = activity.origin.station == activity.destination.station
        track = activity.origin.node_track if at_station else activity.section_track
        start_time = Y[activity.origin.id]
        end_time = Y[activity.destination.id]
        occupancy = track_usage[track]

        for occ_start, occ_end, occ_dir, occ_train, occ_headway in occupancy:
            if occ_train == train.id:
                continue
            if (occ_train == "D") and (occ_end >= start_time) and (occ_start < end_time):
                # Disruption
                event = activity.origin
                if verbose > 0:
                    if event:
                        logger.debug(
                            f"Cancelling at {EAG.print_event_info(event)} because of disruption "
                            f"[{occ_start}, {occ_end}]"
                        )
                    else:
                        logger.debug(f"Cancelling at {None} because of disruption [{occ_start}, {occ_end}]")
                break
            if at_station:
                if occ_end + EAG.minimum_separation_time <= start_time:
                    continue
                if occ_start >= end_time + (EAG.minimum_separation_time if occ_train != "D" else 0):
                    break
                event = activity.origin
                if verbose > 0:
                    logger.debug(
                        f"Cancelling at {EAG.print_event_info(event)} because of conflict with train"
                        f"{occ_train} [{occ_start}, {occ_end}]"
                    )
                break
            else:
                direction = determine_direction(activity, track)
                minimum_headway = get_minimum_headway(EAG, train)
                if occ_dir == direction:
                    if (occ_start <= start_time - minimum_headway) and (occ_end <= end_time - minimum_headway):
                        continue
                    if (occ_start - occ_headway >= start_time) and (occ_end - occ_headway >= end_time):
                        break
                    if verbose > 0:
                        if event:
                            logger.debug(
                                f"Cancelling at {EAG.print_event_info(event)} because of conflict with train "
                                f"{occ_train} (sd) [{occ_start}, {occ_end}]"
                            )
                        else:
                            logger.debug(
                                f"Cancelling at {None} because of conflict with train "
                                f"{occ_train} (sd) [{occ_start}, {occ_end}]"
                            )

                    event = activity.origin
                    break
                else:
                    if occ_end <= start_time - minimum_headway:
                        continue
                    if occ_start - occ_headway >= end_time:
                        break
                    event = activity.origin
                    if verbose > 0:
                        if event:
                            logger.debug(
                                f"Cancelling at {EAG.print_event_info(event)} because of conflict with train "
                                f"{occ_train} (od) [{occ_start}, {occ_end}]"
                            )
                        else:
                            logger.debug(
                                f"Cancelling at {None} because of conflict with train "
                                f"{occ_train} (od) [{occ_start}, {occ_end}]"
                            )
                    break
        if event is not None:
            break

    if event is not None:
        return cancel_train_at(EAG, X, Y, Z, PHI, train, event, verbose)
    else:
        # No conflict detected
        return X, Y, Z, PHI


def cancel_if_too_delayed(EAG: EARailwayNetwork, X: dict, Y: dict, Z: dict, PHI: dict, train: Train, verbose=0):
    """Cancel train if it is delayed beyond the allowed threshold"""
    previous_activity = None

    if train.capacity > 0:
        max_delay = EAG.passenger_train_max_delay
    else:
        max_delay = EAG.freight_train_max_delay

    for activity in EAG.get_ordered_activities_train(train):
        if activity.aggregated or X[activity.id] < 0.5 or activity.origin.node_type != "regular":
            continue

        if float(Y[activity.origin.id]) > float(activity.origin.scheduled_time) + max_delay:
            if verbose > 0:
                logger.debug(
                    f"Cancelling train {train.id} at station {activity.origin.station.id} (origin, "
                    f"{activity.origin.event_type}) because it is delayed for more than {max_delay} minutes"
                )
            if activity.activity_type == "starting":
                logger.debug(f"Cancelling train {train.id} completely")
                return cancel_train_completely(EAG, X, Y, Z, PHI, train)
            elif previous_activity is not None:
                logger.debug(  # type: ignore[unreachable]
                    f"Cancelling train {train.id} at station {previous_activity.destination.station.id} "
                    f"(destination, {previous_activity.destination.event_type}) "
                    f"because it is delayed for more than {max_delay} minutes"
                )
                return cancel_train_at(EAG, X, Y, Z, PHI, train, previous_activity.destination, verbose)
            else:
                logger.debug(f"Cancelling train {train.id} completely")
                return cancel_train_completely(EAG, X, Y, Z, PHI, train)

        elif Y[activity.destination.id] > activity.destination.scheduled_time + max_delay:
            if verbose > 0:
                logger.debug(
                    f"Cancelling train {train.id} at station {activity.destination.station.id} "
                    f"(destination, {activity.origin.event_type})"
                    f"because it is delayed for more than {max_delay} minutes"
                )
            if activity.activity_type == "starting":
                logger.debug(f"Cancelling train {train.id} completely")
                return cancel_train_completely(EAG, X, Y, Z, PHI, train)
            else:
                logger.debug(
                    f"Cancelling train {train.id} at station {activity.origin.station.id} "
                    f"(origin, {activity.origin.event_type})"
                    f"because it is delayed for more than {max_delay} minutes"
                )
                return cancel_train_at(EAG, X, Y, Z, PHI, train, activity.origin, verbose)
        previous_activity = activity

    if verbose > 1:
        logger.debug(f"All delays below {max_delay} minutes")
    return X, Y, Z, PHI


def cancel_partially_operator(
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
    """Set train to timetable, then cancel at first conflict"""
    Xplus, Yplus, Zplus, PHIplus = set_train_to_timetable(EAG, Xplusd, Yplusd, Zplusd, PHIplusd, train)
    Xplus, Yplus, Zplus, PHIplus = cancel_train_at_first_conflict(EAG, Xplus, Yplus, Zplus, PHIplus, train)
    return Xplus, Yplus, Zplus, PHIplus
