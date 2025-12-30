import logging
from collections import defaultdict
from typing import Any, Optional, Tuple, Union, cast

logger = logging.getLogger(__name__)


class Train:
    def __init__(
        self,
        train_id: int,
        capacity: int,
        train_path_nodes: Optional[list[Any]] = None,
        begin_at: Optional[int] = None,
        end_at: Optional[int] = None,
        code: Optional[str] = None,
        visited_stations: Optional[dict] = None,
    ):
        self.id = train_id
        self.capacity = capacity
        self.train_path_nodes = train_path_nodes
        self.begin_at = begin_at
        self.end_at = end_at
        self.code = code
        self.visited_stations = visited_stations


class Bus:
    def __init__(self, bus_id: int, capacity: int):
        self.id = bus_id
        self.capacity = capacity


class Station:
    def __init__(
        self, station_id: int, code: str, node_tracks: list[int], shunting_yard_capacity: bool, junction: Optional[bool]
    ):
        self.id = station_id
        self.code = code
        self.node_tracks = [NodeTrack(nt) for nt in node_tracks]
        self.shunting_yard_capacity = shunting_yard_capacity
        self.junction = junction


class NodeTrack:
    def __init__(self, node_track_id: int):
        self.id = node_track_id


class SectionTrack:
    def __init__(
        self,
        track_id: str,
        origin: Station,
        destination: Station,
        distance: float,
        travel_time: dict,
        road: Optional[bool] = False,
    ):
        self.id = track_id
        self.origin = origin
        self.destination = destination
        self.distance = distance
        self.travel_time = travel_time  # in minutes
        self.road = road  # True if it is a section track for buses


class PassengerGroup:
    def __init__(
        self, group_id: int, origin: Station, destination: Station, time: float, num_passengers: int, priority: int
    ):
        self.id = group_id
        self.origin = origin
        self.destination = destination
        self.time = time  # desired start time
        self.num_passengers = num_passengers
        self.priority = priority


class Event:
    def __init__(
        self,
        id: int,
        event_type: str,
        scheduled_time: float,
        station: Station,
        node_track: Optional[NodeTrack],
        node_track_planned: Optional[NodeTrack],
        train: Union[Train, Bus],
        node_type: str,
        aggregated: bool,
        in_timetable: bool,
        train_path_node_id: Optional[int] = None,
    ):
        self.id = id
        self.event_type = event_type  # departure or arrival event
        self.scheduled_time = scheduled_time
        self.station = station
        self.node_track = node_track
        self.node_track_planned = node_track_planned
        self.train = train
        self.node_type = node_type  # regular, rerouting, short-turning, emergency, passenger origin, or
        # passenger destination, train origin, train destination
        self.aggregated = aggregated  # if True, used for the passenger routing graph
        self.in_timetable = (
            in_timetable  # True if this activity is part of the original timetable. False for aggregated events.
        )
        self.train_path_node_id = train_path_node_id


class Activity:
    def __init__(
        self,
        id: int,
        origin: Event,
        destination: Event,
        passenger_group: Optional[PassengerGroup],  # only for access, egress, and penalty activities
        activity_type: str,
        section_track: Optional[SectionTrack],
        aggregated: bool,
        in_timetable: bool,
        intermediate_stations: Optional[list[Station]] = None,
        section_track_planned: Optional[SectionTrack] = None,
    ):
        self.id = id
        self.origin = origin
        self.destination = destination
        self.activity_type = activity_type
        self.passenger_group = passenger_group
        self.section_track = section_track
        self.section_track_planned = section_track_planned
        self.aggregated = aggregated
        self.intermediate_stations = intermediate_stations  # list of stations between origin and destination
        self.in_timetable = (
            in_timetable  # True if this activity is part of the original timetable. False for aggregated events.
        )


class Disruption:
    def __init__(
        self,
        start_time: int,
        end_time: int,
        node_tracks: Optional[list[NodeTrack]] = None,
        section_tracks: Optional[list[SectionTrack]] = None,
    ):

        if not node_tracks and not section_tracks:
            raise ValueError("No node track or section track specified for the disruption scenario")
        self.start_time = start_time
        self.end_time = end_time
        self.node_tracks = node_tracks
        self.section_tracks = section_tracks


class EARailwayNetwork:
    def __init__(self, start_time_window: int, end_time_window: int, **kwargs: dict):
        self.trains: list[Train] = []
        self.buses: list[Bus] = []
        self.events: list[Event] = []
        self.events_ids: list[int] = []
        self.activities: list[Activity] = []
        self.activities_ids: list[int] = []
        self.stations: list[Station] = []
        self.section_tracks: list[SectionTrack] = []
        self.passengers_groups: list[PassengerGroup] = []
        self.start_time_window = start_time_window
        self.end_time_window = end_time_window
        self.time_horizon = end_time_window - start_time_window
        self.disruption_scenario: Optional[Disruption] = None
        self.incoming_tracks: defaultdict[Tuple[int, int], list] = defaultdict(list)
        self.outgoing_tracks: defaultdict[Tuple[int, int], list] = defaultdict(list)

        # Parameters
        self.waiting_time: float = cast(float, kwargs["waiting_time"])
        self.short_turning: float = cast(float, kwargs["short_turning"])
        self.cost_per_km = kwargs["cost_per_km"]
        self.beta_1 = kwargs["beta_1"]
        self.beta_2 = kwargs["beta_2"]
        self.beta_3 = kwargs["beta_3"]
        self.beta_4 = kwargs["beta_4"]
        self.penalty_cost = kwargs["penalty_cost"]
        self.km_cost_emergency_bus: float = cast(float, kwargs["km_cost_emergency_bus"])
        self.bus_capacity: int = cast(int, kwargs["bus_capacity"])
        self.minimum_transfer_time: int = cast(int, kwargs["minimum_transfer_time"])
        self.maximum_transfer_time: int = cast(int, kwargs["maximum_transfer_time"])
        self.minimum_headway_passenger_trains = cast(float, kwargs["minimum_headway_passenger_trains"])
        self.minimum_headway_freight_trains = cast(float, kwargs["minimum_headway_freight_trains"])
        self.minimum_separation_time: int = cast(int, kwargs["minimum_separation_time"])
        self.passenger_train_max_delay: int = cast(int, kwargs["passenger_train_max_delay"])
        self.freight_train_max_delay: int = cast(int, kwargs["freight_train_max_delay"])
        self.delta_1: float = cast(float, kwargs["delta_1"])
        self.delta_2: float = cast(float, kwargs["delta_2"])
        self.delta_3: float = cast(float, kwargs["delta_3"])
        self.delta_4: float = cast(float, kwargs["delta_4"])
        self.delta_5: float = cast(float, kwargs["delta_5"])
        self.delta_6: float = cast(float, kwargs["delta_6"])
        self.base_date = kwargs["base_date"]
        self.time_extra = kwargs["time_extra"]

        # Preprocessing info
        self.A_train_running_similar: defaultdict[Tuple[Station, Station, int, Train], list[Activity]] = defaultdict(
            list
        )
        self.agg_to_disagg_events: dict[Event, list[Event]] = {}
        self.disagg_to_agg_events: dict[Event, Event] = {}
        self.agg_to_disagg_activities: dict[Activity, list[Activity]] = {}
        self.disagg_to_agg_activities: dict[Activity, Activity] = {}
        self.nb_pass_through_activities: int = 0
        self.nb_waiting_activities: int = 0
        self.A_plus: defaultdict[Event, list[Activity]] = defaultdict(list)
        self.A_minus: defaultdict[Event, list[Activity]] = defaultdict(list)
        self.A_minus_agg: defaultdict[Event, list[Activity]] = defaultdict(list)
        self.A_plus_agg: defaultdict[Event, list[Activity]] = defaultdict(list)
        self.events_with_different_node_tracks: list[list[Event]] = []
        self.regular_aggregated_events: list[Event] = []
        self.regular_rerouting_turning_aggregated_events: list[Event] = []
        self.regular_disaggregated_events: list[Event] = []
        self.regular_rerouting_turning_events: list[Event] = []
        self.categorized_activities: dict[str, list[Activity]] = {}
        self.grouped_activities: dict[str, list[Activity]] = {}
        self.starting_activities_dict: dict[Train, list[Activity]] = {}
        self.A_waiting_plus: dict[Event, list[Activity]] = {}
        self.A_waiting_minus: dict[Event, list[Activity]] = {}
        self.arrival_time_train_end: dict[Union[Train, Bus], float] = {}
        self.A_waiting_pass_through_dict: defaultdict[Tuple[Station, NodeTrack], list[Activity]] = defaultdict(list)
        self.train_running_dict: defaultdict[SectionTrack, list[Activity]] = defaultdict(list)
        self.A_access: dict[PassengerGroup, list[Activity]] = {}
        self.A_egress: dict[PassengerGroup, list[Activity]] = {}
        self.code_to_id: dict[str, int] = {}
        self.A_train: defaultdict[Train, list[Activity]] = defaultdict(list)
        self.A_train_aggregated: defaultdict[Train, list[Activity]] = defaultdict(list)
        self.E_train: defaultdict[Train, list[Event]] = defaultdict(list)
        self.E_train_aggregated: defaultdict[Train, list[Event]] = defaultdict(list)
        self.id_to_code: dict = {}
        self.all_paths_dict: dict = {}
        self.similar_short_turning: list = []

    def add_preprocessing_info(self, preprocess):
        self.categorized_activities = preprocess["categorized_activities"]
        self.grouped_activities = preprocess["grouped_activities"]
        self.regular_aggregated_events = preprocess["regular_aggregated_events"]
        self.regular_rerouting_turning_aggregated_events = preprocess["regular_rerouting_turning_aggregated_events"]
        self.regular_rerouting_turning_events = preprocess["regular_rerouting_turning_events"]
        self.starting_activities_dict = preprocess["starting_activities_dict"]
        self.A_waiting_plus = preprocess["A_waiting_plus"]
        self.A_waiting_minus = preprocess["A_waiting_minus"]
        self.regular_disaggregated_events = preprocess["regular_disaggregated_events"]
        self.arrival_time_train_end = preprocess["arrival_time_train_end"]
        self.A_waiting_pass_through_dict = preprocess["A_waiting_pass_through_dict"]
        self.train_running_dict = preprocess["train_running_dict"]
        self.A_access = preprocess["A_access"]
        self.A_egress = preprocess["A_egress"]

    def add_disruption_scenario(self, disruption_scenario: Disruption):
        if not isinstance(disruption_scenario, Disruption):
            raise TypeError("Expected a Disruption object")
        if not self.disruption_scenario:
            self.disruption_scenario = disruption_scenario
        else:
            raise ValueError("At most one disruption scenario can be considered")

    def add_train(self, train: Train):
        if not isinstance(train, Train):
            raise TypeError("Expected a Train object")
        self.trains.append(train)

    def add_bus(self, bus: Bus):
        if not isinstance(bus, Bus):
            raise TypeError("Expected a Bus object")
        self.buses.append(bus)

    def add_station(self, station: Station):
        if not isinstance(station, Station):
            raise TypeError("Expected a Station object")
        self.stations.append(station)

    def add_section_track(self, section_track: SectionTrack):
        if not isinstance(section_track, SectionTrack):
            raise TypeError("Expected a SectionTrack object")
        already_existent = False
        origin_station = section_track.origin
        destination_station = section_track.destination
        for existent_track in self.section_tracks:
            if existent_track.id == section_track.id and existent_track.road == section_track.road:
                if (existent_track.origin == origin_station and existent_track.destination == destination_station) or (
                    existent_track.destination == origin_station and existent_track.origin == destination_station
                ):
                    already_existent = True
                else:
                    raise ValueError(f"Duplicate section track id {section_track.id} between different station pairs")
        if not already_existent:
            self.section_tracks.append(section_track)
        else:
            logger.debug(
                f"An identical link between the station {origin_station.id} and station "
                f"{destination_station.id} already exists."
            )

    def add_event(self, event: Event):
        if not isinstance(event, Event):
            raise TypeError("Expected an Event object")
        if event.id in self.events_ids:
            raise ValueError("Event id already used")
        self.events.append(event)
        self.events_ids.append(int(event.id))

    def add_passengers_group(self, passengers_group: PassengerGroup):
        if not isinstance(passengers_group, PassengerGroup):
            raise TypeError("Expected a PassengerGroup object")
        self.passengers_groups.append(passengers_group)

    def add_activities(self, activity: Union[Activity, list[Activity]]):

        if isinstance(activity, list) and all(isinstance(a, Activity) for a in activity):
            a_ids = [int(a.id) for a in activity]
            if len(set(a_ids) & set(self.activities_ids)) > 0:
                raise ValueError("At least one activity id already used")
            self.activities.extend(activity)
            self.activities_ids.extend(a_ids)
        elif isinstance(activity, Activity):
            if activity.id in self.activities_ids:
                raise ValueError("Activity id already used")
            self.activities.append(activity)
            self.activities_ids.append(int(activity.id))
        else:
            raise TypeError("Expected an Activity or a list[Activity] object")

    def add_A_train_item(self, key: Train, value: Activity):
        if isinstance(key, Train) and isinstance(value, Activity):
            self.A_train[key].append(value)
        else:
            raise TypeError("Expected a Train key and an Activity value")

    def add_A_train_aggregated_item(self, key: Train, value: Activity):
        if isinstance(key, Train) and isinstance(value, Activity):
            self.A_train_aggregated[key].append(value)
        else:
            raise TypeError("Expected a Train key and an Activity value")

    def add_E_train_item(self, key: Train, value: Event):
        if isinstance(key, Train) and isinstance(value, Event):
            self.E_train[key].append(value)
        else:
            raise TypeError("Expected a Train key and an Event value")

    def add_E_train_aggregated_item(self, key: Train, value: Event):
        if isinstance(key, Train) and isinstance(value, Event):
            self.E_train_aggregated[key].append(value)
        else:
            raise TypeError("Expected a Train key and an Event value")

    def add_events_with_different_node_tracks(self, event_lst: list[Event]):
        if isinstance(event_lst, list) and all(isinstance(e, Event) for e in event_lst):
            self.events_with_different_node_tracks.append(event_lst)
        else:
            raise TypeError("Expected a list[Event] object")

    def add_A_train_running_similar_item(self, key: Tuple[Station, Station, int, Train], value: list[Activity]):
        if isinstance(key, tuple) and isinstance(value, list):
            self.A_train_running_similar[key] = value
        else:
            raise TypeError("Expected a Tuple[Station, Station, int, Train] key and a list[Activity] value")

    def add_A_plus_item(self, key: Event, value: Activity | list[Activity]):
        if isinstance(key, Event) and isinstance(value, list) and all(isinstance(a, Activity) for a in value):
            if not key.aggregated:
                self.A_plus[key] = value
            else:
                raise TypeError("Expected a disaggregated Event")
        elif isinstance(key, Event) and isinstance(value, Activity):
            if not key.aggregated:
                self.A_plus[key].append(value)
            else:
                raise TypeError("Expected a disaggregated Event")
        else:
            raise TypeError("Expected a Event key and an Activity or a list[Activity] value")

    def add_A_minus_item(self, key: Event, value: Activity | list[Activity]):
        if isinstance(key, Event) and isinstance(value, list) and all(isinstance(a, Activity) for a in value):
            if not key.aggregated:
                self.A_minus[key] = value
            else:
                raise TypeError("Expected a disaggregated Event")
        elif isinstance(key, Event) and isinstance(value, Activity):
            if not key.aggregated:
                self.A_minus[key].append(value)
            else:
                raise TypeError("Expected a disaggregated Event")
        else:
            raise TypeError("Expected a Event key and an Activity or a list[Activity] value")

    def add_A_plus_agg_item(self, key: Event, value: Activity | list[Activity]):
        if isinstance(key, Event) and isinstance(value, list) and all(isinstance(a, Activity) for a in value):
            if key.aggregated:
                self.A_plus_agg[key] = value
            else:
                raise TypeError("Expected an aggregated Event")
        elif isinstance(key, Event) and isinstance(value, Activity):
            if key.aggregated:
                self.A_plus_agg[key].append(value)
            else:
                raise TypeError("Expected an aggregated Event")
        else:
            raise TypeError("Expected a Event key and an Activity or a list[Activity] value")

    def add_A_minus_agg_item(self, key: Event, value: Activity | list[Activity]):
        if isinstance(key, Event) and isinstance(value, list) and all(isinstance(a, Activity) for a in value):
            if key.aggregated:
                self.A_minus_agg[key] = value
            else:
                raise TypeError("Expected an aggregated Event")
        elif isinstance(key, Event) and isinstance(value, Activity):
            if key.aggregated:
                self.A_minus_agg[key].append(value)
            else:
                raise TypeError("Expected an aggregated Event")
        else:
            raise TypeError("Expected a Event key and an Activity or a list[Activity] value")

    def add_agg_to_disagg_events_item(self, key: Event, value: list[Event]):
        if isinstance(key, Event) and isinstance(value, list) and all(isinstance(e, Event) for e in value):
            if key.aggregated and all(not e.aggregated for e in value):
                self.agg_to_disagg_events[key] = value
            else:
                raise TypeError("Expected an aggregated key Event and disaggregated value Events")
        else:
            raise TypeError("Expected an Event key and an list[Event] value")

    def add_disagg_to_agg_events_item(self, key: Event, value: Event):
        if isinstance(key, Event) and isinstance(value, Event):
            if not key.aggregated and value.aggregated:
                self.disagg_to_agg_events[key] = value
            else:
                raise TypeError("Expected a disaggregated key Event and an aggregated value Event")
        else:
            raise TypeError("Expected an Event key and an Event value")

    def add_agg_to_disagg_activities_item(self, key: Activity, value: list[Activity]):
        if isinstance(key, Activity) and isinstance(value, list) and all(isinstance(a, Activity) for a in value):
            if key.aggregated and all(not a.aggregated for a in value):
                self.agg_to_disagg_activities[key] = value
            else:
                raise TypeError("Expected an aggregated key Activity and disaggregated value Activities")
        else:
            raise TypeError("Expected an Activity key and an list[Activity] value")

    def add_disagg_to_agg_activities_item(self, key: Activity, value: Activity):
        if isinstance(key, Activity) and isinstance(value, Activity):
            if not key.aggregated and value.aggregated:
                self.disagg_to_agg_activities[key] = value
            else:
                raise TypeError("Expected a disaggregated key Activity and an aggregated value Activity")
        else:
            raise TypeError("Expected an Activity key and an Activity value")

    def add_incoming_outcoming_tracks(self, incoming_tracks, outgoing_tracks):
        self.incoming_tracks = incoming_tracks
        self.outgoing_tracks = outgoing_tracks

    def get_station_by_id(self, station_id: int):
        """Returns the station object with the given station_id, or None if not found."""
        return next((station for station in self.stations if int(station.id) == int(station_id)), None)

    def get_train_by_id(self, train_id: int):
        """Returns the train object with the given train_id, or None if not found."""
        return next((train for train in self.trains if int(train.id) == int(train_id)), None)

    def get_event_by_id(self, event_id: int):
        """Returns the event object with the given event_id, or None if not found."""
        return next((event for event in self.events if int(event.id) == int(event_id)), None)

    def get_event_by_attributes(
        self,
        event_type: str,
        station: Station,
        node_track: Optional[NodeTrack],
        train: Train | Bus,
        node_type: str,
        aggregated: bool,
    ):
        if node_track:
            return next(
                (
                    event
                    for event in self.events
                    if (
                        event.event_type == event_type
                        and event.station == station
                        and event.node_track == node_track
                        and event.train == train
                        and event.node_type == node_type
                        and event.aggregated == aggregated
                    )
                ),
                None,
            )
        else:
            return next(
                (
                    event
                    for event in self.events
                    if (
                        event.event_type == event_type
                        and event.station == station
                        and event.train == train
                        and event.node_type == node_type
                        and event.aggregated == aggregated
                    )
                ),
                None,
            )

    def get_activity_by_id(self, activity_id: Union[int, list[int]]):
        """Returns the activity object with the given activity_id, or None if not found."""
        if isinstance(activity_id, list):
            acts_objs = []
            for a in activity_id:
                acts_objs.append(next((a_ for a_ in self.activities if int(a_.id) == int(a)), None))
            return acts_objs
        else:
            return next((activity for activity in self.activities if int(activity.id) == int(activity_id)), None)

    def get_group_by_id(self, group_id: int):
        """Returns the passenger group object with the given group_id, or None if not found."""
        return next((group for group in self.passengers_groups if int(group.id) == int(group_id)), None)

    def get_section_track_by_id(self, track_id: str):
        """Returns the track object with the given track_id, or None if not found."""
        return next((track for track in self.section_tracks if str(track.id) == str(track_id)), None)

    def get_section_tracks(self, origin, destination, road):
        tracks = []
        for t in self.section_tracks:
            if (t.origin == origin and t.destination == destination and t.road == road) or (
                t.destination == origin and t.origin == destination and t.road == road
            ):
                tracks.append(t)
        return tracks

    def get_node_track_by_id(self, node_track_id: int, station: Station):
        """Returns the node track object with the given node_track_id, or None if not found."""
        return next(
            (node_track for node_track in station.node_tracks if int(node_track.id) == int(node_track_id)), None
        )

    def get_tracks_between(self, origin: int, destination: int) -> list:
        """Returns a list of section tracks connecting the given origin and destination stations."""
        return [
            track
            for track in self.section_tracks
            if (int(track.origin.id) == int(origin) and int(track.destination.id) == int(destination))
            or (int(track.origin.id) == int(destination) and int(track.destination.id) == int(origin))
        ]

    def get_ordered_events_train(self, train: Train) -> list:
        return sorted(
            [event for event in self.events if event.train == train and event.scheduled_time],
            key=lambda event: event.scheduled_time,
        )

    def print_event_info(self, event: Event):
        if event.station:
            station = event.station.id
        else:
            station = None
        if event.node_track:
            node_track = event.node_track.id
        else:
            node_track = None
        if event.train:
            train = event.train.id
        else:
            train = None
        return (
            f"event id: {event.id}, event type {event.event_type}, station {station}, station is a junction? "
            f"{event.station.junction}, node track {node_track}, train {train}, scheduled_time "
            f"{event.scheduled_time}, node type {event.node_type}, aggregated? {event.aggregated}, "
            f"in timetable? {event.in_timetable}"
        )

    def print_activity_info(self, activity: Activity):
        if activity.passenger_group:
            group = activity.passenger_group.id
        else:
            group = None
        if activity.section_track:
            track = activity.section_track.id
        else:
            track = None

        if activity.intermediate_stations:
            inter_stations = []
            for station in activity.intermediate_stations:
                inter_stations.append(station.id)
        else:
            inter_stations = None

        return (
            f"activity id: {activity.id}, activity origin [{self.print_event_info(activity.origin)}], "
            f"activity destination [{self.print_event_info(activity.destination)}], passenger group {group}, "
            f"activity_type: {activity.activity_type}, section track {track}, aggreagted? {activity.aggregated}, "
            f"intermediate stations {inter_stations}, in timetable? {activity.in_timetable}"
        )

    def get_events_per_train(self, train: Train) -> list:
        """Returns a list of events for a specific train."""
        return [event for event in self.events if event.train == train]

    def get_agg_events_per_train(self, train: Train) -> list:
        """Returns a list of events for a specific train."""
        return [event for event in self.events if event.train == train and event.aggregated]

    def get_activities_per_train(self, train: Train) -> list:
        """Returns a list of activities for a specific train."""
        return [activity for activity in self.activities if activity.origin.train == train]

    def get_ordered_activities_train(self, train: Train) -> list:
        activity_priority = {"pass-through": 0, "train waiting": 1, "short-turning": 2, "train running": 3, "ending": 4}
        return sorted(
            [
                activity
                for activity in self.activities
                if activity.origin.train == train
                and activity.origin.scheduled_time
                and activity.destination.scheduled_time
            ],
            key=lambda activity: (
                round(activity.origin.scheduled_time, 6),
                activity_priority.get(activity.activity_type, 99),
            ),
        )

    def get_ordered_disagg_activities_train(self, train: Train) -> list:
        activity_priority = {"pass-through": 0, "train waiting": 1, "short-turning": 2, "train running": 3, "ending": 4}
        return sorted(
            [
                activity
                for activity in self.activities
                if not activity.aggregated
                and activity.origin.train == train
                and activity.origin.scheduled_time
                and activity.destination.scheduled_time
            ],
            key=lambda activity: (
                round(activity.origin.scheduled_time, 6),
                activity_priority.get(activity.activity_type, 99),
            ),
        )

    def get_ordered_agg_activities_train(self, train: Train) -> list:
        return sorted(
            [
                activity
                for activity in self.activities
                if activity.aggregated
                and activity.origin.train == train
                and activity.origin.scheduled_time
                and activity.destination.scheduled_time
            ],
            key=lambda activity: (activity.origin.scheduled_time),
        )

    def last_event_train(self, train: Train) -> Optional[Event]:
        """Returns the last event of a specific train."""
        events = self.get_events_per_train(train)
        if events:
            return max((e for e in events if e.scheduled_time is not None), key=lambda e: e.scheduled_time)
        return None

    def get_incoming_station(self, event: Event):
        """Returns a list of incoming events for a specific event."""
        activities = [activity for activity in self.activities if activity.destination == event]
        if activities:
            return activities[0].origin
        else:
            return None

    def get_outgoing_station(self, event: Event):
        """Returns a list of outgoing events for a specific event."""
        activities = [activity for activity in self.activities if activity.origin == event]
        if activities:
            return activities[0].destination
        else:
            return None

    def get_tracks_between_stations(self, activity: Activity) -> list:
        """Returns section tracks between origin and destination stations (incl. bifurcations)."""
        if activity.intermediate_stations:
            return [
                track
                for track in self.section_tracks
                if (
                    (track.origin == activity.origin.station and track.destination in activity.intermediate_stations)
                    or (track.destination == activity.origin.station and track.origin in activity.intermediate_stations)
                    or (
                        track.origin in activity.intermediate_stations
                        and track.destination == activity.destination.station
                    )
                    or (
                        track.destination in activity.intermediate_stations
                        and track.origin == activity.destination.station
                    )
                    or (
                        track.origin in activity.intermediate_stations
                        and track.destination in activity.intermediate_stations
                    )
                )
            ]
        else:
            return [
                track
                for track in self.section_tracks
                if (
                    (track.origin == activity.origin.station and track.destination == activity.destination.station)
                    or (track.destination == activity.origin.station and track.origin == activity.destination.station)
                )
            ]

    def get_trains_by_code(self, train_code: str):
        return [train for train in self.trains if (train.code == train_code)]

    def get_ordered_trains_at_section_track(self, tracks: list[SectionTrack]):
        """
        Returns the list of trains that pass through any of the given section tracks,
        sorted by their scheduled departure time from the origin station.

        :param tracks: List of SectionTrack objects to consider
        :return: List of Train objects ordered by scheduled time
        """
        passing_trains = []
        track_set = set(tracks)

        for activity in self.activities:
            if activity.section_track in track_set and activity.in_timetable:
                train = activity.origin.train
                scheduled_time = activity.origin.scheduled_time
                passing_trains.append((scheduled_time, train))

        # Sort by scheduled time
        passing_trains.sort(key=lambda x: x[0])

        # Extract and return the trains in order
        return [train for _, train in passing_trains]

    def get_stations_per_train(self, train: Train):
        """Returns a list of stations visited by the given train."""
        visited_stations = set()  # Set to avoid duplicates

        # Iterate through the train's path nodes or events to gather stations
        for a in self.A_train[train]:
            if a.origin.station not in visited_stations:
                visited_stations.add(a.origin.station)
            if a.destination.station not in visited_stations:
                visited_stations.add(a.destination.station)

        # Return the list of visited stations sorted by their ID or code
        return visited_stations

    def add_similar_short_turning(self, similar_act: list):
        if len(similar_act) > 0:
            self.similar_short_turning.append(similar_act)
