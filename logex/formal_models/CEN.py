from typing import Collection, Union, Hashable

from . import Net

class ConditionEventNet(Net):
    
    """
    CEN: a net with capacities of exactly 1.

    Args:
        places_aka_conditions (Collection[Union[int, str]]): 
            places (round states)
        initial_marking (_type_): 
            places which are active initially
        transitions_aka_events (Collection[tuple]):
            events (square states)
        connections_aka_dependencies (Collection):
            connections between states (both round and square)
        name (str, optional): 
            a name for PTN. Defaults to None.
    """

    def __init__(self,
                 places_aka_conditions: Collection[Union[int, str]],
                 initial_marking: Collection[str] | dict | None,
                 transitions_aka_events: Collection[tuple],
                 connections_aka_dependencies: Collection,
                 name: str = None):


        # Checking the input datatypes and unifying them

        # Check name
        assert name is None or isinstance(name, str), 'name must be a string'
        
        # Check places AKA conditions -> set of strings
        assert isinstance(places_aka_conditions, Collection), 'places/conditions must be a Collection'
        for place in places_aka_conditions:
            assert isinstance(place, Hashable), f'place/condition must be Hashable, not {type(place)} ({place})'
        places_aka_conditions = set(list(map(str, places_aka_conditions)))

        # Check transitions AKA events -> set of strings
        assert isinstance(transitions_aka_events, Collection), 'events/transitions must be a Collection'
        for event in transitions_aka_events:
            assert isinstance(event, Hashable), f'transition/event must be Hashable, not {type(event)} ({event})'
        transitions_aka_events = set(list(map(str, transitions_aka_events)))

        # Check initial marking -> dict
        assert isinstance(initial_marking, dict | Collection | None), 'initial_marking must be a dict / Collection / None'
        if initial_marking is None:
            initial_marking = {place: 0 for place in places_aka_conditions}
        elif not isinstance(initial_marking, dict):
            initial_marking = set(list(map(str, initial_marking)))
            for init_mark in initial_marking:
                assert init_mark in places_aka_conditions, f'unknown place/condition in initial_marking: {init_mark}'
            initial_marking = {str(init_mark): 1 for init_mark in initial_marking}
        else:
            for place, value in initial_marking.items():
                assert str(place) in places_aka_conditions, f'unknown place/condition in initial_marking: {place}'
                assert isinstance(value, int), f'values of initial_marking must be of type int, not {type(value)} ({value})'
            initial_marking = {str(key): value for key, value in initial_marking.items()}
        default_marking = {place: 0 for place in places_aka_conditions - set(initial_marking.keys())}
        initial_marking.update(default_marking)

        # Check connection -> set of 2-tuples (place, event) or (event, place)
        assert isinstance(connections_aka_dependencies, Collection), 'connections/dependencies must be a Collection'
        for connection in connections_aka_dependencies:
            assert isinstance(connection, tuple) and len(connection) == 2, f'connections/dependencies must be 2-tuples, not {len(connection)} ({connection})'
            assert (
                (str(connection[0]) in places_aka_conditions and str(connection[1]) in transitions_aka_events) or 
                (str(connection[1]) in places_aka_conditions and str(connection[0]) in transitions_aka_events)
                    ), f'Connection {connection} is not valid. Must connect place to event or event to place'
        connections_aka_dependencies = {(str(conn[0]), str(conn[1])) for conn in connections_aka_dependencies}

        capacities = dict() 
        default_place_capacities = {place: 1 for place in places_aka_conditions}
        default_connection_capacities = {conn: 1 for conn in connections_aka_dependencies}
        capacities.update(default_place_capacities)
        capacities.update(default_connection_capacities)

        # Assert that initial marking respects capacities:
        for place, value in initial_marking.items():
            assert capacities[place] >= value, f'Capacity of place {place} is {capacities[place]} in CEN but trying to set it to {value}'

        # Main attributes of PTN
        super().__init__(places_aka_conditions, initial_marking, transitions_aka_events, connections_aka_dependencies, capacities)
        self.conditions = self.places
        self.initial_marking = {k: v for k, v in initial_marking.items()}
        self.class_name = 'CEN'
        self.rotate_edge_labels = True
        self.name = name if name else 'N'


    def __str__(self):
        string = f'CEN {self.name} = (C, I, E, G)\n'
        string += f'Conditions C  = {self.conditions}\n'
        string += f'Initial Marking I = {self.initial_marking}\n'
        string += f'Events E = {self.events}\n'
        string += 'Connection graph G = \n'
        connections = sorted(list(map(str, self.connections)))
        max_length = max(list(map(len, connections)))
        per_row = 40 // max_length
        for i in range(0, len(connections), per_row):
            upper_i = i*per_row+per_row
            string += '\t' + ', '.join(connections[i:min(upper_i, len(connections))])
            if upper_i <= len(connections):
                string += ',\n'
            else:
                string += '\n'
    
        return string

    def __repr__(self):
        return str(self)
    