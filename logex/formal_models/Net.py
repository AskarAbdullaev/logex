from typing import Collection, Union, Optional
from math import sqrt, ceil
import os
from matplotlib import pyplot as plt
import graphviz as gv
from PIL import Image


class Net():
    """
    Generic class for Net structures:
    - 2 types of nodes: places/conditions and events/transitions
    - each place and transition has a capacity
    - tokens can be stored in places and passed through the net under certain conditions
    - 2 child classes: PTN and CEN
    """

    Token = '●'

    def __init__(self,
                 places: set[str],
                 marking: dict[str: int],
                 events: set[str],
                 connections: set[tuple[str, str]],
                 capacities: dict[Union[tuple, str]: int] = None,
                 name: str = None):

        # Main attributes of PTN
        self.places = places
        self.events = events
        self.connections = connections
        self.capacities = capacities
        self.marking = marking
        self.class_name = 'Net'
        self.rotate_edge_labels = True
        self.name = name if name else 'N'
        self.initial_marking = {p: 0 for p in self.places}

    def reset_marking(self):
        """
        Return to initial marking
        """
        self.marking = {k: v for k, v in self.initial_marking.items()}
        return
    
    def set_marking(self, marking: Union[Collection, dict]):
        """
        Redistribute tokens according to the provided marking.

        Args:
            marking (Union[Collection, dict]): 
                a dictionary of places and number of tokens. If just a collection: 
                each mentioned place is assigned 1 token.
        """

        assert isinstance(marking, Collection | dict), f'marking must be of type Collection | dict, not {type(marking)}'
        if not isinstance(marking, dict):
            markings_to_dict = {}
            for place in marking:
                markings_to_dict.update({str(place): 1})
            marking = markings_to_dict
        
        for place, marks in marking.items():
            assert str(place) in self.places, f'no such place in places: {place}'
            assert isinstance(marks, int), f'marking values must be int, not {type(marks)} ({marks})'
            assert self.capacities[str(place)] >= marks, f'max capacity for place {place} is {self.capacities[place]} but trying to set {marks}'
            self.marking.update({str(place): marks})
        marking = {str(place): marks for place, marks in marking.items()}
        default_marking = {place: 0 for place in self.places - set(marking.keys())}
        self.marking.update(default_marking)
        return
    
    def event_is_possible(self, event: str, return_new_marking: bool = False) -> bool | tuple[bool, dict | None]:
        """
        Checks if event is possible under current marking.
        Important:
            return_new_marking allows to return only a potentially obtained marking,
            but it does not execute the event and thus, does not change the actual marking of
            the PTN.

        Args:
            event (str): 
                name of the event to check
            return_new_marking (bool, optional): 
                allows to return a dictionary with a potential marking after the given event is fired. 
                If event is not possible, None is returned. Defaults to False.

        Returns:
            bool | tuple[bool, dict | None]
        """

        assert str(event) in self.events, f'No such event {event} in events'
        assert isinstance(return_new_marking, bool), f'return_new_marking must be boolean, not {type(return_new_marking)}'

        possible = True

        # Check that preconditions have enough tokens to fire the event
        preconditions = {connection for connection in self.connections if connection[1] == event}
        new_marking = self.marking.copy()
        for pre in preconditions:
            tokens_required = self.capacities[pre]
            tokens_available = self.marking[pre[0]]
            if tokens_required > tokens_available:
                possible = False
                break
            else:
                new_marking.update({pre[0]: tokens_available - tokens_required})

        if possible:
            postconditions = {connection for connection in self.connections if connection[0] == event}
            for post in postconditions:
                tokens_current = new_marking[post[1]]
                tokens_added = self.capacities[post]
                tokens_maximum = self.capacities[post[1]]
                if tokens_current + tokens_added > tokens_maximum:
                    possible = False
                    break
                else:
                    new_marking.update({post[1]: tokens_current + tokens_added})

        if return_new_marking:
            return possible, new_marking if new_marking else None
        else:
            return possible
        
    def fire_event(self, event: str):
        """
        
        Tries to fire the given event. Returns False if it is not possible.

        Args:
            event (str): 
                name of the event to fire

        Returns:
            None | False
        """

        possible, new_marking = self.event_is_possible(event, True)
        if not possible:
            return False
        self.marking = new_marking.copy()
        return
    
    def possible_events(self):
        """
        Returns a set of events which are possible to fire with the current marking.
        """
        possible_events = set()
        for event in self.events:
            if self.event_is_possible(event):
                possible_events.add(event)

        return possible_events
    
    def is_deadlock(self, marking: dict = None):
        """
        Checks if the current state of PTN is a deadlock or
        if the provided marking is a deadlock.

        Args:
            marking (dict, optional): marking to check. Defaults to None.
        """

        # Check and unify the marking provided (or take the current one)
        if marking is not None:
            assert isinstance(marking, Collection | dict), f'marking must be of type Collection | dict, not {type(marking)}'
            if not isinstance(marking, dict):
                markings_to_dict = {}
                for place in marking:
                    markings_to_dict.update({str(place): 1})
                marking = markings_to_dict
            
            for place, marks in marking.items():
                assert str(place) in self.places, f'no such place in places: {place}'
                assert isinstance(marks, int), f'marking values must be int, not {type(marks)} ({marks})'
                assert self.capacities[str(place)] >= marks, f'max capacity for place {place} is {self.capacities[place]} but trying to set {marks}'
                self.marking.update({str(place): marks})
        else:
            marking = self.marking.copy()

        # Check for the deadlock situation (and return back to the current marking)
        deadlock = False

        current_marking = self.marking.copy()
        self.set_marking(marking)
        if not self.possible_events():
            deadlock = True
        self.set_marking(current_marking)

        return deadlock

    def to_lts(self, name: str = None):
        """
        Derives LTS from PTN.

        Parameters:
            name (str): Optional. How to entitle the new LTS instance. 
            By default: LTS(previous name).

        Returns:
            a new LabeledTransitionSystem instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'LTS({self.name})'

        if self.class_name == 'CEN':
            drop_capacity = True
        else:
            drop_capacity = False

        # Store current marking to reset afterwards
        current_marking = self.marking.copy()

        def marking_to_state(marking: dict):
            """
            Helper function to create a LTS state name from PTN current marking
            """
            state_name = sorted(marking.items(), key=lambda x: x[0])
            if not drop_capacity:
                state_name = '#'.join(f'{s}${v}' for s, v in state_name if int(v) > 0)
            else:
                state_name = '#'.join(f'{s}' for s, v in state_name if int(v) > 0)
            return state_name
        
        def state_to_marking(_state: str):
            """
            Back transformation of state to marking
            """
            mark = _state.split('#')
            result = {place: 0 for place in self.places}
            for m in mark:
                if not drop_capacity:
                    _name, value = m.split('$')
                else:
                    _name, value = m, 1
                if int(value) > 0:
                    result.update({_name: int(value)})
            return result

        all_states = set()
        initial_state = marking_to_state(self.initial_marking)
        added_at_previous_step = {initial_state}
        all_states.add(initial_state)
        all_transitions = set()

        while added_at_previous_step:

            added_at_current_step = set()

            for state in added_at_previous_step:

                self.set_marking(state_to_marking(state))

                for event in self.events:

                    possible, new_marking = self.event_is_possible(event, True)
                    if possible:
                        new_state = marking_to_state(new_marking)
                        added_at_current_step.add(new_state)
                        all_transitions.add((state.replace('#', '').replace('$', ''), 
                                             event, 
                                             new_state.replace('#', '').replace('$', '')))

            added_at_previous_step = added_at_current_step - all_states
            all_states |= added_at_current_step
        
        all_states = {s.replace('#', '').replace('$', '') for s in all_states}

        from . import LabeledTransitionSystem
        lts = LabeledTransitionSystem(
            all_states,
            {initial_state.replace('#', '').replace('$', '')},
            all_transitions,
            self.events,
            name=name
        )

        self.set_marking(current_marking)

        return lts
    
    def __call__(self, event: str):
        """
        Another way to fire an event
        """
        return self.fire_event(event)
    
    def draw(self, 
             title: str = None,
             dpi: int = 200, 
             engine: str = 'dot',
             alignment: str = 'horizontal',
             _dir: str = 'Graphs', 
             _file: str = None, 
             _show: bool = True,
             graph_kwargs: dict = None,
             edge_kwargs: dict = None,
             place_node_kwargs: dict = None,
             event_node_kwargs: dict = None) -> Optional[Image]:

        """
        Allows to draw a Net using "pygraphviz" module and matplotlib figures.
        The png file is saved in the directory.

        Parameters:
            title (str): Optional. How the plot will be entitled (by default, the name of the Automaton).
            dpi (int): Optional. Allows to increase the resolution of png file.
            engine (str): type of graphviz engine ('circo', 'dot', 'fdp', 'neato', 'patchwork', 'sfdp')
            alignment (str): horizontal or vertical alignment. Defaults to horizontal.
            _dir (str): Optional. Directory to save the png file. Defaults to "Graphs"
            _file (str): Optional. File name. Defaults to "self.name_.png"
            _show (bool): Allows to return a PIL.Image object instead of showing the graph. Defaults to False.

            graph_kwargs: other parameters for Graphviz Graph. The most useful:
                - bgcolor: (Canvas background color): 'none' (default), 'blue', 'yellow', etc.
                - color: (basic drawing color)
                - fontcolor: 'black' (default)
                - fontname: "Times-Roman" (dafualt), "Courier New", "Helvetica", "Sans", etc.
                - fontsize: "14" (default)
                - label: name of the graph, "" by default
                - labelloc: "t" (label on top) or "b" (label on bottom, default)
                - landscape: "true" / "false" (default). Landscape mode
                - mclimit: "1" (default). Tolerance to edge intersections, the higher - the less. (Only for "dot" engine).
                - mindist: "1" (default). Minimum distance between nodes in "circo" engine.
                - nodesep: "0.25". Minimum space between adjacent nodes for "dot" engine / size of self-loops for other engines.
                - ratio: "auto" (default), "1", "0.5", etc. Height/Width ratio of the output.
                - splines: "true" (edges circumvent nodes, default), "false" (straight edges), "ortho", "curved"

            edge_kwargs: other parameters passed to GraphViz Edges. Most useful:
                - arrowsize: '1' (default), '2', '3', etc.
                - arrowhead: 'box', 'crow', 'diamond', 'dot', 'inv', 'normal' (default), 'tee', 'vee'
                - decorate: 'true', 'false' (deafult). Allows to connect edge label to the edge.
                - fontcolor: 'black' (default)
                - fontname: "Times-Roman" (dafualt), "Courier New", "Helvetica", "Sans", etc.
                - fontsize: "14" (default)
                - minlen: "1" (default). Minimum length of edges for "dot" engine.
                - style: "dashed", "dotted", "solid", "invis", "bold", "tapered"

            place_node_kwargs / event_node_kwargs: other parameters passed to GraphViz Nodes. Most useful:
                - fillcolor: background color of nodes
                - fixed_size: 'true', 'false' (dafault). To adjust node size wrt content or not.
                - fontcolor: 'black' (default)
                - fontname: "Times-Roman" (dafualt), "Courier New", "Helvetica", "Sans", etc.
                - fontsize: "14" (default)
                - shape: "ellipse" (default), "circle", "square", etc. Shape of the node.
                - style: "dashed", "dotted", "solid", "invis", "bold", "filled", "striped", "wedged", "rounded"
                - xlabel: label to draw outside the node but near it
        """

        # Input validation
        possible_engines = {'circo', 'dot', 'fdp', 'neato', 'patchwork', 'sfdp'}
        assert engine in possible_engines, f'unknown engine "{engine}", choose from {possible_engines}'
        assert alignment in {'horizontal', 'vertical'}, f'unknown alignment "{alignment}", choose from {'horizontal', 'vertical'}'
        assert isinstance(_show, bool), f'_show must be an boolean, not {type(_show)}'
        assert isinstance(dpi, int) and dpi > 50, f'dpi must be an integer larger than 50, not {dpi}'
        assert title is None or isinstance(title, str), 'name must be a string'
        assert isinstance(_dir, str) and len(_dir) > 0, f'_dir name must be a non-empty string, not {_dir}'
        assert not _file or isinstance(_file, str), f'_file name must be string, not {type(_file)}'
        if title is None:
            title = f'{self.class_name}: ' + self.name
        if not _file:
            _file = f'{self.name}_.png'
        elif len(_file.split('.')) > 1 and _file.split('.')[-1] != 'png':
            raise ValueError('file must be of png format')
        elif len(_file.split('.')) == 1:
            _file = _file + '.png'

        # Check kwargs
        if graph_kwargs is None:
            graph_kwargs = {}
        else:
            assert isinstance(graph_kwargs, dict), f'graph_kwargs must be dict, not {type(graph_kwargs)}'
        if edge_kwargs is None:
            edge_kwargs = {}
        else:
            assert isinstance(edge_kwargs, dict), f'graph_kwargs must be dict, not {type(edge_kwargs)}'
        if place_node_kwargs is None:
            place_node_kwargs = {}
        else:
            assert isinstance(place_node_kwargs,
                                  dict), f'graph_kwargs must be dict, not {type(place_node_kwargs)}'
        if event_node_kwargs is None:
            event_node_kwargs = {}
        else:
            assert isinstance(event_node_kwargs, dict), f'graph_kwargs must be dict, not {type(event_node_kwargs)}'

        # Graphviz directed graph instance
        G = gv.Digraph(format='png')
        G.engine = engine
        kwargs = dict()
        kwargs.update(dict(rankdir='LR' if alignment == 'horizontal' else 'TB'))
        kwargs.update(dict(dpi=str(dpi), nodesep='1'))
        kwargs.update(graph_kwargs)
        G.attr(**kwargs)


        place_nodes = sorted(list(self.places))
        event_nodes = sorted(list(self.events))

        # Renewing the set of edges and setting their dict of labels
        edges = self.connections
        labels = self.capacities

        # Adding nodes and edges to the Graph
        for n in place_nodes:
            if self.marking[n]:
                row = ceil(sqrt(2 * self.marking[n]))
                tokens = self.Token * self.marking[n]
                tokens = '\n'.join([tokens[i:min(len(tokens), i + row)] for i in range(0, len(tokens), row)])
            else:
                tokens = ''
            xlabel = f'{n}({self.capacities[n]})' if self.class_name == 'PTN' else n
            kwargs = dict(shape="circle", xlabel=xlabel, labeldistance='0.1', margin='0.0')
            kwargs.update(place_node_kwargs)
            G.node(n, tokens, **kwargs)
        for n in event_nodes:
            kwargs = dict(shape="square")
            kwargs.update(event_node_kwargs)
            G.node(n, n, **kwargs)
        for edge in edges:
            edge_label = str(labels[edge]) if self.class_name == 'PTN' else ''
            G.edge(edge[0], edge[1], label=edge_label, **edge_kwargs)
        
        # Save and show
        path = os.path.join(_dir, _file)
        G.render(_file[:-4], _dir, format='png')
        img = Image.open(path, formats=['png'])
        if _show:
            figsize = (int(sqrt(img.size[0]) / 5), int(sqrt(img.size[1]) / 5))
            _, ax = plt.subplots(1, 1, figsize=figsize)
            ax.imshow(img)
            plt.axis('off')
            ax.set_title(title)
            plt.show()
            return
        else:
            return img
