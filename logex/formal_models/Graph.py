from typing import Hashable, Optional, Union, Sequence
from itertools import permutations
from math import sqrt
import os
import regex as re
from matplotlib import pyplot as plt
import graphviz as gv
from PIL import Image
import networkx as nx
from matplotlib import patches
from random import randint
import warnings

from . import tools


class Graph():
    
    """
    Base class for graph structures. Labeled States are connected via Labeled Transitions. Some states might
    be denoted as initial or final.
    """
    AnySymbol = '*'
    AnySymbol2 = '∗'

    def __init__(self,
                 states: set[str],
                 initial_states: set[str],
                 final_states: set[str],
                 transitions: set[tuple[str, Hashable, str]],
                 symbols: set[str]):

        self.states = states
        self.initial_states = initial_states
        self.final_states = final_states
        self.transitions = transitions
        self.symbols = symbols
        self.alphabet = symbols
        self.class_name = 'Graph'
        self.rotate_edge_labels = False
        self.name = 'A'

    def add_state(self, state: Hashable, initial: bool = False, final: bool = False):
        """
        Allows to add a new state to the Graph. It can be also registered as
        initial state or/and final state by using the corresponding boolean arguments

        If the state is already in states, warning will be printed.
        However, it may still be used to make an already registered state initial
        or final.

        Args:
            state (Hashable): (new) state as a Hashable object
            initial (bool, optional): make the (new) state initial. Defaults to False.
            final (bool, optional): make tha (new) state final. Defaults to False.
        """

        assert isinstance(state, Hashable), f'state must be Hashable object, not {type(state)}'
        assert isinstance(initial, bool), f'initial must be boolean object, not {type(initial)}'
        assert isinstance(final, bool), f'initial must be boolean object, not {type(final)}'
        state = str(state)
        assert len(state) > 0, 'state cannot be empty'
        if state in self.states:
            warnings.warn(f'state {state} is already in the graph')
        self.states.add(state)
        if initial:
            self.initial_states.add(state)
        if final:
            self.final_states.add(state)
        return
    
    def remove_state(self, state: Hashable):
        """
        Allows to remove a state from the Graph.

        If the state is not in the Graph, a warning will be printed.

        Args:
            state (Hashable): a state as a Hashable object for removal
        """

        assert isinstance(state, Hashable), f'state must be Hashable object, not {type(state)}'
        state = str(state)
        assert len(state) > 0, 'state cannot be empty'
        if state not in self.states:
            warnings.warn(f'state {state} is not in the graph')
        else:
            self.states.remove(state)
        if state in self.initial_states:
            self.initial_states.remove(state)
        if state in self.final_states:
            self.final_states.remove(state)
        return
    
    def add_initial(self, state: Hashable):
        """
        Allows to add an initial state. It can be a completely new node,
        or a node which is already in the graph.

        Args:
            state (Hashable): a (new) initial node
        """

        assert isinstance(state, Hashable), f'state must be Hashable object, not {type(state)}'
        state = str(state)
        assert len(state) > 0, 'state cannot be empty'
        self.states.add(state)
        self.initial_states.add(state)
        return

    def add_final(self, state: Hashable):
        """
        Allows to add a final state. It can be a completely new node,
        or a node which is already in the graph.

        Args:
            state (Hashable): a (new) final node
        """

        assert isinstance(state, Hashable), f'state must be Hashable object, not {type(state)}'
        state = str(state)
        assert len(state) > 0, 'state cannot be empty'
        self.states.add(state)
        self.final_states.add(state)
        return

    def remove_initial(self, state: Hashable):
        """
        IMPORTANT: 
        it does not remove a state from the graph but only
        removes it from the subset of initial states.
        To remove state completely, use .remove_state(state)

        If there is no such initial state, a warning will be printed.

        Args:
            state (Hashable): a state to remove from initial states
        """

        assert isinstance(state, Hashable), f'state must be Hashable object, not {type(state)}'
        state = str(state)
        assert len(state) > 0, 'state cannot be empty'
        if state not in self.initial_states:
            warnings.warn(f'state {state} is not in the initial states ({self.initial_states})')
        else:
            self.initial_states.remove(state)
        return

    def remove_final(self, state: Hashable):
        """
        IMPORTANT: 
        it does not remove a state from the graph but only
        removes it from the subset of final states.
        To remove state completely, use .remove_state(state)

        If there is no such final state, a warning will be printed.

        Args:
            state (Hashable): a state to remove from final states
        """

        assert isinstance(state, Hashable), f'state must be Hashable object, not {type(state)}'
        state = str(state)
        assert len(state) > 0, 'state cannot be empty'
        if state not in self.final_states:
            warnings.warn(f'state {state} is not in the final states ({self.final_states})')
        else:
            self.final_states.remove(state)
        return

    def _extract_symbol_from_transition(self, transition: tuple[str, Hashable, str]) -> str:

        assert isinstance(transition, tuple) and len(transition) == 3, f'transition {transition} is not a 3-tuple'
        assert transition[0] in self.states, f'unknown state {transition[0]} in transition {transition}'
        assert transition[2] in self.states, f'unknown state {transition[2]} in transition {transition}'

        return str(transition[1])

    def _prepare_transitions_for_drawing(self, use_a_star: bool = True):

        # As there might be too many arrows, the transitions from state_1 to state_2 are aggregate into
        # a single bundle. Also, "*" symbol is used to avoid over-plotting.

        simplified_transitions = set()
        for state1 in self.states:
            for state2 in self.states:
                symbols = set()
                for symbol in self.alphabet:
                    if (state1, symbol, state2) in self.transitions:
                        symbols.add(symbol)
                if symbols:
                    if len(symbols) == len(self.alphabet) and use_a_star:
                        symbols = [self.AnySymbol]
                    symbols = symbols.pop() if len(symbols) == 1 else ('(' + ', '.join(sorted(symbols)) + ')')
                    simplified_transitions.add((state1, symbols, state2))
        
        return simplified_transitions

    
    def draw(self, title: str = None, dpi: int = 200, 
             use_a_star: bool = True, engine: str = 'dot',
             alignment: str = 'horizontal',
             _dir: str = 'Graphs', _file: str = None, _show: bool = True,
             _show_isolated_states: bool = False,
             graph_kwargs: dict = None,
             edge_kwargs: dict = None,
             node_kwargs: dict = None,
             initial_node_kwargs: dict = None,
             final_node_kwargs: dict = None) -> Optional[Image]:

        """
        Allows to draw an automaton using "pygraphviz" module and matplotlib figures.
        The png file is saved in the directory.

        Parameters:
            title (str): Optional. How the plot will be entitled (by default, the name of the Automaton).
            dpi (int): Optional. Allows to increase the resolution of png file.
            use_a_star (bool): Allows to simplify transition notation with a "*" symbol.
            engine (str): type of graphviz engine ('circo', 'dot', 'fdp', 'neato', 'patchwork', 'sfdp')
            alignment (str): horizontal or vertical alignment. Defaults to horizontal.
            _dir (str): Optional. Directory to save the png file. Defaults to "Graphs"
            _file (str): Optional. File name. Defaults to "self.name_.png"
            _show (bool): Allows to return a PIL.Image object instead of showing the graph. Defaults to False.
            _show_isolated_states (bool): Allows to show the states which have no associated transitions and are not initial states. Defaults to False

            graph_kwargs: other parameters for Graphviz Graph. The most useful:
                - bgcolor: (Canvas background color): 'none' (default), 'blue', 'yellow', etc.
                - color: (basic drawing color)
                - fontcolor: 'black' (default)
                - fontname: "Times-Roman" (dafualt), "Courier New", "Helvetica", "Sans", etc.
                - fontsize: "14" (default)
                - label: name of the graph, "" by default
                - labelloc: "t" (label on top) or "b" (label on bottom, default)
                - landscape: "true" / "false" (default). Landscape mode
                - mclimit: "1" (dafault). Tolerance to edge intersections, the higher - the less. (Only for "dot" engine).
                - mindist: "1" (default). Minimum distance between modes in "circo" engine.
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

            node_kwargs / initial_node_kwargs / final_node_kwargs: other parameters passed to GraphViz Nodes. Most useful:
                - fillcolor: background color of nodes
                - fixed_size: 'true', 'false' (default). To adjust node size wrt content or not.
                - fontcolor: 'black' (default)
                - fontname: "Times-Roman" (dafualt), "Courier New", "Helvetica", "Sans", etc.
                - fontsize: "14" (default)
                - shape: "ellipse" (default), "circle", "square", etc. Shape of the node.
                - style: "dashed", "dotted", "solid", "invis", "bold", "filled", "striped", "wedged", "rounded"
        """

        # Input validation
        possible_engines = {'circo', 'dot', 'fdp', 'neato', 'patchwork', 'sfdp'}
        assert engine in possible_engines, f'unknown engine "{engine}", choose from {possible_engines}'
        assert alignment in {'horizontal', 'vertical'}, f'unknown alignment "{alignment}", choose from {'horizontal', 'vertical'}'
        assert isinstance(_show_isolated_states, bool), f'_show_isolated_states must be boolean, not ({type(_show_isolated_states)})'
        assert isinstance(use_a_star, bool), f'use_a_star must be boolean, not ({type(use_a_star)})'
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
        if node_kwargs is None:
            node_kwargs = {}
        else:
            assert isinstance(node_kwargs, dict), f'graph_kwargs must be dict, not {type(node_kwargs)}'
        if edge_kwargs is None:
            edge_kwargs = {}
        else:
            assert isinstance(edge_kwargs, dict), f'graph_kwargs must be dict, not {type(edge_kwargs)}'
        if initial_node_kwargs is None:
            initial_node_kwargs = {}
        else:
            assert isinstance(initial_node_kwargs, dict), f'graph_kwargs must be dict, not {type(initial_node_kwargs)}'
        if final_node_kwargs is None:
            final_node_kwargs = {}
        else:
            assert isinstance(final_node_kwargs, dict), f'graph_kwargs must be dict, not {type(final_node_kwargs)}'

        simplified_transitions = self._prepare_transitions_for_drawing(use_a_star=use_a_star)

        # Graphviz directed graph instance
        G = gv.Digraph(format='png')
        G.engine = engine
        kwargs = dict(rankdir='LR' if alignment == 'horizontal' else 'TB',
               dpi=str(dpi))
        kwargs.update(graph_kwargs)
        G.attr(**kwargs)

        # Renaming "states" to "nodes" as we are dealing with a graph now
        if _show_isolated_states:
            nodes = list(self.states)
        else:
            states_mentioned_in_transitions = {t[0] for t in self.transitions} | {t[-1] for t in self.transitions}
            nodes = sorted([state for state in self.states if state in states_mentioned_in_transitions])

        # Initial state must have an incoming arrow. To implement this - a "ghost" node
        # is created and a "ghost" transition is added. These additional nodes
        # are marked with a special symbol "@" and will be set invisible while plotting
        for i, initial in enumerate(self.initial_states):
            ghost_node = f"@{i}"
            ghost_transition = (ghost_node, '', initial)
            nodes = [ghost_node] + nodes
            simplified_transitions.add(ghost_transition)

        # Renewing the set of edges and setting their dict of labels
        edges = [(str(t[0]), str(t[2])) for t in simplified_transitions]
        labels = {(str(t[0]), str(t[2])): t[1] for t in simplified_transitions}

        # Adding nodes and edges to the Graph
        if self.class_name == 'Kripke Structure':
            xlabels = dict()
            for s, ann in self.annotations.items():
                xlabels.update({s: ','.join([p if v else f'{p}\u0305' for p, v in ann.items()])})
        else:
            xlabels = {n: '' for n in nodes}
        for n in nodes:
            node_added = False
            if n in self.initial_states:
                kwargs = dict(shape="circle", xlabel=xlabels[n])
                kwargs.update(initial_node_kwargs)
                G.node(n, n, **kwargs)
                node_added = True
            if n in self.final_states:
                kwargs = dict(shape="doublecircle")
                kwargs.update(final_node_kwargs)
                G.node(n, n, **kwargs)
                node_added = True
            if '@' in n and not node_added:
                G.node(n, '', shape="circle", style='invisible')
                node_added = True
            if not node_added:
                kwargs = dict(shape="circle", xlabel=xlabels[n])
                kwargs.update(node_kwargs)
                G.node(n, n, **kwargs)
        for edge in edges:
            G.edge(edge[0], edge[1], label=labels[edge], **edge_kwargs)
        
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

    def draw_with_networkx(self, title: str = None, seed: int = None, 
                           use_a_star: bool = True,
                           _show_isolated_states: bool = False) -> int:

        """
        Allows to draw an automaton using "networkx" module and matplotlib figures.
        The algorithm is not deterministic until the seed is explicitly provided.

        As graph plotting is not very convenient in Python, it must require certain
        attempts to obtain a good-looking plot. To capture a nice seed,
        the method returns the seed it used for plotting. When you are satisfied with 
        the plot, you can copy the returned seed and pass it as an argument to 
        preserve the best result.

        typical usage:
        for _ in range(5):
             seed = automaton.draw()
             print(seed)

        You will get 5 (different) views of the same automaton with the orders which produced them.Any
        After choosing the best-looking one:

        automaton.draw(seed=best_seed)

        Parameters:
            title (str): Optional. How the plot will be entitled (by default, the name of the Automaton).
            seed (int): Optional. Allows to fix the best result.
            use_a_star (bool): Allows to simplify transition notation with a "*" symbol.
            _show_isolated_states (bool): Allows to show the states which have no associated transitions and are not initial states. Defaults to False

        Returns:
            seed used for plotting
        """
        assert isinstance(_show_isolated_states, bool), f'_show_isolated_states must be boolean, not ({type(_show_isolated_states)})'
        assert isinstance(use_a_star, bool), f'use_a_star must be boolean, not ({type(use_a_star)})'
        assert seed is None or isinstance(seed, int), f'seed must be an integer, not {seed}'
        assert title is None or isinstance(title, str), 'name must be a string'
        if title is None:
            title = f'{self.class_name}: ' + self.name

        # As there might be too many arrows, the transitions from state_1 to state_2 are aggregate into
        # a single bundle. Also, "*" symbol is used to avoid over-plotting.
        simplified_transitions = self._prepare_transitions_for_drawing(use_a_star=use_a_star)

        # Defining the original edges
        edges = [(str(t[0]), str(t[2])) for t in simplified_transitions]

        # Networkx directed graph instance
        G = nx.MultiDiGraph()

        # Renaming "states" to "nodes" as we are dealing with a graph now
        if _show_isolated_states:
            nodes = list(self.states)
        else:
            states_mentioned_in_transitions = {t[0] for t in self.transitions} | {t[2] for t in self.transitions}
            nodes = sorted([state for state in self.states if state in states_mentioned_in_transitions])

        # Shuffling nodes implicitly using random seed to obtain different plots
        perm = list(permutations(nodes))
        if not seed:
            seed = randint(0, len(perm))
        nodes = list(perm[seed % len(perm)])

        # Choosing a "central" node - a node which has the most connection with other nodes
        # It is important to make the graph more readable
        central_node, max_score = None, 0
        for node in nodes:
            neighbours = list(filter(lambda t: (t[0] == node or t[2] == node) and t[0] != t[2], self.transitions))
            adjacent = len(set([a[-1] for a in neighbours] + [a[0] for a in neighbours]))
            if adjacent > max_score:
                central_node = node
                max_score = adjacent

        # creating a nested dictionary of preferred node distances:
        # The most attracting force is between central node and its neighbours
        # A bit less force is between any other connected nodes
        # The weakest force is between nodes that are not connected directly
        distances = {}
        for n1 in nodes:
            distances.update({n1: {}})
            for n2 in nodes:
                if (n1, n2) in edges or (n2, n1) in edges:
                    distances[n1].update({n2: 2})
                    if n1 == central_node or n2 == central_node:
                        distances[n1].update({n2: 1})
                else:
                    distances[n1].update({n2: 10})

        # Initial state must have an incoming arrow. To implement this - a "ghost" node
        # is created and a "ghost" transition is added. These additional nodes
        # are marked with a special symbol "@" and will be set invisible while plotting
        for i, initial in enumerate(self.initial_states):
            ghost_node = f"@{i}"
            ghost_transition = (ghost_node, '', initial)
            nodes = [ghost_node] + nodes
            simplified_transitions.add(ghost_transition)

            # Also, this "ghost" node nust be close to the initial state - string attracting force
            distances.update({ghost_node: {initial: 0.5}})

        # Renewing the set of edges and setting their dict of labels
        edges = [(str(t[0]), str(t[2])) for t in simplified_transitions]
        labels = {(str(t[0]), str(t[2])): t[1] for t in simplified_transitions}

        # Adding nodes and edges to the Graph
        _, ax = plt.subplots(1, 1)
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # Default node radius for every node
        node_radius = 500
        node_sizes = [node_radius] * len(nodes)

        # Positions of nodes are computed using networkx implementation of
        # Kamada-Kawai algorithm wrt dict of distances
        pos = nx.kamada_kawai_layout(G, dist=distances)

        # Final states must have double circles around them. To achieve it -
        # there additional "ghost" nodes with smaller radius plotted exactly above 
        # the final states with a transparent body
        for i, state in enumerate(self.final_states):
            ghost_node = f'%{i}'
            G.add_node(ghost_node)
            nodes = [ghost_node] + nodes
            node_sizes.append(int(node_radius * 0.7))
            pos.update({ghost_node: pos[state]})

        # Drawing nodes while hiding the "@" ghosts
        nx.draw_networkx_nodes(G, pos, 
                               node_color='white', 
                               linewidths=[1 if not '@' in n else 0 for n in G.nodes], 
                               ax=ax, 
                               edgecolors='black',
                               node_size=node_sizes, 
                               node_shape='o',
                               alpha=[1 if not '@' in n else 0 for n in G.nodes])
        
        # Printing node labels while hiding ghost nodes
        nx.draw_networkx_labels(G, pos, 
                                font_size={n: int(0.08 * node_radius / (len(n) + 1)) for n in G.nodes},
                                labels={n: n if not ('%' in n or '@' in n) else ''  for n in G.nodes}, 
                                ax=ax)
        
        # Defining the curvature of edges and their labels as well as arrow style (especially for self-loops)
        connectionstyle = "arc3,rad=0.05"
        labelstyle = "arc3,rad=0.1"
        arrowstyle = patches.ArrowStyle.CurveFilledB(head_length=0.8, head_width=0.2)
        self_loop_style = patches.ArrowStyle._Curve()

        # Drawing edges wrt their styles
        nx.draw_networkx_edges(G, pos, 
                               ax=ax, 
                               arrowsize=10,
                               connectionstyle=[connectionstyle if edge[:2] not in list(nx.selfloop_edges(G)) else "arc3" for edge in G.edges] , 
                               arrowstyle=[arrowstyle if edge[:2] not in list(nx.selfloop_edges(G)) else self_loop_style for edge in G.edges]
                            )
        
        # Adding edge labels
        nx.draw_networkx_edge_labels(G, pos, 
                                     edge_labels=labels, 
                                     font_color='black', 
                                     ax=ax, 
                                     font_size=int(12 * node_radius / 500),
                                     label_pos=0.5, 
                                     bbox={"alpha": 0}, 
                                     connectionstyle=labelstyle,
                                     verticalalignment='center',
                                     rotate=self.rotate_edge_labels)
        
        
        ax.set_title(title + f' (seed={seed})')
        plt.show()

        return seed

    def __call__(self,
                 word: Sequence,
                 return_path: bool = False) -> Union[bool, tuple[bool, str | None]]:
        """
        Check is the word is accepted / belongs to the language of Automaton

        Args:
            word (str): word to check
            return_path (bool): allows to show the path that leads to a final state

        Returns:
            bool | tuple[bool, str | None]
        """

        assert isinstance(word, Sequence), f'word must be Sequence, not with {type(word)} ({word})'
        if word == tools.EMPTY_WORD:
            word = ''

        accepted = False
        layers = []

        if not self.initial_states:
            return False

        # Trying from every initial state
        for initial_state in self.initial_states:

            # Initialize a set of states reachable at the current step of word traversing
            currently_in = {initial_state}
            if return_path:
                layers.append(currently_in)

            # Iterating along the word letters
            for letter in word:

                # Searching for all the states reachable from the current set of stats with respect to the current letter
                valid_transitions = list(filter(lambda t: t[0] in currently_in and letter == self._extract_symbol_from_transition(t), 
                                                self.transitions))
                currently_in = set([t[2] for t in valid_transitions])

                # If new letter cannot be processed at all - quit the algorithm (word is not accepted)
                if not currently_in:
                    break

                if return_path:
                    layers.append(currently_in)

            # Checking if we arrive to at least one final state after the whole word is traversed
            if self.final_states & currently_in:
                accepted = True
                break

        # Constructing an exemplary path for a word through the automaton (only if accepted)
        if return_path:
            if not accepted:
                return False, None
            path = []
            path.append('END')
            path.append((self.final_states & layers[-1]).pop())
            for letter, layer in zip(word[::-1], layers[:-1][::-1]):
                next_state = path[-1]
                example = list(filter(lambda t: t[0] in layer and letter == self._extract_symbol_from_transition(t) and t[2] == next_state, 
                                      self.transitions))[0]
                path.append(str(letter))
                path.append(example[0])
            path.append('START')
            path = path[::-1]
            states_encountered = set(path[1::2])
            path = ' -> '.join([f'[{p}]' if i % 2 == 1 else str(p) for i, p in enumerate(path)])

            # Creating short notations for long self-loops
            while True:
                changed = False
                for state in states_encountered:
                    loop = re.findall(r'\[' +str(state) +r'\] -> [^\]]+ -> \[' + str(state) + r'\]', path, overlapped=True)
                    if not loop:
                        continue
                    else:
                        n, trial = 0, []
                        for n in list(range(len(loop) + 1))[::-1]:
                            trial = r'\[' + str(state) + r'\]' + (r' -> [^\]]+ -> \[' + str(state) + r'\]') * n
                            trial = re.findall(trial, path)
                            if trial:
                                break

                        if n > 1:
                            changed = True
                            for t in trial:
                                symbols = set([x for x in t.split(' -> ') if '[' not in x])
                                pattern = f'([{str(state)}] -> ' + "|".join(symbols) + f' -> [{str(state)}])x' + str(n)
                                path = path.replace(t, pattern)
                            break
                if not changed:
                    break

            return True, path

        return accepted

    @property
    def is_complete(self) -> bool:
        """
        Verifies if the automaton is complete
        """

        # Checking the there are initial states
        if not bool(self.initial_states):
            return False

        # Checking that every state has at least one transition per symbol
        for state in self.states:
            for alpha in self.alphabet:
                if not bool(list(filter(lambda t: t[0] == state and str(t[1]) == str(alpha), self.transitions))):
                    return False

        return True

    def is_complete_explain(self) -> bool:
        """
        Verifies if the automaton is complete and gives short explanation
        """

        if not bool(self.initial_states):
            print(f'{self.name} is not complete because there are no initial states')
            return False
        for state in self.states:
            for alpha in self.alphabet:
                if not bool(list(filter(lambda t: t[0] == state and str(t[1]) == str(alpha), self.transitions))):

                    print(f'{self.name} is not complete because there exist no transitions from state "{state}" using symbol "{alpha}"')
                    return False

        print(f'{self.name} is complete because from every state there exist transitions using every symbol from the alphabet')
        return True

    @property
    def is_deterministic(self) -> bool:
        """
        Verifies if the automaton is deterministic
        """

        # Checking that there are not multiple initial states
        if len(self.initial_states) > 1:
            return False
        
        # Checking that every state has no more than 1 transition with respect to symbol
        for state in self.states:
            for alpha in self.alphabet:
                if len(list(filter(lambda t: t[0] == state and str(t[1]) == str(alpha), self.transitions))) > 1:
                    return False

        return True

    def is_deterministic_explain(self) -> bool:
        """
        Verifies if the automaton is deterministic and gives short explanation
        """

        if len(self.initial_states) > 1:
            print(f'{self.name} is not deterministic because there are {len(self.initial_states)} initial states (more than 1)')
            return False
        for state in self.states:
            for alpha in self.alphabet:
                tt = list(filter(lambda t: t[0] == state and str(t[1]) == str(alpha), self.transitions))
                n = len(tt)
                if n > 1:
                    print(f'{self.name} is not deterministic because from state "{state}" there are {n} transitions using symbol "{alpha}"\nnamely {tt}')
                    return False

        print(f'{self.name} is deterministic because for every state there is at most 1 transition through every symbol')

        return True