from typing import Collection, Union, Hashable, Sequence
from itertools import product
import regex as re

from . import Graph
from . import tools

class LabeledTransitionSystem(Graph):
    """
    The simplest extension of Graph super-class. Just a collection of states with possibly
    initial states and NO final states. 
    Transitions connect pairs of states through arbitrary actions.

    Args:
        states (Collection[Union[int, str]]): 
            collection of states
        initial_states (Collection[Union[int, str]] | str | None): 
            initial states
        transitions (Collection[tuple]): 
            collection of 3-tuple transitions
        actions (Collection, optional): 
            collection of actions. Can be obtained from transitions. Defaults to None.
        name (str, optional): 
            a name for LTS. Defaults to None.
    """

    def __init__(self,
                 states: Collection[Union[int, str]],
                 initial_states: Collection[Union[int, str]] | str | None,
                 transitions: Collection[tuple],
                 actions: Collection = None,
                 name: str = None):

        # Checking the input datatypes and unifying them

        # Check name
        assert name is None or isinstance(name, str), 'name must be a string'
        
        # Check states -> set of strings
        assert isinstance(states, Collection), 'states must be a Collection'
        for state in states:
            assert isinstance(state, Hashable), f'state must be Hashable, not {type(state)} ({state})'
        states = set(list(map(str, states)))

        # Check initial states -> set of strings -> present in states set
        assert isinstance(initial_states, Collection | str | None), 'states must be a Collection / str / None'
        if isinstance(initial_states, str):
            initial_states = {initial_states}
        elif initial_states is None:
            initial_states = set()
        else:
            initial_states = set(list(map(str, initial_states)))
        for state in initial_states:
            assert state in states, f'initial state {state} is not in states'
            
        # Check transitions -> set of 3-tuples (str, str, str) -> check the existence of states
        actions_in_transitions = set()
        assert isinstance(transitions, Collection), 'transitions must be a Collection'
        for transition in transitions:
            assert len(transition) == 3, f'transitions must be 3-tuples, not {len(transition)} ({transition})'
            assert isinstance(transition[1], Union[str, int]), f'transition condition must be int or str, not {type(transition[1])} ({transition})'
            assert str(transition[0]) in states, f'In transition {transition} starting state {transition[0]} is not defined in states'
            assert str(transition[2]) in states, f'In transition {transition} destination state {transition[2]} is not defined in states'
            actions_in_transitions.add(str(transition[1]))
        transitions = {(str(t[0]), str(t[1]), str(t[2])) for t in transitions}

        # Check alphabet if provided -> set of strings
        assert not actions or isinstance(actions, Collection), 'alphabet must be None or a Collection'
        if actions:
            for action in actions:
                assert isinstance(action, Union[int, str]), f'alphabet symbols must be of type str or int, not {type(action)} ({action})'
            actions = {str(a) for a in actions} - {self.AnySymbol, self.AnySymbol2}
        if not actions:
            actions = actions_in_transitions - {self.AnySymbol, self.AnySymbol2}
            assert len(transitions) > 0 and actions, 'alphabet is not provided and cannot be derived from the transitions'

        # Check that transitions have symbols from the alphabet and replace "*" with several transitions
        additional_transitions = set()
        junk_transitions = set()
        for transition in transitions:
            start, a, end = transition
            if a in {self.AnySymbol, self.AnySymbol2}:
                junk_transitions.add(transition)
                for action in actions:
                    additional_transitions.add((start, action, end))
            else:
                assert a in actions, f'symbol {a} in transition {transition} is not in the alphabet'
        transitions -= junk_transitions
        transitions |= additional_transitions

        # Main attributes of Finite Automaton
        super().__init__(states, initial_states, set(), transitions, actions)
        self.alphabet = actions
        self.actions = actions
        self.class_name = 'LTS'
        self.rotate_edge_labels = True
        self.name = name if name else 'A'

    def __str__(self):
        string = f'LTS {self.name} = (S, I, Σ, T)\n'
        string += f'States S = {self.states}\n'
        string += f'Actions Σ = {self.alphabet}\n'
        string += f'Initial states I = {self.initial_states}\n'
        string += 'Transitions T = \n'
        transitions = list(map(str, self.transitions))
        max_length = max(list(map(len, transitions)))
        per_row = 40 // max_length
        for i in range(0, len(transitions), per_row):
            upper_i = i*per_row+per_row
            string += '\t' + ', '.join(transitions[i:min(upper_i, len(transitions))])
            if upper_i <= len(transitions):
                string += ',\n'
            else:
                string += '\n'

        return string

    def __repr__(self):
        return str(self)
    
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

        accepted = True
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
                    accepted = False
                    break

                if return_path:
                    layers.append(currently_in)


        # Constructing an exemplary path for a word through the automaton (only if accepted)
        if return_path:
            if not accepted:
                return False, None
            path = []
            path.append('END')
            path.append(layers[-1].pop())
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
    
    def draw(self, title: str = None, dpi: int = 200, 
             engine: str = 'dot',
             alignment: str = 'horizontal',
             _dir: str = 'Graphs', _file: str = None, _show: bool = True,
             _show_isolated_states: bool = False,
             graph_kwargs: dict = None,
             edge_kwargs: dict = None,
             node_kwargs: dict = None,
             initial_node_kwargs: dict = None,
             final_node_kwargs: dict = None):
        
        # A wrapper for super-class .draw() method
        # Check kwargs
        if graph_kwargs is None:
            graph_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}
        if edge_kwargs is None:
            edge_kwargs = {}
        if initial_node_kwargs is None:
            initial_node_kwargs = {}
        if final_node_kwargs is None:
            final_node_kwargs = {}
        
        node_kwargs.update(dict(shape='ellipse'))
        initial_node_kwargs.update(dict(shape='ellipse'))
        return super().draw(title = title, dpi = dpi,
             engine = engine,
             alignment = alignment,
             _dir = _dir, _file = _file, _show = _show,
             _show_isolated_states = _show_isolated_states,
             graph_kwargs = graph_kwargs,
             edge_kwargs = edge_kwargs,
             node_kwargs = node_kwargs,
             initial_node_kwargs = initial_node_kwargs,
             final_node_kwargs = final_node_kwargs, use_a_star=False)
    
    def to_finite_automaton(self, name: str = None, make_final: Collection = None):
        """
        Derives an automaton from LTS.

        Parameters:
            name (str): Optional. How to entitle the new FA instance. 
            By default: FA(previous name).

        Returns:
            a new Automaton instance
        """

        assert make_final is None or isinstance(make_final, Collection), f'make_final must be Collectio, not {type(make_final)}'
        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'FA({self.name})'

        if make_final is None:
            make_final = set()
        elif isinstance(make_final, str):
            make_final = {make_final}

        final_states = set()
        for s in make_final:
            assert s in self.states, f'no such state in LTS {s}'
            final_states.add(s)

        from . import Automaton
        fa = Automaton(
            self.states,
            self.initial_states,
            final_states,
            self.transitions,
            self.symbols,
            name=name
        )

        return fa
    
    def deadlocks(self):
        """
        Finds and returns all "deadlock" states: states which have no outgoing transitions
        """

        deadlock_states = set()
        for state in self.states:
            out_transitions = {t for t in self.transitions if t[0] == state}
            if not out_transitions:
                deadlock_states.add(state)

        return deadlock_states
    
    def to_kripke(self, name: str = None):
        """
        Derives a Kripke from a (complete!) LTS.

        Parameters:
            name (str): Optional. How to entitle the new Kripke instance. 
            By default: Kripke(previous name).

        Returns:
            a new Kripke instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'Kripke({self.name})'

        assert self.is_complete, 'LTS is not complete, Kripke structure is not possible to construct'

        kripke_initial = set(product(list(self.initial_states), list(self.alphabet)))
        kripke_initial = {f'{s[0]}@{s[1]}' for s in kripke_initial}
        kripke_transitions = set()
        kripke_states = kripke_initial.copy()

        added_at_previous_step = kripke_initial.copy()

        while added_at_previous_step:

            added_at_current_step = set()

            # Finally, a new transition is added going to the newly created node.
            for state in added_at_previous_step:

                valid_transitions = list(filter(lambda t: t[0] == state.split('@')[0] and t[1] == state.split('@')[1], self.transitions))
                new_states = [t[2] for t in valid_transitions]
                new_states = product(new_states, list(self.alphabet))
                new_states = {f'{s[0]}@{s[1]}' for s in new_states}

                for new_state in new_states:
                    kripke_transitions.add((state, new_state))
                    added_at_current_step.add(new_state)

            added_at_previous_step = added_at_current_step - kripke_states
            kripke_states |= added_at_previous_step

        annotations = {s.replace('@', ';'): s.split('@')[1] for s in kripke_states}
        kripke_states = {s.replace('@', ';') for s in kripke_states}
        kripke_initial = {s.replace('@', ';') for s in kripke_initial}
        kripke_transitions = {(t[0].replace('@', ';'), t[1].replace('@', ';')) for t in kripke_transitions}

        from . import Kripke
        kripke = Kripke(
            kripke_states,
            kripke_initial,
            kripke_transitions,
            annotations,
            name=name
        )

        return kripke
