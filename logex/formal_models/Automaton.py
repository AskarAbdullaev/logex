from typing import Collection, Union, Hashable, Any
from itertools import product

from . import Graph
from . import tools

class Automaton(Graph):

    """
    
    Main class for Finite Automatons (there is another class for Oracles).
    It accepts the following notation:

    fa = Automaton(
            states: 
                collection (preferably set) of states as Hashable objects (strings, integers etc.),
            initial_states: 
                subset of states defined as initial states
                if there are no initial states, still provide an empty collection
                if there is a single initial state, it can be optionally provided as an object (not as a collection),
            final_states:
                subset of states defined as final states
                if there are no final states, still provide an empty collection
                if there is a single final state, it can be optionally provided as an object (not as a collection)
            transitions:
                collection (preferably set) of transitions with a certain form:
                each transition is a 3-tuple defined as:
                                (state_1, symbol, state_2)
                symbol can be replaced by "*" if you mean that any symbol from the alphabet if valid here
            alphabet:
                collection (preferably set) of symbols allowed. It is an optional argument. If alphabet is not provided,
                it will be restored from the transitions. However, if there are only "*" symbols in the transitions,
                restoration will not be possible.
            name:
                name of the automaton. By default, it is named 'A'.
        )

    Remark:
        - symbols of the alphabet can be strings or integers;

    Example:
    fa = Automaton({'A','B', 'C', 'D'},
                        {'A'},
                        {'D'}, 
                        {('A', 'b', 'B'), ('A', 'a', 'C'), ('B', '*', 'D'), ('C', '*', 'D'), ('D', '*', 'D')}, 
                        {'a','b'},
                        "my_automaton")

    Automaton has a typical string representation of the form A = (S, I, Σ, T, F).

    Once created automaton becomes a Callable object.
    You can pass an arbitrary word and check if it is accepted (True) or not accepted (False):

    fa('bbb')
    True
    fa('aaa')
    False

    It is possible to check if automaton is deterministic or complete:

    Properties:
        .is_complete
        .is_deterministic
        .is_power_automaton
    Both return just a boolean answer True or False

    Functions:
        .is_complete_explain()
        .is_deterministic_explain()
    Also return boolean answer but also print down a short explanation

    Methods:

    .draw(title=None, seed=None) -> seed

        Allows to draw an automaton using "networkx" module and matplotlib figures. 
        Uses networkx implementation of Kamada-Kawai algorithm. Might require several attempts
        to obtain a good-looking graph. Returns a seed used. After a nice plot is created,
        copy the seed and pass as an argument to preserve the result.

    .to_power_automaton(name=None)

        Allows to derive a power automaton. If already, just a copy is returned.

    .complement_automaton(name=None)

        Allows to derive a complement automaton. At first a PA will be created.

    .to_oracle(name=None)

        Returns a simple Oracle() instance.

    .to_optimized_oracle(name=None)

        Returns an optimized oracle instance.
    """

    

    def __init__(self,
                 states: Collection[Union[int, str]],
                 initial_states: Collection[Union[int, str]] | str | None,
                 final_states: Collection[Union[int, str]] | str | None,
                 transitions: Collection[tuple],
                 alphabet: Collection = None,
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

        # Check final states -> set of strings -> present in states set
        assert isinstance(final_states, Collection | str | None), 'states must be a Collection / str / None'
        if isinstance(final_states, str):
            final_states = {final_states}
        elif final_states is None:
            final_states = set()
        else:
            final_states = set(list(map(str, final_states)))
        for state in final_states:
            assert state in states, f'final state {state} is not in states'
            
        # Check transitions -> set of 3-tuples (str, str, str) -> check the existence of states
        symbols_in_transitions = set()
        assert isinstance(transitions, Collection), 'transitions must be a Collection'
        for transition in transitions:
            assert len(transition) == 3, f'transitions must be 3-tuples, not {len(transition)} ({transition})'
            assert isinstance(transition[1], Union[str, int]), f'transition condition must be int or str, not {type(transition[1])} ({transition})'
            assert str(transition[0]) in states, f'In transition {transition} starting state {transition[0]} is not defined in states'
            assert str(transition[2]) in states, f'In transition {transition} destination state {transition[2]} is not defined in states'
            symbols_in_transitions.add(str(transition[1]))
        transitions = {(str(t[0]), str(t[1]), str(t[2])) for t in transitions}

        # Check alphabet if provided -> set of strings
        assert not alphabet or isinstance(alphabet, Collection), 'alphabet must be None or a Collection'
        if alphabet:
            for symbol in alphabet:
                assert isinstance(symbol, Union[int, str]), f'alphabet symbols must be of type str or int, not {type(symbol)} ({symbol})'
            alphabet = {str(a) for a in alphabet} - {self.AnySymbol, self.AnySymbol2}
        if not alphabet:
            alphabet = symbols_in_transitions - {self.AnySymbol, self.AnySymbol2}
            assert len(transitions) > 0 and alphabet, 'alphabet is not provided and cannot be derived from the transitions'

        # Check that transitions have symbols from the alphabet and replace "*" with several transitions
        additional_transitions = set()
        junk_transitions = set()
        for transition in transitions:
            start, t, end = transition
            if t in {self.AnySymbol, self.AnySymbol2}:
                junk_transitions.add(transition)
                for symbol in alphabet:
                    additional_transitions.add((start, symbol, end))
            else:
                assert t in alphabet, f'symbol {t} in transition {transition} is not in the alphabet'
        transitions -= junk_transitions
        transitions |= additional_transitions

        # Main attributes of Finite Automaton
        super().__init__(states, initial_states, final_states, transitions, alphabet)
        self.alphabet = alphabet
        self.class_name = 'Finite Automaton'
        self.rotate_edge_labels = False
        self.name = name if name else 'A'

    def __str__(self):
        string = f'Automaton {self.name} = (S, I, Σ, T, F)\n'
        string += f'Alphabet Σ = {self.alphabet}\n'
        string += f'States S = {self.states}\n'
        string += f'Initial states I = {self.initial_states}\n'
        string += f'Final states F = {self.final_states}\n'
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

    @property
    def is_power_automaton(self) -> bool:
        """
        Checks if the automaton is deterministic and complete

        Returns:
            bool
        """
        return self.is_complete and self.is_deterministic

    def to_power_automaton(self, name: str = None):
        """
        Allows to derive a PowerAutomaton.

        Parameters:
            name (str): Optional. How to entitle the new Automaton instance. By default: P(previous name).

        Returns:
            a new Automaton instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'P({self.name})'

        # If it is already a power automaton: just return a copy
        if self.is_power_automaton:
            return Automaton(self.states, self.initial_states, self.final_states, self.transitions, self.alphabet)

        # Helper functions to switch between old and new state names
        def concatenate(states_to_merge):
            states_to_merge = list(set(states_to_merge))
            _is_final = False
            for s in states_to_merge:
                if s in self.final_states:
                    _is_final = True
                    break
            return ';'.join(sorted(states_to_merge)), _is_final

        def unbind(power_state):
            return power_state.split(';')

        # Defining parameters of the future PA
        power_alphabet = self.alphabet
        power_states, power_transitions, power_final_states, power_initial_state = set(), set(), set(), set()

        # Bundling initial states together
        initial, is_final = concatenate(self.initial_states)
        power_initial_state = initial
        power_states.add(initial)

        # For every new state it is immediately checked if it belongs to final states of PA
        if is_final:
            power_final_states.add(initial)

        # PA construction is performed step-by-step
        # At each step we take the set of newly added state at previous step "added_at_previous_step"
        # and going one step further, creating a set: "added_at_current_step"
        # The algorithm stops if no more states were added after a step
        added_at_previous_step = {initial}

        while added_at_previous_step:

            added_at_current_step = set()

            # Check every (state, symbol) pair to see what nodes can be reached and concatenate them
            # The newly concatenate node os added to the "added_at_current_step" and also checked
            # for being a final state.
            # Finally, a new transition is added going to the newly created node.
            for state in added_at_previous_step:
                for symbol in power_alphabet:

                    valid_transitions = list(filter(lambda t: t[0] in unbind(state) and t[1] == symbol, self.transitions))
                    if valid_transitions:
                        new_power_state, is_final = concatenate([v_t[2] for v_t in valid_transitions])
                    else:
                        new_power_state, is_final = tools.EMPTY_STATE, False

                    if is_final:
                        power_final_states.add(new_power_state)

                    new_transition = (state.replace(';', ','), symbol, new_power_state.replace(';', ','))
                    power_transitions.add(new_transition)

                    added_at_current_step.add(new_power_state)

            added_at_previous_step = added_at_current_step - power_states
            power_states |= added_at_previous_step


        power_states = {s.replace(';', ',') for s in power_states}
        power_initial_state = power_initial_state.replace(';', ',')
        power_final_states = {s.replace(';', ',') for s in power_final_states}
        return Automaton(power_states, power_initial_state, power_final_states,
                            power_transitions, power_alphabet, name)

    def to_complement(self, name: str = None):

        """
        Derives the Complement automaton.
        
        Parameters:
            name (str): Optional. How to entitle the new Automaton instance. By default: C(previous name).

        Returns:
            a new Automaton instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'C({self.name})'

        # If self is already deterministic and complete, we can just invert final states a return a copy
        if self.is_power_automaton:

            complement = Automaton(
                self.states,
                self.initial_states,
                self.states - self.final_states,
                self.transitions,
                self.alphabet,
                name=name
            )

            return complement

        # Else, we at first create a PA and then invert ints final states
        pa = self.to_power_automaton()
        complement = Automaton(
                pa.states,
                pa.initial_states,
                pa.states - pa.final_states,
                pa.transitions,
                pa.alphabet,
                name=name
            )

        return complement

    def to_oracle(self, name: str = None):

        """
        Derives a simple oracle.

        Parameters:
            name (str): Optional. How to entitle the new Oracle instance. By default: O(previous name).

        Returns:
            a new Oracle instance
        """
        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'O({self.name})'

        # Just replace transition symbol with (symbol, destination) pair
        from . import Oracle
        oracle = Oracle(
            self.states,
            self.initial_states,
            self.final_states,
            {(t[0], (t[1], t[2]), t[2]) for t in self.transitions},
            set(product(self.alphabet, self.states)),
            name = name
        )
        oracle.is_optimized = False

        return oracle

    def to_optimized_oracle(self, name: str = None):
        """
        Derives an optimized oracle.

        Parameters:
            name (str): Optional. How to entitle the new Oracle instance. By default: OptO(previous name).

        Returns:
            a new Oracle instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'OptO({self.name})'

        # At first, derive a simple Oracle() instance
        oracle = self.to_oracle()

        # Convert to an optimized oracle
        opt_oracle = oracle.to_optimized_oracle(name=name)
        opt_oracle.is_optimized = True

        return opt_oracle
    
    def to_io_automaton(self, name: str = None):
        """
        Derives an I/O automaton.

        Parameters:
            name (str): Optional. How to entitle the new IOA instance. By default: IO(previous name).

        Returns:
            a new IOAutomaton instance
        """

        # Name correctness
        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'IO({self.name})'

        # At first, create a Power Automaton
        pa = self.to_power_automaton()
        input_alphabet = pa.alphabet
        output_alphabet = {'0', '1'}
        states = pa.states
        initial_state = list(pa.initial_states)[0]

        # Transitions quitting the final states must have output '1', otherwise - '0'
        transitions = set()
        for transition in pa.transitions:
            if transition[0] in pa.final_states:
                transitions.add((transition[0], (transition[1], '1'), transition[2]))
            else:
                transitions.add((transition[0], (transition[1], '0'), transition[2]))

        from . import IOAutomaton
        ioa = IOAutomaton(
            states,
            initial_state,
            transitions,
            input_alphabet,
            output_alphabet,
            name
        )

        return ioa

    def product(self, fa: Any, name: str = None, complete_notation: bool = False):
        """
        Constructing a new Automaton as a product of the current one and a given one.

        Args:
            fa (Any): 
                second automaton instance
            name (str, optional): 
                a new name for the product automaton. Defaults to None.
            complete_notation (bool, optional): 
                allows to show all states, even unreachable from initial states. Defaults to False.

        Returns:
            a new instance of Automaton
        """
        
        assert isinstance(fa, Automaton), f'product is defined for 2 finite automata, not {type(fa)}'
        assert name is None or isinstance(name, str), 'name must be a string'
        assert self.alphabet == fa.alphabet, 'alphabet is not the same, product is not possible'
        assert isinstance(complete_notation, bool), f'complete_notation must of type bool, not {type(complete_notation)}'
        if name is None:
            name = f'{self.name}*{fa.name}'

        # Helper functions to switch between old and new state names
        def concatenate(states_to_merge):
            _is_final = False
            if states_to_merge[0] in self.final_states and states_to_merge[1] in fa.final_states:
                _is_final = True
            return ';'.join(states_to_merge), _is_final

        def unbind(_state):
            return _state.split(';')

        # Defining parameters of the future PA
        product_alphabet = self.alphabet
        product_states, product_transitions, product_final_states, product_initial_states = set(), set(), set(), set()

        # Creating initial states
        initial_cross_product = set(product(self.initial_states, fa.initial_states))
        for initial in initial_cross_product:
            new_initial, is_final = concatenate(initial)
            product_initial_states.add(new_initial)
            if is_final:
                product_final_states.add(initial)
        product_states |= product_initial_states

        if not product_initial_states:
            raise ValueError('There must be initial states in both Automata to construct their product')

        added_at_previous_step = product_initial_states

        while added_at_previous_step:

            added_at_current_step = set()

            # Check every (state, symbol) pair to see what nodes can be reached and concatenate them
            # The newly concatenate node os added to the "added_at_current_step" and also checked
            # for being a final state.
            # Finally, a new transition is added going to the newly created node.
            for state in added_at_previous_step:
                for symbol in product_alphabet:

                    state_1, state_2 = unbind(state)
                    valid_transitions_1 = set(filter(lambda t: t[0] == state_1 and symbol == t[1], 
                                                     self.transitions))
                    valid_transitions_2 = set(filter(lambda t: t[0] == state_2 and symbol == t[1], 
                                                     fa.transitions))

                    if not valid_transitions_1 or not valid_transitions_2:
                        continue
                    
                    states_1, states_2 = {t[2] for t in valid_transitions_1}, {t[2] for t in valid_transitions_2}

                    raw_pairs = product(states_1, states_2)
                    for pair in raw_pairs:
                        new_state, is_final = concatenate(pair)

                        if is_final:
                            product_final_states.add(new_state)

                        new_transition = (state.replace(';', ','), symbol, new_state.replace(';', ','))
                        product_transitions.add(new_transition)
                        added_at_current_step.add(new_state)

            added_at_previous_step = added_at_current_step - product_states
            product_states |= added_at_previous_step

        
        product_states = {s.replace(';', ',') for s in product_states}
        product_initial_states = {s.replace(';', ',') for s in product_initial_states}
        product_final_states = {s.replace(';', ',') for s in product_final_states}

        # If complete notation is required - enrich new automaton with every possible
        # state (combination of previous automatons)
        if complete_notation:
            for pair in product(self.states, fa.states):
                cross_state, is_final = concatenate(pair)
                product_states.add(cross_state.replace(';', ','))
                if is_final:
                    product_final_states.add(cross_state.replace(';', ','))

        return Automaton(product_states, product_initial_states, product_final_states,
                            product_transitions, product_alphabet, name)
    
    def to_lts(self, name: str = None):

        """
        Derives LTS from Automaton. (Final states info will be lost!)

        Parameters:
            name (str): Optional. How to entitle the new LTS instance.
            By default: LTS(previous name).

        Returns:
            a new LabeledTransitionSystem instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'LTS({self.name})'

        from . import LabeledTransitionSystem
        lts = LabeledTransitionSystem(
            self.states,
            self.initial_states,
            self.transitions,
            self.alphabet,
            name=name
        )

        return lts
    
    def to_kripke(self, name: str = None, mode: str = 'i+s+o', linker: str = '', 
                  no_annotations: bool = False, no_initial_states: bool = False):

        """
        Derives Kripke from Automaton. (Final states info will be lost!)

        Parameters:
            name (str): Optional. How to entitle the new Kripke instance.
            By default: Kripke(previous name).

        Returns:
            a new LabeledTransitionSystem instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'Kripke({self.name})'

        io = self.to_io_automaton()
        kripke = io.to_kripke(name=name, mode=mode, linker=linker, 
                              no_annotations=no_annotations, no_initial_states=no_initial_states)

        return kripke