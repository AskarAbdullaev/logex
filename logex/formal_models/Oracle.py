from typing import Collection, Union, Hashable
from itertools import product

from . import Graph

class Oracle(Graph):

    """
    
    Main class for Oracle (there is another class for Finite Automata).
    It accepts the following notation (however it is mainly designed to automatically constructed from Finite Automata):

    fa = Oracle(
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
                                (state_1, (symbol, index), state_2)
                index can be an integer for Optimized Oracles or strings (or other Hashable object) for a simple Oracle
                (in the latter case, indices must be states)
                Index can also be replaced by "*" if any index is valid.
            alphabet:
                collection (preferably set) of 2-tuples allowed. It is an optional argument. If alphabet is not provided,
                it will be restored from the transitions. As long as there might be very large alphabets for Oracles,
                using this argument is not recommended.
            name:
                name of the oracle. By default, it is named 'A'.
        )

    Remark:
        - symbols of the alphabet (first entries of tuples) can be strings or integers;

    Example:
    oracle = Oracle({'A','B', 'C', 'D'},
                        {'A'},
                        {'D'}, 
                        {('C', ('b', 'D'), 'D'), ('D', ('b', 'D'), 'D'), ('A', ('a', 'C'), 'C'), ('A', ('b', 'B'), 'B'),
	                            ('D', ('a', 'D'), 'D'), ('B', ('b', 'D'), 'D')}, 
                        "my_oracle")

    Oracle has a typical string representation of the form A = (S, I, Σ, T, F).

    Once created oracle becomes a Callable object.
    You can pass an arbitrary word and check if it is accepted (True) or not accepted (False):

    oracle('bbb')
    True
    oracle('aaa')
    False

    It is possible to check if oracle is deterministic or complete:

    Properties:
        .is_complete
        .is_deterministic
        .is_optimized
    Return just a boolean answer True or False

    Functions:
        .is_complete_explain()
        .is_deterministic_explain()
    Also return boolean answer but also print down a short explanation

    Methods:

    .draw(title=None, seed=None) -> seed

        Allows to draw an oracle using "networkx" module and matplotlib figures. 
        Uses networkx implementation of Kamada-Kawai algorithm. Might require several attempts
        to obtain a good-looking graph. Returns a seed used. After a nice plot is created,
        copy the seed and pass as an argument to preserve the result.

    .complement_automaton(name=None)

        Allows to derive a complement automaton. At first a PA will be created.

    .to_finite_automaton(name=None)

        Returns a Finite Automaton instance derived from the oracle

    .to_optimized_oracle(name=None)

        Returns an optimized oracle instance.
    """

    AnySymbol = '*'
    AnySymbol2 = '∗'
    EmptyState = '∅'
    EmptyWord = '𝜖'

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
            assert state in states, f'initial state {state} is not in states'
            
        # Check transitions -> set of 3-tuples (str, tuple[str, str], str) -> check the existence of states
        symbols_in_transitions = set()
        indices_in_transitions = set()
        assert isinstance(transitions, Collection), 'transitions must be a Collection'
        for transition in transitions:
            assert len(transition) == 3, f'transitions must be 3-tuples, not {len(transition)} ({transition})'
            assert str(transition[0]) in states, f'In transition {transition} starting state {transition[0]} is not defined in states'
            assert str(transition[2]) in states, f'In transition {transition} destination state {transition[2]} is not defined in states'
            assert isinstance(transition[1], tuple) and len(transition[1]) == 2, f'transition[1]  must be a 2-tuple, not {type(transition[1])} ({transition})'
            symbol, index = transition[1]
            symbols_in_transitions.add(str(symbol))
            indices_in_transitions.add(str(index))
        transitions = {(str(t[0]), (str(t[1][0]), str(t[1][1])), str(t[2])) for t in transitions}

        # Check alphabet if provided -> set of tuples[str, str]
        assert not alphabet or isinstance(alphabet, Collection), 'alphabet must be None or a Collection'
        if alphabet:
            for alpha in alphabet:
                assert isinstance(alpha, tuple) and len(alpha) == 2, f'alphabet entry must be a 2-tuple, not {type(alpha)} ({alpha})'
                symbol, index = alpha
                assert isinstance(symbol, Union[int, str]), f'alphabet symbols must be of type str or int, not {type(symbol)} ({symbol})'
            alphabet = {(str(a[0]), str(a[1])) for a in alphabet if self.AnySymbol not in a and self.AnySymbol2 not in a}
        if not alphabet:
            symbols_in_transitions -= {self.AnySymbol, self.AnySymbol2}
            indices_in_transitions -= {self.AnySymbol, self.AnySymbol2}
            alphabet = set(product(list(symbols_in_transitions), list(indices_in_transitions)))
            assert len(transitions) > 0 and alphabet, 'alphabet is not provided and cannot be derived from the transitions'
        symbols = {a[0] for a in alphabet}
        indices = {a[1] for a in alphabet}

        # Check that transitions have symbols from the alphabet and replace "*" with several transitions
        additional_transitions = set()
        junk_transitions = set()
        for transition in transitions:
            start, t, end = transition
            symbol, index = t
            if symbol in {self.AnySymbol, self.AnySymbol2} or index in {self.AnySymbol, self.AnySymbol2}:
                junk_transitions.add(transition)
                for s in symbols:
                    for i in indices:
                        additional_transitions.add((start, (s if symbol in {self.AnySymbol, self.AnySymbol2} else symbol,
                                                            i if index in {self.AnySymbol, self.AnySymbol2} else index), end))
            else:
                assert t in alphabet, f'transition symbol {t} in transition {transition} is not in the alphabet'
        transitions -= junk_transitions
        transitions |= additional_transitions

        # Main attributes of Finite Automaton
        super().__init__(states, initial_states, final_states, transitions, symbols)
        self.alphabet = alphabet
        self.class_name = 'Oracle'
        self.rotate_edge_labels = True
        self.name = name if name else 'A'

        if len(indices) > len(states) - 1:
            self.is_optimized = True
        else:
            self.is_optimized = False

    def __str__(self):
        string = f'Oracle {self.name} = (S, I, Σ, T, F)\n'
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
    
    def _extract_symbol_from_transition(self, transition):
        """
        Helper function to extract the transition symbol
        """
        return transition[1][0]
    
    def _prepare_transitions_for_drawing(self, use_a_star = True):
        """
        Helper function to merge similar transitions into labels
        """
        
        simplified_transitions = set()

        if not self.is_optimized:

            # For non-optimized Oracles
            for state1 in self.states:
                for state2 in self.states:
                    symbols = set()
                    for symbol in self.alphabet:
                        if (state1, symbol, state2) in self.transitions:
                            symbols.add(symbol)
                    if symbols:
                        if len(symbols) == len(self.symbols) and use_a_star:
                            strings = {f'({self.AnySymbol},{state2})'}
                        else:
                            strings = {f'({s[0]},{s[1]})' for s in symbols}
                        symbols = (strings.pop() if len(strings) == 1
                                    else ', '.join(sorted(strings)))
                        simplified_transitions.add((state1, symbols, state2))
        else:

            # For optimized Oracles:
            max_n = 0
            for state in self.states:
                for symbol in self.symbols:
                    n = len(list(filter(lambda t: t[0] == state and symbol == self._extract_symbol_from_transition(t),
                                        self.transitions)))
                    if n > max_n:
                        max_n = n

            for state1 in self.states:
                for state2 in self.states:
                    symbols = set()
                    for symbol in self.alphabet:
                        if (state1, symbol, state2) in self.transitions:
                            symbols.add(symbol)
                    string = ''
                    if not symbols:
                        continue
                    for u_s in self.symbols:
                        grouped_symbols = [s for s in symbols if s[0] == u_s]
                        if len(grouped_symbols) == max_n and use_a_star:
                            string += f'({u_s},{self.AnySymbol})'
                        elif grouped_symbols:
                            strings = {f'({u_s},{s[1]})' for s in grouped_symbols}
                            string += (strings.pop() if len(strings) == 1
                                        else ', '.join(sorted(strings)))
                    simplified_transitions.add((state1, string, state2))
        return simplified_transitions
    
    def to_optimized_oracle(self, name: str = None):
        """
        Derives an optimized oracle.

        Parameters:
            name (str): Optional. How to entitle the new Oracle instance. 
            By default: OptO(previous name).

        Returns:
            a new Oracle instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'Opt {self.name}'

        # Find the maximum number of same-symbol transitions from a single state
        max_n = 0
        for state in self.states:
            for symbol in self.symbols:
                n = len(list(filter(lambda t: t[0] == state and symbol == self._extract_symbol_from_transition(t),
                                    self.transitions)))
                if n > max_n:
                    max_n = n

        # For every state-symbol pair, find their m successors (neighbours).
        # Assign modulo indices.
        changed_transitions = set()
        for state in self.states:
            for symbol in self.symbols:
                neighbours = list(filter(lambda t: t[0] == state and symbol == self._extract_symbol_from_transition(t),
                                         self.transitions))
                m = len(neighbours)
                if m == 0:
                    continue
                groups = [list(range(max_n))[i::m] for i in range(m)]
                for group, n in zip(groups, neighbours):
                    for index in group:
                        changed_transitions.add((n[0], (symbol, index), n[-1]))

        # Return the new instance
        opt_oracle = Oracle(
            self.states,
            self.initial_states,
            self.final_states,
            changed_transitions,
            set(product(self.symbols, list(range(max_n)))),
            name=name
        )
        opt_oracle.is_optimized = True
        return opt_oracle

    def to_finite_automaton(self, name: str = None):
        """
        Derives an automaton from oracle.

        Parameters:
            name (str): Optional. How to entitle the new FA instance. 
            By default: FA(previous name).

        Returns:
            a new Automaton instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = self.name.replace('Opt ', '').replace('(', '').replace(')', '')

        from . import Automaton
        fa = Automaton(
            self.states,
            self.initial_states,
            self.final_states,
            {(t[0], self._extract_symbol_from_transition(t), t[2]) for t in self.transitions},
            self.symbols,
            name=name
        )

        return fa
    
    def to_complement(self, name: str = None):
        """
        Derives a complement oracle from oracle.

        Parameters:
            name (str): Optional. How to entitle the new Oracle instance. 
            By default: C(previous name).

        Returns:
            a new fm.Oracle instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'C({name})'

        fa = self.to_finite_automaton()
        c_fa = fa.to_complement()
        if self.is_optimized:
            c_oracle = c_fa.to_optimized_oracle(name=name)
        else:
            c_oracle = c_fa.to_oracle(name=name)

        return c_oracle
