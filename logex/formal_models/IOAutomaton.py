from typing import Collection, Union, Hashable, Sequence
from itertools import product

from . import Graph
from . import tools

class IOAutomaton(Graph):
    """
    I/O Automaton class. What makes it different from FA:
    - only 1 initial state;
    - transitions must also contain an output: (state_1, (in, out), state_2). 
    The notation is very similar to Oracles.

    Args:
        states (Collection[Union[int, str]]): states of I/O Automaton
        initial_state (str): exactly one initial state
        transitions (Collection[tuple]): transitions
        input_alphabet (Collection): mandatory input alphabet
        output_alphabet (Collection): mandatory output alphabet
        name (str, optional): a name for I/O Automaton. Defaults to None.
    """

    def __init__(self,
                 states: Collection[Union[int, str]],
                 initial_state: str,
                 transitions: Collection[tuple],
                 input_alphabet: Collection,
                 output_alphabet: Collection,
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
        assert isinstance(initial_state, str), f'initial_state nust be of type str, not {type(initial_state)}'
        initial_state = {initial_state}
        for state in initial_state:
            assert state in states, f'initial state {state} is not in states'

        # Check input alphabet
        assert isinstance(input_alphabet, Collection), f'input_alphabet must be a Collection, not {type(input_alphabet)}'
        for symbol in input_alphabet:
            assert isinstance(symbol, Hashable), f'alphabet symbols must be Hashable, not {type(symbol)} ({symbol})'
        input_alphabet = set(list(map(str, input_alphabet)))

        # Check output alphabet
        assert isinstance(output_alphabet, Collection), f'output_alphabet must be a Collection, not {type(output_alphabet)}'
        for symbol in output_alphabet:
            assert isinstance(symbol, Hashable), f'alphabet symbols must be Hashable, not {type(symbol)} ({symbol})'
        output_alphabet = set(list(map(str, output_alphabet)))
            
        # Check transitions -> set of 3-tuples (str, str, str) -> check the existence of states
        assert isinstance(transitions, Collection), 'transitions must be a Collection'
        for transition in transitions:
            assert len(transition) == 3, f'transitions must be 3-tuples, not {len(transition)} ({transition})'
            assert isinstance(transition[1], tuple) and len(transition[1]) == 2, f'transition condition must be a 2-tuple, not {type(transition[1])} ({transition})'
            assert str(transition[0]) in states, f'In transition {transition} starting state {transition[0]} is not defined in states'
            assert str(transition[2]) in states, f'In transition {transition} destination state {transition[2]} is not defined in states'
        transitions = {(str(t[0]), (str(t[1][0]), str(t[1][1])), str(t[2])) for t in transitions}

        # Check that transitions have symbols from the alphabet and replace "*" with several transitions
        additional_transitions = set()
        junk_transitions = set()
        for transition in transitions:
            start, t, end = transition
            in_, out_ = t
            if in_ in {self.AnySymbol, self.AnySymbol2} or out_ in {self.AnySymbol, self.AnySymbol2}:
                junk_transitions.add(transition)
                for i in input_alphabet:
                    for o in output_alphabet:
                        additional_transitions.add((start, (i if in_ in {self.AnySymbol, self.AnySymbol2} else in_,
                                                            o if out_ in {self.AnySymbol, self.AnySymbol2} else out_), end))
            else:
                assert in_ in input_alphabet, f'input symbol {in_} in transition {transition} is not in the input alphabet'
                assert out_ in output_alphabet, f'output symbol {out_} in transition {transition} is not in the output alphabet'
        transitions -= junk_transitions
        transitions |= additional_transitions

        # Checking the completeness of automaton
        for transition in transitions:
            start, t, _ = transition
            for symbol in input_alphabet:
                state_plus_symbol = {transition for transition in transitions 
                                     if transition[0] == start and symbol == self._extract_symbol_from_transition(transition)}
                assert len(state_plus_symbol) <= 1, f'there are several transitions from state {start} using input {symbol}'
            
        # Main attributes of Finite Automaton
        super().__init__(states, initial_state, set(), transitions, input_alphabet | output_alphabet)
        self.input_alphabet = input_alphabet
        self.output_alphabet = output_alphabet
        self.class_name = 'I/O Automaton'
        self.rotate_edge_labels = False
        self.name = name if name else 'A'

    def __str__(self):
        string = f'I/O Automaton {self.name} = (S, i, Σ, T , Θ, O)\n'
        string += f'Input Alphabet Σ = {self.input_alphabet}\n'
        string += f'Output Alphabet Θ = {self.output_alphabet}\n'
        string += f'States S = {self.states}\n'
        string += f'Initial state i = {list(self.initial_states)[0]}\n'
        output_function = {f'O({t[0], t[1][0]}) = {t[1][1]}' for t in self.transitions}
        string += 'Output function O =\n'
        for o_f in output_function:
            string += f'\t{o_f}\n'
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
        return transition[1][0]
    
    def _prepare_transitions_for_drawing(self, use_a_star = True):
        """
        Helper function to merge similar transitions into labels
        
        use_a_star is a remnant of super-class method and does nothing in this case
        """
        
        simplified_transitions = set()

        # For non-optimized Oracles
        for state1 in self.states:
            for state2 in self.states:
                symbols = set()
                for in_ in self.input_alphabet:
                    for out_ in self.output_alphabet:
                        if (state1, (in_, out_), state2) in self.transitions:
                            symbols.add((in_, out_))
                if symbols:
                    symbols = ', '.join([f'{s[0]}/{s[1]}' for s in symbols])
                    simplified_transitions.add((state1, symbols, state2))
    
        return simplified_transitions
    
    def __call__(self, word: Sequence, return_path: bool = False) -> str | bool | tuple[str | bool, str]:
        """
        Pass a word to the I/O Automaton

        Args:
            word (str): word to check
            return_path (bool): allows to show the path

        Returns:
            str of outputs
        """

        assert isinstance(word, Sequence), f'word must be str, not with {type(word)} ({word})'
        if word == tools.EMPTY_WORD:
            word = ''
        for letter in word:
            assert letter in self.input_alphabet, f'letter {letter} in the word in not in the input alphabet'

        # Initialize a set of states reachable at the current step of word traversing
        currently_in = list(self.initial_states)[0]

        output = ''
        path = [currently_in]

         # Iterating along the word letters
        for letter in word:

            # Searching for all the states reachable from the current set of stats with respect to the current letter
            valid_transitions = list(filter(lambda t: t[0] == currently_in and letter == self._extract_symbol_from_transition(t),
                                            self.transitions))
            if not valid_transitions:
                return False
            
            currently_in = valid_transitions[0][2]
            output += valid_transitions[0][1][1]

            if return_path:
                path.append(f'{letter}("{output[-1]}")')
                path.append(currently_in)

        # Constructing an exemplary path for a word through the automaton (only if accepted)
        if return_path:
            path = ['START'] + path + ['END']
            path = ' -> '.join([f'[{p}]' if i % 2 == 1 else str(p) for i, p in enumerate(path)])
            return output, path

        return output
    
    def to_finite_automaton(self, name: str = None, 
                            target_outputs: Collection = None):
        """
        Derives a finite automaton from I/O automaton.

        Parameters:
            name (str): Optional. How to entitle the new FA instance. 
            By default: FA(previous name).
            target_outputs: outputs that indicate final states. By default: "1".

        Returns:
            a new Automaton instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = self.name.replace('Opt ', '').replace('(', '').replace(')', '')
        if target_outputs is not None:
            assert isinstance(target_outputs, Collection), f"target_output must be a Collection, not {type(target_outputs)}"
            target_outputs = set(target_outputs)
            for output in target_outputs:
                assert output in self.output_alphabet, f'there is no symbol {output} in the output alphabet'
        else:
            target_outputs = {"1"}
        

        final_states = set()
        for transition in self.transitions:
            if transition[1][1] in target_outputs:
                final_states.add(transition[0])

        alphabet = {f'({t[1][0]},{t[1][1]})' for t in self.transitions}

        from . import Automaton
        fa = Automaton(
            self.states,
            self.initial_states,
            final_states,
            {(t[0], f'({t[1][0]},{t[1][1]})', t[2]) for t in self.transitions},
            alphabet,
            name=name
        )

        return fa
    
    def to_kripke(self, name: str = None, mode: str = 'i+s+o', linker: str = '', 
                  no_annotations: bool = False, no_initial_states: bool = False):
        """
        Derives a Kripke from a I/O Automaton.

        Parameters:
            name (str): Optional. How to entitle the new Kripke instance. 
            mode (str): mode of concatenation state, input and output. By default: "i+s+o".
            linker (str): a symbol to concatenate state names with. By default - empty string.
            no_annotations (bool): allows to suppress automatic annotations. Defaults to False.
            no_initial_states (bool): allows to suppress initial states;. Defaults to False
            By default: Kripke(previous name).

        Returns:
            a new Kripke instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'Kripke({self.name})'
        assert isinstance(mode, str), f'mode must be a str, not {type(mode)}'
        assert len(mode.split('+')) == 3, f'mode must consist of letter "s", "i", "o" concatenated with a "+"'
        mode = mode.split('+')
        assert set(mode) == {'i', 's', 'o'}, f'mode must consist of letter "s", "i", "o" concatenated with a "+"'
        assert isinstance(linker, str), f'linker must be a string'

        def reconstruct_state_names(s: str):
            s_, i_, o_ = s.split('@')
            info = {
                "s": s_, "i": i_, "o": o_
            }
            new_name = linker.join([info[x] for x in mode])
            return new_name


        kripke_initial_raw = set(product(list(self.initial_states), list(self.input_alphabet)))
        kripke_initial_raw = {f'{s[0]}@{s[1]}' for s in kripke_initial_raw}
        kripke_initial = set()
        for initial in kripke_initial_raw:
            state, input_ = initial.split('@')
            output = list(filter(lambda t: t[0] == state and t[1][0] == input_, self.transitions))[0]
            output = output[1][1]
            kripke_initial.add(initial + '@' + output)

        kripke_transitions = set()
        kripke_states = kripke_initial.copy()

        added_at_previous_step = kripke_initial.copy()

        while added_at_previous_step:

            added_at_current_step = set()

            # Finally, a new transition is added going to the newly created node.
            for state in added_at_previous_step:

                valid_transition = list(filter(lambda t: t[0] == state.split('@')[0] and t[1][0] == state.split('@')[1][0], self.transitions))[0]

                s, (i, o), new_s = valid_transition

                for i in self.input_alphabet:

                    output = list(filter(lambda t: t[0] == new_s and t[1][0] == i, self.transitions))[0]
                    output = output[1][1]
                    new_state = f'{new_s}@{i}@{output}'
                    kripke_transitions.add((state, new_state))
                    added_at_current_step.add(new_state)

            added_at_previous_step = added_at_current_step - kripke_states
            kripke_states |= added_at_previous_step

        if not no_annotations:
            annotations = {reconstruct_state_names(s): s.split('@')[1] for s in kripke_states}
        else:
            annotations = None
        kripke_states = {reconstruct_state_names(s) for s in kripke_states}
        if not no_initial_states:
            kripke_initial = {reconstruct_state_names(s) for s in kripke_initial}
        else:
            kripke_initial = None
        kripke_transitions = {(reconstruct_state_names(t[0]), reconstruct_state_names(t[1])) for t in kripke_transitions}

        from . import Kripke
        kripke = Kripke(
            kripke_states,
            kripke_initial,
            kripke_transitions,
            annotations,
            name=name
        )

        return kripke