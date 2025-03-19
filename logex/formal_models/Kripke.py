from typing import Collection, Union, Hashable, Any, Sequence
import regex as re

from . import Graph
from . import tools

class Kripke(Graph):
    """
    Kripke structure instance, sub-class of Graph() but with no transition labels 
    and sates annotated with propositional variables

    Args:
        states (Collection[Union[int, str]]): states as a collection
        initial_states (Collection[Union[int, str]] | str | None): initial states
        transitions (Collection[tuple]):transitions as 2-tuples (state_1, state_2)
        annotations (dict[Union[str, dict, Collection]], optional): annotations as a dict. Defaults to None.
        name (str, optional): name for Kripke Structure. Defaults to None.
    """

    def __init__(self,
                 states: Collection[Union[int, str]],
                 initial_states: Collection[Union[int, str]] | str | None,
                 transitions: Collection[tuple],
                 annotations: dict[Union[str, dict, Collection]] = None,
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
            
        # Check transitions -> set of 2-tuples (state_1, state_2)
        assert isinstance(transitions, Collection), 'transitions must be a Collection'
        for transition in transitions:
            assert isinstance(transition, tuple) and len(transition) == 2, f'transitions must be 2-tuples, not {len(transition)} ({transition})'
            assert str(transition[0]) in states and str(transition[1]) in states, f'Transition {transition} include unknown states'
        transitions = {(str(transition[0]), str(transition[1])) for transition in transitions}

        # Check annotations -> dict[dict]
        propositionals = set()
        assert isinstance(annotations, dict | None), 'annotations must be a dict'
        if annotations is None:
            annotations = {s: {} for s in states}
        else:
            for state, annotation in annotations.items():
                assert str(state) in states, f'Unknown state {state} in annotations'
                assert isinstance(annotation, str | dict | Collection), f'each annotation must be a dict, not {type(annotation)}'
                if isinstance(annotation, str):
                    annotations[state] = {annotation: 1}
                elif not isinstance(annotation, dict) and isinstance(annotation, Collection):
                    annotations[state] = {str(prop): 1 for prop in annotation}
                for key, value in annotations[state].items():
                    assert isinstance(key, str), f'propositionals must be str, not {type(key)} ({key})'
                    assert isinstance(value, bool) or value in [1, 0], f'propositional values can be boolean or 1/0, not {type(value)} ({value})'
                    propositionals.add(key)
                    annotations[state][key] = bool(value)
        annotations = {
            str(s): {
                k: bool(v) for k, v in annotation.items()
                }
                       for s, annotation in annotations.items()
                       }
        for state in states:
            if state not in annotations.keys():
                annotations.update({state: {}})
            annotation = annotations[state]
            for prop in propositionals:
                if prop not in annotation.keys():
                    annotations[state].update({prop: False})

        # Main attributes of Finite Automaton
        super().__init__(states, initial_states, set(), transitions, set())
        self.propositionals = propositionals
        self.annotations = annotations
        self.class_name = 'Kripke Structure'
        self.name = name if name else 'K'
        self.traces = {}

    def _extract_symbol_from_transition(self, transition):
        return ''
    
    def _prepare_transitions_for_drawing(self, use_a_star = True):
        return {(t[0], '', t[1]) for t in self.transitions}

    def __str__(self):
        string = f'Kripke {self.name} = (S, I, T, L)\n'
        string += f'States S = {self.states}\n'
        string += f'Initial states I = {self.initial_states}\n'
        string += f'Propositionals = {self.propositionals}'
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
        string += 'Annotations L = \n'
        for key, ann in self.annotations.items():
            if ann:
                string += f'\t{key}: {str(ann)}\n'
        if string[-5:] == 'L = \n':
            string = string[:-1] + 'none\n'

        return string

    def __repr__(self):
        return str(self)
    
    def get_traces(self, from_state: str = None):
        """
        A function to create infinite traces from the Kripke structure
        It uses Breadth-First Search while pruning infinite loops
        and generating traces from them.
        """

        assert self.initial_states, 'Kripke Structure must have initial states to generate traces'
        assert isinstance(from_state, str) or from_state is None, f'from_state must be a str, not {type(from_state)}'
        assert from_state is None or from_state in self.states, f'There is no such state in Kripke Structure: "{from_state}"'

        if from_state in self.traces.keys():
            return self.traces[from_state]

        if not from_state:
            from_state = self.initial_states
        else:
            from_state = {from_state}

        def bfs(node: str, all_traces: set):
            """
            Helper function
            """
            

            neighbours = list(filter(lambda t: t[0] == node, self.transitions))
            neighbours = set([n[1] for n in neighbours])
            # print(f'BFS from node {node}: {neighbours}')

            new_traces = set()
            changed_traces = set()
            for trace in all_traces:
                if trace.is_finite and trace[trace.length - 1] == node:
                    changed_traces.add(trace)
                    # print(f'\tValid trace to append to: {trace}')
                    for n in neighbours:
                        if n in trace.unique():
                            # print('LOOP FOUND !!')
                            new_trace = KripkeTrace(parent=self)
                            loop, l = False, []
                            for s in trace:
                                if s == n:
                                    loop = True
                                if not loop:
                                    new_trace.add_state(s)
                                if loop:
                                    l.append(s)
                            new_trace.add_loop(tuple(l), 'inf')
                            new_traces.add(new_trace)
                        else:
                            new_trace = trace.copy()
                            new_trace.add_state(n)
                            new_traces.add(new_trace)
                        # print(f'\t\tNew trace made: {new_trace}')
            
            all_traces -= changed_traces

            if len(new_traces) == 0:
                return
            
            all_traces |= new_traces
            for n in neighbours:
                bfs(n, all_traces)

        set_of_all_traces = set()
        for i in from_state:
            set_of_traces = set()
            initial_trace = KripkeTrace(parent=self)
            initial_trace.add_state(i)
            set_of_traces.add(initial_trace)
            bfs(i, set_of_traces)
            self.traces.update({i: set_of_traces})
            set_of_all_traces |= set_of_traces

        return set_of_all_traces
    
    def evaluate_LTL(self, formula: str) -> bool:
            """
            Recursive algorithm to evaluate LTL formula with respect to the Kripke Structure.
            Make sure that letters "X", "U", "G", "F" only appear as operators.
            Also, the following operators are accepted:

            logical not: 
                ¬ / ! / not / NOT
            logical or: 
                ∨ / | / or / OR
            logical and: 
                ∧ / & / and / AND
            logical implication: 
                → / => / -> / implies / IMPLIES

            Args:
                formula (str): formula in LTL to evaluate

            Returns:
                bool
            """
            
            formula = formula.replace(' ', '')
            infinite_traces = self.get_traces()
            for trace in infinite_traces:
                if not trace.evaluate_LTL(formula):
                    return False
            return True
    
    def evaluate_CTL(self, formula: str) -> bool:
            """
            Recursive algorithm to evaluate CTL formula for the whole Kripke Structure.
            Make sure that letters "X", "U", "G", "F", "A", "E" only appear as operators.
            Also, the following operators are accepted:

            logical not: 
                ¬ / ! / not / NOT
            logical or: 
                ∨ / | / or / OR
            logical and: 
                ∧ / & / and / AND
            logical implication: 
                → / => / -> / implies / IMPLIES

            Args:
                formula (str): formula in CTL to evaluate

            Returns:
                bool
            """
            for state in self.states:
                if not self.evaluate_CTL_for_state(formula=formula, state=state):
                    return False
            return True

    def evaluate_CTL_for_state(self, formula: str, state: str) -> bool:
            """
            Recursive algorithm to evaluate CTL formula with respect to the Kripke Structure.
            Make sure that letters "X", "U", "G", "F", "A", "E" only appear as operators.
            Also, the following operators are accepted:

            logical not: 
                ¬ / ! / not / NOT
            logical or: 
                ∨ / | / or / OR
            logical and: 
                ∧ / & / and / AND
            logical implication: 
                → / => / -> / implies / IMPLIES

            Args:
                formula (str): formula in CTL to evaluate
                state (str): state to check the formula for.

            Returns:
                bool
            """
            assert isinstance(state, str), f'state must be a str, not {type(state)}'
            assert state in self.states, f'There is no such state in Kripke Structure: "{state}"'
            # report = True

            formula = formula.replace(' ', '')
            for symbol, replacement in tools.DICT.items():
                formula = formula.replace(symbol, replacement)
            U = re.findall(r'A\(.+U.+\)|E\(.+U.+\)', formula)
            for u in U:
                a_e = u[0]
                u_prime = u[2:-1].replace('U', a_e + 'U')
                formula = formula.replace(u, u_prime)
            outer, (left, right) = tools._get_outermost_connective(formula, mode='CTL')
            left, right = tools._clean_parenthesis(left, mode='CTL'), tools._clean_parenthesis(right, mode='CTL')
            # if report is on:
            #     print(f'Analysing: {formula}')
            match outer:
                case '!':
                    return not self.evaluate_CTL_for_state(right, state)
                case '|':
                    return self.evaluate_CTL_for_state(left, state) or self.evaluate_CTL_for_state(right, state)
                case '&':
                    return self.evaluate_CTL_for_state(left, state) and self.evaluate_CTL_for_state(right, state)
                case '→':
                    return not self.evaluate_CTL_for_state(left, state) or self.evaluate_CTL_for_state(right, state)
                case 'AG':
                    # All paths with all valid states
                    all_traces = self.get_traces(from_state=state)
                    result = True
                    for t in all_traces:
                        trace_result = True
                        for i in range(t.practical_length):
                            if not self.evaluate_CTL_for_state(right, t[i]):
                                trace_result = False
                                break
                        if not trace_result:
                            result = False
                            break
                    return result
                case 'EG':
                    # At least one path with all valid states
                    all_traces = self.get_traces(from_state=state)
                    result = False
                    for t in all_traces:
                        trace_result = True
                        for i in range(t.practical_length):
                            if not self.evaluate_CTL_for_state(right, t[i]):
                                trace_result = False
                                break
                        if trace_result:
                            result = True
                            break
                    return result
                case 'AF':
                    # All path have at least one valid state
                    all_traces = self.get_traces(from_state=state)
                    result = True
                    for t in all_traces:
                        trace_result = False
                        for i in range(t.practical_length):
                            if self.evaluate_CTL_for_state(right, t[i]):
                                trace_result = True
                                break
                        if not trace_result:
                            result = False
                            break
                    return result
                case 'EF':
                    # At least one path has at least one valid state
                    all_traces = self.get_traces(from_state=state)
                    result = False
                    for t in all_traces:
                        trace_result = False
                        for i in range(t.practical_length):
                            if self.evaluate_CTL_for_state(right, t[i]):
                                trace_result = True
                                break
                        if trace_result:
                            result = True
                            break
                    return result
                case 'AX':
                    # All neighbours are valid
                    all_neighbours = {t[1] for t in self.transitions if t[0] == state}
                    result = True
                    for n in all_neighbours:
                        if not self.evaluate_CTL_for_state(right, n):
                            result = False
                            break
                    return result
                case 'EX':
                    # At least one neighbour is valid
                    all_neighbours = {t[1] for t in self.transitions if t[0] == state}
                    result = False
                    for n in all_neighbours:
                        if self.evaluate_CTL_for_state(right, n):
                            result = True
                            break
                    return result
                case 'AU':
                    # For all traces U is applied
                    all_traces = self.get_traces(from_state=state)
                    result = True
                    for t in all_traces:
                        trace_result = True
                        for i in range(t.practical_length):
                            if self.evaluate_CTL_for_state(right, state=t[i]):
                                break
                            if not self.evaluate_CTL_for_state(left, state=t[i]):
                                trace_result = False
                                break
                        if not trace_result:
                            result = False
                            break
                    return result
                case 'EU':
                    # At least for one trace U is applied
                    all_traces = self.get_traces(from_state=state)
                    result = False
                    for t in all_traces:
                        trace_result = True
                        for i in range(t.practical_length):
                            if self.evaluate_CTL_for_state(right, state=t[i]):
                                break
                            if not self.evaluate_CTL_for_state(left, state=t[i]):
                                trace_result = False
                                break
                        if trace_result:
                            result = True
                            break
                    return result
                case '':
                    result = self.annotations[state][formula]
                    # print(state, formula, result)
                    return result

    def to_encoding(self, font: str = 'limboole',
                    steps: int = 1, 
                    next_step_marker: str = '_next',
                    pretty: bool = False) -> str:
        """
        Converting Kripke Structure into propositional encoding.

        Args:
            font (str, optional): type of logical symbols to use ('limboole' or 'fancy'). Defaults to 'limboole'.
            steps (int, optional): how many steps forward to encode. Defaults to 1.
            next_step_marker (str, optional): what is the symbol that connects a variable to a step number. Defaults to '-next'.
            pretty (bool, optional): allows to create more readable but less compact encoding. Defaults to False.

        Returns:
            str: encoding
        """
        if font.lower() == 'limboole':
            if pretty:
                AND, OR, IMPLIES, NOT = ' & ', ' | ', ' -> ', ' !'
                TRUE, FALSE = 'T', 'F'
            else:
                AND, OR, IMPLIES, NOT = '&', '|', '->', '!'
                TRUE, FALSE = 'T', 'F'
        elif font.lower() == 'fancy':
            if pretty:
                AND, OR, IMPLIES, NOT = ' ∧ ', ' ∨ ', ' → ', ' ¬'
                TRUE, FALSE = 'T', 'F'
            else:
                AND, OR, IMPLIES, NOT = '∧', '∨', '→', '¬'
                TRUE, FALSE = 'T', 'F'
        else:
            raise AssertionError(f'font can be "limboole" or "fancy", not {font}')

        assert isinstance(steps, int) and steps >= 1, f'steps must be an integer at least 1, not ({steps})'
        assert isinstance(next_step_marker, str), f'next_step_marker must be a str, not {type(next_step_marker)}'
        assert isinstance(pretty, bool), f'pretty must be boolean, not ({type(pretty)})'
            
        # Check that if state has indeed a unique annotation (and generate encodings for states)
        assert self.propositionals, 'Kripkr without propositional is not convertable to encoding'
        unique_encodings = set()
        state_encodings = {}
        for state, ann in self.annotations.items():
            enc = sorted(ann.items(), key=lambda x: x[0])
            enc = [symbol if value else f'{NOT}{symbol}' for symbol, value in enc]
            enc = AND.join(enc)
            if enc in unique_encodings:
                raise AssertionError('the are sevral states with the same annotations - encoding is not possible!')
            unique_encodings.add(enc)
            state_encodings.update({state: enc})

        # Dictionary of transitions
        neighbours = {}
        for state in self.states:
            neighbours.update({state: []})
            for state_ in self.states:
                if (state, state_) in self.transitions:
                    neighbours[state].append(state_)

        encoding = ''
        if pretty:
            encoding += '% Intoducing T and F constants:\n'
        encoding += TRUE + AND + NOT + FALSE
        if pretty:
            encoding += '\n'

        if pretty:
            encoding += '% 0 step encoding:\n'
        encoding += AND + '(' + TRUE + IMPLIES + '(' + OR.join(f'({s})' for s in state_encodings.values()) + '))'
        if pretty:
            encoding += '\n'

        for step in range(steps):

            if pretty:
                encoding += f'\n% Encoding for step {step+1}:\n'

            suffix = next_step_marker + str(step+1)
            new_step_encodings = {
                state: AND.join([p.removesuffix(next_step_marker + str(step)) + suffix for p in enc.split(AND)])
                for state, enc in state_encodings.items()
                }
            
            implications = []
            for state, nn in neighbours.items():
                implication = '((' + state_encodings[state] + ')' + IMPLIES
                if not nn:
                    implication += FALSE + ')'
                else:
                    next_states = [] 
                    for n in nn:
                        next_states.append(f'({new_step_encodings[n]})')
                    next_states = OR.join(next_states)
                    implication += '(' + next_states + '))'
                implications.append(implication)

            if pretty:
                encoding += AND + ('\n' + AND).join(implications) + '\n'
            else:
                encoding += AND + AND.join(implications)
            
            state_encodings = new_step_encodings.copy()

        return encoding

    def state_enconding(self, state: str,
                        time_step: int = 0,
                        font: str = 'limboole',
                        next_step_marker: str = '_next',
                        pretty: bool = False) -> str:
        """
        Converting Kripke Structure into propositional encoding.

        Args:
            state (str): what state to encode
            font (str, optional): type of logical symbols to use ('limboole' or 'fancy'). Defaults to 'limboole'.
            time_step (int, optional): at what time step to encode a state. Defaults to 0.
            next_step_marker (str, optional): what is the symbol that connects a variable to a step number. Defaults to '-next'.
            pretty (bool, optional): allows to create more readable but less compact encoding. Defaults to False.

        Returns:
            str: encoding
        """
        if font.lower() == 'limboole':
            if pretty:
                AND, OR, IMPLIES, NOT = ' & ', ' | ', ' -> ', '!'
                TRUE, FALSE = 'TRUE', 'FALSE'
            else:
                AND, OR, IMPLIES, NOT = '&', '|', '->', '!'
                TRUE, FALSE = 'TRUE', 'FALSE'
        elif font.lower() == 'fancy':
            if pretty:
                AND, OR, IMPLIES, NOT = ' ∧ ', ' ∨ ', ' → ', '¬'
                TRUE, FALSE = '⊤', '⊥'
            else:
                AND, OR, IMPLIES, NOT = '∧', '∨', '→', '¬'
                TRUE, FALSE = '⊤', '⊥'
        else:
            raise AssertionError(f'font can be "limboole" or "fancy", not {font}')

        assert isinstance(time_step, int) and time_step >= 0, f'time_step must be an integer at least 0, not ({time_step})'
        assert isinstance(next_step_marker, str), f'next_step_marker must be a str, not {type(next_step_marker)}'
        assert isinstance(pretty, bool), f'pretty must be boolean, not ({type(pretty)})'
        assert self.propositionals, 'Kripke without propositional is not convertable to encoding'
        assert state in self.states, f'there is no such state in states ({state})'
        
        if time_step == 0:
            suffix = ''
        else:
            suffix = next_step_marker + str(time_step)
        ann = sorted(self.annotations[state].items(), key=lambda x: x[0])
        encoding = '('+ AND.join([(p + suffix) if value else (NOT + p + suffix)
                                  for p, value in ann]) +')'
        return encoding

class KripkeTrace():
    """
    Helper class for registering and processing traces of kripke functions. Based on the list.
    It is able to store infinite traces with loops.
    """

    def __init__(self, parent: Any, input_sequence: str = None):

        assert isinstance(parent, Kripke), f'parent of KripkeTrace must be Kripke instance, not {type(parent)}'

        self.list = []
        self.start = None
        self.is_finite = True
        self.loops = []
        self.length = 0
        self.inf = int(2 ** 32)
        self.parent = parent

        assert input_sequence is None or isinstance(input_sequence,
                                                    str), f'input_sequence must be a string, not {type(input_sequence)}'
        if input_sequence:
            input_sequence = input_sequence.replace(' ', '').split(',')
            pointer = 0
            while pointer < len(input_sequence):
                letter = input_sequence[pointer]
                if '(' not in letter and ')' not in letter:
                    self.add_state(letter)
                    pointer += 1
                if '(' in letter:
                    loop = []
                    while True:
                        l = input_sequence[pointer].replace('(', '')
                        if ')' not in l:
                            loop.append(l)
                            pointer += 1
                        else:
                            l, x = l.split(')')
                            loop.append(l)
                            pointer += 1
                            self.add_loop(tuple(loop), int(x) if x not in ['w', 'inf'] else x)
                            break

    def __str__(self):
        string = f'Kripke Trace π of length {self.length if self.is_finite else "inf"}:\n[START] -> '
        for state in self.list:
            if isinstance(state, str):
                string += state + ' -> '
            else:
                string += f'({", ".join(state[0])}){state[1]} -> '
        string += '[END]'
        return string

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def copy(self):
        new_trace = KripkeTrace(
            parent=self.parent
        )
        new_trace.list = self.list.copy()
        new_trace.is_finite = self.is_finite
        new_trace.loops = self.loops.copy()
        new_trace.length = self.length

        return new_trace

    def unique(self) -> set:
        """
        Returns all unique states contained within the trace

        Returns:
            set
        """
        unique_states = set()
        for state in self.list:
            if isinstance(state, str):
                unique_states.add(state)
            else:
                unique_states |= {state[0]}
        return unique_states

    def __len__(self):
        """
        Returns true length for finite trace and 2**32 for infinite.

        Returns:
            _type_: _description_
        """
        if self.is_finite:
            return self.length
        else:
            return self.inf

    def add_state(self, state: str):
        """
        Allows to append a new state to the end of finite trace.

        Args:
            state (str): state of a parent Kripke structure
        """
        assert self.is_finite, 'trace is already infinite, no states can be added'
        assert state in self.parent.states, f'there is no state {state} in the parent Kripke Structure'
        if self.length > 0:
            assert (self[self.length - 1],
                    state) in self.parent.transitions, f'there is no transition from state {self[self.length - 1]} to state {state} in the parent Kripke Structure'
        self.list.append(state)
        self.length += 1

    def add_loop(self, loop: Sequence, times: Union[int, str]):
        """
        Allows to append a new loop of states to the end of finite trace.

        Args:
            loop (Sequence): loop of states
            times (int): how many times to repeat the loop. Also can be "inf" or "w" for infinite loops
        """
        assert self.is_finite, 'trace is already infinite, no states can be added'
        assert isinstance(loop, Sequence), f'loop must be a Sequence, not {type(loop)}'
        if isinstance(loop, str):
            loop = [loop]
        loop = list(loop)

        for state in loop:
            assert state in self.parent.states, f'there is no state {state} in the parent Kripke Structure'

        if self.length > 0:
            assert (self[self.length - 1], loop[
                0]) in self.parent.transitions, f'there is no transition from state {self[self.length - 1]} to state {loop[0]} in the parent Kripke Structure'
        if len(loop) > 1:
            # print(loop)
            for state_1, state_2 in zip(loop[:-1], loop[1:]):
                assert (state_1,
                        state_2) in self.parent.transitions, f'there is no transition from state {state_1} to state {state_2} in the parent Kripke Structure'

        if times in ['inf', 'w']:
            self.is_finite = False
            self.list.append((loop, 'w'))
            self.loops.append((self.length, loop, 'w', len(self.list) - 1))
        elif isinstance(times, int) and times >= 2:
            self.list.append((loop, times))
            self.loops.append((self.length, loop, self.length + len(loop) * times, len(self.list) - 1))
            self.length += len(loop) * times
        else:
            raise AssertionError(f'times can be a str "w" / "inf" or an integer at least 2, not {times}')

    def __getitem__(self, index):
        """
        Key processing, also handling infinite cases.
        """

        pointer = 0
        for state in self.list:
            if isinstance(state, str):
                if pointer == index:
                    return state
                pointer += 1
            elif state[1] == 'w':
                delta = index - pointer
                result = state[0][delta % len(state[0])]
                return result
            else:
                delta = index - pointer
                if len(state[0]) * state[1] > delta:
                    result = state[0][delta % len(state[0])]
                    return result
                else:
                    pointer += len(state[0]) * state[1]

    @property
    def practical_length(self):
        """
        Effective length of a trace to be checked.
        """
        if self.is_finite:
            return self.length
        else:
            return self.length + len(self.loops[-1][1]) * 2

    def __iter__(self):

        for i in range(len(self)):
            yield self[i]

    def evaluate_LTL(self, formula: str,
                     from_state: int = 0,
                     depth: int = 0) -> bool:
        """
        Recursive algorithm to evaluate LTL formula with respect to the trace.
        Make sure that letters "X", "U", "G", "F" only appear as operators.
        Also, the following operators are accepted:

        logical not:
            ¬ / ! / not / NOT
        logical or:
            ∨ / | / or / OR
        logical and:
            ∧ / & / and / AND
        logical implication:
            → / => / -> / implies / IMPLIES

        Args:
            formula (str): formula in LTL to evaluate
            from_state (int, optional): from which state (index) to begin. Defaults to 0.
            depth (int, optional): technical argument. Defaults to 0.

        Returns:
            bool
        """

        formula = formula.replace(' ', '')
        outer, (left, right) = tools._get_outermost_connective(formula)
        left, right = tools._clean_parenthesis(left), tools._clean_parenthesis(right)
        # if outer:
        #     print('\t'* depth + f'Outer Operator: {outer} ({left} _ {right})')
        match outer:
            case '!':
                return not self.evaluate_LTL(right, from_state=from_state, depth=depth + 1)
            case '|':
                return self.evaluate_LTL(left, from_state=from_state, depth=depth + 1) or self.evaluate_LTL(right,
                                                                                                            from_state=from_state,
                                                                                                            depth=depth + 1)
            case '&':
                return self.evaluate_LTL(left, from_state=from_state, depth=depth + 1) and self.evaluate_LTL(right,
                                                                                                             from_state=from_state,
                                                                                                             depth=depth + 1)
            case '→':
                return not self.evaluate_LTL(left, from_state=from_state, depth=depth + 1) or self.evaluate_LTL(right,
                                                                                                                from_state=from_state,
                                                                                                                depth=depth + 1)
            case 'X':
                return self.evaluate_LTL(right, from_state=from_state + 1, depth=depth + 1)
            case 'G':
                assert left == '', f'Invalid LTL, check part: {left}'
                return all(self.evaluate_LTL(right, from_state=from_state + i, depth=depth + 1) for i in
                           range(self.practical_length))
            case 'F':
                assert left == '', f'Invalid LTL, check part: {left}'
                return any(self.evaluate_LTL(right, from_state=from_state + i, depth=depth + 1) for i in
                           range(self.practical_length))
            case 'U':
                result = True
                for i in range(self.practical_length):
                    if self.evaluate_LTL(right, from_state=from_state + i, depth=depth + 1):
                        break
                    if not self.evaluate_LTL(left, from_state=from_state + i, depth=depth + 1):
                        result = False
                        break
                return result
            case '':
                state_0 = self[from_state]
                result = bool(self.parent.annotations[state_0][formula])
                # print('\t'* depth + f'evaluating symbol {formula} in state N{from_state} ({self[from_state]}) = {result}')
                return result