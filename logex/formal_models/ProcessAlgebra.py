from typing import Collection, Union
from itertools import product
import regex as re

from . import tools

class ProcessAlgebra():
    """
    A special class to define a PA as a set of sequential rules.

    Rules must be a collection of strings of the form: "Process1 = expression1"
    By syntax, processes must start with uppercase letter, while actions - with lowercase.

    Apart from '.' connector, there is a "+" connector allowed.

    Also, parallel operator "||" can be used to join several independent processes together.

    Examples of valid rules:

    "P = a.c.P"
    "R = e.R"
    "Q = a.c.f.P + b.f.Q"
    "T = q.p.(f.P + b.P)"
    "Parallel = P || R"
    """

    def __init__(self, 
                 rules: Union[Collection[str], str], name: str = None):


        # Checking the input datatypes and unifying them

        # Check name
        assert name is None or isinstance(name, str), 'name must be a string'
        
        if isinstance(rules, str):
            rules = {rules}

        processed_rules = {}
        for rule in rules:
            assert isinstance(rule, str), f'each rule must be of type str, not {type(rule)} ({rule})'
            assert '=' in rule and '=' not in rule.replace('=', ''), f'there must be exactly 1 equality sign "=", not: {rule}'
            rule = rule.replace(' ', '')
            process, expression = rule.split('=')
            assert '.' not in process and process[0].isupper(), f'left part of the rule must an upper letter process with not "."s, not {process}'
            for symbol in expression:
                assert symbol.isalnum() or symbol in {'+', '.', '|', '(', ')'}, f'only possible letters for right-hand side of rules are letters, "+", "||", ".", "()", revise {rule}'
                assert '|' not in expression.replace('||', ''), f'parallel operator must use "||", revise {rule}'
            processed_rules.update({process: expression})

        # Identify valid processes
        valid_processes = set(p for p in processed_rules.keys() if '||' not in processed_rules[p])
        parallel_processes = set(processed_rules.keys()) - valid_processes
        new_ = set()
        for p in valid_processes | parallel_processes:
            processes_in_expression = set([('.' + pp).split('.')[-1] for pp in re.split(r'\+|\|\|', processed_rules[p].replace('(', '').replace(')', ''))])
            for new_process in processes_in_expression - valid_processes:
                processed_rules.update({new_process: tools.EMPTY_STATE})
            new_ |= processes_in_expression
        valid_processes |= new_

        # Identify isolated groups (sets of processes that are in the same graph structure)
        rule_sets = []
        for rule in processed_rules.items():
            if '||' in rule[1]:
                    continue
            symbols = ('.' + rule[0] + '.' + rule[1] + '.').replace('+', '.').replace('(', '.').replace(')', '.')
            symbols = set(symbols.split('.'))
            rule_sets.append(symbols & valid_processes)

        groups = [rule_sets[0]]
        for rule_set in rule_sets[1:]:
            group_found = False
            for g in groups:
                if g & rule_set:
                    g |= rule_set
                    group_found = True
                    break
            if not group_found:
                groups.append(rule_set)

        # Create Rendez-Vous dict:
        rendez_vous = dict()
        group_actions = dict()
        all_actions = set()
        for group in groups:
            actions = set()
            for rule in processed_rules.items():
                if '||' in rule[1]:
                    continue
                dotted_rule = ('.' + rule[0] + '.' + rule[1] + '.').replace('+', '.').replace('(', '.').replace(')', '.').replace('||', '.')
                if not any(f'.{g}.' in dotted_rule for g in group):
                    continue
                actions |= set(dotted_rule.split('.'))
            actions -= set(processed_rules.keys())
            actions -= {tools.EMPTY_STATE}
            actions -= {''}
            group_actions.update({tuple(sorted(group)): actions})
            all_actions |= actions

        for action in all_actions:
            theta = set()
            for g, g_a in group_actions.items():
                if action in g_a:
                    theta.add(tuple(sorted(g)))
            rendez_vous.update({action: theta})

        # If there are parallel expressions: check that they indeed parallelize disjoint processes
        for process in parallel_processes:
            process = processed_rules[process]
            split_process = process.split('||')
            groups_copy = groups.copy()
            assert len(split_process) <= len(groups), f'there are only {len(groups)} independent processes but {len(split_process)} parallels were given'

            for sp in split_process:
                symbols = ('.' + sp + '.').replace('+', '.').replace('(', '.').replace(')', '.')
                symbols = set(symbols.split('.'))
                processed_involved = symbols & valid_processes

                group_found = False
                for g in groups:
                    if not processed_involved - g:
                        group_found = True
                        assert g in groups_copy, f'Some processes in {processed_involved} cannot be parallelized as already involved in another process'
                        groups_copy = [gg for gg in groups_copy if gg != g]
                        break
                if not group_found:
                    raise AssertionError(f'processes {processed_involved} are from disjoint graphs but passed with "+" choice operator')
        
        self.rules = processed_rules
        self.processes = set(processed_rules.keys())
        self.groups = {tuple(sorted(g)) for g in groups}
        self.rendez_vous = rendez_vous
        self.group_actions = group_actions
        self.actions = all_actions
        self.class_name = 'PA'
        self.name = name if name else 'PA'

    def __str__(self):
        string = f'Process Algebra {self.name}\n'
        string += f'Processes  = {self.processes}\n'
        string += f'Independent processes = {self.groups}\n'
        string += f'Actions with rendez-vous rule = {self.rendez_vous}\n'
        string += 'Rules =\n'
        for rule in self.rules.items():
            string += f'\t{rule[0]} = {rule[1]}\n'
        return string

    def __repr__(self):
        return str(self)
    
    def apply_action_to_process(self, process: str, action: str):
        """
        Applying an action to a desired process

        Args:
            process (str): process
            action (str): event/action

        Returns:
            _type_: _description_
        """

        assert isinstance(process, str), 'process must be a string'
        assert isinstance(action, str), 'action must be a string'
        assert action in self.actions, f'no such action "{action}"'
        process = process.replace(' ', '')
        action = action.replace(' ', '')

        outer, parts = tools._get_outermost_connective(process, mode="PA")
        # print(outer, parts)

        # print(f'Trying to apply "{action}" to "{process}"')
        # print(f'Outer: {outer}, parts: {parts}')

        match outer:
            case '':
                assert process in self.processes, f'no such process "{process}"'
                process = self.rules[process]
                if process == tools.EMPTY_STATE:
                    return []
                else:
                    return self.apply_action_to_process(process, action)

            case '.':
                assert parts[0] in self.actions, f'unknown action {parts[0]}'
                if action == parts[0]:
                    return [parts[1]]
                return []
            
            case '+':
                result = []
                for part in parts:
                    result.extend(self.apply_action_to_process(part, action))
                return result
            
            case '||':
                results = []

                rendez_vous = self.rendez_vous[action] 

                # Register what processed were triggered and what groups were involved
                groups_involved = set()
                groups_triggered = set()

                for part in parts:
                    
                    symbols = set(('.' + part + '.').replace('+', '.').replace('(', '.').replace(')', '.').split('.'))
                    processed_involved = symbols & self.processes
                    for g in self.groups:
                        if not processed_involved - set(g):
                            group_involved = g
                            break
                    groups_involved.add(group_involved)

                    trial_result = self.apply_action_to_process(part, action)
                    if trial_result:
                        results.append(trial_result)
                        groups_triggered.add(group_involved)
                    else:
                        results.append([part])

                # If no triggered processes = no changes (can skip)
                if not groups_triggered:
                    return []
                
                # Now, we need to check that action triggered respects the rendez-vous rule
                groups_must_be_triggered = rendez_vous & groups_involved
                if groups_triggered != groups_must_be_triggered:
                    # print(f'\nRANDEZ-VOUZ failed for action {action} and process {process}')
                    # print(f'Must have triggered: {groups_must_be_triggered}')
                    # print(f'Only triggered: {groups_triggered}\n')
                    return []
                
                results = product(*results)
                results = set(['||'.join(r) for r in results])
                return results
            case _:
                raise ValueError(f'unknown connective {outer}')

        
    def to_lts(self, name: str = None, initial: str = None):
        """
        Derives LTS from Progress Algebra.

        Parameters:
            name (str): Optional. How to entitle the new LTS instance.
            initial (str): initial process state to start with
            By default: LTS(previous name).

        Returns:
            a new LabeledTransitionSystem instance
        """

        assert name is None or isinstance(name, str), 'name must be a string'
        if name is None:
            name = f'LTS({self.name})'

        assert initial is None or isinstance(initial, str), 'initial must be a string'
        if initial is None:
            initial = sorted(self.processes)[0]
        else:
            initial = initial.replace(' ', '')
        assert initial in self.processes, f'no such process "{initial}"'

        all_states = set()
        added_at_previous_step = {initial}
        all_states.add(initial)
        all_transitions = set()

        while added_at_previous_step:

            added_at_current_step = set()

            for state in added_at_previous_step:

                for action in self.actions:

                    new_states = self.apply_action_to_process(state, action)

                    if new_states:
                        if isinstance(new_states, str):
                            new_states = [new_states]
                        for ns in new_states:
                            for key, value in self.rules.items():
                                if value == ns:
                                    ns = key
                                    break
                            added_at_current_step.add(ns)
                            all_transitions.add((state, action, ns))

            added_at_previous_step = added_at_current_step - all_states
            all_states |= added_at_current_step

        from . import LabeledTransitionSystem
        lts = LabeledTransitionSystem(
            all_states,
            {initial},
            all_transitions,
            self.actions,
            name=name
        )

        return lts