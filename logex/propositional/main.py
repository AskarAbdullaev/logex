from typing import Union, Sequence
import regex as re
from itertools import product, combinations
import numpy as np
from random import shuffle
import warnings

from . import tools
from . import const
from .formula import Formula

def to_nnf(formula: Union[str, Formula]):
    """
    Convert to Negation Normal Form (NNF)
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    return tools.simplify(formula)

def to_cnf(formula: Union[str, Formula]):
    """
    Convert to CNF
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.to_cnf()
    else:
        formula = Formula(formula)
        return formula.to_cnf().string

def is_cnf(formula: Union[str, Formula]):
    """
    Check if formula is in CNF
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula.check_cnf()

def to_dnf(formula: Union[str, Formula]):
    """
    Convert to DNF
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.to_dnf()
    else:
        formula = Formula(formula)
        return formula.to_dnf().string

def to_aig(formula: Union[str, Formula]):
    """
    Convert to the Add-Inverter Graph form (AIG)
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.to_aig()
    else:
        formula = Formula(formula)
        return formula.to_aig().string

def tree_of_operations(formula: Union[str, Formula]):
    """
    Draw a tree of operations
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        formula = formula.string

    root, _ = tools.construct_tree_of_operations(formula, _clean=True)
    tree = tools.draw_tree(root, space=6, linker=6)

    report = ''
    report += ' TREE OF OPERATIONS '.center(60, '#') + '\n'
    report += f'Formula: {formula}' + '\n'
    report += 'Tree:' + '\n\n'
    report += tree + '\n'
    report += '#' * 60
    tools.log(report)

    return

def bcp(formula: Union[str, Formula], literal: str, _check_cnf: bool = True, generate_report: bool = True):
    """
    Binary Constraint Propagation (BCP)

    Args:
        formula (Union[str, Formula]): formula in CNF
        literal (str): literal to perform BCP on
        _check_cnf (bool, optional): check that inpt is in CNF. Defaults to True.
        generate_report (bool, optional): allows report. Defaults to True.

    Returns:
        new formula
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    input_is_str = False
    if isinstance(formula, str):
        input_is_str = True
        formula = Formula(formula, as_cnf=True)
    if _check_cnf:
        assert formula.check_cnf(), 'input formula must be in CNF'
    
    result = set()

    negated_literal = tools.negated(literal)

    report = ''
    if generate_report:
        report += ' BCP '.center(60, '#') + '\n'
        report += f'Original CNF: {formula}' + '\n'
        report += f'BCP on literal {literal}' + '\n'

    for clause in formula.clauses:
        report += f'({clause})' + ' ==> '
        literals = clause.split(const.OR)
        if literal in literals:
            report += const.TRUE + '\n'
            continue
        elif negated_literal in literals:
            _clause = const.OR.join(sorted([l for l in literals if l != negated_literal]))
            if _clause == '':
                report += const.FALSE + '\n'
                if generate_report:
                    report += f'\nResult: {const.FALSE}' + '\n'
                    report += '#' * 60
                    tools.log(report)
                if input_is_str:
                    return const.FALSE
                return Formula(const.FALSE)
            new_clause = const.OR.join(sorted([l for l in literals if l != negated_literal]))
            result.add(const.OR.join(sorted([l for l in literals if l != negated_literal])))
        else:
            new_clause = const.OR.join(sorted([l for l in literals]))
            result.add(const.OR.join(sorted([l for l in literals])))
        report += f'({new_clause})' + '\n'

    result = const.AND.join([f'({c})' for c in result])
    if result == '':
        if generate_report:
            report += f'\nResult: {const.TRUE}' + '\n'
            report += '#' * 60
            tools.log(report)
        if input_is_str:
            return const.TRUE
        return Formula(const.TRUE)

    if generate_report:
        report += f'\nResult: {result}' + '\n'
        report += '#' * 60
        print(report)

    if input_is_str:
        return result
    
    return Formula(result, as_cnf=True)

def dpll(formula: str, _check_cnf: bool = True, generate_report: bool = True):
    """
    DPLL algorithm

    Args:
        formula (str): formula in CNF
        order_of_variables (list[str], optional): order in which to build a tree. Defaults to None.
        _check_cnf (bool, optional): check that input is in CNF. Defaults to True.
        generate_report (bool, optional): allows report. Defaults to True.

    Returns:
        tree
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula, as_cnf=True)
    if _check_cnf:
        assert formula.check_cnf(), 'input formula must be in CNF'

    if formula.string in [const.TRUE, const.FALSE]:
        results_ = []

    else:

        def best_variable(f: Formula):
            if f.n_vars == 0:
                return f.string
            frequencies = [f.string.count(v) for v in f.variables]
            best = max(frequencies)
            return f.variables[frequencies.index(best)]
        
        results_ = []
        best_var = best_variable(formula)
        clauses = formula.clauses.copy()
        root = tools.node(content=best_var, connective=best_var, is_root=True, left_part=[])
        nodes = [root]

        def recursive_dpll(current_formula: Formula, var: str, current_node: tools.node, depth: int = 0, clauses=clauses):

            if current_node.is_leaf:
                return

            child_right = bcp(current_formula, var, _check_cnf=False, generate_report=False)
            results_.append(f'BCP on "{var}":' + '\t' * (depth + 1) + child_right.string)
            var_right = best_variable(child_right)

            next_clause = set(current_node.left_part) | {tools.negated(var)}
            for c in clauses:
                as_set = set(c.split(const.OR))
                if len(as_set) == len(as_set & next_clause):
                    next_clause = as_set
                    next_clause = list(next_clause)
                    break
            
            node_right = tools.node(content=child_right, connective=var_right, left_part=current_node.left_part + [tools.negated(var)])
            current_node.right_child = node_right
            node_right.parent = current_node
            if child_right.n_vars == 0:
                node_right.is_leaf = True
                node_right.left_part = next_clause
            nodes.append(node_right)
            recursive_dpll(child_right, var_right, node_right, depth=depth+1, clauses=clauses)

            child_left = bcp(current_formula, tools.negated(var), _check_cnf=False, generate_report=False)
            results_.append(f'BCP on "{tools.negated(var)}":' + '\t' * (depth + 1) + child_left.string)
            var_left = best_variable(child_left)

            next_clause = set(current_node.left_part) | {var}
            for c in clauses:
                as_set = set(c.split(const.OR))
                if len(as_set) == len(as_set & next_clause):
                    next_clause = as_set
                    next_clause = list(next_clause)
                    break
            
            node_left = tools.node(content=child_left, connective=var_left, left_part=current_node.left_part + [var])
            current_node.left_child = node_left
            node_left.parent = current_node
            if child_left.n_vars == 0:
                node_left.is_leaf = True
                node_left.left_part = next_clause
            nodes.append(node_left)
            recursive_dpll(child_left, var_left, node_left, depth=depth+1, clauses=clauses)

        recursive_dpll(formula, best_var, root)

    sat, unsat = 0, 0
    for r in results_:
        if const.TRUE in r:
            sat += 1
        elif const.FALSE in r:
            unsat += 1

    if sat == 0:
        res = 'UNSAT'
    elif unsat == 0:
        res = 'VALID'
    else:
        res = 'SAT'

    drawing = tools.draw_tree(root, space=5, linker=6, tight_layout=True, left_dashed=True)

    report = ''
    if generate_report:
        report += ' DPLL '.center(60, '#') + '\n'
        report += f'\nFormula: {formula.string}' + '\n'
        report += '\n'.join(results_)
        report += f'\nResult: {res}' + '\n'
        report += drawing + '\n'
        report += '#' * 60
        tools.log(report)
    
    return res


def evaluate(formula: Union[str, Formula], assignment: Union[dict, Sequence]):
    """
    Allows to evaluate formula under certain assignment
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula(assignment)


def get_truth_table(formula: Union[str, Formula]):
    """
    Explicit truth table for small formulas
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula.get_truth_table()

def is_valid(formula: Union[str, Formula]):
    """
    Validity check
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.is_valid()
    else:
        formula = Formula(formula)
        return formula.is_valid()


def is_tautology(formula: Union[str, Formula]):
    """
    Validity check
    """
    if _is_clause(formula):
        if isinstance(formula, Formula):
            formula = formula.string
        return len(set(tools.literals(formula))) > len(set(tools.literals(formula.replace(const.NOT, ''))))

    return is_valid(formula)


def is_sat(formula: Union[str, Formula]):
    """
    SAT check
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.is_sat()
    else:
        formula = Formula(formula)
        return formula.is_sat()


def is_unsat(formula: Union[str, Formula]):
    """
    UNSAT check
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.is_unsat()
    else:
        formula = Formula(formula)
        return formula.is_unsat()

def is_refutable(formula: Union[str, Formula]):
    """
    Refutability check
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.is_refutable()
    else:
        formula = Formula(formula)
        return formula.is_refutable()

def get_model(formula: Union[str, Formula]):
    """
    Returns a model if exists
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula.get_model()


def get_all_models(formula: str):
    """
    Returns all models for small formulas
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula.get_all_models()

def get_falsifying(formula: Union[str, Formula]):
    """
    Returns falsifying assignment
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula.get_falsifying()


def get_all_falsifying(formula: Union[str, Formula]):
    """
    Returns all falsifying assignments for small formulas
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula)
    return formula.get_all_falsifying()

def graph_coloring(vertices: list, edges: list[tuple], num_colors: int, generate_report: bool = True):
    """
    Returns encoding for a graph coloring problem

    Args:
        vertices (list): list of vertices of the problem
        edges (list[tuple]): list of connections between vertices (edges)
        num_colors (int): number of colors allowed
        generate_report (bool, optional): to generate report or not. Defaults to True.

    Returns:
        encoding
    """

    assert len(vertices) >= 2, 'There must be at least 2 vertices'
    for edge in edges:
        assert len(edge) == 2, 'Each edge must be a 2-tuple'
        assert edge[0] in vertices, f'{edge[0]} is not in provided vertices'
        assert edge[1] in vertices, f'{edge[1]} is not in provided vertices'
    assert isinstance(num_colors, int) and num_colors >= 1, 'num_colors must be a positive integer'

    vertices = sorted(set(vertices))

    report = ''
    if generate_report:
        report += ' GRAPH COLORING '.center(60, '#') + '\n'
        report += f'Vertices: {vertices}' + '\n'
        report += f'Edges: {edges}' + '\n'
        report += f'Number of colors: {num_colors}' + '\n\n'

    formula = []

    # At least one color per vertex
    for vertex in vertices:
        formula.append(
            '(' + const.OR.join([vertex + str(color) for color in range(1, num_colors + 1)]) + ')'
        )
    if generate_report:
        report += 'Each vertex must have at least one color:' + '\n'
        report += const.AND.join(formula[-len(vertices):]) + '\n\n'

    # At most one color per vertex
    for vertex in vertices:
        for color1 in range(1, num_colors + 1):
            for color2 in range(color1 + 1, num_colors + 1):
                formula.append(
                    '(' + f'{const.NOT}{vertex}{color1}{const.OR}{const.NOT}{vertex}{color2}' + ')'
                )
    if generate_report:
        report += 'Each vertex must have at most one color:' + '\n'
        report += const.AND.join(formula[-int(len(vertices) * num_colors * (num_colors - 1) / 2):]) + '\n\n'

    # Different adjacent colors:
    for edge in edges:
        for color in range(1, num_colors + 1):
            formula.append(
                '(' + f'{const.NOT}{edge[0]}{color}{const.OR}{const.NOT}{edge[1]}{color}' + ')'
            )

    if generate_report:
        report += 'Adjacent vertices cannot have the same color:' + '\n'
        report += const.AND.join(formula[-int(len(vertices) * num_colors):]) + '\n\n'

    formula = const.AND.join(formula)

    if generate_report:
        report += 'Result: ' + formula + '\n'
        report += '#' * 60
        print(report)

    return formula

def _is_clause(formula: Union[str, Formula]):
    """
    Checks if a formula is a clause
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        string = formula.string
    else:
        string = formula[:]
        string = tools.clean(string)
    
    for connective in [const.AND, const.IMPLIES, const.XOR, const.EQUIVALENT]:
        if connective in string:
            return False
    if const.NOT + '(' in string:
        return False

    return True

def _is_cube(formula: Union[str, Formula]):
    """
    Checks if a formula is a clause
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        string = formula.string
    else:
        string = formula[:]
        string = tools.clean(string)
    
    for connective in [const.OR, const.IMPLIES, const.XOR, const.EQUIVALENT]:
        if connective in string:
            return False
    if const.NOT + '(' in string:
        return False

    return True


def possible_pivots(formula: Union[str, Formula], formula_1: Union[str, Formula] = None,
                    _supress_clean: bool = False,
                    _supress_check: bool = False):
    """
    Returns possible pivots for resolution.

    If a single formula is provided, it is supposed that it is in CNF form. All possible pivots
    for its clauses will be returned.

    If 2 formulas are provided, they are supposed to be clauses.

    Args:
        formula (Union[str, Formula]): formula in cnf
        formula_1 (Union[str, Formula], optional): _description_. Defaults to None.
        _supress_clean (bool, optional): _description_. Defaults to False.
        _supress_check (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"

    if formula_1:
        assert isinstance(formula_1, str) or isinstance(formula_1, Formula), f"only str / Formula types are accepted, not {type(formula_1)}"
        if isinstance(formula, Formula):
            c1 = formula.string
            vars1 = formula.variables
        else:
            c1 = formula[:]
            vars1 = tools.variables(c1)
            if not _supress_clean:
                c1 = tools.clean(c1)

        if isinstance(formula_1, Formula):
            c2 = formula_1.string
            vars2 = formula_1.variables
        else:
            c2 = formula_1[:]
            vars2 = tools.variables(c2)
            if not _supress_clean:
                c2 = tools.clean(c2)
        
        if not _supress_check:
            if not _is_clause(c1):
                raise ValueError('clause_1 is not a valid clause')
            if not _is_clause(c2):
                raise ValueError('clause_2 is not a valid clause')

        common_variables = [v for v in vars1 if v in vars2]
        possible = []
        for v in common_variables:
            not_v = tools.negated(v)
            if (v in c1 and not_v in c2) or (v in c2 and not_v in c1):
                if v not in possible:
                    possible.append(v)
        return possible

    if isinstance(formula, str):
        formula = Formula(formula, as_cnf=True)

    if not _supress_check:
        assert formula.check_cnf(), 'formula must be in CNF to find pivots'

    possible = []
    for v in formula.variables:
        l, not_l, both = 0, 0, 0
        for c in formula.clauses:
            if tools.negated(v) in c.split(const.OR) and v in c.split(const.OR):
                both += 1
            else:
                if tools.negated(v) in c.split(const.OR):
                    not_l += 1
                if v in c.split(const.OR):
                    l += 1
            if both > 0 and (l > 0 or not_l > 0):
                possible.append(v)
                break
            if both == 0 and (l > 0 and not_l > 0):
                possible.append(v)
                break

    return possible


def binary_resolution(clause_1: Union[str, Formula], clause_2: Union[str, Formula], 
                      pivot: str = None,
                      generate_report: bool = True,
                      allow_tautology: bool = True,
                      return_all: bool = False,
                      _supress_clean: bool = False,
                      _supress_check: bool = False):
    """
    Binary resolution between 2 clauses.

    Args:
        clause_1 (Union[str, Formula]): first clause
        clause_2 (Union[str, Formula]): second clause
        pivot (str, optional): pivot to resolve on. Defaults to None.
        generate_report (bool, optional): allows a report. Defaults to True.
        allow_tautology (bool, optional): allow tautological clauses as a result. Defaults to True.
        return_all (bool, optional): return all possible results. Defaults to False.
        _supress_clean (bool, optional): omit input cleaning. Defaults to False.
        _supress_check (bool, optional): omit input checking. Defaults to False.

    Returns:
        a single result or all possible results as a list
    """

    report = ''
    if generate_report:
        report += ' RESOLUTION '.center(60, '#') + '\n'
        report += ("[ single" if not return_all else "[ multiple") + " output ] "
        report += "[ tautology" + ("" if allow_tautology else " not") + " allowed ]\n"
        report += f'Clause 1: {clause_1}' + '\n'
        report += f'Clause 2: {clause_2}' + '\n'

    assert isinstance(clause_1, str) or isinstance(clause_1, Formula), f"only str / Formula types are accepted, not {type(clause_1)}"
    assert isinstance(clause_2, str) or isinstance(clause_2, Formula), f"only str / Formula types are accepted, not {type(clause_2)}"

    if isinstance(clause_1, Formula):
        c1 = clause_1.string
    else:
        c1 = clause_1[:]
        if not _supress_clean:
            c1 = tools.clean(c1)

    if isinstance(clause_2, Formula):
        c2 = clause_2.string
    else:
        c2 = clause_2[:]
        if not _supress_clean:
            c2 = tools.clean(c2)

    if not _supress_check:
        if not _is_clause(c1):
            raise ValueError('clause_1 is not a valid clause')
        if not _is_clause(c2):
            raise ValueError('clause_2 is not a valid clause')

    if pivot is None:
        possible = possible_pivots(clause_1, clause_2)
        if possible:
            if generate_report:
                report += f'Possible pivots are: {possible}\n'
        else:
            if generate_report:
                report += 'There are no possible pivots\n'
                report += 'Resolvent: None\n'
                report += '#' * 60
                tools.log(report)
            return [] if return_all else False

        resolvents = []
        for i, pp in enumerate(possible):
            resolvent = binary_resolution(clause_1, clause_2, pp, generate_report=False,
                                                  return_all=False, allow_tautology=allow_tautology)
            if resolvent not in resolvents:
                resolvents.append(resolvent)
            if generate_report:
                report += f'RES(Clause 1, Clause 2; on "{pp}") = {resolvent}\n'
            if not return_all and i == 0:
                break
        if generate_report:
            report += '#' * 60
            print(report)
        return resolvents if return_all else resolvents[0]

    if generate_report:
        report += f'Pivot: "{pivot}"\n'
    negated_pivot = tools.negated(pivot)

    c1 = c1.replace('(', '').replace(')', '').split(const.OR)
    c2 = c2.replace('(', '').replace(')', '').split(const.OR)
    if pivot in c1 and negated_pivot in c2:
        if generate_report:
            report += f'{pivot} is removed from clause_1, "{negated_pivot}" is removed from clause_2\n'
        resolvent = {l for l in c1 if l != pivot} | {l for l in c2 if l != negated_pivot}
        resolvent = list(dict.fromkeys(resolvent).keys())
        resolvent = const.OR.join(resolvent)
    elif pivot in c2 and negated_pivot in c1:
        if generate_report:
            report += f'{negated_pivot} is removed from clause_1, "{pivot}" is removed from clause_2\n'
        resolvent = {l for l in c1 if l != negated_pivot} | {l for l in c2 if l != pivot}
        resolvent = list(dict.fromkeys(resolvent).keys())
        resolvent = const.OR.join(resolvent)
    else:
        if generate_report:
            report += 'These 2 clauses are not resolvable\n'
        resolvent = False

    if resolvent in {'()', ''}:
        resolvent = const.FALSE

    elif not allow_tautology and isinstance(resolvent, str):
        if len(set(tools.literals(resolvent))) > len(set(tools.literals(resolvent.replace(const.NOT, '')))):
            resolvent = const.TRUE

    if generate_report:
        report += f'Resolvent: {resolvent if isinstance(resolvent, str) else "None"}\n'
        report += '#' * 60
        print(report)
    
    return resolvent


def resolution(formula: Union[str, Formula], formula_2: Union[str, Formula] = None, pivot: str = None, generate_report: bool = True,
               allow_tautology: bool = True,
               return_all: bool = False):
    """
    Binary resolution between 2 clauses.

    Args:
        formula (Union[str, Formula]): formula in CNF or a clause
        formula_2 (Union[str, Formula], optiona;): second clause. Defaults to None.
        pivot (str, optional): pivot to resolve on. Defaults to None.
        generate_report (bool, optional): allows a report. Defaults to True.
        allow_tautology (bool, optional): allow tautological clauses as a result. Defaults to True.
        return_all (bool, optional): return all possible results. Defaults to False.

    Returns:
        a single result or all possible results as a list
    """

    report = ' RESOLUTION '.center(60, '#') + '\n'
    report += ("[ single" if not return_all else "[ multiple") + " output ] "
    report += "[ tautology" + ("" if allow_tautology else " not") + " allowed ]\n"

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"

    if formula_2:
        assert isinstance(formula_2, str) or isinstance(formula_2, Formula), f"only str / Formula types are accepted, not {type(formula_2)}"
        if isinstance(formula_2, Formula):
            formula_2 = formula_2.string

        if isinstance(formula, Formula):
            formula = formula.string

        assert _is_clause(
            formula), 'If 2 formulas are provided, each of them must be a clause (error in formula 1)'
        assert _is_clause(
            formula_2), 'If 2 formulas are provided, each of them must be a clause (error in formula 2)'
        result = binary_resolution(formula, formula_2, pivot, generate_report, allow_tautology, return_all)
        return result

    if isinstance(formula, str):
        formula = Formula(formula, as_cnf=True)
    assert formula.check_cnf(), 'formula must be in CNF to apply resolution'

    if generate_report:
        report += f'CNF: {formula}\n'

    if len(formula.clauses) == 1:
        if generate_report:
            report += f'Resolution id not possible for a single clause\n'
            report += f'Result: None\n'
            report += '#' * 60 + '\n'
            tools.log(report)
        return [] if return_all else False

    if pivot is None:
        possible = possible_pivots(formula)
        if generate_report:
            report += f'Pivot is not provided\n'
            report += f'Possible pivots are: {possible}\n'
    else:
        if generate_report:
            report += f'Pivot: "{pivot}"\n'
        possible = [pivot]

    if not possible:
        if generate_report:
            report += f'Result: None\n'
            report += '#' * 60 + '\n'
            tools.log(report)
        return [] if return_all else False

    results = []
    for pivot in possible:

        resolvents = []
        for c1, c2 in combinations(formula.clauses, 2):

            resolvent = binary_resolution(c1, c2, pivot, False, allow_tautology)
            if resolvent and resolvent not in resolvents:
                resolvents.append(resolvent)
                if generate_report:
                    if len(resolvents) == 1:
                        report += f'Pivot "{pivot}" generates the following resolvents:\n'
                    report += f'\tRES(({c1}), ({c2}), on "{pivot}") = {resolvent}\n'
        results.extend(resolvents)
        if not return_all:
            break

    if generate_report:
        if not results:
            report += 'No resolvents can be derived\n'
        report += '#' * 60 + '\n'
        tools.log(report)
    return results

def resolution_refutation(formula: Union[str, Formula], mode: str = 'bfs'):
    """
    Allows to construct resolution refutation of a CNF formula
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula, as_cnf=True)
    assert formula.check_cnf(), 'formula must be in CNF to apply resolution'
    assert formula.is_unsat(), 'formula is SAT, refutation is not possible'
    assert mode.lower() in {'bfs'}, 'mode should bfs'
    
    report = ' RESOLUTION REFUTATION '.center(60, '#') + '\n'

    if len(formula.clauses) == 1:
        report += f'Resolution refutation is impossible with 1 clause\n'
        report += f'Result: None\n'
        report += '#' * 60 + '\n'
        print(report)
        return
    
    if mode.lower() == 'bfs':

        length_limit = max([len(c.split(const.OR)) for c in formula.clauses]) * 2 - 2
        upper_depth_limit = 10
        current_depth_limit = 1

        while current_depth_limit < upper_depth_limit:

            all_clauses = list(formula.clauses)
            # print(f'Trying with current depth limit = {current_depth_limit}')
            depth, solved, derived = 0, False, []
            index_table = [(-1, -1, '', i) for i in range(len(formula.clauses))]

            while not solved:

                derived = []
                depth += 1

                for c1, c2 in sorted(combinations(all_clauses, 2), key=lambda pair: len(pair[0]) * len(pair[1])):
                    if solved:
                        break
                    possible = possible_pivots(c1, c2, _supress_clean=True, _supress_check=True)
                    if not possible:
                        continue
                    for pp in possible:
                        if solved:
                            break
                        res = binary_resolution(c1, c2, pp, False, False, _supress_clean=True,
                                                        _supress_check=True)
                        
                        if not res or res == const.TRUE or res == f'({const.TRUE})':
                            continue
                        if res == const.FALSE:
                            solved = True
                            # print('SOLVED')
                        if len(tools.literals(res)) > min(length_limit, current_depth_limit - depth):
                            continue
                        if res and res not in all_clauses and res not in derived:
                            derived.append(res)
                            index_table.append(
                                (all_clauses.index(c1), all_clauses.index(c2), pp, len(all_clauses) + len(derived) - 1))

                # print(f'Depth {depth}: {derived}')
                all_clauses = all_clauses + derived

                if not derived:
                    current_depth_limit += 1
                    break

            if solved:
                print(f'{len(index_table) - len(formula.clauses)} trial resolutions were performed\nFound at depth {depth}\n\n')
                break

        if not solved:
            raise RuntimeError('''Unfortunately, the program failed to construct a resolution refutation for the given
                                  CNF. Make sure that the CNF is indeed UNSAT by using is_unsat or by using 
                                  external programs like Limboole. If you are sure that your input is USAT, augment
                                  the maximum length of derived.''')

        root = tools.node(content=const.FALSE, connective=const.FALSE, data=index_table[-1], is_root=True)
        all_nodes = [root]

        def backtracking(n: tools.node):

            left, right, _, index = n.data
            n.content = all_clauses[index].replace('(', '').replace(')', '')
            n.connective = all_clauses[index].replace('(', '').replace(')', '')

            if left == right == -1:
                n.is_leaf = True
                return

            right_child = tools.node(content=all_clauses[right], connective=all_clauses[right],
                                    data=index_table[right], parent=n)
            all_nodes.append(right_child)
            left_child = tools.node(content=all_clauses[left], connective=all_clauses[left],
                                    data=index_table[left], parent=n)
            all_nodes.append(left_child)
            n.left_child = left_child
            n.right_child = right_child
            backtracking(right_child)
            backtracking(left_child)

        backtracking(root)

    formatter = lambda s: f'{s}' if not s in formula.clauses else f'[{s}]'

    tree = tools.draw_tree(root, space=10, linker=12, tight_layout=True, formatter=formatter)
    tree = '\n'.join(tree.split('\n')[::-1]).replace('‾', '_').replace('\\', '@').replace('/', '&')
    tree = tree.replace('@', '/').replace('&', '\\')
    print(tree)


def is_blocked_on_literal(clause: Union[str, Formula], in_formula: Union[str, Formula], literal: str, 
                          _suppress_check: bool = False, generate_report: bool = True):
    """
    Check if a clause is blocked in formula on a literal

    Args:
        clause (Union[str, Formula]): _description_
        in_formula (Union[str, Formula]): _description_
        literal (str): _description_
        _suppress_check (bool, optional): _description_. Defaults to False.
    """

    assert isinstance(clause, str) or isinstance(clause, Formula), f"only str / Formula types are accepted, not {type(clause)}"
    if isinstance(clause, str):
        clause = Formula(clause, as_cnf=True)
    assert _is_clause(clause.string), 'clause is not a valid clause'

    assert isinstance(in_formula, str) or isinstance(in_formula, Formula), f"only str / Formula types are accepted, not {type(in_formula)}"
    if isinstance(in_formula, str):
        in_formula = Formula(in_formula, as_cnf=True)
    
    if not _suppress_check:
        assert in_formula.check_cnf(), 'in_formula must be in CNF'

    assert isinstance(literal, str) and literal in clause.literals, f'literal "{literal}" is not in clause'

    report = ' BLOCKED CLAUSE '.center(60, '#') + '\n'
    report += f'Checking if clause: {clause}\n'
    report += f'is blocked in formula: {in_formula}\nOn literal "{literal}"\n\n'
        
    not_literal = tools.negated(literal)
    if not_literal not in in_formula.literals:
        return True
    
    result = True
    for c in in_formula.clauses:
        if not_literal in c.split(const.OR):
            resolvent = binary_resolution(c, clause, generate_report=False)
            report += f'Possible RES with clause ({c}) = ({resolvent}) '
            if not is_tautology(resolvent):
                result = False
                report += '(non-tautology)\n'
                break
            report += '(tautology)\n'
    
    report += f'\nResult: {"BLOCKED" if result else "NOT BLOCKED"}\n'
    report += '#' * 60 + '\n'
    if generate_report:
        tools.log(report)

    return result


def is_blocked(clause: Union[str, Formula], in_formula: Union[str, Formula],
               _suppress_check: bool = False, generate_report: bool = True):
    """
    Check if a clause is blocked in formula

    Args:
        clause (Union[str, Formula]): _description_
        in_formula (Union[str, Formula]): _description_
        _suppress_check (bool, optional): _description_. Defaults to False.
    """

    assert isinstance(clause, str) or isinstance(clause, Formula), f"only str / Formula types are accepted, not {type(clause)}"
    if isinstance(clause, str):
        clause = Formula(clause, as_cnf=True)
    assert _is_clause(clause.string), 'clause is not a valid clause'

    assert isinstance(in_formula, str) or isinstance(in_formula, Formula), f"only str / Formula types are accepted, not {type(in_formula)}"
    if isinstance(in_formula, str):
        in_formula = Formula(in_formula, as_cnf=True)

    if not _suppress_check:
        assert in_formula.check_cnf(), 'in_formula must be in CNF'

    report = ' BLOCKED CLAUSE '.center(60, '#') + '\n'
    report += f'Checking if clause: {clause}\n'
    report += f'is blocked in formula: {in_formula}\n\n'

    blocking_literals = []
    for literal in clause.literals:
        block = is_blocked_on_literal(clause, in_formula, literal, False, False)
        report += f'On literal {literal}: {block}\n'
        if block:
            blocking_literals.append(literal)

    report += '\nResult: ' + (f'BLOCKED on literals {blocking_literals}'
                             if blocking_literals else "NOT BLOCKED") + '\n'
    report += '#' * 60 + '\n'
    if generate_report:
        print(report)

    return blocking_literals


def blocked_clauses(formula: Union[str, Formula], generate_report: bool = True):
    """
    Find blocked clauses in formula

    Args:
        clause (Union[str, Formula]): _description_
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula, as_cnf=True)

    assert formula.check_cnf(), 'formula must be in CNF'

    report = ' BLOCKED CLAUSE '.center(60, '#') + '\n'
    report += f'Searching for blocked clauses in: {formula}\n\n'


    blocked = {}
    for clause in formula.clauses:
        literals = is_blocked(clause, formula, _suppress_check=True,
                              generate_report=False)
        if not literals:
            literals = False
        blocked.update({clause: literals})
        if literals:
            report += f'Clause {clause} is blocked on literals {literals}\n'
        else:
            report += f'Clause {clause} is not blocked\n'

    if generate_report:
        report += f'\nResult {len(blocked)}/{len(formula.clauses)} clauses are blocked\n'
        report += '#' * 60 + '\n'
        print(report)
    return blocked

def blocked_clauses_elimination(formula: Union[str, Formula], generate_report: bool = True):
    """
    Step-wise blocked clause elimination

    Args:
        clause (Union[str, Formula]): _description_
    """

    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, str):
        formula = Formula(formula, as_cnf=True)

    assert formula.check_cnf(), 'formula must be in CNF'

    report = ' BCE '.center(60, '#') + '\n'

    residual = Formula(formula.string, as_cnf=True)
    step = 1
    while True:

        blocked = blocked_clauses(residual, generate_report=False)

        report += f'Step {step})\n'
        report += f'\tCurrent residual: {residual}\n'
        report += f'\tBlocked clauses: {blocked}\n'

        if all(not v for v in blocked.values()):
            break

        take_one = list(blocked.items())[0][0]

        report += f'\tEliminating: {take_one}\n\n'

        if len(residual.clauses) > 1:
            residual = Formula(
                const.AND.join(f'({c})' for c in residual.clauses - {take_one.replace('(', '').replace(')', '')}),
                as_cnf = True
            )
        else:
            residual = Formula(const.TRUE, as_cnf=True)
        step += 1

    report += f'Result: {residual}\n'
    report += '#' * 60 + '\n'
    if generate_report:
        print(report)

    return residual
