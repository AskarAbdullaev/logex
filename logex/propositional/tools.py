import regex as re
from typing import Any, Callable, Union
import numpy as np
import datetime
from math import ceil

from .formula import Formula
from . import const


class node():
    """
    Helper class for tree of operations
    """

    def __init__(self,
                 content: str = None,
                 connective: str = None,
                 left_part: str = None,
                 right_part: str = None,
                 parent: Any = None,
                 left_child: Any = None,
                 right_child: Any = None,
                 is_leaf: bool = False,
                 is_root: bool = False,
                 is_clause: bool = False,
                 is_cnf: bool = False,
                 data: Any = None):
        (self.content, self.connective, self.left_part, self.right_part, self.parent, self.left_child,
         self.right_child, self.is_leaf, self.is_root, self.is_clause, self.is_cnf, self.data) = (content,
                                                                                                  connective,
                                                                                                  left_part,
                                                                                                  right_part,
                                                                                                  parent,
                                                                                                  left_child,
                                                                                                  right_child,
                                                                                                  is_leaf, is_root,
                                                                                                  is_clause, is_cnf,
                                                                                                  data)

    def __str__(self):
        return f'[{self.content}:  {self.left_part} ({self.connective}) {self.right_part}]'

    def __repr__(self):
        return str(self)
    
def clean(string: Union[str, Formula]):
    """
    Small pipeline for string cleaning
    """
    assert isinstance(string, str) or isinstance(string, Formula), f"only str / Formula types are accepted, not {type(string)}"
    if isinstance(string, Formula):
        return string.string
    
    cleaned_str = standardize(string)
    cleaned_str = cleaned_str.replace(' ', '').replace('\t', '')
    cleaned_str = ''.join([s.split('%')[0] for s in cleaned_str.split('\n')])
    cleaned_str = clean_parenthesis(cleaned_str)
    return cleaned_str

def standardize(string: str):
    """
    Making operations standard
    """
    assert isinstance(string, str) or isinstance(string, Formula), f"only str / Formula types are accepted, not {type(string)}"
    if isinstance(string, Formula):
        return string.string

    standard_str = string[:]
    for symbol, standard_symbol in const.CONNECTIVES_DICT.items():
        if symbol in standard_str:
            standard_str = standard_str.replace(symbol, standard_symbol)
    return standard_str

    
def clean_parenthesis(string: str, _clean: bool = True):
    """
    Removing redundant parenthesis

    if _clean = False, supresses initial cleaning
    """
    assert isinstance(string, str) or isinstance(string, Formula), f"only str / Formula types are accepted, not {type(string)}"
    if isinstance(string, Formula):
        return string.string
    assert isinstance(_clean, bool), f"_clean must be boolean, not {type(_clean)}"

    cleaned_string = string[:]
    if _clean:
        cleaned_string = standardize(cleaned_string)

    counter = 0
    for symbol in cleaned_string:
        if symbol == '(':
            counter += 1
        elif symbol == ')':
            counter -= 1
    if counter != 0:
        raise ValueError('Parenthesis are not matching!')
    all_enclosures = []
    for i, symbol in enumerate(cleaned_string):
        if symbol == '(':
            new_enclosure = '('
            counter = 1
            for symbol_ in cleaned_string[i + 1:]:
                if symbol_ == '(':
                    counter += 1
                elif symbol_ == ')':
                    counter -= 1
                new_enclosure += symbol_
                if counter == 0:
                    break
            if counter != 0:
                raise ValueError('Parenthesis are not matching!')
            all_enclosures.append(new_enclosure)

    if cleaned_string in all_enclosures:
        cleaned_string = cleaned_string[1:-1]

    for enclosure in all_enclosures:
        if '(' + enclosure + ')' in all_enclosures:
            cleaned_string = cleaned_string.replace('(' + enclosure + ')', enclosure)

    for enclosure in all_enclosures:
        if not any(connective in enclosure for connective in const.CONNECTIVES[1:]):
            cleaned_string = cleaned_string.replace(enclosure, enclosure[1:-1])

    while 2 * const.NOT in cleaned_string:
        cleaned_string = cleaned_string.replace(2 * const.NOT, '')
    return cleaned_string

def from_txt(file: str, encoding: str = 'utf-8', **formula_kwargs):
    """
    Allows to load a formula from txt file
    """
    assert isinstance(file, str), f"file must be a string, not {type(file)}"
    if len(file.split('.')) == 1:
        file = file + '.txt'
    elif len(file.split('.')) == 2:
        assert file.split('.')[1] == 'txt', f'file must be .txt, not .{file.split('.')[1]}'
    else:
        raise AssertionError('invalid file name')
    
    with open(file, 'r', encoding=encoding) as f:
        formula = Formula(f.read(), **formula_kwargs)

    return formula

def get_outermost_connective(string: Union[str, Formula], _clean: bool = True) -> tuple[str, tuple[str, str]]:
    """
    Takes a string, cleans it and finds the outermost logical connective with respect to the
    conventional precedence rules;
    Returns tuple of the form: 
        outermost_connective, (left-hand side, right-hand side)
    """
    assert isinstance(string, str) or isinstance(string, Formula), f"only str / Formula types are accepted, not {type(string)}"
    if isinstance(string, Formula):
        string  = string.string
    elif _clean:
        string = clean(string)

    found = set(re.findall(const.COMBINED, string))
    if not found:
        return '', (string.replace('(', '').replace(')', ''), '')
    if len(found) == 1:
        
        conn = found.pop()
        split_string = string.replace('(', '').replace(')', '').split(conn)
        left, outer, right = split_string[0], conn, conn.join(split_string[1:])
        return outer, (left, right)

    special_symbol = '@'
    enclosures = []
    counter, _, new_enclosure = 0, np.zeros(len(string)), ''
    for i, symbol in enumerate(string):
        if symbol == '(':
            counter += 1
            if counter == 1:
                new_enclosure = '('
            else:
                new_enclosure += '('
        elif symbol == ')':
            counter -= 1
            if counter == 0:
                new_enclosure += ')'
                enclosures.append(new_enclosure[:])
                new_enclosure = ''
            else:
                new_enclosure += ')'
        elif counter >= 1:
            new_enclosure += symbol

    replacement_dict = {}
    for i, enclosure in enumerate(enclosures):
        string = string.replace(enclosure, special_symbol + chr(9312 + i))
        replacement_dict.update({special_symbol + chr(9312 + i): enclosure})  # store enclosure replacements

    split_string = [x for x in re.split(const.COMBINED, string) if x]  # split by logical connectives

    if not any(x in split_string for x in const.CONNECTIVES):
        # If there are no logical connectives - return the atomic formula
        outermost_connective = ''
        left_part, right_part = ''.join(split_string), ''
    elif not any(x in split_string for x in const.CONNECTIVES[1:]) and split_string[0] == const.NOT:
        # If the only connective is NOT in the beginning
        outermost_connective = const.NOT
        left_part, right_part = '', ''.join(split_string[1:])
    else:
        # Else find the first connective by precedence and split the formula at its index
        for connective in const.CONNECTIVES[1:]:
            for i, symbol in enumerate(split_string):
                if symbol == connective:
                    outermost_connective = connective
                    left_part, right_part = ''.join(split_string[:i]), ''.join(split_string[i + 1:])

    # Back substitution of enclosures
    for special_symbol, enclosure in replacement_dict.items():
        left_part = left_part.replace(special_symbol, enclosure)
        right_part = right_part.replace(special_symbol, enclosure)

    return outermost_connective, (left_part, right_part)

def construct_tree_of_operations(string: Union[str, Formula], _clean: bool = True):
    """
    Allows to construct a tree of operations.
    """
    assert isinstance(string, str) or isinstance(string, Formula), f"only str / Formula types are accepted, not {type(string)}"
    if isinstance(string, Formula):
        string  = string.string
    elif _clean:
        string = clean(string)

    nodes = set()

    outer, (lhs, rhs) = get_outermost_connective(string)
    root = node(string, outer, lhs, rhs, is_root=True)
    nodes.add(root)

    _construct_tree_of_operations(root, nodes)

    return root, nodes


def _construct_tree_of_operations(_node: Any, set_of_nodes: set):
    """
    Recursively constructs a tree of logical operations;
    """

    if _node.left_part:
        outer, (lhs, rhs) = get_outermost_connective(clean_parenthesis(_node.left_part))
        lhs, rhs = clean_parenthesis(lhs), clean_parenthesis(rhs)
        if not outer:
            new_node = node(_node.left_part, left_part=_node.left_part, parent=_node, is_leaf=True)
            _node.left_child = new_node
            set_of_nodes.add(new_node)
        else:
            new_node = node(_node.left_part, outer, lhs, rhs, _node)
            set_of_nodes.add(new_node)
            _node.left_child = new_node
            _construct_tree_of_operations(new_node, set_of_nodes=set_of_nodes)
    if _node.right_part:
        outer, (lhs, rhs) = get_outermost_connective(clean_parenthesis(_node.right_part))
        lhs, rhs = clean_parenthesis(lhs), clean_parenthesis(rhs)
        if not outer:
            new_node = node(_node.right_part, left_part=_node.right_part, parent=_node, is_leaf=True)
            set_of_nodes.add(new_node)
            _node.right_child = new_node
        else:
            new_node = node(_node.right_part, outer, lhs, rhs, _node)
            set_of_nodes.add(new_node)
            _node.right_child = new_node
            _construct_tree_of_operations(new_node, set_of_nodes=set_of_nodes)


def reconstruct_from_tree(_node: Any):
    """
    Allows to unparse the tree of operations starting from a certain node
    """

    if _node is None:
        return ''
    if _node.is_leaf:
        return _node.content
    return '(' + reconstruct_from_tree(_node.left_child) + _node.connective + reconstruct_from_tree(
        _node.right_child) + ')'


def simplify(formula: Union[str, Formula]):
    """
    Get rid of implications and biconditionals. Push negations to the literals

    Args:
        formula (str): _description_
        generate_report (bool): allows to generate report

    Returns:
        _type_: _description_
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.to_nnf()
    else:
        formula = Formula(formula)
        return formula.to_nnf().string

def variables(formula: Union[str, Formula]):
    """
    Extracts variables from expression
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.variables
    
    string = formula[:]
    string = clean(string)
    for symbol in const.CONNECTIVES + ['(', ')', const.TRUE, const.FALSE]:
        string = string.replace(symbol, ' ')
    string = re.sub(r'\s{2,200}', ' ', string)
    _variables = set(string.split(' '))
    return sorted([v for v in _variables if v])

def literals(formula: str):
    """
    Exctracts lliterals from expression
    """
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        return formula.literals
    
    string = formula[:]
    string = clean(string)
    for symbol in const.CONNECTIVES[1:] + ['(', ')', const.TRUE, const.FALSE]:
        string = string.replace(symbol, ' ')
    string = re.sub(r'\s{2,200}', ' ', string)
    _literals = set(string.split(' '))
    return sorted([v for v in _literals if v], key=lambda v: v.replace(const.NOT, ''))

def negated(formula: Union[str, Formula]):
    """
    Returns negated formula
    """
    
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        formula = formula.string
    
    string = formula[:]
    string = clean(string)
    outer, _ = get_outermost_connective(string)
    if outer == const.NOT:
        string = string[1:]
    else:
        string = const.NOT + f'({string})'
    return clean_parenthesis(string)

def translate(formula: Union[str, Formula], font: str = 'limboole'):
    """
    Allows to write a formula in different notation

    Args:
        formula (Union[str, Formula]): formula
        font (str, optional): mode of representation: 'limboole' / 'fancy' / 'default' / 'words'. Defaults to 'limboole'.

    Returns:
        _type_: _description_
    """

    assert font.lower() in ['limboole', 'fancy', 'default', 'words'], "unknown font, consider 'limboole', 'standard', 'default', 'words'"
    assert isinstance(formula, str) or isinstance(formula, Formula), f"only str / Formula types are accepted, not {type(formula)}"
    if isinstance(formula, Formula):
        formula = formula.string

    string = formula[:]
    for c, r in const.CONNECTIVES_DICT.items():
        string = string.replace(c, r)

    match font.lower():
        case 'limboole':
            for c, r in const.LIMBOOLE.items():
                string = string.replace(c, r)
        case 'fancy':
            for c, r in const.FANCY.items():
                string = string.replace(c, r)
        case 'default':
            for c, r in const.DEFAULT.items():
                string = string.replace(c, r)
        case 'words':
            for c, r in const.WORDS.items():
                string = string.replace(c, r)
                string = string.replace('  ', ' ')

    return string


def log(report: str):
    """
    Helper function to print and log a report (if required)
    """
    print(report)
    with open(const.OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write('\n\n\n' + str(datetime.datetime.now()) + '\n')
        f.write(report)

def draw_tree(root,
              space: int = 4,
              linker: int = 4,
              tight_layout: bool = True,
              left_dashed: bool = False,
              formatter: Callable = lambda s: f'({s})'):
    """
    Helper function for tree drawing

    Args:
        root (_type_): root node
        space (int, optional): minimum sace between nodes. Defaults to 4.
        linker (int, optional): minimum length of one linker shoulder. Defaults to 4.
        tight_layout (bool, optional): allows tight layout. Defaults to False.
        left_dashed (bool, optional): allows to print left linkers with dashed lines . Defaults to False.
        formatter (Callable, optional): formatter for node contents. Defaults to lambda s: f'({s})'.

    Returns:
       printable tree
    """

    # Depth are different from "heights" and correspond to layers of the tree
    depths = {id(root): 0}

    def get_depths(_node, _d: int = 0):
        depths.update({id(_node): _d})
        if _node.left_child:
            get_depths(_node.left_child, _d + 1)
        if _node.right_child:
            get_depths(_node.right_child, _d + 1)

    get_depths(root)
    max_height = max([v for (_, v) in depths.items()])

    drawing = []
    for i in range(0, max_height * 2, 2):
        height = i // 2
        if height == 0:
            length_of_linker = linker
        else:
            length_of_linker = (linker * 2 + space) * 2 ** (height - 1) // 2
        drawing.append([" " * length_of_linker] * 2 ** (max_height - height))
        drawing.append([" " * length_of_linker] * 2 ** (max_height - height))
    drawing.append(f'({root.connective})')
    drawing = drawing[::-1]

    def recursive_drawing(_node, path=None):

        if id(_node) == id(root):
            path = []

        column = ((2 ** depths[id(_node)]) - 1) / 2
        for ii, turn in enumerate(path):
            if turn == 'L':
                column -= (2 ** (depths[id(_node)] - ii)) / 4
            else:
                column += (2 ** (depths[id(_node)] - ii)) / 4

        length = len(drawing[depths[id(_node)] * 2 - 1][0])

        if not _node.is_leaf:
            value = formatter(_node.connective)
        else:
            value = formatter(_node.content)
        if id(_node) != id(root) and _node.parent.left_child and id(_node.parent.left_child) == id(_node):
            drawing[depths[id(_node)] * 2 - 1][int(column)] = " " * (len(value) - 1) + '/' + (
                '˙' if left_dashed else '‾') * (length - len(value))
            drawing[depths[id(_node)] * 2][int(column)] = value + " " * (length - len(value))
        if id(_node) != id(root) and _node.parent.right_child and id(_node.parent.right_child) == id(_node):
            drawing[depths[id(_node)] * 2 - 1][int(column)] = '‾' * (length - len(value)) + '\\' + " " * (
                        len(value) - 1)
            drawing[depths[id(_node)] * 2][int(column)] = " " * (length - len(value)) + value
        if _node.left_child:
            recursive_drawing(_node.left_child, path=[*path, "L"])
        if _node.right_child:
            recursive_drawing(_node.right_child, path=[*path, "R"])

    recursive_drawing(root)

    # The final drawing preparation - aligning all the rows according to the longest (deepest) layer
    longest_line = (2 * len(drawing[-1]) * linker + (len(drawing[-1]) - 1) * space + max_height) // 2
    for i, d in enumerate(drawing[1:], start=1):
        height = (i - 1) // 2
        paired = [(d[j] + d[j + 1]).center(longest_line // (2 ** height)) for j in range(0, len(d), 2)]
        lengths = [len(pair) for pair in paired]
        error = (longest_line - sum(lengths)) / (len(paired) - 1) if len(paired) > 1 else 0
        e, k, start = 0, 0, (len(paired) - 1) // 2 + 1
        while k < len(paired) - 1:
            e += error
            adjust = ceil(e)
            paired[start] = paired[start] + " " * adjust
            e -= adjust
            start = start + (k + 1) * (-1) ** k
            k += 1
        drawing[i] = "".join(paired)

        # drawing[i] = "".join([(d[j] + d[j+1]).center(longest_line // (2 ** height)) for j in range(0, len(d), 2)])
        drawing[i] = drawing[i] + " " * (longest_line - len(drawing[i]))
    drawing[0] = drawing[0].center(longest_line)

    # Additional modification which deletes unnecessarily long linking lines and makes the tree look more compact
    if tight_layout:
        reduction = []
        for i in range(longest_line):
            non_empty = set()
            for j in range(len(drawing)):
                if drawing[j][i] != " ":
                    non_empty.update({drawing[j][i]})
            if not non_empty - {"‾", "˙"}:
                reduction.append(i)
            else:
                continue

        if len(reduction) >= linker - 1:
            revised_reduction = []
            sequence = [reduction[0]]
            for i, x in enumerate(reduction[linker - 1:], start=linker - 1):
                if x == reduction[i - 1] + 1:
                    sequence.append(x)
                else:
                    if len(sequence) >= 2:
                        revised_reduction.extend(sequence[2:])
                    sequence = []
            for i in range(len(drawing)):
                drawing[i] = "".join([drawing[i][j] for j in range(len(drawing[i])) if j not in revised_reduction])

    drawing = '\n'.join(drawing)

    return drawing
