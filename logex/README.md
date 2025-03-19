# Logex

Utilities for studying logic. 
Based on lectures from JKU Institute for Symbolic Artificial Intelligence (https://www.jku.at/en/institute-for-symbolic-artificial-intelligence/)
Currently has sub-modules:

- propositional
- equational
- formal_models

## Dependencies

- graphviz=12.0.0
- matplotlib=3.10.1
- networkx=3.4.2
- numpy=2.2.3
- pillow=11.1.0
- pygraphviz=1.14
- pysat=3.2.1
- regex=2024.11.6

## propositional

Basic tools to play with propositional (predicate) formulas, including:

- formulas cleaning;
- get outermost connective;
- extract variables;
- extract literals;
- get negated;
- simplify;
- translate using other logical symbols;
- evaluate formula;
- get truth table;
- check for tautology;
- check for UNSAT / SAT / VALID / REFUTABLE;
- get tree of operations;
- get model / falsifying assignment
- convert to CNF / NNF / DNF / AIG
- apply BCP;
- apply DPLL
- find pivots;
- apply binary resolution / resolution / resolution refutation;
- check is clause is blocked;
- find blocked clauses;
- apply blocked clause elimination;

More datailed explanation is provided in the notebook "proposition.ipynb".

## equational

A small sub-module to experiment with equational reasoning. Allows to create
Term instances, match them, search for MGU, LGG, critical pairs, perform Knuth-Bendix completion.

More datailed explanation is provided in the notebook "equational.ipynb".

## formal_models

Sub-module for (mostly visual) studying of basic concepts of formal models.

Includes the following classes:

- fm.Automaton (determinism / completeness / power automatons / complement automatons / product automatons / accepted language)
- fm.Oracle (oracles and optimized oracles)
- fm.IOAutomaton
- fm.LabeledTransitionSystem
- fm.Kripke (LTL, CTL checking)
- fm.ConditionEventNet
- fm.PlaceTransitionNet
- fm.DigitalCircuit
- fm. KripkeTrace (LTL checking)
- fm.ProcessAlgebra

... and conversions between classes

More datailed explanation is provided in the notebook "formal_models.ipynb".

## Authors

Askar Abdullaev 
([https://github.com/AskarAbdullaev/](https://github.com/AskarAbdullaev/logex/))
