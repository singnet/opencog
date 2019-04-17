"""
learnable pln formula experiments
"""

import unittest
import torch
import torch.optim as optim

from opencog.ure import BackwardChainer
from opencog.atomspace import AtomSpace, types
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from pln import  TTruthValue
from pln import get_ttv, set_ttv
from opencog.bindlink import execute_atom, evaluate_atom
import numpy as np


atomspace = AtomSpace()
initialize_opencog(atomspace)
rule_base = ConceptNode('PLN')
ExecutionLink(SchemaNode("URE:maximum-iterations"), 
              rule_base, NumberNode('10'))


colors = ["green", "red", "yellow"]
fruits = ["apple", "banana"]
Pcolorfruit = [[0.45, 0.50, 0.05],
               [0.10, 0.00, 0.90]]

w_size = 4
w = torch.autograd.Variable(torch.normal(torch.zeros(w_size),
                                         0.1 * torch.ones(w_size)),
                            requires_grad=True)
res_string = "weights \nA*AB: {0}, A: {1}, AB: {2}, bias: {3}"


def weighted_modus_ponens_formula(B, AB, A):
    tA = get_ttv(A)
    tAB = get_ttv(AB)
    #print("Input: A=", tA, " AB=", tAB);
    new_tv = torch.sigmoid(w[0]*tA*tAB + w[1]*tA + w[2]*tAB + w[3])
    set_ttv(B, new_tv)
    #print("Result: B=", new_tv)
    return B


def params(train_implication):
    param = [w]
    for color in colors:
        for fruit in fruits:
            #if we want to train TTVs:
            if train_implication:
                ttv = torch.autograd.Variable(torch.Tensor((0.5, 0.5)),
                                              requires_grad=True)
                param += [ttv]
            else:
                ttv = TTruthValue(Pcolorfruit[fruits.index(fruit)]
                                  [colors.index(color)], 0.5)
            set_ttv(ImplicationLink(PredicateNode(fruit),         
                                    PredicateNode(color)), ttv)
    return param


def build_modus_ponens():
    mp_rule = BindLink(
            VariableList(
                TypedVariableLink(VariableNode("$P1"), TypeNode("PredicateNode")),
                TypedVariableLink(VariableNode("$C1"), TypeNode("ConceptNode")),
                TypedVariableLink(VariableNode("$P2"), TypeNode("PredicateNode"))),
            AndLink(
                EvaluationLink(VariableNode("$P1"), VariableNode("$C1")),
                ImplicationLink(VariableNode("$P1"),VariableNode("$P2"))),
            ExecutionOutputLink(
                GroundedSchemaNode("py: weighted_modus_ponens_formula"),
                ListLink(
                    EvaluationLink(VariableNode("$P2"), VariableNode("$C1")),
                    ImplicationLink(VariableNode("$P1"),
                                    VariableNode("$P2")),
                    EvaluationLink(VariableNode("$P1"), VariableNode("$C1")))))

    DefineLink(DefinedSchemaNode("modus_ponens_predicates"), mp_rule)
    MemberLink(DefinedSchemaNode("modus_ponens_predicates"), 
               ConceptNode("PLN"))


def main():
    build_modus_ponens()
    # train implication or just weigths of modus_ponens_formula
    train_implication = False
    if train_implication:
        print("train implication and modus_ponens_formula' weights")
    else:
        print("train only modus_ponens_formula' weights")

    optimizer = optim.SGD(params(train_implication), lr=1e-1)
    initial_w = w.clone()
    print('initial ' + res_string.format(*initial_w))
    print("inital implication links")
    for color in colors:
        for fruit in fruits:
            print(ImplicationLink(PredicateNode(fruit),         
                                  PredicateNode(color)))

    for i in range(300):
        fruit_id = np.random.randint(len(fruits))
        color_id = np.random.choice(len(colors), p=Pcolorfruit[fruit_id])
        fruit = fruits[fruit_id]
        color = colors[color_id]
        x = fruit + "-" + str(i)
        set_ttv(EvaluationLink(PredicateNode(fruit), 
                               ConceptNode(x)),
                TTruthValue(1.0, 1.0))
        optimizer.zero_grad()
        loss = torch.zeros(1)
        for c in colors:
            bc = BackwardChainer(atomspace, rule_base,
                                EvaluationLink(PredicateNode(c), ConceptNode(x)))
            bc.do_chain()
            evlink = bc.get_results().out[0]
            p = get_ttv(evlink)
            # print(evlink)
            if color == c:
                loss = loss - torch.log(p[0])
            else:
                loss = loss - torch.log(1-p[0])
        loss.backward()
        optimizer.step()
    print("implication links")
    for color in colors:
        for fruit in fruits:
            print(ImplicationLink(PredicateNode(fruit),         
                                  PredicateNode(color)))
    print("after training")
    print(res_string.format(w[0], w[1], w[2], w[2], w[3]))


if __name__ == '__main__':
    main()
