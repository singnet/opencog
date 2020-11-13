"""
Example of optimizing truth values of implication links with pytorch
"""

import torch
import torch.optim as optim

from opencog.ure import BackwardChainer
from opencog.atomspace import AtomSpace, types
from opencog.utilities import initialize_opencog, finalize_opencog
from opencog.type_constructors import *
from opencog.atomspace import TensorTruthValue as TTruthValue
from opencog.bindlink import execute_atom
import numpy as np

atomspace = AtomSpace()
initialize_opencog(atomspace)
rule_base = ConceptNode('PLN')
ExecutionLink(SchemaNode("URE:maximum-iterations"), rule_base, NumberNode('30'))


colors = ["green", "red", "yellow"]
fruits = ["apple", "banana"]
P_color_given_fruit = [[0.45, 0.50, 0.05],
                       [0.10, 0.00, 0.90]]


def modus_ponens_simplified_formula(B, AB, A):
    new_tv = A.tv.torch() * AB.tv.torch()
    B.tv = TTruthValue(new_tv)
    return B


def params():
    params = []
    for color in colors:
        for fruit in fruits:
            ttv = torch.autograd.Variable(torch.Tensor((0.5, 0.5)), requires_grad=True)
            ImplicationLink(PredicateNode(fruit), 
                            PredicateNode(color)).tv = TTruthValue(ttv)
            params += [ttv]
    return params


def generate_modus_ponens():
    mp_rule = BindLink(
            VariableList(
                TypedVariableLink(VariableNode("$P1"), TypeNode("PredicateNode")),
                TypedVariableLink(VariableNode("$C1"), TypeNode("ConceptNode")),
                TypedVariableLink(VariableNode("$P2"), TypeNode("PredicateNode"))),
            AndLink(
                EvaluationLink(VariableNode("$P1"), VariableNode("$C1")),
                ImplicationLink(VariableNode("$P1"),VariableNode("$P2"))),
            ExecutionOutputLink(
                GroundedSchemaNode("py: modus_ponens_simplified_formula"),
                ListLink(
                    EvaluationLink(VariableNode("$P2"), VariableNode("$C1")),
                    ImplicationLink(VariableNode("$P1"),VariableNode("$P2")),
                    EvaluationLink(VariableNode("$P1"), VariableNode("$C1")))))

    DefineLink(DefinedSchemaNode("modus_ponens_predicates"), mp_rule)
    MemberLink(DefinedSchemaNode("modus_ponens_predicates"), ConceptNode("PLN"))


def main():
    generate_modus_ponens()
    optimizer = optim.SGD(params(), lr=1e-2)
    for i in range(100):
        fruit_id = np.random.randint(len(fruits))
        color_id = np.random.choice(len(colors), p=P_color_given_fruit[fruit_id])
        fruit = fruits[fruit_id]
        color = colors[color_id]
        x = fruit + "-" + str(i)

        # a particular fruit, color observation
        EvaluationLink(PredicateNode(fruit), 
                            ConceptNode(x)).tv = TTruthValue(1.0, 1.0)

        optimizer.zero_grad()
        loss = torch.zeros(1)
        for c in colors:
            # compute P(fruit|color) 
            bc = BackwardChainer(atomspace, rule_base,
                                EvaluationLink(PredicateNode(c), ConceptNode(x)))
            bc.do_chain()
            evlink = bc.get_results().out[0]
            p_fruit_given_color = evlink.tv
            print(evlink)
            if color == c:
                loss = loss - torch.log(p_fruit_given_color.mean)
            else:
                loss = loss - torch.log(1 - p_fruit_given_color.mean)
        print('loss: ' + str(loss))
        loss.backward()
        optimizer.step()
    print("results after train:")
    for color in colors:
       for fruit in fruits:
            print(ImplicationLink(PredicateNode(fruit), PredicateNode(color)))


if __name__ == '__main__':
    main()
