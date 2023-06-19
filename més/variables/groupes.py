from typing import Union

from .relation import Relation
from .variable import Variable
from ..contexte import contexte


class GroupeVars(object):
    def __init__(soimême, nom: str, *variables: Variable):
        soimême.nom = nom
        soimême.variables = variables

    def dépend_de(soimême, *variables: Union["Variable", "GroupeVars"]):
        for mod in contexte:
            for variable in variables:
                relation = Relation(indépendante=variable, dépendante=soimême)
                mod.spécifier_relation(relation)

    def __iter__(soimême):
        for v in soimême.variables:
            yield v

    def __contains__(soimême, item):
        return item in soimême.variables

    def __str__(soimême):
        return soimême.nom
