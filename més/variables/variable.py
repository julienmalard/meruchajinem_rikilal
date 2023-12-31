from __future__ import annotations

from typing import Union, TYPE_CHECKING, Optional, Any

import pandas as pd
import pymc as pm

from .relation import Relation
from ..contexte import contexte

if TYPE_CHECKING:
    from .groupes import GroupeVars


class Variable(object):
    def __init__(soimême, nom: str):
        soimême.nom = nom

    def dépend_de(soimême, *variables: Union["Variable", "GroupeVars"]):
        for mod in contexte:
            for variable in variables:
                relation = Relation(indépendante=variable, dépendante=soimême)
                mod.spécifier_relation(relation)

    def générer_mu(soimême, dépendances: dict[str, Any]):
        mu = None
        for d in dépendances:
            a = pm.Normal(
                name=nom_coefficient_relation(d, soimême),
                mu=0, sigma=100
            )

            if mu is None:
                mu = a * dépendances[d]
            else:
                mu = mu + a * dépendances[d]

        return mu

    def préparer_données(soimême, données: pd.Series):
        return données

    def générer_variable_pm(soimême, dépendances: dict[str, Any], données: Optional[pd.Series]):
        raise NotImplementedError()

    def __str__(soimême):
        return soimême.nom


def nom_coefficient_relation(indépendant: Variable, dépendant: Variable) -> str:
    return 'rel_' + str(indépendant) + '_envers_' + str(dépendant)
