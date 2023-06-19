from numbers import Number
from typing import Optional, Any

import numpy as np
import pandas as pd
import pymc as pm

from .variable import Variable


class VariableÉchelle(Variable):
    def __init__(soimême, nom, n_catégories: Optional[Number] = None):
        super().__init__(nom)
        soimême.n_catégories = n_catégories

    def générer_variable_pm(soimême, dépendances: dict[str, Any], données: Optional[pd.Series]):
        if not dépendances:
            return données
        mu = soimême.générer_mu(dépendances)
        n_catégories = soimême.n_catégories or len(données.unique())

        divisions = pm.Normal(
            name='divisions_' + soimême.nom, mu=np.arange(-1, n_catégories - 2), sigma=10, shape=n_catégories - 1,
            transform=pm.distributions.transforms.univariate_ordered
        )
        return pm.OrderedLogistic(name=soimême.nom, cutpoints=divisions, eta=mu, observed=données, compute_p=False)


class VariableBooléenne(VariableÉchelle):
    def générer_variable_pm(soimême, dépendances: dict[str, Any], données: Optional[pd.Series]):
        if not dépendances:
            return données
        mu = soimême.générer_mu(dépendances)
        b = pm.Normal(
            name='b_' + soimême.nom, mu=0, sigma=100
        )
        return pm.Bernoulli(name=soimême.nom, logit_p=mu + b, observed=données)

    def __init__(soimême, nom):
        super().__init__(nom, n_catégories=2)
