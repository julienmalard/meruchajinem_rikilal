from typing import Any, Optional

import pandas as pd
import pymc as pm

from .variable import Variable


class VariableCatégorique(Variable):
    def __init__(soimême, nom):
        super().__init__(nom)

    def générer_variable_pm(soimême, dépendances: dict[str, Any], données: Optional[pd.Series]):
        if not dépendances:
            return données
        mu = soimême.générer_mu(dépendances)

        ét = pm.HalfNormal(name='ét_' + soimême.nom, sigma=10)
        b = pm.Normal(
            name='b_' + soimême.nom, mu=0, sigma=100
        )
        return pm.Categorical(name=soimême.nom, mu=mu + b, sigma=ét, observed=données)
