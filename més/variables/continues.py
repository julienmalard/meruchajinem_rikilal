from typing import Optional, Any

import pandas as pd
import pymc as pm

from .variable import Variable


class VariableContinue(Variable):
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
        return pm.Normal(name=soimême.nom, mu=mu + b, sigma=ét, observed=données)

    def préparer_données(soimême, données: pd.Series):
        return (données - données.mean()) / données.std()


class VariablePositive(VariableContinue):
    def __init__(soimême, nom):
        super().__init__(nom)

    def générer_variable_pm(soimême, dépendances: dict[str, Any], données: Optional[pd.Series]):
        if not dépendances:
            return données
        mu = soimême.générer_mu(dépendances)

        ét = pm.HalfNormal(name='ét_' + soimême.nom, sigma=10)
        b = pm.HalfNormal(
            name='b_' + soimême.nom, sigma=100
        )
        return pm.TruncatedNormal(name=soimême.nom, mu=mu + b, sigma=ét, lower=0, observed=données)

    def préparer_données(soimême, données: pd.Series):
        return données / données.std()
