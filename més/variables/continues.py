from numbers import Number
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

        ét = pm.HalfCauchy(name='ét_' + soimême.nom, beta=5)
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

        ét = pm.HalfCauchy(name='ét_' + soimême.nom, beta=5)
        b = pm.Normal(
            name='b_' + soimême.nom, sigma=100
        )
        return pm.LogNormal(name=soimême.nom, mu=mu + b, sigma=ét, observed=données)

    def préparer_données(soimême, données: pd.Series):
        return données / données.std() + 0.01


class VariableBornée(VariableContinue):
    def __init__(soimême, nom, minimum: Optional[Number] = None, maximum: Optional[Number] = None):
        super().__init__(nom)
        soimême._bornes = [minimum, maximum]

    def générer_variable_pm(soimême, dépendances: dict[str, Any], données: Optional[pd.Series]):
        if not dépendances:
            return données
        mu = soimême.générer_mu(dépendances)

        ét = pm.HalfCauchy(name='ét_' + soimême.nom, beta=5)
        b = pm.Normal(
            name='b_' + soimême.nom, mu=0, sigma=100
        )
        return pm.LogitNormal(name=soimême.nom, mu=mu+b, sigma=ét, observed=données)

    def obt_bornes(soimême, données: Optional[pd.Series]) -> list[Number, Number]:
        minimum, maximum = soimême._bornes
        if données is not None:
            minimum = min(données) if minimum is None else minimum
            maximum = max(données) if maximum is None else maximum
        minimum = 0 if minimum is None else minimum
        maximum = 1 if maximum is None else maximum
        return minimum, maximum

    def préparer_données(soimême, données: pd.Series):
        bornes = soimême.obt_bornes(données)
        données_ajustées = (données - bornes[0]) / (bornes[1] - bornes[0])
        return données_ajustées * 0.99 + 0.005  # Pour éviter les problèmes avec le logit plus tard
