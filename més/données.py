from typing import Optional, Union

import pandas as pd
import xarray as xr

from .variables import Variable


class Données(object):
    def __init__(
            soimême,
            nom: str,
            données: Union[str, pd.DataFrame],
            colonnes_var: dict[str, Union[str, Variable]],
            col_région: str,
            année: Optional[int] = None,
            forme_sig: Optional[str] = None
    ):
        soimême.nom = nom
        soimême.colonnes_var = colonnes_var
        soimême.col_région = col_région
        soimême.année = année
        soimême.forme_sig = forme_sig

        if isinstance(données, pd.DataFrame):
            soimême.données_pd = données
        else:
            ext = données.split(".")[1]
            if ext == 'dta':
                soimême.données_pd = pd.read_stata(données)
            elif ext == 'csv':
                soimême.données_pd = pd.read_csv(données)

        soimême.données_pd = soimême.données_pd.loc[:, soimême.colonnes_var.keys()].dropna()

    def obtenir(soimême, variable: Variable):
        col_variable = next(c for c, v in soimême.colonnes_var.items() if str(variable) == str(v))
        données = soimême.données_pd[col_variable]
        données = variable.préparer_données(données)
        return données

    def __contains__(soimême, variable):
        return str(variable) in [str(v) for v in soimême.colonnes_var.values()]
