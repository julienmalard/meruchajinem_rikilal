from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Union
import pandas as pd
if TYPE_CHECKING:
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

    def obtenir(soimême, variable: Union[str, Variable]):
        col_variable = next(c for c, v in soimême.colonnes_var.items() if str(variable) == str(v))
        return soimême.données_pd[col_variable]

    def __contains__(soimême, variable):
        return str(variable) in [str(v) for v in soimême.colonnes_var.values()]
