from __future__ import annotations

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..variables import Variable, GroupeVars


class Relation(object):
    def __init__(soimême, indépendante: Union[Variable, GroupeVars], dépendante: Union[Variable, GroupeVars]):
        soimême.indépendante = indépendante
        soimême.dépendante = dépendante

    def __lt__(soimême, autre: Relation):
        if soimême.indépendante != autre.indépendante:
            return str(soimême.indépendante) < str(autre.indépendante)
        else:
            return str(soimême.dépendante) < str(autre.dépendante)
