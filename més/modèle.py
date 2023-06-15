from __future__ import annotations

import os.path
from typing import Union, Any

import arviz as az
import pymc as pm

from .contexte import contexte
from .données import Données
from .variables import GroupeVars, Relation, Variable

DOSSIER_RÉSULTATS = 'résultats'


class Modèle(object):
    def __init__(soimême, nom: str):
        soimême.nom = nom
        soimême.relations: list[Relation] = []

    def spécifier_relation(soimême, relation: Relation):
        soimême.relations.append(relation)

    def appliquer(soimême, données: Données) -> ModèleCalibré:
        mod = ModèleCalibré(soimême, données)
        return mod

    def __enter__(soimême):
        contexte.append(soimême)

    def __exit__(soimême, exc_type, exc_val, exc_tb):
        contexte.remove(soimême)


class ModèleCalibré(object):
    def __init__(soimême, modèle: Modèle, données: Données):
        soimême.modèle = modèle
        soimême.données = données

    def impacte(soimême, dépendante: Union[GroupeVars, Variable], indépendante: Union[GroupeVars, Variable], ):
        var_dépendante = soimême.résoudre_variable(dépendante)
        var_indépendante = soimême.résoudre_variable(indépendante)
        trace = soimême.obtenir_calibration()

    def obtenir_calibration(soimême):
        fichier_calibs = soimême.obtenir_fichier_calibs()
        if not os.path.isfile(fichier_calibs):
            soimême.calibrer()

        return az.from_netcdf(fichier_calibs)

    def calibrer(soimême):
        fichier_calibs = soimême.obtenir_fichier_calibs()
        with pm.Model():
            soimême.créer_modèle()
            trace = pm.sample()
        az.to_netcdf(trace, fichier_calibs)

    def obtenir_fichier_calibs(soimême):
        return os.path.join(DOSSIER_RÉSULTATS, 'traces', soimême.modèle.nom, soimême.données.nom + 'ncdf')

    def résoudre_variable(soimême, variable: Union[GroupeVars, Variable]) -> Variable:
        if isinstance(variable, Variable):
            return variable
        try:
            return next(v for v in variable if v.nom in soimême.données)
        except StopIteration:
            raise ValueError(f"Aucune variable disponible dans les données pour groupe {variable.nom}")

    def créer_modèle(soimême):
        variables = soimême.résoudre_variables()
        résolues: dict[str, Any] = {}
        n_non_résolues = len(variables)

        while n_non_résolues > 0:
            prêtes = [v for v in variables if all(d.nom in résolues for d in soimême.dépendances(v))]
            for v in prêtes:
                résolues[v.nom] = v.générer_variable_pm(
                    {c: vl for c, vl in résolues.items() if c in [str(x) for x in soimême.dépendances(v)]},
                    soimême.données.obtenir(v)
                )
                variables.remove(v)

            if n_non_résolues == len(variables) != 0:
                raise ValueError(
                    f"Connexions circulaires : {', '.join([v.nom for v in variables if v.nom not in résolues])}"
                )

            n_non_résolues = len(variables)

    def dépendances(soimême, variable: Variable) -> list[Variable]:
        dépendances: list[Variable] = []
        for r in soimême.modèle.relations:
            if soimême.résoudre_variable(r.dépendante) is variable:
                dépendances.append(soimême.résoudre_variable(r.indépendante))

        return dépendances

    def résoudre_variables(soimême) -> list[Variable]:
        variables: list[Variable] = []

        for r in soimême.modèle.relations:
            for v in [r.dépendante, r.indépendante]:
                if (résolue := soimême.résoudre_variable(v)) not in variables:
                    variables.append(résolue)
        return variables
