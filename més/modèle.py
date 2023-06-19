from __future__ import annotations

import hashlib
import os.path
from os import makedirs
from typing import Union, Any, Optional, TypedDict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pymc as pm

from .contexte import contexte
from .données import Données
from .variables import GroupeVars, Relation, Variable
from .variables.variable import nom_coefficient_relation

DOSSIER_RÉSULTATS = 'résultats'


class Impacte(TypedDict):
    cheminement: list[Variable]
    nom: str
    dist: np.ndarray
    composantes: list[dict[str, Union[Variable, np.ndarray]]]


class Modèle(object):
    def __init__(soimême, nom: str):
        soimême.nom = nom
        soimême.relations: list[Relation] = []

    def spécifier_relation(soimême, relation: Relation):
        soimême.relations.append(relation)

    def appliquer(soimême, données: Données) -> ModèleCalibré:
        mod = ModèleCalibré(soimême, données)
        return mod

    def empreinte_structure(soimême) -> str:
        return hashlib.md5(
            ";".join(f'{r.indépendante}->{r.dépendante}' for r in sorted(soimême.relations)).encode()
        ).hexdigest()[:10]

    def __enter__(soimême):
        contexte.append(soimême)

    def __exit__(soimême, exc_type, exc_val, exc_tb):
        contexte.remove(soimême)


class ModèleCalibré(object):
    def __init__(soimême, modèle: Modèle, données: Données):
        soimême.modèle = modèle
        soimême.données = données

    def impacte(
            soimême,
            dépendante: Union[GroupeVars, Variable],
            indépendante: Union[GroupeVars, Variable]
    ) -> list[Impacte]:
        cheminements = soimême.cheminements(dépendante, indépendante)
        trace = soimême.obtenir_calibration()

        impactes: list[Impacte] = []
        for ch in cheminements:
            impacte = np.array(1.)
            composantes = []
            for i, v in enumerate(ch[1:]):
                coefficient = nom_coefficient_relation(ch[i], v)
                impacte_ch = trace.posterior[coefficient].values

                impacte = impacte * impacte_ch
                composantes.append({
                    "dépendante": v,
                    "indépendante": ch[i],
                    "dist": impacte_ch
                })
            impactes.append(Impacte(
                cheminement=ch,
                nom=" -> ".join(str(v) for v in ch),
                dist=impacte,
                composantes=composantes
            ))

        return impactes

    def dessiner_impacte(soimême):
        trace = soimême.obtenir_calibration()

        relations = soimême.modèle.relations
        variables = list(set([
            x for r in relations for x in [
                soimême.résoudre_variable(r.dépendante),
                soimême.résoudre_variable(r.indépendante)
            ]
        ]))
        liens = {
            "source": [],
            "target": [],
            "value": []
        }

        def trace_relation(de: Variable, à: Variable) -> np.ndarray:
            var_de = soimême.résoudre_variable(de)
            var_à = soimême.résoudre_variable(à)
            coef = nom_coefficient_relation(var_de, var_à)
            return trace.posterior[coef].values

        def force_relation(de: Variable, à: Variable) -> float:
            return abs(trace_relation(de, à).mean())

        # Normaliser les impactes
        facteurs = {}
        while len(facteurs) < len(variables):
            non_normalisées = [v for v in variables if str(v) not in facteurs]
            prêtes = [
                v for v in non_normalisées if not any(
                    soimême.résoudre_variable(r.indépendante) is v and soimême.résoudre_variable(r.dépendante) in non_normalisées for r in relations
                )
            ]

            for p in prêtes:
                causes_de_p = [soimême.résoudre_variable(r.indépendante) for r in relations if soimême.résoudre_variable(r.dépendante) is p]
                dépendantes_de_p = [soimême.résoudre_variable(r.dépendante) for r in relations if soimême.résoudre_variable(r.indépendante) is p]
                taille_sortie_p = np.sum([
                    force_relation(de=p, à=d) * facteurs[str(d)] for d in dépendantes_de_p
                ]) or 1
                taille_entrée_p = np.sum([
                    force_relation(de=c, à=p) for c in causes_de_p
                ]) or taille_sortie_p
                facteur_p = taille_sortie_p / taille_entrée_p
                facteurs[str(p)] = facteur_p

        for r in relations:
            var_r_indépendante = soimême.résoudre_variable(r.indépendante)
            var_r_dépendante = soimême.résoudre_variable(r.dépendante)

            liens["source"].append(variables.index(var_r_indépendante))
            liens["target"].append(variables.index(var_r_dépendante))
            liens["value"].append(
                force_relation(var_r_indépendante, var_r_dépendante) * facteurs[str(var_r_dépendante)]
            )

        fig = go.Figure(data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 15,
                    "line": dict(color="black", width=0.5),
                    "label": [str(v) for v in variables]
                },
                link=liens
            )
        ])

        fig.update_layout(title_text=soimême.modèle.nom)
        fig.write_image(soimême.obtenir_fichier_graphiques(f"sankey.jpeg"))

    def cheminements(
            soimême, indépendante: Union[GroupeVars, Variable], dépendante: Union[GroupeVars, Variable]
    ) -> list[list[Variable]]:
        var_dépendante = soimême.résoudre_variable(dépendante)
        var_indépendante = soimême.résoudre_variable(indépendante)
        relations = soimême.modèle.relations

        def chercher_cheminement(de: Variable, à: Variable, base: Optional[list[Variable]] = None):
            cheminements: list[list[Variable]] = []
            relations_de = [r for r in relations if soimême.résoudre_variable(r.indépendante) is de]
            base = base or [de]
            for r in relations_de:
                dépendante_r = soimême.résoudre_variable(r.dépendante)
                if dépendante_r is à:
                    cheminements.append([*base, dépendante_r])
                else:
                    for ch in chercher_cheminement(de=dépendante_r, à=à, base=[*base, dépendante_r]):
                        cheminements.append(ch)
            return cheminements

        return chercher_cheminement(var_indépendante, var_dépendante)

    def dessiner_traces(soimême):
        trace = soimême.obtenir_calibration()

        for v in trace.posterior:
            az.plot_trace(trace, [v])
            fig = plt.gcf()
            fig.suptitle(f"Trace {soimême.données.nom}, {v}")

            nom_fichier = os.path.join("traces", v)
            fichier_figure = os.path.join(soimême.obtenir_fichier_graphiques(nom_fichier))
            fig.savefig(fichier_figure)
            plt.close(fig)

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

        dossier_calibs = os.path.dirname(fichier_calibs)
        if not os.path.isdir(dossier_calibs):
            makedirs(dossier_calibs)

        az.to_netcdf(trace, fichier_calibs)

    def obtenir_fichier_calibs(soimême):
        empreinte_structure = soimême.modèle.empreinte_structure()
        return os.path.join(DOSSIER_RÉSULTATS, 'calibs', soimême.modèle.nom + "_" + empreinte_structure,
                            soimême.données.nom + '.ncdf')

    def obtenir_fichier_graphiques(soimême, nom_fichier: str) -> str:
        empreinte_structure = soimême.modèle.empreinte_structure()
        fichier_graphique = os.path.join(
            DOSSIER_RÉSULTATS, 'figures', soimême.modèle.nom + "_" + empreinte_structure, soimême.données.nom,
            nom_fichier
        )
        dossier_graphique = os.path.dirname(fichier_graphique)
        if not os.path.isdir(dossier_graphique):
            os.makedirs(dossier_graphique)
        return fichier_graphique

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
