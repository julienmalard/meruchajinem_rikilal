import numpy as np

from més.données import Données
from més.modèle import Modèle
from més.variables import VariableÉchelle, VariableBooléenne, VariableCatégorique, VariablePositive, GroupeVars
from més.variables.continues import VariableBornée

if __name__ == "__main__":
    isa = VariableÉchelle('isa', 4)
    malnutrition = VariableBooléenne('malnutrition')
    santé = GroupeVars('santé', isa, malnutrition)

    brèche_pauvreté = VariableBornée('brèche pauvreté')
    catégorie_richesse = VariableÉchelle('pauvreté', 5)
    pauvreté = GroupeVars('pauvreté', brèche_pauvreté, catégorie_richesse)

    niveau_études = VariableÉchelle('niveau études')
    années_études = VariablePositive('années études')
    études = GroupeVars('études', niveau_études, années_études)

    langue_maternelle = VariableCatégorique('langue maternelle')
    parle_langue_dominante = VariableBooléenne('parle langue dominante')
    parle_langue_dominante_fréquemment = VariableBooléenne('parle langue dominante fréquemment')
    langue_maternelle_dominante = VariableBooléenne('langue maternelle dominante')
    langue = GroupeVars(
        'langue',
        langue_maternelle,
        parle_langue_dominante_fréquemment,
        parle_langue_dominante,
        langue_maternelle_dominante
    )

    indigène = VariableBooléenne('indigène')
    groupe_ethnique = VariableCatégorique('groupe ethnique')
    ethnicité = GroupeVars('ethnicité', indigène, groupe_ethnique)

    rural = VariableBooléenne('rural')
    genre = VariableCatégorique('genre')
    seule = VariableBooléenne('parent seul')

    mod = Modèle('principal')

    with mod:
        santé.dépend_de(pauvreté, études, langue)  # , seule, genre, ethnicité, rural)
        pauvreté.dépend_de(études, langue)  # , seule, genre, ethnicité, rural)
        # études.dépend_de(langue, seule, genre, ethnicité, rural)
        # langue.dépend_de(ethnicité, genre, rural)
        # seule.dépend_de(ethnicité, rural)
        # ethnicité.dépend_de(rural)

    Iximulew = Données(
        'Iximulew',
        données="Ruxe'el tzij roma SLAN.csv",
        colonnes_var={
            'ISA': isa,
            'brecha.pobr.ingresos': brèche_pauvreté,
            'leng.frec.cstlñ': parle_langue_dominante_fréquemment,
            'habla.cstlñ': parle_langue_dominante,
            'etnia.cstlñ': indigène,
            'jefa.mujer': genre,
            'rural': rural,
            'solteroa': seule,
            'educación.adultos': années_études
        }, col_région='COL_MUNI', année=2011,
        forme_sig='Munis.shp'
    )
    mod_guatemala = mod.appliquer(Iximulew)
    l_p_guatemala = mod_guatemala.impacte(études, santé)
    print({x["nom"]: [np.mean(x["dist"]), np.std(x["dist"])] for x in l_p_guatemala})
    print({x["nom"]: [np.mean(x["dist"]), np.std(x["dist"])] for x in mod_guatemala.impacte(pauvreté, santé)})
    mod_guatemala.dessiner_impacte()

    mod_guatemala.dessiner_traces()

    exit(0)
    భారత = Données(
        'భారత', 'fichier.nc', {
            'pauvreté': 'COL_PAUVRETÉ',
            'langue maternelle': '',
            'malnutrition': '',
            'groupe ethnique': ''
        }, col_région='COL_RÉGION'
    )


    mod_inde = mod.appliquer(భారత)
    l_p_inde = mod_inde.impacte(langue, pauvreté)
    mod_inde.prévalence(pauvreté)
    mod_inde.impacte(langue, santé)

    భారత.carte(l_p_inde)
    Monde.carte(l_p_inde, l_p_guatemala)
