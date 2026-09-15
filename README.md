# DVF Scout · V2 Cloud

V2 de l'application Streamlit d'analyse immobilière à partir des données publiques françaises.

## Nouveautés V2

- DPE public ADEME avec score de rapprochement et niveau de confiance.
- Interface en une seule page verticale, sans onglets.
- Cartes de synthèse plus lisibles.
- Bloc « À retenir ».
- Réglages simples : rayon, période, tolérance de surface, tolérance sur les pièces.
- Mode expert rangé dans un menu déroulant.
- Score de robustesse de l'échantillon DVF.
- Carte enrichie avec le bien étudié, même numéro, même rue et autres comparables.
- Bouton de copie de l'argumentaire.
- Les éléments qualitatifs restent descriptifs : aucune prime de prix n'est inventée automatiquement.

## Mettre à jour l'application déjà hébergée sur Streamlit

Si la V1 est déjà en ligne via GitHub + Streamlit :

1. Décompressez le ZIP V2.
2. Ouvrez votre dépôt GitHub `dvf-scout`.
3. Remplacez les fichiers `streamlit_app.py`, `core.py`, `requirements.txt` et `README.md` par ceux de cette V2.
4. Validez avec **Commit changes**.
5. Streamlit détecte normalement le commit et redéploie automatiquement l'application. Attendez quelques minutes puis rechargez l'URL existante.

Le fichier d'entrée reste : `streamlit_app.py`.

## Sources

- DGFiP / data.gouv.fr : Demandes de valeurs foncières géolocalisées.
- IGN / Géoplateforme : géocodage d'adresse.
- ADEME : DPE logements existants depuis juillet 2021.

## Important

Le DPE est rapproché automatiquement par adresse, identifiant BAN, surface et type de bien. Dans une copropriété, plusieurs DPE peuvent correspondre à la même adresse. L'application affiche donc un niveau de confiance et permet de voir les principaux candidats au lieu de présenter un rapprochement incertain comme certain.
