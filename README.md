# Examen BentoML

Afin de reproduire le projet vous devez suivre les étapes suivantes:

- Forker le projet sur votre compte github

- Cloner le projet sur votre machine

- Créer un environnement virtuel puis l'activer en source

- Install Bentoml, docker, requirements

- executer les fichiers en python3. Afin de sauvegarder dans BentoML votre modèle:
    /home/votre_nom/examen_bentoml/src/prepare_data.py
    /home/votre_nom/examen_bentoml/src/train_model.py
- Se positionner à la racine de examen BentoML puis lancer les instructions suivantes:
  - <bentoml build>
  - <bentoml list>
  - <docker --version>
  - <bentoml containerize ridge_service:latest>
  - <docker run -p 3000:3000 ridge_service:TAG_DE_VOTRE_BENTO>

Bon travail! Vous n'avez plus qu'à tester l'API rest avec postman par exemple
