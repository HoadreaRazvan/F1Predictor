# 🏎️ F1 Podium Predictor

Aplicație de Machine Learning care prezice **podiumul (P1, P2, P3)** în Formula 1 —
atât per cursă, cât și clasamentul final de campionat — folosind **3 algoritmi
implementați complet de la zero** (fără scikit-learn).

## ✨ Caracteristici

- **3 modele ML scrise manual peste NumPy** (zero biblioteci ML gata făcute):
  - **Regresie Logistică** — gradient descent, cross-entropy, regularizare L2, ponderare pe clase
  - **Arbore de Decizie** — CART cu criteriul **entropie + information gain**
  - **Random Forest** — bagging + subset aleator de feature-uri, peste arborele propriu
- **Date reale FastF1** (2018–2025, 8 sezoane, ~3500 rezultate) — încărcate **offline** din cache
- **Feature engineering fără data leakage** — toate feature-urile sunt strict cauzale
- **Validare temporală train/validare/test pe ani** — pentru sezonul `S`: antrenare pe `≤ S-2`,
  validare pe `S-1`, test pe `S`
- **Predicție dublă** — podium per cursă + agregare în clasament de campionat
- **Metrici implementate manual** + grafice PNG (matplotlib)
- Interfață **CLI** completă

## 📦 Instalare

```bash
pip install -r requirements.txt
```

Dependențe: `fastf1` (doar încărcarea datelor), `numpy`, `pandas`, `matplotlib`.
Datele sunt deja în `data/cache/` (cache FastF1) — **nu e nevoie de internet**.

## 🚀 Utilizare

Toate comenzile se rulează prin `python main.py <comandă>`:

```bash
# 1. Construiește datasetul (rezultate + feature-uri) din cache
python main.py build-data

# 2. Validare temporală (train ≤ S-2, validare S-1, test S) pentru toate modelele (+ grafice)
python main.py evaluate --model all

# 3. Antrenează și salvează un model (până la un sezon-prag)
python main.py train --model forest --train-until 2024

# 4. Prezice podiumul unei curse
python main.py predict-race --season 2025 --round 1 --model forest

# 5. Prezice un sezon întreg: podium per cursă + clasament campionat
python main.py predict-season --season 2025 --model forest

# 6. Efectul curselor in-sezon: compară performanța cu vs fără cursele deja disputate din sezon
python main.py insezon --season 2025 --model all
```

`--model` acceptă `logreg`, `tree`, `forest` (sau `all` la `evaluate`/`train`/`insezon`).

### Efectul curselor in-sezon (comanda `insezon`)
`python main.py insezon --season 2025` compară, pe fiecare rundă `R` din sezon, două regimuri:
**fără in-sezon** (model antrenat doar pe sezoanele anterioare) și **cu in-sezon** (model antrenat
pe sezoanele anterioare + cursele 2025 de dinaintea rundei R). Generează în `outputs/plots/` câte
un grafic pe metrică (`insezon_2025_accuracy.png`, `..._podium_hit_rate.png`, `..._exact_podium.png`,
`..._winner_acc.png`) cu **media cumulativă** pe runde — culoarea = modelul, linia continuă = cu
in-sezon, linia întreruptă = fără — astfel încât diferența de performanță crește vizibil pe parcursul
sezonului. Salvează și `outputs/insezon_2025.csv`.

## 📊 Rezultate (medii pe sezoanele de test 2020–2025)

| Model | Acuratețe | Podium hit-rate | Podium exact | Câștigător (Top-1) |
|-------|:---------:|:---------------:|:------------:|:------------------:|
| Regresie Logistică | 0.867 | 0.672 | 0.240 | **0.536** |
| Arbore de Decizie  | 0.872 | 0.581 | 0.167 | 0.388 |
| **Random Forest**  | **0.903** | **0.673** | **0.253** | 0.471 |

- **Podium hit-rate** ≈ 0.67 → în medie **~2 din 3** piloți de pe podium sunt prezişi corect.
- **Podium exact** = fracția curselor în care toți cei 3 piloți de pe podium sunt nimeriți (ca set).
- **Câștigător (Top-1)** = fracția curselor în care P1 e prezis corect.

> Notă: valorile se obțin rulând `python main.py evaluate --model all`.

## 🧠 Cum funcționează

### Problema
Pentru fiecare `(cursă, pilot)` antrenăm un clasificator binar: *va termina pe podium?*
(`is_podium = 1` dacă locul ≤ 3). Fiindcă un clasificator per pilot nu garantează exact 3
pozitivi, pentru o cursă calculăm `P(podium)` pentru toți piloții, îi **sortăm descrescător**
și luăm primii 3 ca podium prezis (cea mai mare probabilitate = P1).

### Fără data leakage
Feature-urile pentru cursa R folosesc **doar** rezultate din curse strict anterioare:
formă rulantă (ultimele N curse), puncte sezon-to-date, forța echipei, istoric pe circuit,
poziția pe grid/calificări, vreme. Cursele se procesează cronologic, iar starea istorică se
actualizează **după** calcularea feature-urilor.

### Validare temporală train/validare/test pe ani
Pentru fiecare sezon de test `S`: antrenăm pe sezoanele `≤ S-2` (adică `< S-1`), ținem sezonul
`S-1` ca **validare** și raportăm performanța finală pe sezonul nevăzut `S`. Astfel prezicem mereu
"viitorul" cu modele antrenate doar pe "trecut", iar `S-1` rămâne ținut deoparte. Sezoane de test:
2020–2025.

## 📈 Grafice (comanda `evaluate`)

`python main.py evaluate --model all` salvează în `outputs/plots/` grafice cu performanța celor
3 modele pe cele 4 metrici (acuratețe, podium hit-rate, podium exact, câștigător Top-1):

- **Câte un grafic pe metrică**, defalcat pe sezoanele de test: `accuracy.png`,
  `podium_hit_rate.png`, `exact_podium.png`, `winner_acc.png` (bare grupate, 3 modele per sezon).
- **Un grafic-sumar comparativ** (`metric_summary.png`) — cele 4 metrici pe axa X, mediate pe
  sezoanele de test.
- **Importanța feature-urilor** per model (`feature_importance_<model>.png`).

Opțiuni: `--model {logreg,tree,forest,all}`, `--no-plots`.

## 🗂️ Structura proiectului

```
src/f1pred/
├── config.py              # căi, sezoane, hiperparametri, sistemul de puncte
├── data/                  # încărcare FastF1 offline + tabel "lung"
├── features/engineering.py# feature-uri cauzale + etichetă is_podium
├── models/                # logistic_regression, decision_tree, random_forest (de la zero)
├── evaluation/            # metrici manuale + validare temporală train/validare/test
├── predict/               # podium per cursă + clasament campionat
├── viz/plots.py           # grafice PNG
└── cli.py                 # interfața de linie de comandă
tests/                     # teste unitare (modele, metrici, anti-leakage)
data/cache/                # cache FastF1 (2018–2025)
outputs/                   # CSV-uri, modele salvate, grafice (generat)
```

## 🧪 Teste

```bash
pip install pytest
python -m pytest tests/ -q
```

Testele verifică corectitudinea celor 3 algoritmi pe seturi-jucărie, metricile și —
important — **garanția anti-leakage** a feature-urilor.

## 📝 Note

- Toți algoritmii ML, metricile și antrenarea sunt scrise manual. NumPy e folosit doar pentru
  algebră vectorizată, pandas pentru date, FastF1 doar pentru încărcarea datelor, matplotlib
  doar pentru grafice — **niciunul nu e bibliotecă de ML**.
- Cache-ul FastF1 e citit în mod offline strict (`Cache.offline_mode`), deci aplicația rulează
  fără internet.
