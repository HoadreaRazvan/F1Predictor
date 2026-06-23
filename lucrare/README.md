# Lucrare de licență — cod LaTeX

Lucrarea „Comparare algoritmi de predicție cu învățare automată" în format LaTeX.

## Fișiere

- `lucrare.tex` — documentul principal (preambul + toate capitolele)
- `referinte.bib` — bibliografia (format biblatex/IEEE, compatibil cu export din Zotero)

Figurile sunt incluse direct din `../outputs/plots/` (generate cu `python main.py experiments`),
prin `\graphicspath`. Nu trebuie copiate manual.

## Cum compilezi

### Varianta 1 — Overleaf (nu necesită instalare)

1. Creează un proiect nou pe [overleaf.com](https://www.overleaf.com).
2. Încarcă `lucrare.tex` și `referinte.bib`.
3. Creează un folder `figuri/` în proiect și încarcă acolo figurile PNG din `outputs/plots/`
   (preambulul caută deja figurile și în `figuri/`).
4. În *Menu → Settings*, setează **Compiler: XeLaTeX** (obligatoriu — fontspec NU merge pe pdfLaTeX).
5. Apasă *Recompile*. Bibliografia (biber) rulează automat.

> **Fonturile sunt deja tratate automat.** Times New Roman și Consolas nu există pe Overleaf;
> documentul cade automat pe clonele *TeX Gyre Termes* și *Inconsolata* (livrate cu orice
> distribuție TeX). Local, pe Windows, vor fi folosite fonturile reale.

### Depanare: „Your compile timed out" pe Overleaf

Cauza obișnuită era căutarea fonturilor Windows lipsă — rezolvată prin fallback-ul automat de mai
sus. Dacă tot expiră:

- asigură-te că ai setat compilatorul pe **XeLaTeX** (nu pdfLaTeX/LuaLaTeX);
- verifică faptul că figurile PNG sunt încărcate (lipsa lor dă erori, nu timeout, dar încetinește);
- alternativ, compilează **local** (Varianta 2) — nu există limită de timp și ai fonturile reale.

### Varianta 2 — local (MiKTeX sau TeX Live pe Windows)

Necesită **XeLaTeX** + **biber** (pentru Times New Roman și Consolas reale, instalate pe Windows):

```
xelatex lucrare.tex
biber   lucrare
xelatex lucrare.tex
xelatex lucrare.tex
```

Sau, mai simplu, cu `latexmk`:

```
latexmk -xelatex lucrare.tex
```

## Conformitate cu cerințele de redactare

- Font de bază Times New Roman 12pt, spațiere 1,5 rânduri, aliniere justified, margini 2 cm.
- Cod cu font monospace Consolas 10pt, pe fundal alb, la 1 rând, cu evidențiere de sintaxă
  (fără poze cu cod).
- Figuri numerotate automat „Figura X.Y", centrate, caption sub figură; toate sunt referite în text.
- Tabele numerotate automat „Tabelul X.Y", centrate, caption deasupra; toate sunt referite în text.
- Bibliografie IEEE generată automat; fiecare intrare este referită în text.
- Numerotarea paginilor începe de la cuprins (pagina de titlu nu este numerotată).
