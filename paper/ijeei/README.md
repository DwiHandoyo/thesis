# IJEEI Paper — CQRS + LQR

Sumber LaTeX paper IJEEI yang dikompresi dari tesis. Disusun modular agar mudah diubah saat data eksperimen final.

## Struktur

```
paper/ijeei/
├── paper.tex                # Entry point. Compile file ini.
├── config/
│   └── ijeei.sty            # Style: A4, Times 12pt, 1.5 spasi, heading 1./A./1)
├── data/
│   ├── info.tex             # Judul, penulis, afiliasi
│   └── data.tex             # Semua angka eksperimen sebagai macro
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_proposed_method.tex
│   ├── 04_experiment.tex
│   ├── 05_results.tex       # Berisi placeholder TBD untuk angka final
│   └── 06_conclusion.tex
├── figures/                 # 7-10 gambar terpilih dari thesis/src/resources
├── references.bib           # BibTeX, gaya IEEE
└── template.pdf             # Template asli IJEEI sebagai referensi
```

## Cara Compile

```bash
cd paper/ijeei
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
```

## Cara Update saat Data Final

Semua angka headline disentralisasi di `data/data.tex` sebagai `\newcommand`.
Setelah eksperimen final, edit file itu saja — paper otomatis ikut update.

Contoh:
```latex
\newcommand{\meanQueueLQR}{1234}     % ganti dari TBD ke nilai final
\newcommand{\regretLQR}{0.05}
```

Ada juga blok `% TODO` di [sections/05_results.tex](sections/05_results.tex)
yang menandai paragraf yang perlu ditulis ulang dengan diskusi kuantitatif.

## Bagian yang Sengaja Disisakan (Cek Sebelum Submit)

1. **Sitasi `morris2023` dan `rahman2010`** di `references.bib` masih placeholder, ganti dengan referensi tepat dari `chapters/references.tex`.
2. **Hasil per skenario beban** (ramp, impulse, periodic, low) di `sections/05_results.tex` — ada `% TODO` blok satu paragraf per pola.
3. **Limitations & threats to validity** di akhir `sections/05_results.tex`.
4. **Analisis residual identifikasi sistem** ($R^2$, eigenvalue $A$) di `sections/05_results.tex` subbab A.
5. **Author CV + foto** (per template, sebelum daftar pustaka) belum ditambahkan.
6. **Email afiliasi pembimbing** di `data/info.tex` masih placeholder.

## Format IJEEI yang Sudah Diikuti

- A4, single column, Times New Roman 12pt
- 1.5 line spacing (untuk review version)
- Section 1, 2, 3 (bold) + Subsection A, B (italic) + Subsubsection 1) (italic)
- Caption: Figure di bawah gambar, Table di atas tabel
- Abstract italic, 100-200 kata
- Keywords italic
- Sitasi IEEE numbered `[1]`
- Equation centered, nomor di kanan
- Daftar pustaka IEEEtran style

## Estimasi Panjang

Target 12-15 halaman saat 1.5 spasi. Konversi ke versi final B5 (oleh editor IJEEI) akan memadatkan tampilan ~20%.

## Submit

1. Buka [https://www.ijeei.org/](https://www.ijeei.org/) → menu *How to Submit*
2. Register akun di OJS
3. Upload `paper.pdf` + supplementary (kalau ada)
4. Copyright Transfer Agreement (ditandatangani corresponding author)
5. APC USD 250 dibayar setelah accepted
