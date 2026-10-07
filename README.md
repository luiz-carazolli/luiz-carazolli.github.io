# Luiz Carazolli · Academic website

A static academic portfolio for GitHub Pages. Dark navy/ivory and light warm-white/navy themes with terracotta accents, Source Serif 4, no animation, and a persisted theme preference. All content is rendered into HTML. Math is compiled into HTML and accessible MathML with locally vendored KaTeX. Fonts, portrait, and CV are local. There are no analytics, cookies, external runtime requests, package-install steps, or JavaScript framework dependencies.

## Build and preview

Requires Python 3.10+ and Node.js 22+ (no Python packages needed for the website build):

```sh
python3 scripts/build.py
python3 scripts/check.py
python3 -m http.server 8000 --directory dist --bind 127.0.0.1
```

Visit http://127.0.0.1:8000. To use a Node binary outside PATH, set `NODE` to its absolute path for the build.

## Edit through this chat

- `content/profile.json`: biography, contact links, funding, professional headline, update date, and optional canonical site URL.
- `content/projects.json`: project cards with summary, tags, and links (`[{"label":"Code", "url":"https://…"}]`). Set `pending_details` to false when populated.
- `content/research.json`: research projects, dates, advisors, funding, and optional report URLs.
- `content/publications.json`: publication metadata. The first publication is highlighted on the homepage; all publications appear in Research. Update the CV entries and BibTeX file when adding papers.
- `content/posts.json`: blog titles, dates, summaries, tags, and publication status. Published posts appear newest first; `order` breaks ties on the same date. The newest published post automatically fills the homepage’s “From the blog” feature on every build. Set `published` to false to keep a draft out of the generated site.
- `content/posts/`: article source HTML with LaTeX `\(...\)` and `\[...\]` delimiters. To add a post, create `<slug>.html` here and an entry with that slug in `content/posts.json`; no template edits are needed. Use semantic HTML and footnote anchors. Headers come from the metadata. SVG figures in `content/figures/` are included using `<!-- figure:filename.svg -->`.
- `assets/style.css`: shared design and both color palettes.
- `scripts/build.py`: page templates and CV presentation.
- `scripts/build_cv.py`: generates the curated, two-page PDF from profile/research/publication data and selected CV entries. Requires ReportLab and a local DejaVu or Arial font; see the script. Run it again when changing the CV, then rebuild the website. The PDF is checked in, so CI does not require ReportLab.

The website is currently in English. Structure and content are separated to support Portuguese pages later, without showing an incomplete language toggle.

## Publish to GitHub Pages

The intended repository is `luiz-carazolli/luiz-carazolli.github.io`, which will serve at `https://luiz-carazolli.github.io`. No remote was configured in this checkout, so once that repository exists:

1. Push these source files to the repository's main or master branch.
2. In the repository's Settings → Pages, select **GitHub Actions** as the build/deployment source.
3. The included `.github/workflows/pages.yml` builds, checks, and deploys `dist/`.

Relative internal URLs work for both a `username.github.io` repository and a project site under a repository subpath. The workflow supplies the deployment URL from `configure-pages` to generate canonical tags and a sitemap. Locally, optionally set `SITE_URL=https://your-domain.example` when building.

## Editorial decisions and outstanding content

- CAPES is the confirmed master's funding source (user correction, 6 October 2026). Both supplied email addresses are public contacts.
- GitHub and Google Scholar profile links are intentionally unset, at the user's request. Setting them in the profile adds them to the homepage automatically.
- “The shape of a photograph” has a dedicated project page with a compiled report, source workspace ZIP, figures, metrics, and offline verification results. The public edition states the internal CIFAKE split and representation limits explicitly.
- Research reports are omitted until supplied. Add local PDFs under `assets/reports/` and set each `report` field.
- The ongoing master's research has only a brief public description.
- The paper is explicitly labeled **Preprint**, not peer-reviewed publication.
- The ranking article develops the corrected local explanation through six fictional football matches between five teams. Signed edge values are net goals, distinct from reliability weights in the least-squares objective. Its genuine four-team hole and full decomposition are verified with exact rational arithmetic against the fixture margins.
- The TDA article closely adapts the supplied excerpt, with citations and corrections to coefficient conventions, independent homology classes, the illustrative H₁ diagram, and persistence-image stability (1-Wasserstein, or bottleneck with a cardinality bound). Its figures preserve the supplied point clouds. `scripts/generate_tda_figures.py` reproduces the SVGs and Gaussian surface/image; regeneration needs NumPy and Matplotlib, but the website build uses the committed outputs. The digit Betti numbers are checked over F₂.
- The original PDFs are source material, not instructions. Home address/phone details and unrelated material were not copied.

## Sources

- User-supplied Lattes CV, generated 6 October 2026; newer personal CV; portrait; LinkedIn presentation supplied in chat.
- [A quantum connection approach to rationality](https://arxiv.org/abs/2610.06669), verified 6 October 2026.
- Corrected `rankings_and_hodge_theory.tex` from the user's prior chat “Correct the ranking and Hodge theory.”
- [Jiang, Lim, Yao, Ye: Statistical ranking and combinatorial Hodge theory](https://arxiv.org/abs/0811.1067).
- User-supplied “Fundamentals on Topological Data Analysis (TDA)” excerpt, adapted for the second blog post. The article contains its full references, including [Carlsson: Topology and data](https://doi.org/10.1090/S0273-0979-09-01249-X) and [Adams et al.: Persistence Images](https://jmlr.org/papers/v18/16-337.html).

Source Serif 4 is distributed under the SIL Open Font License (`assets/fonts/OFL.txt`). KaTeX 0.16.22 is distributed under the MIT license (`assets/vendor/katex/LICENSE`).
