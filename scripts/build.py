#!/usr/bin/env python3
"""Build an accessible, dependency-free static portfolio into dist/."""
from pathlib import Path
from html import escape as e
from urllib.parse import urljoin
import json, os, shutil, subprocess, re, hashlib
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist'
STYLE_VERSION = hashlib.sha256((ROOT/'assets/style.css').read_bytes()).hexdigest()[:12]
def read(name):
    return json.loads((ROOT / 'content' / (name + '.json')).read_text())
profile, projects, research, publications = [read(n) for n in ('profile','projects','research','publications')]
SITE_URL = os.environ.get('SITE_URL', profile.get('site_url') or '').rstrip('/')
PAPER = publications[0]
POSTS = sorted((p for p in read('posts') if p.get('published', True)), key=lambda p: (date.fromisoformat(p['date']), p.get('order', 0)), reverse=True)
LATEST_POST = POSTS[0]
NAV = [('index.html','Home'),('research.html','Research'),('projects.html','Projects & Code'),('cv.html','CV'),('blog.html','Blog')]

def social():
    links = [(profile.get(k),label) for k,label in [('github','GitHub'),('scholar','Google Scholar'),('lattes','Lattes'),('linkedin','LinkedIn')]]
    return '<div class="social-links">' + ''.join(f'<a href="{e(url)}">{label}<span aria-hidden="true" class="external">↗</span></a>' for url,label in links if url) + f'<a href="mailto:{e(profile["email"])}">Email<span aria-hidden="true" class="external">↗</span></a></div>'

def page(path, title, description, body, active='index.html', math=False):
    prefix = '../' * (len(Path(path).parts)-1)
    nav = ''.join(f'<a href="{prefix}{url}"' + (' aria-current="page"' if active == url else '') + f'>{label}</a>' for url,label in NAV)
    canonical = f'<link rel="canonical" href="{e(SITE_URL+"/"+path)}"><meta property="og:url" content="{e(SITE_URL+"/"+path)}">' if SITE_URL else ''
    image = f'<meta property="og:image" content="{e(SITE_URL)}/assets/luiz-carazolli.jpeg"><meta property="og:image:alt" content="Portrait of Luiz Carazolli">' if SITE_URL else ''
    schema = {'@context':'https://schema.org','@type':'Person','name':profile['full_name'],'alternateName':profile['name'],'jobTitle':profile['role'],'affiliation':{'@type':'CollegeOrUniversity','name':'University of Campinas'},'sameAs':[profile[k] for k in ['github','scholar','linkedin','lattes'] if profile.get(k)]}
    if SITE_URL: schema['url'] = SITE_URL
    html = f'''<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)} | Luiz Carazolli</title><meta name="description" content="{e(description)}"><meta name="theme-color" content="#111c29">
<meta property="og:title" content="{e(title)} | Luiz Carazolli"><meta property="og:description" content="{e(description)}"><meta property="og:type" content="{'article' if math else 'website'}"><meta property="og:locale" content="en_US"><meta name="twitter:card" content="summary">{canonical}{image}
<link rel="icon" type="image/svg+xml" href="{prefix}assets/favicon.svg"><link rel="stylesheet" href="{prefix}assets/style.css?v={STYLE_VERSION}">
{f'<link rel="stylesheet" href="{prefix}assets/vendor/katex/katex.min.css">' if math else ''}
<script src="{prefix}assets/theme.js"></script><script src="{prefix}assets/site.js" defer></script>
<script type="application/ld+json">{json.dumps(schema,ensure_ascii=False).replace('<',chr(92)+'u003c')}</script></head>
<body><a class="skip" href="#main">Skip to content</a><header class="site-header wrap"><div class="header-inner"><a class="wordmark" href="{prefix}index.html" aria-label="Luiz Carazolli, home">Luiz Carazolli<em>.</em></a><nav aria-label="Main navigation">{nav}</nav><button class="theme-toggle" type="button" data-theme-toggle hidden><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16Z" fill="currentColor" stroke="none"/></svg><span>Light mode</span></button></div></header>
<main id="main" class="wrap">{body}</main><footer class="site-footer wrap"><p><span class="footer-name">Luiz Carazolli</span> · Mathematics &amp; research</p><p>Updated {e(profile['updated'])} · <a href="mailto:{e(profile['email'])}">Get in touch</a></p></footer></body></html>'''
    dest=OUT/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(html)

def intro(kicker,title,lead):
    return f'<header class="page-intro"><p class="eyebrow">{kicker}</p><h1>{title}</h1><p class="lead">{lead}</p></header>'

def paper_links(paper):
    citation = '<a href="assets/publication.bib" download>Download citation</a>' if paper['id'] == PAPER['id'] else ''
    return f'<div class="link-row"><a href="{paper["arxiv"]}">Read on arXiv <span aria-hidden="true">↗</span></a><a href="{paper["pdf"]}">Paper PDF <span aria-hidden="true">↗</span></a>{citation}</div>'

def research_rows():
    result=''
    for r in research:
        detail = f'<p class="meta">Supervision: {e(r["advisors"])}</p>' if r['advisors'] else ''
        funding = profile['masters_funding'] if r['status']=='Ongoing' else r['funding']
        if funding: detail += f'<p class="meta">Funding: {e(funding)}</p>'
        if r.get('report'): detail+=f'<div class="link-row"><a href="{e(r["report"])}">Research report</a></div>'
        result+=f'<article class="research-row"><div><p class="year">{e(r["years"])}</p><span class="status {"ongoing" if r["status"]=="Ongoing" else ""}">{r["status"]}</span></div><div><p class="meta">{e(r["kind"])} · {e(r["institution"])}</p><h3>{e(r["title"])}</h3><p>{e(r["summary"])}</p>{detail}</div></article>'
    return result

def publication(paper):
    return f'<article class="publication" id="{paper["id"]}"><p class="eyebrow">{paper["year"]} · {paper["status"]}</p><h3>{e(paper["title"])}</h3><p class="authors">{e(paper["authors"])}</p><p class="description">{e(paper["summary"])}</p><p class="meta">{paper["identifier"]} · Submitted {paper["date"]}</p>{paper_links(paper)}</article>'

def cv_entry(title,meta,text=''):
    return f'<div class="cv-entry"><h3>{title}</h3><p class="meta">{meta}</p>{f"<p>{text}</p>" if text else ""}</div>'

def cv_section(title,content):
    return f'<section class="cv-section"><h2>{title}</h2><div>{content}</div></section>'

def build():
    OUT.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'assets',OUT/'assets',dirs_exist_ok=True)
    (OUT/'.nojekyll').write_text('')
    home=f'''<section class="hero"><div><p class="eyebrow">Geometry · Topology · Data · AI</p><h1>Luiz Carazolli</h1><p class="headline">{e(profile['headline'])}</p><p class="role">{e(profile['role'])}<br>{e(profile['institution'])}</p>{social()}</div><figure class="portrait"><img src="assets/luiz-carazolli.jpeg" alt="Luiz Carazolli" width="265" height="310" fetchpriority="high"><figcaption>{e(profile['location'])}</figcaption></figure></section>
<section class="home-section" aria-labelledby="about-title"><h2 class="section-label" id="about-title"><span class="section-no">01</span>About me</h2><div><p class="about-copy">{e(profile['about'])}</p><p class="interests"><span>Geometry &amp; topology</span><span>Topological data analysis</span><span>AI</span></p></div></section>
<section class="home-section" aria-labelledby="latest-title"><h2 class="section-label" id="latest-title"><span class="section-no">02</span>Latest research</h2><article class="featured-paper"><div class="paper-label"><span>New preprint</span><time datetime="2026-10-05">05 October 2026</time></div><h3><a href="{PAPER['arxiv']}">{e(PAPER['title'])}</a></h3><p>{e(PAPER['summary'])}</p><p>{e(PAPER['authors'])}</p><div class="link-row"><a href="{PAPER['arxiv']}">Read the paper <span aria-hidden="true">↗</span></a><a href="research.html#publications">All publications <span aria-hidden="true">→</span></a></div></article></section>
<section class="home-section" aria-labelledby="explore-title"><h2 class="section-label" id="explore-title"><span class="section-no">03</span>Explore my work</h2><div class="work-grid"><article><p class="eyebrow">Current research</p><h3>Orbifold Quantum Cohomology</h3><p>Master’s research at Unicamp, under Prof. Dr. Lino Grama and Dr. Leonardo Cavenaghi.</p><div class="link-row"><a href="research.html">Research projects <span aria-hidden="true">→</span></a></div></article><article><p class="eyebrow">From the blog</p><h3>{e(LATEST_POST["title"])}</h3><p>{e(LATEST_POST["summary"])}</p><div class="link-row"><a href="blog/{LATEST_POST["slug"]}.html">Read the article <span aria-hidden="true">→</span></a></div></article></div></section>
<section class="contact-strip"><div><p>Let’s connect.</p><p class="meta">Geometry, topology, and the overlap of mathematics with AI &amp; data science.</p></div><a class="button" href="mailto:{e(profile['email'])}">Get in touch <span aria-hidden="true">↗</span></a></section>'''
    page('index.html','Mathematics & Research',"Luiz Carazolli — mathematics master’s candidate at Unicamp. Research in geometry and topology, with interests in data science and AI.",home)
    cards=''
    for i,p in enumerate(projects,1):
        tags='<ul class="tags">'+''.join(f'<li>{e(t)}</li>' for t in p['tags'])+'</ul>' if p['tags'] else ''
        links='<div class="link-row">'+''.join(f'<a href="{e(l["url"])}">{e(l["label"])} ↗</a>' for l in p['links'])+'</div>' if p['links'] else ''
        cards+=f'<article class="project-card" id="{e(p["slug"])}"><p class="eyebrow">Project {i:02d}</p><h2>{e(p["title"])}</h2><p class="muted">{e(p["summary"])}</p>{tags}{links}{"<p class=notice>Details forthcoming.</p>" if p.get("pending_details") else ""}</article>'
    page('projects.html','Projects & Code','Selected projects and code by Luiz Carazolli.',intro('Selected work','Projects &amp; Code','A home for projects, computational work, and the ideas behind them.')+f'<div class="page-body">{cards}</div>','projects.html')
    for project in projects:
        if project.get('page'):
            source = ROOT/'content/projects'/f'{project["slug"]}.html'
            page(project['page'], project['title'], project['summary'], source.read_text(), 'projects.html', True)
    body=intro('Academic work','Research','Geometry and topology, with experience in both pure mathematics and applied AI research.')
    body+='<div class="page-body"><section aria-labelledby="publications"><div class="section-heading"><h2 id="publications">Publications &amp; Preprints</h2></div>'+''.join(publication(p) for p in publications)+'</section><section aria-labelledby="research-projects"><div class="section-heading"><h2 id="research-projects">Research projects</h2><span class="meta">2022–present</span></div>'+research_rows()+'</section></div>'
    page('research.html','Research','Research projects and publications in geometry, topology, and artificial intelligence.',body,'research.html')
    body=intro('Academic & professional background','Curriculum Vitae',e(profile['full_name']))
    body+=f'<div class="page-body"><div class="cv-top"><p class="meta">{e(profile["role"])} · Unicamp<br><a href="mailto:{profile["email"]}">{profile["email"]}</a><br><a href="mailto:{profile["secondary_email"]}">{profile["secondary_email"]}</a></p><a class="button primary" href="assets/luiz-carazolli-cv.pdf" download="Luiz-Carazolli-CV.pdf">Download CV <span>PDF ↓</span></a></div>'
    body+=cv_section('Profile',f'<p>{e(profile["about"])}</p>')
    body+=cv_section('Education',cv_entry('M.Sc. in Mathematics','2025–present · University of Campinas (Unicamp)','Research: Orbifold Quantum Cohomology.<br>Advisors: Prof. Dr. Lino Grama and Dr. Leonardo Cavenaghi.<br>Funding: CAPES.')+cv_entry('B.Sc. in Mathematics','2021–2024 · University of Campinas (Unicamp)'))
    body+=cv_section('Research experience',''.join(cv_entry(e(r['title']),e(r['years']+' · '+r['institution']),e(r['summary'])+(f'<br>Supervision: {e(r["advisors"])}.' if r['advisors'] else '')+(f'<br>Funding: {e(r["funding"])}.' if r['funding'] else '')) for r in research[1:]))
    body+=cv_section('Publications',cv_entry(e(PAPER['title']),f'{PAPER["year"]} · Preprint',e(PAPER['authors'])+f'<br><a href="{PAPER["arxiv"]}">{PAPER["identifier"]}</a>'))
    body+=cv_section('Teaching',cv_entry('Teaching Assistant · General Topology','2026 · IMECC, University of Campinas','Graduate-level course.')+cv_entry('Teaching Assistant · Introduction to Analysis','2025 · IMECC, University of Campinas','Undergraduate-level course.'))
    body+=cv_section('Technical skills','<p>Python · R · Mathematica · LaTeX</p><p>Academic and project-oriented research, mathematical writing, and teaching.</p>')
    body+=cv_section('Selected talks','<ul><li><strong>Introduction to Orbifolds</strong> · Seminar, 2024.</li><li><strong>The Gauss–Bonnet Theorem via the Moving-Frame Method</strong> · 34th Brazilian Mathematics Colloquium, 2023.</li><li><strong>Differential Forms and the Geometry of Surfaces</strong> · Unicamp Undergraduate Research Congress, 2023.</li></ul>')
    body+=cv_section('Selected training','<ul><li>Foundations of Business Strategy · University of Virginia, 2025.</li><li>Differentiable Manifolds · UFMG, 2024.</li><li>Introduction to Banach Spaces · IMPA, 2023.</li><li>General Topology · UFMG, 2023.</li></ul>')
    body+=cv_section('Languages','<p>Portuguese (native) · English (C2 proficiency, 2017) · Spanish and Italian (intermediate, as reported in CV).</p>')
    body+=cv_section('Selected honors','<p>Gold medals: National Science Olympiad, Brazilian Astronomy Olympiad, and Unicamp Mathematics Olympiad (2020). Silver medal: Brazilian Physics Olympiad (2020).</p>')
    body+=f'<p class="meta" style="margin-top:25px">Full academic record: <a href="{profile["lattes"]}">Lattes CV ↗</a> · <a href="{profile["linkedin"]}">LinkedIn ↗</a></p></div>'
    page('cv.html','Curriculum Vitae','Education, research, publications, teaching, and technical skills. Download Luiz Carazolli’s CV as a PDF.',body,'cv.html')
    entries=''
    for post in POSTS:
        url='blog/'+post['slug']+'.html'
        tags=''.join(f'<li>{e(tag)}</li>' for tag in post['tags'])
        display_date=date.fromisoformat(post['date']).strftime('%d %B %Y')
        entries+=f'<article class="blog-item"><div class="meta"><time datetime="{post["date"]}">{display_date}</time><br>{e(post["category"])}</div><div><h2><a href="{url}">{e(post["title"])}</a></h2><p>{e(post["summary"])}</p><ul class="tags">{tags}</ul><div class="link-row"><a href="{url}">Read the article <span aria-hidden="true">→</span></a></div></div></article>'
        article=(ROOT/'content/posts'/f'{post["slug"]}.html').read_text()
        article=re.sub(r'<!-- figure:([a-z0-9-]+[.]svg) -->',lambda match:(ROOT/'content/figures'/match[1]).read_text(),article)
        header=f'<header class="article-header"><p class="article-back"><a href="../blog.html">← All writing</a></p><p class="eyebrow">{e(post["eyebrow"])}</p><h1>{e(post["title"])}</h1><p class="lead">{e(post["subtitle"])}</p><p class="meta">Luiz Carazolli · <time datetime="{post["date"]}">{display_date}</time> · {e(post["format"])}</p></header>'
        page(url,post['title'],post['summary'],header+article,'blog.html',True)
    body=intro('Notes & explanations','Blog','Mathematical ideas, worked through carefully.')+f'<div class="page-body">{entries}</div>'
    page('blog.html','Blog','Explanations connecting mathematics, geometry, topology, and data science.',body,'blog.html')
    page('404.html','Page not found','This page could not be found.',intro('404','A missing page.','The page you’re looking for may have moved.')+'<div class="page-body"><a href="index.html">Return home →</a></div>')
    if SITE_URL:
        paths=[p.relative_to(OUT).as_posix() for p in OUT.rglob('*.html') if p.name!='404.html']
        (OUT/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+''.join('<url><loc>'+e(SITE_URL+'/'+p)+'</loc></url>' for p in sorted(paths))+'</urlset>')
        (OUT/'robots.txt').write_text('User-agent: *\nAllow: /\nSitemap: '+SITE_URL+'/sitemap.xml\n')
    else:
        for filename in ['sitemap.xml','robots.txt']:
            (OUT/filename).unlink(missing_ok=True)
    subprocess.run([os.environ.get('NODE','node'),str(ROOT/'scripts/render-math.cjs')],check=True)
    print(f'Built {len(list(OUT.rglob("*.html")))} pages in {OUT}')

if __name__=='__main__': build()
