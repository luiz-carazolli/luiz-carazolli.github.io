#!/usr/bin/env python3
"""Check every local link/fragment/asset and the compiled mathematical example."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
from fractions import Fraction as F
import re, json
ROOT = Path(__file__).resolve().parents[1] / 'dist'
class Page(HTMLParser):
    def __init__(self, path):
        super().__init__();self.ids=set();self.refs=[];self.h1=0;self.lang=None;self.path=path
        self.feed(path.read_text())
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids, f'Duplicate id {a["id"]}: {self.path}'
            self.ids.add(a['id'])
        if tag=='h1':self.h1+=1
        if tag=='html':self.lang=a.get('lang')
        if tag=='img':assert a.get('alt'),f'Missing image alt: {self.path}'
        for key in ('href','src'):
            if key in a:self.refs.append(a[key])
pages={p.resolve():Page(p) for p in ROOT.rglob('*.html')}
for path,page in pages.items():
    assert page.h1==1, f'{path}: expected one h1'
    assert page.lang=='en'
    for ref in page.refs:
        u=urlparse(ref)
        if u.scheme or u.netloc:continue
        target=(path.parent/unquote(u.path)).resolve() if u.path else path
        if target.is_dir():target=target/'index.html'
        assert target.is_relative_to(ROOT.resolve()),f'Link escapes site: {ref}'
        assert target.exists(),f'Broken link in {path}: {ref}'
        if u.fragment and target in pages:assert unquote(u.fragment) in pages[target].ids,f'Broken fragment: {ref}'
for css in ROOT.rglob('*.css'):
    for url in re.findall(r'url\([\s\'"]*([^\)\'"\s]+)',css.read_text()):
        if url.startswith(('data:','http:','https:')):continue
        # Bundled KaTeX also lists optional legacy font formats; current browsers use woff2.
        if url.endswith(('.ttf','.woff')) and 'vendor/katex' in str(css):continue
        assert (css.parent/url).exists(),f'Missing CSS asset: {css}: {url}'
posts=json.loads((ROOT.parent/'content/posts.json').read_text())
published=[p for p in posts if p.get('published',True)]
index=(ROOT/'blog.html').read_text()
home=(ROOT/'index.html').read_text()
equations=0
for post in published:
    slug=post['slug']
    blog=(ROOT/f'blog/{slug}.html').read_text()
    source=(ROOT.parent/f'content/posts/{slug}.html').read_text()
    expected=len(re.findall(r'\\\(|\\\[',source))
    assert blog.count('class="katex-mathml"')==expected, f'Equation count mismatch: {slug}'
    assert 'katex-error' not in blog and r'\[' not in blog and r'\(' not in blog
    assert f'href="blog/{slug}.html"' in index, f'Post absent from index: {slug}'
    equations+=expected
newest=max(published,key=lambda p:(p['date'],p.get('order',0)))
assert f'href="blog/{newest["slug"]}.html"' in home, 'Homepage must feature newest published post'
from generate_tda_figures import DIGITS, rips
assert [rips(p)[2] for p in DIGITS]==[(1,0),(1,1),(1,2)]
edges=[(0,1),(1,2),(0,2),(2,3),(3,4),(4,0)]
s=list(map(F,[0,2,3,2,1]))
football=(ROOT.parent/'content/posts/statistical-rankings-hodge-theory.html').read_text()
matches=re.findall(r'data-match="([A-E]{2})" data-goals-i="(\d+)" data-goals-j="(\d+)"',football)
assert [m[0] for m in matches]==['AB','BC','AC','CD','DE','EA']
y=[F(int(goals_j)-int(goals_i)) for _,goals_i,goals_j in matches]
assert y==list(map(F,[3,2,4,1,1,1]))
g=[s[j]-s[i] for i,j in edges]
c=[F(1,3),F(1,3),-F(1,3),F(0),F(0),F(0)]
h=[F(2,3),F(2,3),F(4,3),F(2),F(2),F(2)]
assert y==[a+b+d for a,b,d in zip(g,c,h)]
def dot(a,b):return sum(x*z for x,z in zip(a,b))
assert dot(g,c)==dot(g,h)==dot(c,h)==0
assert h[0]+h[1]-h[2]==0
balance=[F(0)]*5
for (i,j),v in zip(edges,h):balance[i]-=v;balance[j]+=v
assert balance==[0]*5
assert sum(h[2:])==F(22,3)
assert [dot(v,v) for v in [y,g,c,h]]==[F(32),F(17),F(1,3),F(44,3)]
print(f'Passed: {len(pages)} pages, local links, anchors, assets, {equations} compiled formulas, latest-post feature, digit homology, and exact Hodge decomposition.')
