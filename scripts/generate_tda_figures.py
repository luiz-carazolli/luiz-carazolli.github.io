#!/usr/bin/env python3
"""Reproduce the TDA article figures. Optional regeneration needs NumPy/Matplotlib.

The website build uses the committed SVG/PNG outputs and needs neither package.
"""
from pathlib import Path
from itertools import combinations
from math import hypot, erf, sqrt
from html import escape

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'content/figures'
DIGITS = [
    [(0,.90),(.10,.80),(0,.70),(0,.52),(0,.34),(0,.16),(0,-.02),(0,-.20),(-.08,-.32),(0,-.32),(.08,-.32)],
    [(0,.85),(.18,.82),(.32,.70),(.40,.52),(.40,.30),(.34,.10),(.22,-.04),(0,-.10),(-.22,-.04),(-.34,.10),(-.40,.30),(-.40,.52),(-.32,.70),(-.18,.82)],
    [(0,.92),(.18,.89),(.30,.77),(.34,.61),(.30,.47),(.18,.37),(0,.33),(-.18,.37),(-.30,.47),(-.34,.61),(-.30,.77),(-.18,.89),(.18,.29),(.30,.18),(.34,.02),(.30,-.14),(.18,-.27),(0,-.31),(-.18,-.27),(-.30,-.14),(-.34,.02),(-.30,.18),(-.18,.29)],
]
EPSILON = .24

def rank_f2(columns):
    pivots = {}
    for column in columns:
        while column:
            pivot = column.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = column
                break
            column ^= pivots[pivot]
    return len(pivots)

def rips(points):
    edges = [(i,j) for i,j in combinations(range(len(points)),2)
             if hypot(points[i][0]-points[j][0], points[i][1]-points[j][1]) <= EPSILON]
    lookup = {edge:i for i,edge in enumerate(edges)}
    triangles = [t for t in combinations(range(len(points)),3)
                 if all(e in lookup for e in combinations(t,2))]
    rank1 = rank_f2([(1<<i) | (1<<j) for i,j in edges])
    rank2 = rank_f2([sum(1<<lookup[e] for e in combinations(t,2)) for t in triangles])
    return edges, triangles, (len(points)-rank1, len(edges)-rank1-rank2)

def svg(name, width, height, title, description, body, extra=''):
    FIGURES.mkdir(exist_ok=True)
    (FIGURES/name).write_text(
        f'<svg class="tda-diagram {extra}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{name}-title {name}-desc" xmlns="http://www.w3.org/2000/svg">'
        f'<title id="{name}-title">{escape(title)}</title><desc id="{name}-desc">{escape(description)}</desc>{body}</svg>')

def line(a,b,cls='tda-edge'):
    return f'<line class="{cls}" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}"/>'

def text(x,y,value,cls='tda-label',anchor='middle'):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(value)}</text>'

def polygon(points):
    return '<polygon class="tda-face" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'"/>'

def point(p):
    return f'<circle class="tda-point" cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="3.3"/>'

def generate_vectors():
    for with_rips in (False,True):
        body = ''
        for panel,(digit,points) in enumerate(zip(('1','0','8'),DIGITS)):
            coords = [(110+panel*220+x*150,180-y*150) for x,y in points]
            if with_rips:
                edges,triangles,betti = rips(points)
                assert betti == (1,panel), (digit,betti)
                for x,y in coords:
                    body += f'<circle class="tda-ball" cx="{x:.2f}" cy="{y:.2f}" r="{EPSILON*75}"/>'
                body += ''.join(polygon([coords[i] for i in t]) for t in triangles)
                body += ''.join(line(coords[i],coords[j]) for i,j in edges)
            body += ''.join(point(p) for p in coords)
            body += text(110+220*panel,267,digit)
            if with_rips:
                body += text(110+220*panel,295,f'β₀ = 1 · β₁ = {panel}','tda-caption')
        svg('tda-rips-digits.svg' if with_rips else 'tda-digits.svg',660,310 if with_rips else 285,
            'Rips complexes of the digits 1, 0, and 8' if with_rips else 'Three digit point clouds',
            'At epsilon 0.24 the connected complexes have respectively zero, one, and two independent loops. Shaded disks have radius epsilon/2. All edges and triangles satisfying the Rips rule are drawn.' if with_rips else 'Discrete samples trace the outlines of 1, 0, and 8.',body)
    corners = {'A':(55,38),'B':(215,38),'C':(215,198),'D':(55,198),'E':(280,118)}
    for name,filled,extended in [('tda-square.svg',False,False),('tda-filled-square.svg',True,False),('tda-equivalent-cycles.svg',False,True)]:
        body = ''
        if filled:
            body += polygon([corners[k] for k in 'ABC'])+polygon([corners[k] for k in 'ACD'])
        if extended:
            body += polygon([corners[k] for k in 'BEC'])
        edges = ['AB','BC','CD','DA'] + (['AC'] if filled else []) + (['BE','EC'] if extended else [])
        body += ''.join(line(corners[a],corners[b]) for a,b in edges)
        for k in ('ABCDE' if extended else 'ABCD'):
            x,y = corners[k]
            body += point((x,y))+text(x+(19 if k=='E' else 0),y+(-15 if k in 'AB' else 27 if k in 'CD' else 5),k)
        svg(name,325 if extended else 270,240,'Equivalent cycles' if extended else 'A filled square' if filled else 'An unfilled square',
            'The outer detour through E and the square differ by the boundary of filled triangle BEC.' if extended else 'Two triangles with diagonal AC fill the square loop.' if filled else 'Four vertices and four edges form a loop enclosing an unfilled region.',body,'tda-small')
    p = lambda x,y:(70+82*x,455-82*y)
    body = ''
    for i in range(6):
        body += line(p(i,0),p(i,5),'tda-grid')+line(p(0,i),p(5,i),'tda-grid')
        body += text(p(i,0)[0],477,str(i),'tda-caption')
        body += text(55,p(0,i)[1]+5,str(i),'tda-caption','end')
    body += line(p(0,0),p(5.2,0),'tda-axis')+line(p(0,0),p(0,5.2),'tda-axis')
    body += line(p(0,0),p(5,5),'tda-diagonal')
    body += text(286,508,'Birth','tda-label')+text(35,20,'Death','tda-label','start')
    body += text(433,69,'d = b','tda-caption')
    body += line(p(1,1),p(1,3),'tda-persistence')
    body += text(143,296,'d − b = 2','tda-caption','end')
    for x,y in [(1,3),(2,2.5),(1.5,4)]:
        a,b = p(x,y)
        body += f'<circle class="tda-diagram-point" cx="{a}" cy="{b}" r="5"/>'
        body += text(a-12 if x==2 else a+13,b+23 if x==2 else b-10,
                     f'({x:g}, {y:g})','tda-caption','end' if x==2 else 'start')
    svg('tda-persistence-diagram.svg',530,525,'An illustrative persistence diagram',
        'Three points have birth-death coordinates (1,3), (2,2.5), and (1.5,4). The diagonal is dashed. A vertical segment of length two marks the first point’s persistence.',body)

def generate_surface():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'text.color':'#172638','axes.labelcolor':'#172638','xtick.color':'#56616c','ytick.color':'#56616c','axes.edgecolor':'#b0aaa2'})
    points = [(1,2),(2,.5),(1.5,2.5)]
    sigma = .3
    x = np.linspace(0,3,170); y = np.linspace(0,3.5,180)
    xx,yy = np.meshgrid(x,y)
    rho = sum(p/(2*np.pi*sigma**2)*np.exp(-((xx-b)**2+(yy-p)**2)/(2*sigma**2)) for b,p in points)
    xe=np.linspace(0,3,19); ye=np.linspace(0,3.5,19)
    cdf = lambda z:.5*(1+erf(z/(sigma*sqrt(2))))
    values=np.zeros((18,18))
    for j in range(18):
        for i in range(18):
            values[j,i]=sum(p*(cdf(xe[i+1]-b)-cdf(xe[i]-b))*(cdf(ye[j+1]-p)-cdf(ye[j]-p)) for b,p in points)
    assert values.min() >= 0 and 4.8 < values.sum() < 5
    cmap=LinearSegmentedColormap.from_list('navy-terracotta',['#faf9f6','#cfb5a7','#b9694a','#172638'])
    fig=plt.figure(figsize=(10,4.5),facecolor='#faf9f6')
    ax=fig.add_subplot(121,projection='3d',facecolor='#faf9f6')
    ax.plot_surface(xx,yy,rho,cmap=cmap,linewidth=0,antialiased=True,rstride=2,cstride=2)
    ax.set(xlabel='Birth',ylabel='Persistence',zlabel='Density',xlim=(0,3),ylim=(0,3.5))
    ax.set_title('Persistence surface',pad=15,fontsize=13)
    ax.view_init(elev=28,azim=-60)
    ax.tick_params(labelsize=8)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
    bx=fig.add_subplot(122,facecolor='#faf9f6')
    im=bx.pcolormesh(xe,ye,values,cmap=cmap,shading='flat',edgecolors='#ffffff50',linewidth=.3)
    bx.set(xlabel='Birth',ylabel='Persistence',aspect='equal',xlim=(0,3),ylim=(0,3.5))
    bx.set_title('Persistence image',pad=15,fontsize=13)
    bar=fig.colorbar(im,ax=bx,shrink=.76,pad=.04)
    bar.set_label('Integrated weight',fontsize=9);bar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.02,right=.96,bottom=.15,top=.85,wspace=.18)
    fig.text(.5,.025,'Gaussian width σ = 0.3   ·   Weight w(b, p) = min(p, 3)   ·   18 × 18 cells',ha='center',fontsize=9,color='#56616c')
    dest=ROOT/'assets/tda/persistence-surface.png';dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest,dpi=140,facecolor=fig.get_facecolor());plt.close(fig)

if __name__ == '__main__':
    generate_vectors()
    print('Verified digit Betti numbers over F₂:', [rips(p)[2] for p in DIGITS])
    generate_surface()
