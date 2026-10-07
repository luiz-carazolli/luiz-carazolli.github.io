#!/usr/bin/env python3
"""Regenerate the curated CV with ReportLab. Website CI uses the committed PDF."""
from pathlib import Path
import json, os
from html import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'content/profile.json').read_text())
R=json.loads((ROOT/'content/research.json').read_text())
PUB=json.loads((ROOT/'content/publications.json').read_text())[0]
fontroot=Path(os.environ.get('CV_FONT_DIR','/System/Library/Fonts/Supplemental'))
fonts={'Body':'Georgia.ttf','BodyBold':'Georgia Bold.ttf','Label':'Arial.ttf'}
if not (fontroot/fonts['Body']).exists():
    fontroot=Path('/usr/share/fonts/truetype/dejavu')
    fonts={'Body':'DejaVuSerif.ttf','BodyBold':'DejaVuSerif-Bold.ttf','Label':'DejaVuSans.ttf'}
for name,file in fonts.items():pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
ink=HexColor('#172638'); accent=HexColor('#99482f'); muted=HexColor('#56616c')
styles={
 'title':ParagraphStyle('title',fontName='Body',fontSize=27,leading=33,textColor=ink,spaceAfter=6),
 'subtitle':ParagraphStyle('subtitle',fontName='Label',fontSize=10,leading=15,textColor=muted,spaceAfter=9),
 'contact':ParagraphStyle('contact',fontName='Label',fontSize=8.5,leading=13,textColor=muted,spaceAfter=12),
 'section':ParagraphStyle('section',fontName='Body',fontSize=14,leading=18,textColor=accent,spaceBefore=10,spaceAfter=6),
 'heading':ParagraphStyle('heading',fontName='BodyBold',fontSize=10.2,leading=14,textColor=ink,spaceAfter=3),
 'meta':ParagraphStyle('meta',fontName='Label',fontSize=8.5,leading=12,textColor=muted,spaceAfter=3),
 'body':ParagraphStyle('body',fontName='Body',fontSize=9.4,leading=13.3,textColor=ink,spaceAfter=4)
}
story=[]
def p(t,kind='body'):return Paragraph(t,styles[kind])
def section(title):story.append(p(title,'section'))
def entry(title,meta,text=''):
    parts=[p(title,'heading'),p(meta,'meta')]
    if text:parts.append(p(text))
    parts.append(Spacer(1,3));story.append(KeepTogether(parts))
def clean(t):return escape(t).replace('–','-').replace('—','-').replace('’',"'").replace('ℝ','R')
story.extend([p(P['name'],'title'),p('M.Sc. candidate in Mathematics | University of Campinas','subtitle'),p(f'<a href="mailto:{P["email"]}">{P["email"]}</a> · <a href="mailto:{P["secondary_email"]}">{P["secondary_email"]}</a><br/><a href="{P["linkedin"]}">LinkedIn</a> · <a href="{P["lattes"]}">Lattes: 5597894328652593</a>','contact'),HRFlowable(width='100%',thickness=.7,color=accent)])
section('Profile')
story.append(p(clean(P['about'])))
section('Education')
entry('M.Sc. in Mathematics','2025-present | University of Campinas (Unicamp)','Research: Orbifold Quantum Cohomology.<br/>Advisors: Prof. Dr. Lino Grama and Dr. Leonardo Cavenaghi. Funding: CAPES.')
entry('B.Sc. in Mathematics','2021-2024 | University of Campinas (Unicamp)')
section('Research experience')
for r in R[1:]:
    desc=clean(r['summary'])
    if r['advisors']:desc+='<br/>Supervision: '+clean(r['advisors'])+'.'
    if r['funding']:desc+=' Funding: '+clean(r['funding'])+'.'
    entry(clean(r['title']),clean(r['years'])+' | '+clean(r['institution']),desc)
story.append(PageBreak())
story.extend([p('Luiz Carazolli','title'),p('Curriculum Vitae | October 2026','subtitle'),HRFlowable(width='100%',thickness=.7,color=accent)])
section('Publications')
entry(clean(PUB['title']), '2026 | Preprint | arXiv:2610.06669',clean(PUB['authors'])+f'<br/><a href="{PUB["arxiv"]}" color="#99482f">{PUB["arxiv"]}</a>')
section('Teaching')
entry('Teaching Assistant - General Topology','2026 | IMECC, University of Campinas','Graduate-level course.')
entry('Teaching Assistant - Introduction to Analysis','2025 | IMECC, University of Campinas','Undergraduate-level course.')
section('Technical skills')
story.append(p('Python · R · Mathematica · LaTeX<br/>Academic and project-oriented research, mathematical writing, and teaching.'))
section('Selected talks')
entry('Introduction to Orbifolds','2024 | Seminar')
entry('The Gauss-Bonnet Theorem via the Moving-Frame Method','2023 | 34th Brazilian Mathematics Colloquium')
entry('Differential Forms and the Geometry of Surfaces','2023 | Unicamp Undergraduate Research Congress')
section('Selected training')
story.append(p('Foundations of Business Strategy - University of Virginia, 2025.<br/>Differentiable Manifolds - UFMG, 2024.<br/>Introduction to Banach Spaces - IMPA, 2023.<br/>General Topology - UFMG, 2023.'))
section('Languages')
story.append(p('Portuguese (native) · English (C2 proficiency, 2017)<br/>Spanish and Italian (intermediate).'))
section('Selected honors')
story.append(p('Gold medals: National Science Olympiad, Brazilian Astronomy Olympiad, and Unicamp Mathematics Olympiad (2020). Silver medal: Brazilian Physics Olympiad (2020).'))
def footer(c,doc):
    c.saveState();c.setStrokeColor(HexColor('#d5d3cd'));c.line(43,39,552,39)
    c.setFont('Label',8);c.setFillColor(muted);c.drawString(43,26,'Luiz Carazolli | Updated October 2026');c.drawRightString(552,26,str(doc.page));c.restoreState()
output=ROOT/'assets/luiz-carazolli-cv.pdf'
doc=SimpleDocTemplate(str(output),pagesize=(595.28,841.89),rightMargin=43,leftMargin=43,topMargin=35,bottomMargin=52,title='Luiz Carazolli - Curriculum Vitae',author=P['full_name'])
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(output)
