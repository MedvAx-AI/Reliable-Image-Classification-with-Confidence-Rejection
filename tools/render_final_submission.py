"""Render the editable final summary and three presentation slides."""
from pathlib import Path
import json
import re
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
FONTS = Path('C:/Windows/Fonts')
if (FONTS/'calibri.ttf').exists():
    for name, filename in [('Body','calibri.ttf'),('Strong','calibrib.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(FONTS/filename)))
else:
    pdfmetrics.registerFont(pdfmetrics.Font('Body','Helvetica','WinAnsiEncoding'))
    pdfmetrics.registerFont(pdfmetrics.Font('Strong','Helvetica-Bold','WinAnsiEncoding'))
INK = colors.HexColor('#162B40')
TEAL = colors.HexColor('#007D81')
BODY = ParagraphStyle('body',fontName='Body',fontSize=9.3,leading=11.5,textColor=INK,spaceAfter=5)
HEADING = ParagraphStyle('heading',parent=BODY,fontName='Strong',fontSize=11.3,leading=14,spaceBefore=6,spaceAfter=4,textColor=TEAL)
TITLE = ParagraphStyle('title',parent=HEADING,fontSize=19,leading=21,spaceBefore=0,spaceAfter=8,textColor=INK)


def markup(text):
    text = escape(text)
    text = text.replace('&lt;br/&gt;', '<br/>')
    text = re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#007D81">\1</link>',text)
    return text


def footer(c, doc):
    c.setStrokeColor(TEAL)
    c.line(40,35,A4[0]-40,35)
    c.setFont('Body',8)
    c.setFillColor(INK)
    c.drawString(40,23,'Final test | run a9d534f5ff01664d | fixed model and policies')
    c.drawRightString(A4[0]-40,23,f'{doc.page} / 2')


def render_summary():
    lines=(ROOT/'reports/technical_summary.md').read_text(encoding='utf-8').splitlines()
    story=[]
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1
            continue
        if line=='<!-- pagebreak -->':
            story.append(PageBreak())
        elif line.startswith('# '):
            story.append(Paragraph(markup(line[2:]),TITLE))
        elif line.startswith('## '):
            story.append(Paragraph(markup(line[3:]),HEADING))
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[s.strip() for s in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',s) for s in cells):
                    rows.append([Paragraph(markup(s),ParagraphStyle('cell',parent=BODY,fontSize=8.6,leading=10.5)) for s in cells])
                i+=1
            table=Table(rows,colWidths=[122,55,96,96,66,66],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E3F1F1')),
                                      ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,TEAL),
                                      ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7FA')]),
                                      ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                                      ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,6)])
            continue
        elif line.startswith('!['):
            path=re.search(r'\]\(([^)]+)\)',line).group(1)
            story.append(Image(str(ROOT/'reports'/path),width=430,height=430*765/1120,hAlign='LEFT'))
            story.append(Spacer(1,4))
        else:
            story.append(Paragraph(markup(line),BODY))
        i+=1
    path=ROOT/'reports/technical_summary.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=45,
                      title='Reliable Image Classification with Confidence Rejection',author='Project team').build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(path).pages)
    if count!=2:
        raise RuntimeError(f'The summary has {count} pages; adjust layout or text to exactly two.')


def text(c, value, x, y, width, size=20, strong=False, color=INK):
    p=Paragraph(markup(value),ParagraphStyle('slide',fontName='Strong' if strong else 'Body',fontSize=size,leading=size*1.18,textColor=color))
    _,height=p.wrap(width,1000)
    p.drawOn(c,x,y-height)
    return y-height


def slide_base(c, number, title, subtitle):
    c.setFillColor(colors.HexColor('#F4F7FA'))
    c.rect(0,0,960,540,fill=1,stroke=0)
    c.setFillColor(TEAL)
    c.rect(0,530,960,10,fill=1,stroke=0)
    text(c,title,35,500,890,29,True)
    text(c,subtitle,35,458,890,15)
    c.setFont('Body',11)
    c.setFillColor(INK)
    c.drawString(35,16,'Reliable Image Classification with Confidence Rejection | 12 October 2026')
    c.drawRightString(925,16,f'{number} / 3')


def render_slides():
    metrics=json.loads((ROOT/'results/final_metrics.json').read_text())['metrics']
    c=canvas.Canvas(str(ROOT/'reports/slides.pdf'),pagesize=(960,540))
    c.setTitle('Reliable Image Classification - final demonstration')
    c.setAuthor('Project team')
    slide_base(c,1,'When should an image classifier refuse to answer?','Research question: improve accepted accuracy without hiding the coverage cost.')
    text(c,'CIFAR-10 | all 10 classes | seed 42',35,406,880,23,True)
    text(c,'10,000 train → 2,000 model validation → 1,500 calibration → 1,500 policy selection<br/>Final test: 2,000 independent official test images, 200 per class',35,369,875,20)
    text(c,'Frozen ResNet-18 + trained linear head',35,292,880,23,True)
    text(c,'ImageNet weights; resize 256 / center crop 224; 30 head epochs.<br/>Calibration T = 0.95617; raw / calibrated thresholds = 0.56 / 0.57.',35,255,875,20)
    for x,name,description in [(35,'A · Baseline','Raw; always predict'),(270,'B · Rejection','Raw confidence ≥ 0.56'),(505,'C · Calibration','Scaled; always predict'),(740,'D · Combined','Scaled confidence ≥ 0.57')]:
        c.setFillColor(colors.white)
        c.roundRect(x,80,210,100,10,fill=1,stroke=0)
        text(c,name,x+12,165,185,20,True,TEAL)
        text(c,description,x+12,131,185,17)
    c.showPage()
    slide_base(c,2,'Clean rejection helps; corruption breaks reliability','All numbers below are final test. B/D show accepted accuracy and coverage (%).')
    conditions=[('clean','none','Clean'),('gaussian_blur','severe','Blur radius 1.5'),('gaussian_noise','severe','Noise std 0.12'),('jpeg','severe','JPEG quality 15')]
    rows=[['Condition','Full accuracy','B acc. / cov.','D acc. / cov.']]
    for corruption,severity,label in conditions:
        selected={r['condition']:r for r in metrics if (r['corruption'],r['severity'])==(corruption,severity)}
        rows.append([label,f"{selected['A']['full_set_accuracy']*100:.2f}%",*[f"{selected[k]['accepted_accuracy']*100:.2f} / {selected[k]['coverage']*100:.2f}" for k in 'BD']])
    table=Table(rows,colWidths=[250,190,220,220],rowHeights=[38]*5)
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Body'),('FONTSIZE',(0,0),(-1,-1),20),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEDED')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EAF0F4')]),('TEXTCOLOR',(0,0),(-1,-1),INK),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),12)]))
    table.wrapOn(c,900,300)
    table.drawOn(c,35,220)
    text(c,'Clean D: +5.85 points accepted accuracy; 12.10% rejected.',35,198,890,23,True,TEAL)
    text(c,'Clean ECE: 0.0201 → 0.0187; NLL nearly unchanged. Labels stay the same.<br/>Threshold sweeps trade coverage for risk; raw/scaled curves nearly overlap.<br/>The clean 90% target is not a guarantee on corrupted images.',35,157,885,19)
    c.showPage()
    slide_base(c,3,'Live demo: confident errors and useful predictions rejected','Run the saved weights on the selected pixels. No fitting or test-based threshold changes.')
    c.drawImage(str(ROOT/'results/figures/failures.png'),35,58,width=512,height=512*765/1120)
    text(c,'What goes wrong?',585,406,340,23,True,TEAL)
    y=text(c,'Airplane → ship<br/>Ambiguous silhouette/background; confidence ≈ 99.9%.',585,366,340,18)
    y=text(c,'Correct dog → REJECT<br/>56.84% is below frozen 57%.',585,y-24,340,18)
    y=text(c,'Ship → airplane after JPEG<br/>Lost details and blocking; still ≈ 99.0% confident.',585,y-24,340,18)
    text(c,'Visual causes are hypotheses.<br/>Fallback: saved case panel.<br/>Reproducibility: repository README.',585,y-27,340,16)
    c.showPage()
    c.save()
    if len(PdfReader(ROOT/'reports/slides.pdf').pages)!=3:
        raise RuntimeError('Slides must contain exactly three pages.')


if __name__=='__main__':
    render_summary()
    render_slides()
    print('Saved exactly two summary pages and three slides.')
