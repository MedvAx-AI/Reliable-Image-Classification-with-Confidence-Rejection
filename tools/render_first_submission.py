"""Render the first-submission proposal. Requires reportlab and pypdf.

Run from the repository root: python tools/render_first_submission.py
Uses embedded Calibri when available; otherwise uses PDF's built-in Helvetica.
"""

from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports' / 'first_submission.md'
OUTPUT = ROOT / 'reports' / 'first_submission.pdf'
NAVY = colors.HexColor('#183249')
TEAL = colors.HexColor('#167A82')
INK = colors.HexColor('#263342')
MUTED = colors.HexColor('#586778')
PALE = colors.HexColor('#EEF4F7')
RULE = colors.HexColor('#D6E0E6')
FONTS = Path('C:/Windows/Fonts')

if (FONTS / 'calibri.ttf').exists():
    for name, file in [('Report', 'calibri.ttf'), ('Report-Bold', 'calibrib.ttf'),
                       ('Report-Italic', 'calibrii.ttf'), ('Report-BoldItalic', 'calibriz.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
    pdfmetrics.registerFontFamily('Report', normal='Report', bold='Report-Bold',
                                  italic='Report-Italic', boldItalic='Report-BoldItalic')
    FONT, BOLD = 'Report', 'Report-Bold'
else:
    FONT, BOLD = 'Helvetica', 'Helvetica-Bold'

STYLES = {
    'title': ParagraphStyle('title', fontName=BOLD, fontSize=19.5, leading=22.3,
                            textColor=NAVY, spaceAfter=8),
    'meta': ParagraphStyle('meta', fontName=FONT, fontSize=9.2, leading=11.7,
                           textColor=MUTED, spaceAfter=7),
    'body': ParagraphStyle('body', fontName=FONT, fontSize=10.05, leading=12.7,
                           textColor=INK, spaceAfter=5.5),
    'section': ParagraphStyle('section', fontName=BOLD, fontSize=12.4, leading=16,
                              textColor=NAVY, spaceBefore=8.5, spaceAfter=4,
                              keepWithNext=True),
    'cell': ParagraphStyle('cell', fontName=FONT, fontSize=9.1, leading=11.1,
                           textColor=INK, alignment=TA_LEFT),
    'thead': ParagraphStyle('thead', fontName=BOLD, fontSize=9.1, leading=11.1,
                            textColor=colors.white),
    'reference': ParagraphStyle('reference', fontName=FONT, fontSize=8.4, leading=10.2,
                                textColor=MUTED, spaceAfter=2.5),
    'step': ParagraphStyle('step', fontName=FONT, fontSize=10.05, leading=12.7,
                           textColor=INK, leftIndent=13, firstLineIndent=-13,
                           spaceAfter=3.3),
}


def inline(text):
    text = escape(text.strip())
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',
                  lambda m: f'<link href="{m.group(2)}" color="#167A82">{m.group(1)}</link>', text)
    text = re.sub(r'`([^`]+)`', lambda m: f'<font name="Courier" size="8.5">{m.group(1)}</font>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text


def page_decor(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(36, height - 25, width - 72, 3, fill=1, stroke=0)
    canvas.setFont(BOLD, 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(36, height - 41, 'INTRODUCTION TO COMPUTER VISION  /  FIRST SUBMISSION')
    canvas.setStrokeColor(RULE)
    canvas.line(36, 32, width - 36, 32)
    canvas.setFont(FONT, 8)
    canvas.drawString(36, 20, 'Reliable Image Classification with Confidence Rejection')
    canvas.drawRightString(width - 36, 20, f'{doc.page} / 2')
    canvas.restoreState()


def render():
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    story = []
    i = 0
    in_refs = False
    split_done = False
    available = A4[0] - 72
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), STYLES['title']))
            i += 1
            continue
        if line.startswith('## '):
            in_refs = line == '## References'
            story.append(Paragraph(inline(line[3:]), STYLES['section']))
            i += 1
            continue
        if line.startswith('|'):
            raw = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    raw.append(cells)
                i += 1
            if raw[0][0] == 'Subset':
                widths = [115, 54, 60, available - 229]
            elif raw[0][0] == 'Condition':
                widths = [103, 174, available - 277]
            else:
                widths = [85, 204, available - 289]
            rows = [[Paragraph(inline(cell), STYLES['thead' if r == 0 else 'cell'])
                     for cell in row] for r, row in enumerate(raw)]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('LINEBELOW', (0, -1), (-1, -1), 0.5, RULE),
            ]))
            story.append(table)
            if raw[0][0] != 'Condition':
                story.append(Spacer(1, 6))
            continue
        if not split_done and line.startswith('For logits z, confidence'):
            story.extend([PageBreak(), Paragraph('3. Experimental plan (continued)', STYLES['section'])])
            split_done = True
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|')):
            if re.match(r'^\d+\. ', lines[i]):
                break
            para.append(lines[i])
            i += 1
        if para[0].startswith('**First submission:'):
            html = '<br/>'.join(inline(p.rstrip('\\')) for p in para)
            style = STYLES['meta']
        else:
            html = inline(' '.join(p.strip() for p in para))
            style = STYLES['reference' if in_refs else 'step' if re.match(r'^\d+\. ', para[0]) else 'body']
        story.append(Paragraph(html, style))

    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=36, rightMargin=36,
                          topMargin=51, bottomMargin=40,
                          title='Reliable Image Classification with Confidence Rejection - First Submission',
                          author='Telman3000; LeoVesinML; Mysteri0K1ng; MedvAx-AI',
                          subject='Implementation and evaluation proposal, 2 October 2026')
    frame = Frame(36, 40, available, A4[1] - 91, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='report', frames=frame, onPage=page_decor))
    doc.build(story)
    pages = PdfReader(str(OUTPUT)).pages
    print(f'Rendered {OUTPUT.name}: {len(pages)} pages, {OUTPUT.stat().st_size:,} bytes')
    if len(pages) != 2:
        raise RuntimeError('Expected exactly two proposal pages; inspect and adjust layout.')


if __name__ == '__main__':
    render()
