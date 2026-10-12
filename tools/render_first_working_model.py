"""Render reports/first_working_model.md to PDF. Requires reportlab and pypdf.

Run from any directory: python tools/render_first_working_model.py
"""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Preformatted, KeepTogether,
)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/first_working_model.md"
OUTPUT = SOURCE.with_suffix(".pdf")
NAVY = colors.HexColor("#162B40")
TEAL = colors.HexColor("#087D8B")
GRAY = colors.HexColor("#566575")


def fonts():
    directory = Path("C:/Windows/Fonts")
    names = [("Report", "calibri.ttf"), ("ReportBold", "calibrib.ttf"),
             ("ReportItalic", "calibrii.ttf"), ("ReportBoldItalic", "calibriz.ttf")]
    if all((directory / filename).exists() for _, filename in names):
        for name, filename in names:
            pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
        pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold",
                                      italic="ReportItalic", boldItalic="ReportBoldItalic")
        return "Report", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
                  lambda m: '<a href="' + html.escape(m[2], quote=True)
                  + '" color="#087D8B">' + m[1] + '</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', text)
    return text


def render():
    regular, bold = fonts()
    body = ParagraphStyle("Body", fontName=regular, fontSize=9.8, leading=12.0,
                          textColor=NAVY, spaceAfter=6)
    title = ParagraphStyle("Title", parent=body, fontName=bold, fontSize=19,
                           leading=21, spaceAfter=8)
    heading = ParagraphStyle("Heading", parent=body, fontName=bold, fontSize=11.6,
                             leading=14, spaceBefore=7, spaceAfter=5, keepWithNext=True)
    cell = ParagraphStyle("Cell", parent=body, fontSize=9.4, leading=11.3, spaceAfter=0)
    code = ParagraphStyle("Code", fontName="Courier", fontSize=8.5, leading=11,
                          textColor=NAVY, backColor=colors.HexColor("#F0F5F8"),
                          borderPadding=7, spaceBefore=2, spaceAfter=8)
    width = A4[0] - 76
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- pagebreak -->":
            story.append(PageBreak())
            i += 1
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), title))
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), heading))
            i += 1
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(block), code))
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                parts = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", v) for v in parts):
                    rows.append(parts)
                i += 1
            proportions = [0.23, 0.13, 0.13, 0.51] if len(rows[0]) == 4 else [0.64, 0.36]
            table = Table([[Paragraph(inline(v), cell) for v in row] for row in rows],
                          colWidths=[width * p for p in proportions], repeatRows=1,
                          hAlign=TA_LEFT)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DFEDF1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#F5F8FA"), colors.white]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 7)])
        else:
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```", "<!--")):
                paragraph.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(paragraph)), body))

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(TEAL)
        canvas.rect(0, A4[1] - 8, A4[0], 8, fill=1, stroke=0)
        canvas.setFillColor(GRAY)
        canvas.setFont(bold, 8)
        canvas.drawString(38, A4[1] - 30, "INTRODUCTION TO COMPUTER VISION")
        canvas.setFont(regular, 8)
        canvas.drawRightString(A4[0] - 38, A4[1] - 30, "FIRST WORKING MODEL | 13 OCTOBER 2026")
        canvas.setStrokeColor(colors.HexColor("#D5E0E8"))
        canvas.line(38, 33, A4[0] - 38, 33)
        canvas.setFillColor(GRAY)
        canvas.drawString(38, 21, "Reliable Image Classification with Confidence Rejection")
        canvas.drawRightString(A4[0] - 38, 21, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=38, rightMargin=38,
                            topMargin=47, bottomMargin=43,
                            title="Reliable Image Classification with Confidence Rejection - First working model",
                            author="Aleksandr Medvedev, Telman Nuruzov, Leo Vesin, Nikita Shankin")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    reader = PdfReader(str(OUTPUT))
    print(f"Created {OUTPUT} ({len(reader.pages)} pages)")


if __name__ == "__main__":
    render()
