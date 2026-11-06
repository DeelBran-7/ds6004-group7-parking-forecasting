"""Export the reviewed annual Markdown report to a printable PDF.

Run build_report_figures.py first. PDF dependencies: requirements-report.txt.
Pass --font for an embeddable Chinese TrueType font on another operating system.
This script formats existing results; it does not train or evaluate models.
"""

import argparse
import re
import shutil
from html import escape
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "annual"
REPO = "https://github.com/DeelBran-7/ds6004-group7-parking-forecasting"


def url(target):
    if target.startswith(("https://", "http://")):
        return target
    resolved = (REPORT / target).resolve()
    relative = resolved.relative_to(ROOT)
    kind = "tree" if resolved.is_dir() else "blob"
    return f"{REPO}/{kind}/main/{quote(relative.as_posix())}"


def inline(text):
    """Escape content before adding only the report's supported markup."""
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<link href="{url(m[2])}" color="#0077A5">{m[1]}</link>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r'<font color="#0077A5">\1</font>', text)
    return text


def render_pdf(font_path=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from PIL import Image as PILImage

    fonts = [font_path, "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
             "/usr/share/fonts/truetype/arphic/uming.ttc", "C:/Windows/Fonts/msyh.ttc"]
    font = next((Path(p) for p in fonts if p and Path(p).is_file()), None)
    if font is None:
        raise ValueError("Pass --font /path/to/an/embeddable/chinese.ttf")
    pdfmetrics.registerFont(TTFont("ReportBody", str(font)))
    blue, ink, muted, pale, grid = [colors.HexColor(c) for c in ("#0077A5", "#183042", "#526B80", "#EFF6FA", "#DCE6ED")]
    page_width, page_height = A4
    width = page_width - 84
    base = dict(fontName="ReportBody", fontSize=9.5, leading=14.7, textColor=ink, wordWrap="CJK")
    styles = {
        "body": ParagraphStyle("body", **base, spaceAfter=8),
        "small": ParagraphStyle("small", **dict(base, fontSize=8.0, leading=11.5, textColor=muted), spaceAfter=6),
        "lead": ParagraphStyle("lead", **dict(base, fontSize=10.7, leading=16.5)),
        "h1": ParagraphStyle("title", **dict(base, fontSize=22, leading=30), spaceAfter=12),
        "h2": ParagraphStyle("section", **dict(base, fontSize=17, leading=24, textColor=blue), spaceAfter=12),
        "h3": ParagraphStyle("subhead", **dict(base, fontSize=11.5, leading=17, textColor=blue), spaceBefore=5, spaceAfter=6),
        "cell": ParagraphStyle("cell", **dict(base, fontSize=8.2, leading=12.4)),
        "headcell": ParagraphStyle("headcell", **dict(base, fontSize=8.2, leading=12.4, textColor=colors.white)),
    }
    def p(text, kind="body"):
        return Paragraph(inline(text), styles[kind])

    def table(data):
        count = len(data[0])
        ratios = {2: [.25, .75], 3: [.26, .34, .40], 4: [.34, .20, .20, .26], 5: [.18, .25, .19, .19, .19]}[count]
        cells = [[p(cell, "headcell" if i == 0 else "cell") for cell in row] for i, row in enumerate(data)]
        result = Table(cells, colWidths=[width*r for r in ratios], repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), blue), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, pale]),
            ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
            ("LINEBELOW", (0,-1), (-1,-1), .5, grid),
        ]))
        return result

    story, sections = [], 0
    lines = (REPORT / "README.md").read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("## "):
            if sections:
                story.append(PageBreak())
            sections += 1
            story.append(p(line[3:], "h2"))
        elif line.startswith("# "):
            story.append(p(line[2:], "h1"))
        elif line.startswith("### "):
            story.append(p(line[4:], "h3"))
        elif line.startswith("> "):
            box = Table([[p(line[2:], "lead")]], colWidths=[width], hAlign="LEFT")
            box.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), pale), ("BOX", (0,0), (-1,-1), .5, grid),
                ("LEFTPADDING", (0,0), (-1,-1), 10), ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]))
            story += [box, Spacer(1, 10)]
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*)\]\((.*)\)", line)
            if match is None:
                raise ValueError(f"Unsupported image syntax: {line}")
            caption, filename = match.groups()
            image_path = REPORT / filename
            with PILImage.open(image_path) as image:
                w, h = image.size
            limits = {"forecast_mae.png": 210, "data_quality.png": 275, "forecast_by_carpark.png": 310,
                      "full_risk_calibration.png": 260, "recommendation_tradeoff.png": 280}
            scale = min(width/w, limits[image_path.name]/h)
            story += [Image(str(image_path), width=w*scale, height=h*scale), Spacer(1, 4), p(caption, "small")]
        elif line.startswith("|"):
            data = []
            while True:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    data.append(cells)
                if i >= len(lines) or not lines[i].strip().startswith("|"):
                    break
                line = lines[i].strip()
                i += 1
            story += [table(data), Spacer(1, 9)]
        else:
            kind = "small" if line.startswith("[") or line.startswith("分析版本") or not sections else "body"
            story.append(p(line, kind))

    def frame(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(blue)
        canvas.rect(0, page_height-7, page_width, 7, fill=1, stroke=0)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(42, page_height-28, "DS6004  /  GROUP 7  /  SINGAPORE PARKING STUDY")
        canvas.setStrokeColor(grid)
        canvas.line(42, 39, page_width-42, 39)
        canvas.setFont("ReportBody", 7.4)
        canvas.drawString(42, 26, "年度测评 | 2026-11-06 | 数值来源：仓库年度评估汇总")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(page_width-42, 26, f"{doc.page:02d}")
        canvas.restoreState()

    pdf_path = REPORT / "annual_evaluation.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=48, bottomMargin=50, title="DS6004 年度停车预测与推荐测评报告", author="DS6004 Group 7")
    doc.build(story, onFirstPage=frame, onLaterPages=frame)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    for filename in ("forecast_mae.png", "full_risk_calibration.png", "recommendation_tradeoff.png"):
        shutil.copy2(ROOT / "examples/annual/evaluation" / filename, REPORT / "figures" / filename)
    for filename in ("data_quality.png", "forecast_by_carpark.png"):
        if not (REPORT / "figures" / filename).is_file():
            raise ValueError("Run scripts/build_report_figures.py first")
    print(render_pdf(args.font))


if __name__ == "__main__":
    main()
