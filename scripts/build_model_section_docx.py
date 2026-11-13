"""Build the editable English modelling contribution from reviewed report text.

Use the bundled document runtime with python-docx. This formats saved results;
it neither trains models nor changes any evaluated values.
"""
from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/model_section'
SOURCE = REPORT / 'Model_Section_EN.md'
OUTPUT = REPORT / 'DS6004_Group7_Model_Section_Report_EN.docx'


def set_font(style, size, bold=False):
    style.font.name = 'Times New Roman'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.rFonts
    for key in list(fonts.attrib):
        if 'theme' in key.lower(): del fonts.attrib[key]
    for key in ['ascii', 'hAnsi', 'eastAsia', 'cs']:
        fonts.set(qn('w:' + key), 'Times New Roman')
    for node in list(rpr):
        if node.tag in {qn('w:spacing'), qn('w:position')}:
            rpr.remove(node)
    for node in style.element.findall('.//' + qn('w:pBdr')):
        node.getparent().remove(node)


def hyperlink(paragraph, text, target):
    part = paragraph.part
    rid = part.relate_to(target, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True)
    element = OxmlElement('w:hyperlink'); element.set(qn('r:id'), rid)
    run = OxmlElement('w:r'); props = OxmlElement('w:rPr')
    color = OxmlElement('w:color'); color.set(qn('w:val'), '174C73'); props.append(color)
    run.append(props); node = OxmlElement('w:t'); node.text = text; run.append(node)
    element.append(run); paragraph._p.append(element)


def text_runs(paragraph, value):
    pattern = r'https://[^\s]+|\*\*[^*]+\*\*'
    start = 0
    for match in re.finditer(pattern, value):
        paragraph.add_run(value[start:match.start()])
        token = match.group()
        if token.startswith('https://'):
            hyperlink(paragraph, token, token)
        else:
            paragraph.add_run(token[2:-2]).bold = True
        start = match.end()
    paragraph.add_run(value[start:])


def add_table(doc, rows):
    count = len(rows[0])
    widths = {3: [1.12, 3.08, 2.3], 4: [1.05, 3.0, 1.15, 1.3],
              5: [1.65, 1.15, 1.2, 1.2, 1.3],
              6: [.68, .55, 1.05, 1.05, 1.57, 1.6]}[count]
    table = doc.add_table(rows=0, cols=count)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    props = table._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement('w:' + name)
        for key, value in {'val': 'single', 'sz': '4', 'color': 'D9D9D9'}.items():
            border.set(qn('w:' + key), value)
        borders.append(border)
    props.append(borders)
    for index, data in enumerate(rows):
        row = table.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement('w:cantSplit'))
        if index == 0:
            repeat = OxmlElement('w:tblHeader'); repeat.set(qn('w:val'), 'true'); trpr.append(repeat)
        for col, (cell, value, width) in enumerate(zip(row.cells, data, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement('w:tcMar')
            for side, val in [('top', '65'), ('bottom', '65'), ('left', '90'), ('right', '90')]:
                margin = OxmlElement('w:' + side); margin.set(qn('w:w'), val); margin.set(qn('w:type'), 'dxa'); margins.append(margin)
            tcpr.append(margins)
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E7EDF2' if index == 0 else ('F5F7F9' if index % 2 == 0 else 'FFFFFF')); tcpr.append(shade)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col == 0 or count == 3 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(value); run.font.name = 'Times New Roman'; run.font.size = Pt(9.5); run.bold = index == 0
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = Pt(3); p.add_run().font.size = Pt(3)


def build():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(1)
    section.top_margin, section.bottom_margin = Inches(.7), Inches(.7)
    section.header_distance = section.footer_distance = Inches(.3)
    styles = doc.styles
    for name, size, bold in [('Normal', 11, False), ('Title', 21, True), ('Subtitle', 10.5, False),
                              ('Heading 1', 14, True), ('Heading 2', 11.5, True), ('Caption', 9, False)]:
        set_font(styles[name], size, bold)
    styles['Normal'].paragraph_format.line_spacing = 1.08
    styles['Normal'].paragraph_format.space_after = Pt(7)
    styles['Normal'].paragraph_format.widow_control = True
    styles['Title'].paragraph_format.space_after = Pt(9)
    styles['Subtitle'].paragraph_format.space_after = Pt(4)
    for name in ['Heading 1', 'Heading 2']:
        styles[name].paragraph_format.space_before = Pt(8)
        styles[name].paragraph_format.space_after = Pt(7)
        styles[name].paragraph_format.keep_with_next = True
    styles['Caption'].paragraph_format.space_after = Pt(7)
    styles['Caption'].paragraph_format.line_spacing = 1.0

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run('DS6004 Group 7  |  Modelling contribution'); run.font.name = 'Times New Roman'; run.font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run('Page '); run.font.name = 'Times New Roman'; run.font.size = Pt(8)
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)

    lines = SOURCE.read_text().splitlines(); i = 0; in_body = False; new_page = False
    while i < len(lines):
        line = lines[i].strip(); i += 1
        if not line: continue
        if line == '<!-- PAGE -->':
            new_page = True
        elif line.startswith('# '):
            doc.add_paragraph(line[2:], 'Title')
        elif line.startswith('## '):
            p = doc.add_paragraph(line[3:], 'Heading 1')
            p.paragraph_format.page_break_before = new_page
            new_page = False; in_body = True
        elif line.startswith('### '):
            doc.add_paragraph(line[4:], 'Heading 2')
        elif line.startswith('|'):
            rows = []
            while True:
                cells = [x.strip() for x in line.strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', x) for x in cells): rows.append(cells)
                if i >= len(lines) or not lines[i].strip().startswith('|'): break
                line = lines[i].strip(); i += 1
            add_table(doc, rows)
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[(.*)\]\((.*)\)', line)
            p = doc.add_paragraph(); p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_after = Pt(3)
            picture = p.add_run().add_picture(str(REPORT / match[2]), width=Inches(6.5))
            picture._inline.docPr.set('descr', match[1])
            doc.add_paragraph(match[1], 'Caption')
        elif line.startswith('Table '):
            p = doc.add_paragraph(line, 'Caption'); p.paragraph_format.keep_with_next = True
            for run in p.runs: run.bold = True
        else:
            p = doc.add_paragraph(style='Normal' if in_body else 'Subtitle')
            if line.startswith('['):
                p.paragraph_format.space_after = Pt(5)
                p.paragraph_format.line_spacing = 1.0
                p.style = 'Caption'
            text_runs(p, line)
    doc.core_properties.title = 'Parking Occupancy and Arrival Risk Modelling'
    doc.core_properties.subject = 'DS6004 Group 7 final report modelling contribution'
    doc.core_properties.author = 'DS6004 Group 7'
    doc.core_properties.keywords = 'parking; occupancy forecasting; arrival risk; temporal validation'
    doc.core_properties.comments = ''
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    build()
