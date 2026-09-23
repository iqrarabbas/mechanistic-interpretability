import argparse
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def run(text, bold=False, italic=False, code=False):
    properties = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    if code:
        properties.extend(("<w:rFonts w:ascii=\"Courier New\" w:hAnsi=\"Courier New\"/>", "<w:sz w:val=\"19\"/>"))
    props = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f"<w:r>{props}<w:t{preserve}>{escape(text)}</w:t></w:r>"


def inline_runs(text):
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)", text)
    output = []
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            output.append(run(part[1:-1], code=True))
        elif part.startswith("**") and part.endswith("**"):
            output.append(run(part[2:-2], bold=True))
        elif part.startswith("*") and part.endswith("*"):
            output.append(run(part[1:-1], italic=True))
        else:
            output.append(run(part))
    return "".join(output)


def paragraph(text="", style=None, indent=None, shading=None):
    properties = []
    if style:
        properties.append(f'<w:pStyle w:val="{style}"/>')
    if indent is not None:
        properties.append(f'<w:ind w:left="{indent}" w:hanging="360"/>')
    if shading:
        properties.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>')
    p_props = f"<w:pPr>{''.join(properties)}</w:pPr>" if properties else ""
    return f"<w:p>{p_props}{inline_runs(text)}</w:p>"


def table(rows):
    widths = [max(len(row[index]) if index < len(row) else 0 for row in rows) for index in range(max(map(len, rows)))]
    total = max(sum(widths), 1)
    grid = "".join(f'<w:gridCol w:w="{max(900, int(9000 * width / total))}"/>' for width in widths)
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell_index in range(len(widths)):
            value = row[cell_index] if cell_index < len(row) else ""
            fill = "D9EAF7" if row_index == 0 else ("F4F7FA" if row_index % 2 == 0 else "FFFFFF")
            cell_width = max(900, int(9000 * widths[cell_index] / total))
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{cell_width}" w:type="dxa"/>'
                f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/></w:tcPr>'
                f'{paragraph(value, style="TableHeader" if row_index == 0 else "TableText")}</w:tc>'
            )
        table_rows.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:left w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:right w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:insideH w:val="single" w:sz="3" w:color="D4DCE3"/>'
        '<w:insideV w:val="single" w:sz="3" w:color="D4DCE3"/></w:tblBorders></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>{''.join(table_rows)}</w:tbl>"
    )


def markdown_to_body(markdown):
    lines = markdown.splitlines()
    body = []
    index = 0
    in_code = False
    code_lines = []
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            if in_code:
                body.append(paragraph("\n".join(code_lines), style="CodeBlock", shading="EEF2F5"))
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", lines[index + 1]):
            rows = [[cell.strip() for cell in line.strip("|").split("|")]]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            body.append(table(rows))
            body.append(paragraph())
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            body.append(paragraph(heading.group(2), style=f"Heading{len(heading.group(1))}"))
        elif line.startswith("> "):
            body.append(paragraph(line[2:], style="Quote"))
        elif re.match(r"^[-*]\s+", line):
            body.append(paragraph(re.sub(r"^[-*]\s+", "• ", line), style="ListParagraph"))
        elif re.match(r"^\d+\.\s+", line):
            body.append(paragraph(line, style="ListParagraph"))
        elif line.strip() == "---":
            body.append('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:color="9AA9B7"/></w:pBdr></w:pPr></w:p>')
        elif line.strip():
            body.append(paragraph(line))
        else:
            body.append(paragraph())
        index += 1
    return "".join(body)


def build_docx(source, destination):
    body = markdown_to_body(source.read_text())
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/><w:cols w:space="708"/></w:sectPr></w:body></w:document>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="160"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="300" w:after="120"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="100"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:color w:val="2F5597"/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="3"/></w:pPr><w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="420" w:hanging="240"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="500" w:right="500"/><w:shd w:val="clear" w:fill="EAF2F8"/></w:pPr><w:rPr><w:i/><w:color w:val="244062"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="CodeBlock"><w:name w:val="Code Block"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="300"/><w:spacing w:before="80" w:after="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="19"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="0"/></w:pPr><w:rPr><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableHeader"><w:name w:val="Table Header"/><w:basedOn w:val="TableText"/><w:rPr><w:b/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    document_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", document_relationships)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build_docx(args.source, args.destination)


if __name__ == "__main__":
    main()
