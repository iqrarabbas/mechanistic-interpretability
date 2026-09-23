from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).parent
SOURCE = ROOT / "sae_mi_robustness_summary_2026-08-20.md"
OUTPUT = ROOT / "SAE_MI_Robustness_Summary_2026-08-20.pdf"


def wrapped_lines(text, width):
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def render_page(pdf, items, page_number):
    figure = plt.figure(figsize=(8.27, 11.69))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    y = 0.955
    for text, style in items:
        size, weight, color, spacing, family = style
        figure.text(
            0.068,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            family=family,
            va="top",
        )
        y -= spacing
    figure.text(
        0.5,
        0.025,
        f"SAE-Guided ViT Robustness Summary  •  {page_number}",
        ha="center",
        fontsize=7.5,
        color="#666666",
    )
    pdf.savefig(figure)
    plt.close(figure)


def main():
    pages = []
    current = []
    used = 0.0

    def add(text, style):
        nonlocal current, used
        spacing = style[3]
        if used + spacing > 0.88 and current:
            pages.append(current)
            current = []
            used = 0.0
        current.append((text, style))
        used += spacing

    for raw in SOURCE.read_text().splitlines():
        line = raw.strip()
        if not line:
            add("", (8.6, "normal", "black", 0.012, "DejaVu Sans"))
        elif line.startswith("# "):
            for part in wrapped_lines(line[2:], 55):
                add(part, (19, "bold", "#17365D", 0.034, "DejaVu Sans"))
        elif line.startswith("## "):
            for part in wrapped_lines(line[3:], 70):
                add(part, (13, "bold", "#1F4E79", 0.025, "DejaVu Sans"))
        elif line.startswith("|"):
            columns = [column.strip() for column in line.strip("|").split("|")]
            if all(set(column) <= {"-", ":"} for column in columns):
                continue
            table_line = "  |  ".join(columns)
            for part in wrapped_lines(table_line, 112):
                add(part, (7.1, "normal", "#222222", 0.0145, "DejaVu Sans Mono"))
        elif line.startswith("- "):
            cleaned = line[2:].replace("**", "").replace("`", "")
            for index, part in enumerate(wrapped_lines(cleaned, 94)):
                add(("• " if index == 0 else "   ") + part, (8.7, "normal", "black", 0.0175, "DejaVu Sans"))
        elif len(line) > 2 and line[0].isdigit() and line[1] == ".":
            cleaned = line.replace("**", "").replace("`", "")
            for part in wrapped_lines(cleaned, 94):
                add(part, (8.7, "normal", "black", 0.0175, "DejaVu Sans"))
        else:
            cleaned = line.replace("**", "").replace("`", "")
            for part in wrapped_lines(cleaned, 94):
                add(part, (8.7, "normal", "black", 0.0175, "DejaVu Sans"))
    if current:
        pages.append(current)

    with PdfPages(OUTPUT) as pdf:
        for page_number, items in enumerate(pages, 1):
            render_page(pdf, items, page_number)
    print(f"Generated {OUTPUT} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
