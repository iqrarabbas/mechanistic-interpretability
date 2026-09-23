from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).parent
SOURCE = ROOT / "todays_sae_blur4_experiments.md"
OUTPUT = ROOT / "SAE_Blur4_Experiments_2026-08-13.pdf"


def wrapped_lines(text, width):
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def render_page(pdf, items, page_number):
    figure = plt.figure(figsize=(8.27, 11.69))
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    y = 0.955
    for text, style in items:
        size, weight, color, spacing = style
        figure.text(
            0.075,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            family="DejaVu Sans Mono" if text.startswith("    ") else "DejaVu Sans",
            va="top",
        )
        y -= spacing
    figure.text(0.5, 0.025, str(page_number), ha="center", fontsize=8, color="#666666")
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
            add("", (9, "normal", "black", 0.014))
        elif line.startswith("# "):
            for part in wrapped_lines(line[2:], 55):
                add(part, (20, "bold", "#17365D", 0.035))
        elif line.startswith("## "):
            for part in wrapped_lines(line[3:], 68):
                add(part, (14, "bold", "#1F4E79", 0.027))
        elif line.startswith("### "):
            for part in wrapped_lines(line[4:], 75):
                add(part, (11, "bold", "#2F5597", 0.022))
        elif line.startswith("|"):
            columns = [column.strip() for column in line.strip("|").split("|")]
            if all(set(column) <= {"-", ":"} for column in columns):
                continue
            table_line = "  |  ".join(columns)
            for part in wrapped_lines(table_line, 104):
                add("    " + part, (7.6, "normal", "#222222", 0.016))
        elif line.startswith("- "):
            for index, part in enumerate(wrapped_lines(line[2:], 88)):
                add(("• " if index == 0 else "   ") + part, (9.2, "normal", "black", 0.019))
        elif line.startswith("`") and line.endswith("`"):
            for part in wrapped_lines(line.strip("`"), 92):
                add("    " + part, (8.5, "normal", "#333333", 0.019))
        else:
            cleaned = line.replace("**", "").replace("`", "")
            for part in wrapped_lines(cleaned, 92):
                add(part, (9.2, "normal", "black", 0.019))
    if current:
        pages.append(current)

    with PdfPages(OUTPUT) as pdf:
        for page_number, items in enumerate(pages, 1):
            render_page(pdf, items, page_number)
    print(f"Generated {OUTPUT} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
