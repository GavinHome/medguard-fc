"""paper_zh.md → paper_zh.pdf（学术排版：宋体正文/粗体标题/booktabs 式表格）。

用法: uv run python paper/build_pdf.py
"""
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame,
                                KeepTogether, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "paper" / "paper_zh.md"
OUT = ROOT / "paper" / "paper_zh.pdf"

pdfmetrics.registerFont(TTFont("Songti", "/System/Library/Fonts/Supplemental/Songti.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SongtiB", "/System/Library/Fonts/Supplemental/Songti.ttc", subfontIndex=1))
registerFontFamily("Songti", normal="Songti", bold="SongtiB", italic="Songti", boldItalic="SongtiB")

PAGE_W, PAGE_H = A4
MARGIN = 62
AVAIL = PAGE_W - 2 * MARGIN

S = {
    "title":  ParagraphStyle("title", fontName="SongtiB", fontSize=17, leading=25, alignment=TA_CENTER, spaceAfter=6),
    "author": ParagraphStyle("author", fontName="Songti", fontSize=10.5, leading=15, alignment=TA_CENTER, spaceAfter=4),
    "h1":     ParagraphStyle("h1", fontName="SongtiB", fontSize=13.5, leading=19, spaceBefore=16, spaceAfter=7, keepWithNext=1),
    "h2":     ParagraphStyle("h2", fontName="SongtiB", fontSize=11.5, leading=16, spaceBefore=11, spaceAfter=5, keepWithNext=1),
    "body":   ParagraphStyle("body", fontName="Songti", fontSize=10.5, leading=17.5, alignment=TA_JUSTIFY,
                             wordWrap="CJK", firstLineIndent=21, spaceAfter=4),
    "body0":  ParagraphStyle("body0", parent=None, fontName="Songti", fontSize=10.5, leading=17.5,
                             alignment=TA_JUSTIFY, wordWrap="CJK", spaceAfter=4),
    "li":     ParagraphStyle("li", fontName="Songti", fontSize=10.5, leading=17, alignment=TA_JUSTIFY,
                             wordWrap="CJK", leftIndent=16, spaceAfter=3),
    "cell":   ParagraphStyle("cell", fontName="Songti", fontSize=9, leading=12.5, wordWrap="CJK"),
    "cellh":  ParagraphStyle("cellh", fontName="SongtiB", fontSize=9, leading=12.5, alignment=TA_CENTER, wordWrap="CJK"),
}


def esc(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace("\u2013", "-")                      # 宋体无 en-dash 字形
    t = t.replace("\u2079", "")                       # ⁹ 由 ×10⁹ 组合处理
    t = re.sub(r"×10\^?9", "×10<super>9</super>", t) if "^" in t else t
    t = t.replace("×10/L", "×10<super>9</super>/L")   # 上标 9 缺字形, 改标签
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
    return t


def make_table(rows):
    ncol = len(rows[0])
    # 列宽按内容最大长度加权, 表格总宽 = 可用宽
    lens = [max(len(r[c]) for r in rows) for c in range(ncol)]
    tot = sum(lens)
    widths = [max(AVAIL * l / tot, 55) for l in lens]
    scale = AVAIL / sum(widths)
    widths = [w * scale for w in widths]
    data = []
    for i, r in enumerate(rows):
        st = S["cellh"] if i == 0 else S["cell"]
        data.append([Paragraph(esc(c), st) for c in r])
    t = Table(data, colWidths=widths, hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Songti", 9)
    canv.drawCentredString(PAGE_W / 2, 34, str(canv.getPageNumber()))
    canv.restoreState()


def main():
    lines = MD.read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip() or ln.strip() == "---":
            i += 1
            continue
        if ln.startswith("## "):
            story.append(CondPageBreak(100))
            story.append(Paragraph(esc(ln[3:]), S["h1"]))
        elif ln.startswith("### "):
            story.append(CondPageBreak(80))
            story.append(Paragraph(esc(ln[4:]), S["h2"]))
        elif ln.startswith("# "):
            story.append(Paragraph(esc(ln[2:]), S["title"]))
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            tbl = make_table(rows)
            group = [tbl, Spacer(1, 8)]
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("h1", "h2"):
                group.insert(0, story.pop())          # 标题与表格绑定, 防孤行标题
            if len(rows) <= 9:
                story.append(KeepTogether(group))
            else:
                story.extend(group)
            continue
        elif re.match(r"^\d+\. ", ln) or ln.startswith("- "):
            story.append(Paragraph(esc(ln), S["li"]))
        elif ln.startswith("**作者**"):
            story.append(Paragraph(esc(ln), S["author"]))
        elif ln.startswith("*") and ln.endswith("*") and not ln.startswith("**"):
            story.append(Paragraph("<i>" + esc(ln.strip("*")) + "</i>", S["body0"]))
        else:
            st = S["body"] if not ln.startswith("**") else S["body0"]
            story.append(Paragraph(esc(ln), st))
        i += 1

    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=52,
                          title="面向中文医疗智能体工具调用的分级故障恢复：评测基准与错误驱动的数据合成",
                          author="（待定）", subject="MedGuard-FC：中文医疗工具调用故障恢复评测与安全微调",
                          creator="medguard-fc")
    doc.addPageTemplates([PageTemplate(id="main", frames=[Frame(MARGIN, 52, AVAIL, PAGE_H - MARGIN - 52)], onPage=footer)])
    doc.build(story)
    print("PDF 生成:", OUT)


if __name__ == "__main__":
    main()
