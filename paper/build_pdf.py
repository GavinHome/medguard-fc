"""paper_zh.md / paper_en.md → 学术排版 PDF。

中文: 宋体正文/粗体标题; 英文: Times 正文, 非 Latin-1 字符(τ→①≈α≥⁹)回退宋体。
用法:
  /path/to/anaconda3/python3 paper/build_pdf.py                    # 中文 paper_zh.pdf
  ... paper/build_pdf.py --lang en                                 # 英文 paper_en.pdf
"""
import argparse
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
PAGE_W, PAGE_H = A4
MARGIN = 62
AVAIL = PAGE_W - 2 * MARGIN

pdfmetrics.registerFont(TTFont("Songti", "/System/Library/Fonts/Supplemental/Songti.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("SongtiB", "/System/Library/Fonts/Supplemental/Songti.ttc", subfontIndex=1))
registerFontFamily("Songti", normal="Songti", bold="SongtiB", italic="Songti", boldItalic="SongtiB")
registerFontFamily("Times-Roman", normal="Times-Roman", bold="Times-Bold",
                   italic="Times-Italic", boldItalic="Times-Bold")

# WinAnsi(Latin-1 扩展)能覆盖的非 ASCII 字符 — 之外的字符回退到宋体
WINANSI_EXTRA = set("–—''\u2018\u2019\u201c\u201d…•‚„†‡ˆ‰Š‹ŒŽ™š›œžŸƒ±×÷§©®°µ¼½¾")


def make_esc(latin: bool):
    """构造 esc(): Markdown 行内格式 + 字符兜底。latin=True 时非 WinAnsi 字符包宋体标签。"""

    def wrap_foreign(t: str) -> str:
        if not latin:
            return t
        out = []
        for c in t:
            if ord(c) > 0x7F and c not in WINANSI_EXTRA:
                out.append(f'<font name="Songti">{c}</font>')
            else:
                out.append(c)
        return "".join(out)

    def esc(t: str) -> str:
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        t = t.replace("\u2079", "")
        t = t.replace("×10/L", "×10<super>9</super>/L")   # 宋体缺 ⁹ 字形, 上下文修正
        t = wrap_foreign(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", t)
        return t

    return esc


def make_styles(lang: str):
    if lang == "zh":
        body_f, head_f, cell_f = "Songti", "SongtiB", "Songti"
        body_kw = dict(fontSize=10.5, leading=17.5, wordWrap="CJK", firstLineIndent=21)
        cell_kw = dict(fontSize=9, leading=12.5, wordWrap="CJK")
        title_kw = dict(fontSize=17, leading=25)
    else:
        body_f, head_f, cell_f = "Times-Roman", "Times-Bold", "Times-Roman"
        body_kw = dict(fontSize=10.5, leading=15.2)
        cell_kw = dict(fontSize=9, leading=12)
        title_kw = dict(fontSize=16.5, leading=21)
    return {
        "title": ParagraphStyle("title", fontName=head_f, alignment=TA_CENTER, spaceAfter=8, **title_kw),
        "author": ParagraphStyle("author", fontName=body_f, fontSize=10.5, leading=15, alignment=TA_CENTER, spaceAfter=4),
        "h1": ParagraphStyle("h1", fontName=head_f, fontSize=13.5 if lang == "zh" else 13, leading=19,
                             spaceBefore=16, spaceAfter=7, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName=head_f, fontSize=11.5 if lang == "zh" else 11, leading=16,
                             spaceBefore=11, spaceAfter=5, keepWithNext=1),
        "body": ParagraphStyle("body", fontName=body_f, alignment=TA_JUSTIFY, spaceAfter=4, **body_kw),
        "body0": ParagraphStyle("body0", fontName=body_f, alignment=TA_JUSTIFY, spaceAfter=4,
                                **{k: v for k, v in body_kw.items() if k != "firstLineIndent"}),
        "li": ParagraphStyle("li", fontName=body_f, alignment=TA_JUSTIFY, leftIndent=16, spaceAfter=3,
                             **{k: v for k, v in body_kw.items() if k != "firstLineIndent"}),
        "cell": ParagraphStyle("cell", fontName=cell_f, alignment=TA_JUSTIFY, **cell_kw),
        "cellh": ParagraphStyle("cellh", fontName=head_f, alignment=TA_CENTER, **cell_kw),
    }


def make_table(rows, S, esc):
    ncol = len(rows[0])
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
    canv.setFont("Times-Roman" if doc._lang == "en" else "Songti", 9)
    canv.drawCentredString(PAGE_W / 2, 34, str(canv.getPageNumber()))
    canv.restoreState()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    md = ROOT / "paper" / ("paper_en.md" if args.lang == "en" else "paper_zh.md")
    out = ROOT / "paper" / ("paper_en.pdf" if args.lang == "en" else "paper_zh.pdf")
    if args.input:
        md = Path(args.input)
    if args.output:
        out = Path(args.output)

    esc = make_esc(args.lang == "en")
    S = make_styles(args.lang)
    lines = md.read_text().splitlines()
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
            tbl = make_table(rows, S, esc)
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
        elif ln.startswith("**作者**") or ln.startswith("**Authors**"):
            story.append(Paragraph(esc(ln), S["author"]))
        elif ln.startswith("*") and ln.endswith("*") and not ln.startswith("**"):
            story.append(Paragraph("<i>" + esc(ln.strip("*")) + "</i>", S["body0"]))
        else:
            st = S["body"] if not ln.startswith("**") else S["body0"]
            story.append(Paragraph(esc(ln), st))
        i += 1

    zh_meta = dict(title="面向中文医疗智能体工具调用的分级故障恢复：评测基准与错误驱动的数据合成",
                   author="Simon Yang",
                   subject="MedGuard-FC：中文医疗工具调用故障恢复评测与安全微调")
    en_meta = dict(title="MedGuard-FC: Benchmarking and Teaching Graded Failure Recovery for Tool-Calling Medical Agents",
                   author="Simon Yang",
                   subject="MedGuard-FC: benchmark and error-driven data synthesis for graded tool-failure recovery in medical agents")
    meta = en_meta if args.lang == "en" else zh_meta
    doc = BaseDocTemplate(str(out), pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=52, **meta)
    doc._lang = args.lang
    doc.addPageTemplates([PageTemplate(id="main", frames=[Frame(MARGIN, 52, AVAIL, PAGE_H - MARGIN - 52)], onPage=footer)])
    doc.build(story)
    print("PDF 生成:", out)


if __name__ == "__main__":
    main()
