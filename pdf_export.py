# pdf_export.py
import os
import tempfile
from xml.sax.saxutils import escape
from typing import List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

# ---------- 字体注册（不变）----------
def register_chinese_font():
    try:
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        for path in font_paths:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont('ChineseFont', path))
                return 'ChineseFont'
        pdfmetrics.registerFont(TTFont('STSong-Light', 'STSong-Light.ttf'))
        return 'STSong-Light'
    except Exception:
        return 'Helvetica'

CHINESE_FONT = register_chinese_font()

# ---------- 图片处理（不压缩）----------
def get_optimised_image(image_path: str, max_width: int = 450):
    """
    直接使用原图，不压缩，设置显示宽度（高度按比例缩放）。
    max_width: 图片在PDF中显示的最大宽度（像素/点，1点≈1/72英寸）。
    原图保持原始分辨率，PDF文件会包含完整图像数据。
    """
    with Image.open(image_path) as img:
        w, h = img.size
        aspect = h / w
        display_width = max_width
        display_height = display_width * aspect
        # 直接返回 RLImage，使用原图路径，不重新保存
        return RLImage(image_path, width=display_width, height=display_height)

# ---------- 带图片的 PDF 导出 ----------
def export_to_pdf(
    image_paths: List[str],
    result_texts: List[str],
    question: str,
    output_path: Optional[str] = None
) -> str:
    if not image_paths:
        return None

    if output_path is None:
        pdf_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        output_path = pdf_temp.name
        pdf_temp.close()

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                 fontName=CHINESE_FONT, fontSize=16, alignment=TA_CENTER, spaceAfter=20)
    question_style = ParagraphStyle('QuestionStyle', parent=styles['Normal'],
                                    fontName=CHINESE_FONT, fontSize=12, leading=14, spaceAfter=30)
    result_style = ParagraphStyle('ResultStyle', parent=styles['Normal'],
                                  fontName=CHINESE_FONT, fontSize=10, leading=12, spaceAfter=10)
    filename_style = ParagraphStyle('FilenameStyle', parent=styles['Heading2'],
                                    fontName=CHINESE_FONT, fontSize=12, spaceAfter=10)

    story = []
    story.append(Paragraph("多模态图片分析报告", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"用户问题：{question}", question_style))
    story.append(Spacer(1, 20))

    for idx, (img_path, result_text) in enumerate(zip(image_paths, result_texts)):
        filename = os.path.basename(img_path)
        story.append(Paragraph(f"图片 {idx+1}: {filename}", filename_style))
        try:
            # 使用不压缩的图片，显示宽度设为 450 点（约 6.25 英寸）
            img = get_optimised_image(img_path, max_width=450)
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"图片加载失败: {str(e)}", result_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph("分析结果：", result_style))
        safe_text = escape(result_text).replace('\n', '<br/>')
        story.append(Paragraph(safe_text, result_style))
        story.append(Spacer(1, 20))
        if idx < len(image_paths) - 1:
            story.append(PageBreak())

    doc.build(story)
    return output_path

# ---------- 纯文本表格导出（兼容旧接口）----------



# ---------- 纯文本表格导出（兼容旧接口） ----------
def export_pdf(records: List[List], output_path: Optional[str] = None) -> str:
    if not records:
        return None
    if output_path is None:
        pdf_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        output_path = pdf_temp.name
        pdf_temp.close()

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                 fontName=CHINESE_FONT, fontSize=16, alignment=TA_CENTER, spaceAfter=20)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'],
                                  fontName=CHINESE_FONT, fontSize=10, leading=14)

    story = []
    story.append(Paragraph("Qwen3.5-VL 推理结果", title_style))
    story.append(Spacer(1, 20))

    data = [["图片", "结果", "耗时 (s)"]]
    for row in records:
        safe_text = escape(str(row[1])).replace('\n', '<br/>')
        data.append([row[0], Paragraph(safe_text, normal_style), str(row[2])])

    col_widths = [2.5 * 72, 4.5 * 72, 1.2 * 72]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), CHINESE_FONT),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(table)
    doc.build(story)
    return output_path