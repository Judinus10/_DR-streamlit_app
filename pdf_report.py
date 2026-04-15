import io
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _safe_text(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _np_to_rl_image(np_img: np.ndarray, width_mm: float = 82) -> RLImage:
    if np_img.ndim == 2:
        pil_img = PILImage.fromarray(np_img.astype(np.uint8), mode="L")
    else:
        pil_img = PILImage.fromarray(np_img.astype(np.uint8))

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    width_px, height_px = pil_img.size
    target_width = width_mm * mm
    target_height = (height_px / max(width_px, 1)) * target_width
    return RLImage(buf, width=target_width, height=target_height)


def _build_clean_kv_table(rows: List[List[str]], col_widths):
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10.5),
                ("LEADING", (0, 0), (-1, -1), 14),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#222222")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _build_summary_table(rows: List[List[str]], col_widths):
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#dbe3ec")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (-1, -1), "Helvetica"),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEADING", (0, 0), (-1, -1), 13),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _build_prob_table(rows: List[List[str]], col_widths):
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef6ff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12344d")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d6dee8")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _build_image_card(label: str, rl_img: RLImage, small_style) -> Table:
    card = Table(
        [[Paragraph(f"<b>{label}</b>", small_style)], [rl_img]],
        colWidths=[84 * mm],
        hAlign="LEFT",
    )
    card.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#dbe3ec")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return card


def build_pdf_report(
    *,
    computed: Dict[str, Dict[str, Any]],
    available_eyes: List[str],
    analysis_input_mode: str,
    class_names: List[str],
    selected_view_mode: str,
    options: Dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=_safe_text(options.get("report_title", "DR Grade Report")),
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#111111"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )

    meta_right_style = ParagraphStyle(
        "MetaRight",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#6b7280"),
        alignment=TA_RIGHT,
        spaceAfter=6,
    )

    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#222222"),
        spaceBefore=8,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.2,
        leading=14,
        textColor=colors.black,
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#5b6470"),
    )

    eye_title_style = ParagraphStyle(
        "EyeTitleStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#12344d"),
        spaceBefore=8,
        spaceAfter=6,
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#b91c1c"),
        alignment=TA_LEFT,
        spaceBefore=2,
    )

    story = []

    report_title = _safe_text(options.get("report_title", "DR Grade Report"))
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # centered heading
    story.append(Paragraph(report_title, title_style))
    # generated time at right
    story.append(Paragraph(f"Generated on: {generated_time}", meta_right_style))
    story.append(Spacer(1, 4))

    # patient details
    story.append(Paragraph("PATIENT DETAILS", section_style))

    left_rows = [
        ["Name:", _safe_text(options.get("patient_name"))],
        ["Patient ID:", _safe_text(options.get("patient_id"))],
    ]
    right_rows = [
        ["Clinician / Examiner:", _safe_text(options.get("clinician_name"))],
        ["Institution:", _safe_text(options.get("institution_name"))],
    ]

    left_table = _build_clean_kv_table(left_rows, [32 * mm, 58 * mm])
    right_table = _build_clean_kv_table(right_rows, [50 * mm, 40 * mm])

    patient_block = Table(
        [[left_table, right_table]],
        colWidths=[90 * mm, 90 * mm],
        hAlign="LEFT",
    )
    patient_block.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(patient_block)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#8b8b8b")))
    story.append(Spacer(1, 8))

    for eye in available_eyes:
        c = computed[eye]
        conf = float(c["probs"][c["pred_idx"]])

        story.append(Paragraph(f"{eye.title()} Eye", eye_title_style))

        if options.get("include_prediction_summary", True):
            summary_rows = [
                ["Predicted Grade", _safe_text(c["pred_name"])],
                ["Confidence", f"{conf * 100:.1f}%"],
            ]
            story.append(_build_summary_table(summary_rows, [50 * mm, 130 * mm]))
            story.append(Spacer(1, 8))

        if options.get("include_probabilities", True):
            prob_rows = [["Class", "Probability"]]
            order = np.argsort(-c["probs"])
            for i in order:
                prob_rows.append([class_names[i], f"{float(c['probs'][i]) * 100:.1f}%"])

            story.append(Paragraph("Class Probabilities", body_style))
            story.append(Spacer(1, 4))
            story.append(_build_prob_table(prob_rows, [100 * mm, 80 * mm]))
            story.append(Spacer(1, 8))

        image_cards = []

        if options.get("include_original_image", False):
            image_cards.append(_build_image_card("Original Image", _np_to_rl_image(c["raw_rgb"]), small_style))

        if options.get("include_gradcam_heatmap", False):
            image_cards.append(_build_image_card("Grad-CAM Heatmap", _np_to_rl_image(c["heatmap_rgb"]), small_style))

        if options.get("include_gradcam_overlay", False):
            image_cards.append(_build_image_card("Grad-CAM Overlay", _np_to_rl_image(c["cam_overlay_rgb"]), small_style))

        if options.get("include_exudates_mask", False):
            ex_mask_img = (c["seg"]["ex_mask"] * 255).astype(np.uint8)
            image_cards.append(_build_image_card("Exudates Mask", _np_to_rl_image(ex_mask_img), small_style))

        if options.get("include_exudates_overlay", False):
            image_cards.append(_build_image_card("Exudates Overlay", _np_to_rl_image(c["seg"]["ex_overlay"]), small_style))

        if options.get("include_haemorrhages_mask", False):
            he_mask_img = (c["seg"]["he_mask"] * 255).astype(np.uint8)
            image_cards.append(_build_image_card("Haemorrhages Mask", _np_to_rl_image(he_mask_img), small_style))

        if options.get("include_haemorrhages_overlay", False):
            image_cards.append(_build_image_card("Haemorrhages Overlay", _np_to_rl_image(c["seg"]["he_overlay"]), small_style))

        if image_cards:
            story.append(Paragraph("Selected Images", body_style))
            story.append(Spacer(1, 4))

            row = []
            for card in image_cards:
                row.append(card)
                if len(row) == 2:
                    image_row = Table([row], colWidths=[87 * mm, 87 * mm], hAlign="LEFT")
                    image_row.setStyle(
                        TableStyle(
                            [
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                ("TOPPADDING", (0, 0), (-1, -1), 0),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                            ]
                        )
                    )
                    story.append(image_row)
                    story.append(Spacer(1, 6))
                    row = []

            if row:
                if len(row) == 1:
                    row.append("")
                image_row = Table([row], colWidths=[87 * mm, 87 * mm], hAlign="LEFT")
                image_row.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                        ]
                    )
                )
                story.append(image_row)
                story.append(Spacer(1, 6))

        story.append(Spacer(1, 6))

    # notes moved near end
    notes_text = _safe_text(options.get("notes"))
    if notes_text != "-":
        story.append(Spacer(1, 2))
        story.append(Paragraph("Notes", section_style))
        story.append(Paragraph(notes_text, body_style))
        story.append(Spacer(1, 6))

    # disclaimer clearly visible
    if options.get("include_disclaimer", True):
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1")))
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Disclaimer: This report is generated by a research/demo system for diabetic retinopathy analysis. "
                "It is not a medical diagnosis and should not be used as a substitute for clinical judgment "
                "or professional ophthalmic evaluation.",
                disclaimer_style,
            )
        )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes