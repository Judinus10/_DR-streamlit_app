import numpy as np
import streamlit as st

import utils_classification as cls_utils


def render_eye_controls(eye: str, cls_model=None):
    st.markdown(f"### {eye.title()} Eye Controls")

    st.selectbox(
        f"{eye.title()} explainability method",
        options=["GradCAM++", "ScoreCAM"],
        key=f"{eye}_analysis_cam_method",
        help="GradCAM++ is the baseline. ScoreCAM is slower but useful for comparison.",
    )

    st.selectbox(
        f"{eye.title()} target class",
        options=list(range(cls_utils.NUM_CLASSES)),
        format_func=lambda i: cls_utils.CLASS_NAMES[i],
        key=f"{eye}_analysis_target_class",
    )

    layer_options = ["(auto)"]
    if cls_model is not None:
        try:
            convs = cls_utils.list_conv2d_layers(cls_model)
            layer_options.extend([name for name, _ in convs])
        except Exception:
            pass

    current_layer = st.session_state.get(f"{eye}_analysis_target_layer", "(auto)")
    if current_layer not in layer_options:
        st.session_state[f"{eye}_analysis_target_layer"] = "(auto)"

    st.selectbox(
        f"{eye.title()} target layer",
        options=layer_options,
        key=f"{eye}_analysis_target_layer",
        help="Auto uses the default selected layer. You can manually test other Conv2D layers here.",
    )

    st.slider(
        f"{eye.title()} overlay strength",
        min_value=0.10,
        max_value=0.80,
        step=0.05,
        key=f"{eye}_analysis_alpha",
    )


def _exp_image(image, pair: bool = False):
    cls = "exp-img-wrap pair" if pair else "exp-img-wrap"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_original_single(c: dict, primary_eye: str):
    st.markdown("### Original Image")
    col1, col2 = st.columns([1.85, 0.95], gap="large")
    with col1:
        st.image(c["raw_rgb"], use_container_width=True)
    with col2:
        conf = float(c["probs"][c["pred_idx"]])
        st.markdown("### View Summary")
        st.markdown(
            f"""
            <div class="pred-card">
                <b>Prediction:</b> {c["pred_name"]}<br>
                <b>Confidence:</b> {conf*100:.1f}%<br><br>
                This is the original uploaded retinal image.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_original_pair(computed: dict):
    st.markdown("### Original Images")
    col1, col2 = st.columns(2, gap="large")

    if "right" in computed:
        with col1:
            st.markdown("#### Right Eye")
            st.image(computed["right"]["raw_rgb"], use_container_width=True)

    if "left" in computed:
        with col2:
            st.markdown("#### Left Eye")
            st.image(computed["left"]["raw_rgb"], use_container_width=True)


def render_explainability_single(c: dict, primary_eye: str):
    st.markdown("### Explainability Comparison")
    st.markdown(f"#### {primary_eye.title()} Eye")

    img1, img2, img3 = st.columns([1, 1, 1], gap="medium")
    with img1:
        _exp_image(c["display_rgb"], pair=False)
    with img2:
        _exp_image(c["heatmap_rgb"], pair=False)
    with img3:
        _exp_image(c["cam_overlay_rgb"], pair=False)


def render_explainability_pair_vertical(computed: dict, cls_model=None):
    st.markdown("### Explainability Comparison")

    if "right" in computed:
        c = computed["right"]
        row_left, row_right = st.columns([4.6, 1.0], gap="large")

        with row_left:
            st.markdown("#### Right Eye")
            img1, img2, img3 = st.columns([1.35, 1.35, 1.35], gap="medium")
            with img1:
                _exp_image(c["display_rgb"], pair=True)
            with img2:
                _exp_image(c["heatmap_rgb"], pair=True)
            with img3:
                _exp_image(c["cam_overlay_rgb"], pair=True)

        with row_right:
            render_eye_controls("right", cls_model)

        st.markdown("---")

    if "left" in computed:
        c = computed["left"]
        row_left, row_right = st.columns([4.6, 1.0], gap="large")

        with row_left:
            st.markdown("#### Left Eye")
            img1, img2, img3 = st.columns([1.35, 1.35, 1.35], gap="medium")
            with img1:
                _exp_image(c["display_rgb"], pair=True)
            with img2:
                _exp_image(c["heatmap_rgb"], pair=True)
            with img3:
                _exp_image(c["cam_overlay_rgb"], pair=True)

        with row_right:
            render_eye_controls("left", cls_model)


def render_ex_single(c: dict, primary_eye: str):
    st.markdown("### Exudates (EX)")
    st.caption(f"Powered by the segmentation model for the {primary_eye.title()} eye.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### EX Mask")
        st.image((c["seg"]["ex_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
    with col2:
        st.markdown("#### EX Overlay")
        st.image(c["seg"]["ex_overlay"], use_container_width=True)


def render_ex_pair(computed: dict):
    st.markdown("### Exudates (EX)")
    col1, col2 = st.columns(2, gap="large")

    if "right" in computed:
        c = computed["right"]
        with col1:
            st.markdown("#### Right Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["ex_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["ex_overlay"], use_container_width=True)

    if "left" in computed:
        c = computed["left"]
        with col2:
            st.markdown("#### Left Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["ex_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["ex_overlay"], use_container_width=True)


def render_he_single(c: dict, primary_eye: str):
    st.markdown("### Haemorrhages (HE)")
    st.caption(f"Powered by the segmentation model for the {primary_eye.title()} eye.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### HE Mask")
        st.image((c["seg"]["he_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
    with col2:
        st.markdown("#### HE Overlay")
        st.image(c["seg"]["he_overlay"], use_container_width=True)


def render_he_pair(computed: dict):
    st.markdown("### Haemorrhages (HE)")
    col1, col2 = st.columns(2, gap="large")

    if "right" in computed:
        c = computed["right"]
        with col1:
            st.markdown("#### Right Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["he_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["he_overlay"], use_container_width=True)

    if "left" in computed:
        c = computed["left"]
        with col2:
            st.markdown("#### Left Eye")
            a, b = st.columns(2, gap="medium")
            with a:
                st.image((c["seg"]["he_mask"] * 255).astype(np.uint8), use_container_width=True, clamp=True)
            with b:
                st.image(c["seg"]["he_overlay"], use_container_width=True)