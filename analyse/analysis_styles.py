import streamlit as st


ANALYSIS_CSS = """
<style>
  [data-testid="stSidebar"] {display: none;}
  [data-testid="stSidebarNav"] {display: none;}

  .switch-note {
      color: #aab7c8;
      font-size: 1rem;
      margin-bottom: 0.6rem;
      text-align: center;
      width: 100%;
  }

  .pred-card {
      padding: 0.95rem 1rem;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      background: rgba(255,255,255,0.02);
  }

  .saved-card {
      padding: 0.85rem 1rem;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 14px;
      background: rgba(255,255,255,0.02);
      margin-bottom: 0.65rem;
  }

  .saved-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem 2rem;
  }

  .saved-col {
      line-height: 1.65;
  }

  .saved-label {
      font-weight: 700;
  }

  /* center wrapper */
  div[data-testid="stRadio"] {
      width: 100% !important;
      display: flex !important;
      justify-content: center !important;
      align-items: center !important;
      margin-top: 0.15rem !important;
      margin-bottom: 1.35rem !important;
  }

  div[data-testid="stRadio"] > div {
      width: 100% !important;
      display: flex !important;
      justify-content: center !important;
      align-items: center !important;
  }

  div[role="radiogroup"] {
      display: inline-flex !important;
      justify-content: center !important;
      align-items: stretch !important;
      flex-wrap: nowrap !important;
      gap: 0 !important;
      margin-left: auto !important;
      margin-right: auto !important;
      border: 1px solid rgba(70, 213, 244, 0.70) !important;
      border-radius: 16px !important;
      overflow: hidden !important;
      background: rgba(7, 28, 38, 0.95) !important;
      box-shadow:
          0 0 0 1px rgba(44, 196, 232, 0.08),
          0 8px 22px rgba(0, 0, 0, 0.18) !important;
  }

  div[role="radiogroup"] > label {
      margin: 0 !important;
      border: none !important;
      border-right: 1px solid rgba(70, 213, 244, 0.28) !important;
      border-radius: 0 !important;
      padding: 0.88rem 2.15rem !important;
      min-width: 210px !important;
      justify-content: center !important;
      align-items: center !important;
      background: rgba(7, 28, 38, 0.95) !important;
      transition:
          background 0.18s ease,
          color 0.18s ease,
          box-shadow 0.18s ease !important;
      cursor: pointer !important;
      position: relative !important;
      box-shadow: none !important;
  }

  div[role="radiogroup"] > label:last-child {
      border-right: none !important;
  }

  div[role="radiogroup"] > label:hover {
      background: rgba(38, 164, 203, 0.14) !important;
  }

  /* hide radio elements */
  div[role="radiogroup"] input,
  div[role="radiogroup"] input[type="radio"],
  div[role="radiogroup"] svg,
  div[role="radiogroup"] [data-testid="stMarkdownContainer"] + div,
  div[role="radiogroup"] label > div:first-child {
      display: none !important;
      visibility: hidden !important;
      width: 0 !important;
      height: 0 !important;
      min-width: 0 !important;
      min-height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;
  }

  div[role="radiogroup"] > label p {
      color: #dff7ff !important;
      font-size: 1.04rem !important;
      font-weight: 650 !important;
      margin: 0 !important;
      text-align: center !important;
      letter-spacing: 0.01em !important;
      transition: color 0.18s ease !important;
  }

  div[role="radiogroup"] > label::after {
      content: none !important;
  }

  /* ACTIVE TAB:
     Use aria-checked on the inner div that Streamlit actually toggles */
  div[role="radiogroup"] > label:has(div[aria-checked="true"]),
  div[role="radiogroup"] > label:has(input[aria-checked="true"]),
  div[role="radiogroup"] > label:has(input:checked),
  div[role="radiogroup"] > label[data-selected="true"],
  div[role="radiogroup"] > label[aria-checked="true"] {
      background: linear-gradient(
          180deg,
          rgba(90, 238, 255, 1.0) 0%,
          rgba(44, 214, 241, 0.98) 100%
      ) !important;
      box-shadow:
          inset 0 0 0 1px rgba(255,255,255,0.18),
          0 0 24px rgba(63, 224, 248, 0.28) !important;
      z-index: 2 !important;
  }

  div[role="radiogroup"] > label:has(div[aria-checked="true"])::after,
  div[role="radiogroup"] > label:has(input[aria-checked="true"])::after,
  div[role="radiogroup"] > label:has(input:checked)::after,
  div[role="radiogroup"] > label[data-selected="true"]::after,
  div[role="radiogroup"] > label[aria-checked="true"]::after {
      content: "";
      position: absolute;
      left: 10%;
      right: 10%;
      top: 4px;
      height: 3px;
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
  }

  div[role="radiogroup"] > label:has(div[aria-checked="true"]) p,
  div[role="radiogroup"] > label:has(input[aria-checked="true"]) p,
  div[role="radiogroup"] > label:has(input:checked) p,
  div[role="radiogroup"] > label[data-selected="true"] p,
  div[role="radiogroup"] > label[aria-checked="true"] p {
      color: #052838 !important;
      font-weight: 800 !important;
  }

  div[data-testid="stImage"] img {
      max-height: none !important;
      width: 100% !important;
      max-width: 100% !important;
      object-fit: contain !important;
      display: block !important;
      margin-left: auto !important;
      margin-right: auto !important;
  }

  .exp-img-wrap {
      width: 100%;
      min-width: 0;
  }

  .exp-img-wrap [data-testid="stImage"] {
      width: 100% !important;
  }

  .exp-img-wrap [data-testid="stImage"] img {
      width: 100% !important;
      max-width: 100% !important;
      height: 420px !important;
      object-fit: contain !important;
      border-radius: 14px !important;
      background: transparent !important;
  }

  .exp-img-wrap.pair [data-testid="stImage"] img {
      height: 460px !important;
  }

  .loader-overlay {
      position: fixed;
      inset: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      background: rgba(3, 8, 20, 0.72);
      z-index: 999999;
      backdrop-filter: blur(2px);
  }

  .loader-box {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 18px 26px;
      border-radius: 16px;
      background: rgba(15, 23, 42, 0.96);
      border: 1px solid rgba(255,255,255,0.09);
      box-shadow: 0 10px 30px rgba(0,0,0,0.30);
      font-size: 20px;
      font-weight: 500;
  }

  .custom-loader {
      width: 24px;
      height: 24px;
      border: 3px solid rgba(255,255,255,0.18);
      border-top: 3px solid rgba(255,255,255,0.95);
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
  }

  @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
  }
</style>
"""


def apply_analysis_styles() -> None:
    st.markdown(ANALYSIS_CSS, unsafe_allow_html=True)


def show_fixed_loader(message: str = "Preparing analysis..."):
    holder = st.empty()
    holder.markdown(
        f"""
        <div class="loader-overlay">
            <div class="loader-box">
                <div class="custom-loader"></div>
                <span>{message}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return holder


def clear_fixed_loader(holder) -> None:
    holder.empty()