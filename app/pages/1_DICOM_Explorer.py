"""
DICOM Metadata Explorer
=======================
Auto-downloads SIIM_SMALL (250 chest X-rays, ~30 MB) on first run,
extracts metadata to a pandas DataFrame, shows summary stats + charts.
"""
import os
import glob
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom

st.set_page_config(page_title="DICOM Explorer", page_icon="📊", layout="wide")
st.title("📊 DICOM Metadata Explorer")
st.markdown(
    "Walks a folder of DICOM files, extracts key metadata, builds a `pandas` DataFrame, "
    "and reports summary statistics — the first step of any clinical ML project."
)

# ---------------------------------------------------------------- helpers
def parse_age(age_str):
    """Convert a DICOM PatientAge string ('045Y' / '21') to years (float) or None."""
    if not age_str or not isinstance(age_str, str):
        return None
    age_str = age_str.strip()
    if not age_str:
        return None
    if age_str[-1].upper() in "YMWD":
        try:
            v = int(age_str[:-1])
        except ValueError:
            return None
        return {"Y": v, "M": v / 12, "W": v * 7 / 365, "D": v / 365}[age_str[-1].upper()]
    try:
        return float(age_str)
    except ValueError:
        return None


def explore_dicom_folder(folder, recursive=True):
    """Walk a folder of DICOMs and return a metadata DataFrame."""
    pattern = os.path.join(folder, "**", "*.dcm") if recursive else os.path.join(folder, "*.dcm")
    paths = sorted(glob.glob(pattern, recursive=recursive))
    rows = []
    progress = st.progress(0.0, text=f"Reading {len(paths)} DICOM files...")
    for i, path in enumerate(paths):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception as e:
            rows.append({"file": os.path.basename(path), "error": str(e)})
            continue
        if not hasattr(ds, "Modality"):
            continue
        rows.append({
            "file":               os.path.basename(path),
            # --- four required fields ---
            "modality":           getattr(ds, "Modality", None),
            "patient_age":        parse_age(getattr(ds, "PatientAge", None)),
            "acquisition_date":   getattr(ds, "AcquisitionDate",
                                  getattr(ds, "StudyDate", None)),
            "body_part_examined": getattr(ds, "BodyPartExamined", None),
            # --- extras ---
            "patient_id":         getattr(ds, "PatientID", None),
            "patient_sex":        getattr(ds, "PatientSex", None),
            "view_position":      getattr(ds, "ViewPosition", None),
            "manufacturer":       getattr(ds, "Manufacturer", None),
            "_path":              path,
        })
        progress.progress((i + 1) / len(paths))
    progress.empty()
    df = pd.DataFrame(rows)
    if "acquisition_date" in df.columns:
        df["acquisition_date"] = pd.to_datetime(
            df["acquisition_date"], format="%Y%m%d", errors="coerce"
        )
    return df


@st.cache_data(show_spinner=False)
def load_dicom_folder_cached(folder, file_count):
    """Cache the DataFrame on (folder, file_count) so re-runs are instant."""
    return explore_dicom_folder(folder, recursive=True)


@st.cache_resource(show_spinner="Downloading SIIM_SMALL (~30 MB)...")
def download_siim_small():
    """Download the SIIM_SMALL dataset via fastai."""
    from fastai.data.external import untar_data, URLs
    return Path(untar_data(URLs.SIIM_SMALL))


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Data source")
source = st.sidebar.radio(
    "Where should the explorer read DICOMs from?",
    ["Auto-download SIIM_SMALL", "Use existing folder path"],
)

folder = None
if source == "Auto-download SIIM_SMALL":
    st.sidebar.markdown(
        "Downloads the 30 MB teaching subset of the SIIM-ACR Pneumothorax "
        "challenge via fastai (no Kaggle credentials needed)."
    )
    if st.sidebar.button("Download & explore", type="primary"):
        st.session_state["dicom_folder"] = str(download_siim_small() / "train")
    if "dicom_folder" in st.session_state:
        folder = st.session_state["dicom_folder"]
else:
    user_path = st.sidebar.text_input("Folder path", value="data/sample_dicoms")
    if st.sidebar.button("Explore", type="primary"):
        st.session_state["dicom_folder"] = user_path
    if "dicom_folder" in st.session_state:
        folder = st.session_state["dicom_folder"]

# ---------------------------------------------------------------- main
if folder is None:
    st.info("👈 Pick a data source in the sidebar and click the button to begin.")
    st.stop()

if not os.path.isdir(folder):
    st.error(f"Folder not found: `{folder}`")
    st.stop()

dcm_count = len(glob.glob(os.path.join(folder, "**", "*.dcm"), recursive=True))
if dcm_count == 0:
    st.warning(f"No `.dcm` files found in `{folder}`")
    st.stop()

st.success(f"Found **{dcm_count}** DICOM files in `{folder}`")

df = load_dicom_folder_cached(folder, dcm_count)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("DICOM files", len(df))
c2.metric("Unique patients", df["patient_id"].nunique() if "patient_id" in df else "—")
c3.metric("Modalities", df["modality"].nunique())
c4.metric("Body parts", df["body_part_examined"].dropna().nunique())

# ---------- Tabs ----------
tab_data, tab_charts, tab_image = st.tabs(["📋 Data", "📈 Charts", "🖼️ Sample image"])

with tab_data:
    st.subheader("Metadata DataFrame")
    st.dataframe(
        df.drop(columns=["_path"]),
        use_container_width=True,
        height=350,
    )
    st.subheader("Summary statistics")
    sub1, sub2 = st.columns(2)
    with sub1:
        st.markdown("**Patient age (years)**")
        st.dataframe(df["patient_age"].describe().round(1).to_frame("value"))
        st.markdown("**Patient sex**")
        st.dataframe(df["patient_sex"].value_counts(dropna=False).to_frame("count"))
    with sub2:
        st.markdown("**Modality**")
        st.dataframe(df["modality"].value_counts(dropna=False).to_frame("count"))
        st.markdown("**Body part examined**")
        st.dataframe(df["body_part_examined"].value_counts(dropna=False).to_frame("count"))

    st.subheader("Missing values per column")
    st.dataframe(df.drop(columns=["_path"]).isna().sum().to_frame("missing"))

with tab_charts:
    st.subheader("Overview")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df["modality"].value_counts().plot.bar(ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("DICOM files by modality")
    axes[0].tick_params(axis="x", rotation=0)
    df["body_part_examined"].value_counts(dropna=False).plot.bar(ax=axes[1], color="darkorange", edgecolor="black")
    axes[1].set_title("Files by body part")
    axes[1].tick_params(axis="x", rotation=15)
    ages = df["patient_age"].dropna()
    if len(ages):
        ages.plot.hist(ax=axes[2], bins=20, color="seagreen", edgecolor="black")
        axes[2].axvline(ages.mean(), color="red", linestyle="--", label=f"mean={ages.mean():.1f}")
        axes[2].axvline(ages.median(), color="blue", linestyle="--", label=f"median={ages.median():.1f}")
        axes[2].legend()
    axes[2].set_title("Patient age distribution")
    axes[2].set_xlabel("Age (years)")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Sex and dates")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sex_counts = df["patient_sex"].value_counts(dropna=False)
    if len(sex_counts):
        axes[0].pie(sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%",
                    colors=["lightcoral", "skyblue", "lightgray"][:len(sex_counts)],
                    startangle=90, wedgeprops={"edgecolor": "black"})
        axes[0].set_title("Patient sex distribution")

    years = df["acquisition_date"].dt.year.value_counts().sort_index()
    if len(years):
        years.plot.bar(ax=axes[1], color="mediumpurple", edgecolor="black")
        axes[1].set_title("Acquisitions per year")
        axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Body part × Modality")
    ct = pd.crosstab(
        df["body_part_examined"].fillna("(missing)"),
        df["modality"],
        margins=True, margins_name="Total",
    )
    st.dataframe(ct)

with tab_image:
    st.subheader("Sample DICOM image")
    st.caption("Pick a file and see the pixel array alongside its metadata.")

    options = df["file"].tolist()
    selected = st.selectbox("DICOM file", options=options, index=0)
    row = df[df["file"] == selected].iloc[0]

    try:
        ds = pydicom.dcmread(row["_path"])
        img = ds.pixel_array
        col_img, col_meta = st.columns([2, 1])
        with col_img:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(selected)
            st.pyplot(fig)
        with col_meta:
            st.markdown("**Metadata**")
            meta = {
                "Modality":      row.get("modality"),
                "Body part":     row.get("body_part_examined"),
                "Patient age":   row.get("patient_age"),
                "Patient sex":   row.get("patient_sex"),
                "View position": row.get("view_position"),
                "Manufacturer":  row.get("manufacturer"),
                "Acquired":      str(row.get("acquisition_date")),
                "Image size":    f"{img.shape[0]} × {img.shape[1]}",
                "Pixel dtype":   str(img.dtype),
            }
            for k, v in meta.items():
                st.write(f"**{k}:** {v if pd.notna(v) else '—'}")
    except Exception as e:
        st.error(f"Could not read pixels from this file: {e}")

st.divider()
st.caption(
    "Dataset: SIIM_SMALL via fastai — "
    "[source](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)"
)
