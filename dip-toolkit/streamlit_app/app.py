"""
Digital Image Processing Toolkit — Streamlit Demo
---------------------------------------------------
Interactive companion to notebooks/dip_toolkit.ipynb.

Run:
    pip install -r requirements.txt
    streamlit run streamlit_app/app.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage, fftpack
from scipy.signal import wiener
from skimage import data, exposure, filters, morphology, segmentation, measure, restoration, color, feature
from skimage.util import random_noise
from sklearn.cluster import KMeans

st.set_page_config(page_title="DIP Toolkit", layout="wide")

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def show_row(images, titles, cmap="gray"):
    cols = st.columns(len(images))
    for col, im, t in zip(cols, images, titles):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(im, cmap=cmap)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
        col.pyplot(fig)
        plt.close(fig)


@st.cache_data
def load_sample(kind):
    if kind == "Medical (MRI phantom)":
        return data.shepp_logan_phantom()
    return data.camera().astype(np.float64) / 255.0


def load_uploaded(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return exposure.rescale_intensity(img.astype(np.float64), out_range=(0, 1))


# ------------------------------------------------------------------ #
# Sidebar — image source & category
# ------------------------------------------------------------------ #
st.sidebar.title("🩻 DIP Toolkit")
st.sidebar.caption("Upload your own image or use a sample, then pick a technique to explore.")

source = st.sidebar.radio("Image source", ["Sample image", "Upload my own"])
if source == "Upload my own":
    uploaded = st.sidebar.file_uploader("Upload (X-ray, MRI, photo...)", type=["png", "jpg", "jpeg", "bmp", "tif"])
    image = load_uploaded(uploaded) if uploaded else load_sample("Medical (MRI phantom)")
else:
    sample_kind = st.sidebar.selectbox("Sample", ["Medical (MRI phantom)", "General (photo)"])
    image = load_sample(sample_kind)

category = st.sidebar.selectbox(
    "Technique category",
    [
        "Fundamentals (Histogram / CLAHE)",
        "Spatial Filtering",
        "Frequency Domain (FFT)",
        "Noise & Restoration",
        "Morphology",
        "Segmentation",
        "Full Pipeline",
    ],
)

st.title("Digital Image Processing Toolkit")

with st.expander("Show original image"):
    st.image(image, clamp=True, width=300)

st.divider()

# ------------------------------------------------------------------ #
# 1. Fundamentals
# ------------------------------------------------------------------ #
if category == "Fundamentals (Histogram / CLAHE)":
    st.header("Histogram Equalization & CLAHE")
    clip = st.slider("CLAHE clip limit", 0.005, 0.05, 0.03, 0.005)

    img_eq = exposure.equalize_hist(image)
    img_clahe = exposure.equalize_adapthist(image, clip_limit=clip)

    show_row([image, img_eq, img_clahe], ["Original", "Global Histogram Equalization", "CLAHE"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, im, t in zip(axes, [image, img_eq, img_clahe], ["Original", "Equalized", "CLAHE"]):
        ax.hist(im.ravel(), bins=256, range=(0, 1), color="steelblue")
        ax.set_title(f"{t} — Histogram", fontsize=10)
    st.pyplot(fig)

# ------------------------------------------------------------------ #
# 2. Spatial Filtering
# ------------------------------------------------------------------ #
elif category == "Spatial Filtering":
    st.header("Smoothing, Sharpening & Edge Detection")

    st.subheader("Smoothing")
    ksize = st.slider("Filter size", 3, 15, 5, 2)
    mean_blur = ndimage.uniform_filter(image, size=ksize)
    gauss_blur = ndimage.gaussian_filter(image, sigma=ksize / 3)
    median_blur = ndimage.median_filter(image, size=ksize)
    show_row([image, mean_blur, gauss_blur, median_blur],
              ["Original", "Mean Filter", "Gaussian Filter", "Median Filter"])

    st.subheader("Sharpening")
    amount = st.slider("Unsharp amount", 0.5, 3.0, 1.5, 0.1)
    unsharp = filters.unsharp_mask(image, radius=3, amount=amount)
    laplacian = ndimage.laplace(gauss_blur)
    sharpened = np.clip(image - laplacian, 0, 1)
    show_row([image, unsharp, sharpened], ["Original", "Unsharp Masking", "Laplacian Sharpening"])

    st.subheader("Edge Detection")
    sigma = st.slider("Canny sigma", 0.5, 4.0, 1.5, 0.1)
    sobel = filters.sobel(image)
    prewitt = filters.prewitt(image)
    canny = feature.canny(image, sigma=sigma)
    show_row([image, sobel, prewitt, canny], ["Original", "Sobel", "Prewitt", "Canny"])

# ------------------------------------------------------------------ #
# 3. Frequency Domain
# ------------------------------------------------------------------ #
elif category == "Frequency Domain (FFT)":
    st.header("2D FFT & Frequency-Domain Filtering")
    cutoff = st.slider("Cutoff radius", 5, 100, 30, 5)

    F = fftpack.fftshift(fftpack.fft2(image))
    mag_spectrum = np.log1p(np.abs(F))

    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    lp_mask = (dist <= cutoff).astype(np.float64)
    hp_mask = (dist > cutoff).astype(np.float64)

    img_lowpass = np.real(fftpack.ifft2(fftpack.ifftshift(F * lp_mask)))
    img_highpass = np.real(fftpack.ifft2(fftpack.ifftshift(F * hp_mask)))
    img_highpass = exposure.rescale_intensity(img_highpass, out_range=(0, 1))

    show_row([image, mag_spectrum], ["Original", "Log Magnitude Spectrum"])
    show_row([image, img_lowpass, img_highpass],
              ["Original", f"Low-pass (cutoff={cutoff})", f"High-pass (cutoff={cutoff})"])

# ------------------------------------------------------------------ #
# 4. Noise & Restoration
# ------------------------------------------------------------------ #
elif category == "Noise & Restoration":
    st.header("Noise Simulation & Restoration")
    noise_type = st.selectbox("Noise type", ["Gaussian", "Salt & Pepper", "Speckle (ultrasound-like)"])
    amount = st.slider("Noise intensity", 0.01, 0.15, 0.05, 0.01)

    if noise_type == "Gaussian":
        noisy = random_noise(image, mode="gaussian", var=amount)
        restored = wiener(noisy, mysize=5)
        show_row([image, noisy, restored], ["Original", "Gaussian Noise", "Wiener Restoration"])
    elif noise_type == "Salt & Pepper":
        noisy = random_noise(image, mode="s&p", amount=amount)
        restored = ndimage.median_filter(noisy, size=3)
        show_row([image, noisy, restored], ["Original", "Salt & Pepper Noise", "Median Filter Restoration"])
    else:
        noisy = random_noise(image, mode="speckle", var=amount)
        restored = restoration.denoise_nl_means(noisy, h=0.08, fast_mode=True, patch_size=5, patch_distance=6)
        show_row([image, noisy, restored], ["Original", "Speckle Noise", "Non-Local Means Restoration"])

# ------------------------------------------------------------------ #
# 5. Morphology
# ------------------------------------------------------------------ #
elif category == "Morphology":
    st.header("Morphological Operations")
    radius = st.slider("Structuring element radius", 1, 10, 3)

    binary = image > filters.threshold_otsu(image)
    selem = morphology.disk(radius)

    eroded = morphology.erosion(binary, selem)
    dilated = morphology.dilation(binary, selem)
    opened = morphology.opening(binary, selem)
    closed = morphology.closing(binary, selem)
    boundary = binary ^ eroded

    show_row([binary, eroded, dilated], ["Binary Mask (Otsu)", "Erosion", "Dilation"])
    show_row([opened, closed, boundary], ["Opening", "Closing", "Boundary Extraction"])

# ------------------------------------------------------------------ #
# 6. Segmentation
# ------------------------------------------------------------------ #
elif category == "Segmentation":
    st.header("Segmentation Methods")

    otsu_mask = image > filters.threshold_otsu(image)

    sigma = st.slider("Canny sigma", 0.5, 4.0, 2.0, 0.1)
    edges = feature.canny(image, sigma=sigma)
    filled = ndimage.binary_fill_holes(edges)

    k = st.slider("K-means clusters", 2, 6, 3)
    flat = image.reshape(-1, 1)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(flat)
    kmeans_seg = km.labels_.reshape(image.shape)

    distance = ndimage.distance_transform_edt(otsu_mask)
    coords = feature.peak_local_max(distance, min_distance=15, labels=otsu_mask)
    mask_markers = np.zeros(distance.shape, dtype=bool)
    mask_markers[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask_markers)
    ws_labels = segmentation.watershed(-distance, markers, mask=otsu_mask)

    show_row([otsu_mask, filled, kmeans_seg, ws_labels],
              ["Otsu Threshold", "Canny + Fill Holes", f"K-Means (k={k})", "Watershed"],
              cmap="nipy_spectral")

    st.subheader("Region-Property Filtering")
    labeled = measure.label(otsu_mask)
    props = measure.regionprops(labeled)
    candidates = sorted([p for p in props if p.area > 50], key=lambda p: p.area, reverse=True)

    st.write(f"Found **{len(props)}** regions, **{len(candidates)}** above the area threshold.")
    if candidates:
        rows = [{"label": p.label, "area": int(p.area), "eccentricity": round(p.eccentricity, 3),
                 "solidity": round(p.solidity, 3)} for p in candidates[:5]]
        st.table(rows)

        target = candidates[0]
        target_mask = labeled == target.label
        overlay = color.gray2rgb(image)
        overlay[target_mask] = [1, 0.2, 0.2]
        show_row([image, target_mask, overlay], ["Original", "Selected Region (largest)", "Overlay"])

# ------------------------------------------------------------------ #
# 7. Full Pipeline
# ------------------------------------------------------------------ #
else:
    st.header("End-to-End Pipeline: Denoise → Enhance → Segment → Overlay")

    noise_var = st.slider("Simulated noise level", 0.001, 0.02, 0.008, 0.001)

    noisy = random_noise(image, mode="gaussian", var=noise_var)
    denoised = restoration.denoise_nl_means(noisy, h=0.06, fast_mode=True)
    enhanced = exposure.equalize_adapthist(denoised, clip_limit=0.02)
    mask = enhanced > filters.threshold_otsu(enhanced)
    clean_mask = morphology.closing(mask, morphology.disk(2))
    clean_mask = morphology.remove_small_objects(clean_mask, min_size=40)

    overlay = color.gray2rgb(image)
    boundary = clean_mask ^ morphology.erosion(clean_mask, morphology.disk(1))
    overlay[boundary] = [1, 0, 0]

    show_row([noisy, denoised, enhanced], ["1. Noisy Input", "2. Denoised (NLM)", "3. Enhanced (CLAHE)"])
    show_row([clean_mask, overlay], ["4-5. Segmented + Cleaned Mask", "6. Final Overlay"])

st.divider()
st.caption("Built with NumPy, OpenCV, scikit-image, SciPy, scikit-learn & Streamlit.")
