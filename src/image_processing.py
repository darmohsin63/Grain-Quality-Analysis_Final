import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# GrainVision - Adaptive Grain Detection & Quality Analysis
#
# Designed for BOTH:
#   - light grains on dark backgrounds
#   - dark/brown grains on light backgrounds
#
# Pipeline:
#   adaptive foreground detection
#       -> morphology
#       -> connected-component filtering
#       -> watershed separation
#       -> feature extraction
#       -> Good / Mixed / Defective classification
#
# NOTE:
# The quality classifier is a preliminary image-processing model.
# It is NOT a scientifically validated agricultural grading standard.
# ============================================================


MIN_AREA = 80
MAX_AREA = 200000

MIN_SOLIDITY = 0.60
MIN_CIRCULARITY = 0.12

# Grain-like objects are normally elongated.
MIN_ASPECT = 1.35
MAX_ASPECT = 8.0

DISTANCE_RATIO = 0.32


def safe_ratio(a, b):
    return float(a / b) if b else 0.0


# ------------------------------------------------------------
# Foreground polarity detection
# ------------------------------------------------------------

def detect_foreground_polarity(gray):
    """
    Determine whether the objects are lighter or darker than the
    background by looking at the image border.

    This fixes the major problem with the previous version:
    the previous algorithm assumed grains were darker than the
    background. Your current test image has LIGHT grains on a
    DARK background.
    """

    h, w = gray.shape

    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1],
    ])

    border_median = float(np.median(border))

    image_median = float(np.median(gray))

    # If the border is dark, foreground is expected to be bright.
    if border_median < image_median:
        return "bright"

    return "dark"


# ------------------------------------------------------------
# Adaptive mask
# ------------------------------------------------------------

def create_grain_mask(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    polarity = detect_foreground_polarity(
        blurred
    )

    # Otsu gives a data-driven threshold instead of a fixed value.
    otsu_value, _ = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    if polarity == "bright":

        # Light grains on dark background.
        _, mask = cv2.threshold(
            blurred,
            otsu_value,
            255,
            cv2.THRESH_BINARY,
        )

    else:

        # Dark grains on light background.
        _, mask = cv2.threshold(
            blurred,
            otsu_value,
            255,
            cv2.THRESH_BINARY_INV,
        )

    # --------------------------------------------------------
    # HSV support
    #
    # Only use saturation as a supporting signal. It must not
    # replace the polarity-aware intensity mask.
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]

    sat_value = float(
        np.percentile(saturation, 70)
    )

    saturation_mask = np.uint8(
        saturation > max(18, sat_value)
    ) * 255

    # Add saturation only inside already-detected foreground.
    # This prevents random coloured text from becoming grains.
    if polarity == "bright":

        support = cv2.bitwise_and(
            mask,
            saturation_mask,
        )

        mask = cv2.bitwise_or(
            mask,
            support,
        )

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    # --------------------------------------------------------
    # Remove components touching the image border.
    #
    # A real grain can theoretically touch the border, but for
    # this application it is safer to exclude incomplete grains.
    # --------------------------------------------------------

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    h, w = mask.shape

    for label in range(1, n):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]

        # Ignore components touching image boundary.
        if x <= 0 or y <= 0 or x + cw >= w - 1 or y + ch >= h - 1:
            continue

        cleaned[labels == label] = 255

    return cleaned


# ------------------------------------------------------------
# Grain plausibility filter
# ------------------------------------------------------------

def is_plausible_grain(contour, image_shape):

    area = cv2.contourArea(contour)

    if area < MIN_AREA or area > MAX_AREA:
        return False

    h_img, w_img = image_shape[:2]

    x, y, w, h = cv2.boundingRect(contour)

    if w < 3 or h < 3:
        return False

    aspect = safe_ratio(
        max(w, h),
        min(w, h),
    )

    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    if perimeter <= 0:
        return False

    circularity = safe_ratio(
        4 * math.pi * area,
        perimeter * perimeter,
    )

    if circularity < MIN_CIRCULARITY:
        return False

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    solidity = safe_ratio(
        area,
        hull_area,
    )

    if solidity < MIN_SOLIDITY:
        return False

    # Reject huge regions.
    if area > h_img * w_img * 0.25:
        return False

    return True


# ------------------------------------------------------------
# Watershed
# ------------------------------------------------------------

def split_touching_grains(mask):

    if cv2.countNonZero(mask) == 0:
        return np.zeros_like(mask), []

    distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    maximum = float(
        distance.max()
    )

    if maximum <= 0:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = [
            c for c in contours
            if is_plausible_grain(c, mask.shape)
        ]

        return mask, contours

    _, sure_fg = cv2.threshold(
        distance,
        DISTANCE_RATIO * maximum,
        255,
        cv2.THRESH_BINARY,
    )

    sure_fg = np.uint8(
        sure_fg
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    sure_bg = cv2.dilate(
        mask,
        kernel,
        iterations=2,
    )

    unknown = cv2.subtract(
        sure_bg,
        sure_fg,
    )

    marker_count, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[unknown == 255] = 0

    watershed_image = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.watershed(
        watershed_image,
        markers,
    )

    separated = np.zeros_like(mask)

    contours = []

    for marker_id in range(
        2,
        marker_count + 1,
    ):

        region = (
            np.uint8(
                markers == marker_id
            ) * 255
        )

        if cv2.countNonZero(region) < MIN_AREA:
            continue

        region_contours, _ = cv2.findContours(
            region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not region_contours:
            continue

        contour = max(
            region_contours,
            key=cv2.contourArea,
        )

        if is_plausible_grain(
            contour,
            mask.shape,
        ):
            cv2.drawContours(
                separated,
                [contour],
                -1,
                255,
                -1,
            )

            contours.append(
                contour
            )

    # Watershed can occasionally fail on an image containing isolated
    # grains. Fall back to ordinary contours in that case.
    if not contours:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = [
            c for c in contours
            if is_plausible_grain(
                c,
                mask.shape,
            )
        ]

        separated = np.zeros_like(mask)

        for contour in contours:
            cv2.drawContours(
                separated,
                [contour],
                -1,
                255,
                -1,
            )

    return separated, contours


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------

def extract_features(
    contour,
    gray,
):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    length = max(
        w,
        h,
    )

    width = min(
        w,
        h,
    )

    aspect_ratio = safe_ratio(
        length,
        width,
    )

    circularity = safe_ratio(
        4 * math.pi * area,
        perimeter * perimeter,
    )

    equivalent_diameter = math.sqrt(
        safe_ratio(
            4 * area,
            math.pi,
        )
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    solidity = safe_ratio(
        area,
        hull_area,
    )

    rect = cv2.minAreaRect(
        contour
    )

    rw, rh = rect[1]

    if rw > 0 and rh > 0:
        oriented_length = max(
            rw,
            rh,
        )

        oriented_width = min(
            rw,
            rh,
        )

    else:
        oriented_length = length
        oriented_width = width

    grain_mask = np.zeros(
        gray.shape,
        dtype=np.uint8,
    )

    cv2.drawContours(
        grain_mask,
        [contour],
        -1,
        255,
        -1,
    )

    mean_intensity, std_intensity = cv2.meanStdDev(
        gray,
        mask=grain_mask,
    )

    return {
        "Area": round(float(area), 2),
        "Perimeter": round(float(perimeter), 2),
        "Length": round(float(oriented_length), 2),
        "Width": round(float(oriented_width), 2),
        "Aspect_Ratio": round(float(aspect_ratio), 3),
        "Circularity": round(float(circularity), 3),
        "Equivalent_Diameter": round(float(equivalent_diameter), 2),
        "Solidity": round(float(solidity), 3),
        "Mean_Intensity": round(float(mean_intensity[0][0]), 2),
        "Intensity_Std": round(float(std_intensity[0][0]), 2),
        "X": int(x),
        "Y": int(y),
        "W": int(w),
        "H": int(h),
    }


# ------------------------------------------------------------
# Preliminary quality classification
# ------------------------------------------------------------

def classify_grain(
    row,
    median_area,
):

    score = 100.0

    area = float(
        row["Area"]
    )

    aspect = float(
        row["Aspect_Ratio"]
    )

    solidity = float(
        row["Solidity"]
    )

    circularity = float(
        row["Circularity"]
    )

    intensity_std = float(
        row["Intensity_Std"]
    )

    # Relative size is better than one universal hard-coded
    # grain-size threshold.
    if median_area:

        ratio = area / median_area

        if ratio < 0.55 or ratio > 1.75:
            score -= 25

        elif ratio < 0.70 or ratio > 1.45:
            score -= 10

    # Shape.
    if aspect < 1.35 or aspect > 6.5:
        score -= 12

    if solidity < 0.78:
        score -= 20

    elif solidity < 0.88:
        score -= 8

    if circularity < 0.20:
        score -= 12

    elif circularity < 0.30:
        score -= 6

    # Surface irregularity indicator.
    if intensity_std > 50:
        score -= 15

    elif intensity_std > 35:
        score -= 7

    score = max(
        0,
        min(
            100,
            round(score),
        ),
    )

    if score >= 78:
        quality = "Good"

    elif score >= 55:
        quality = "Mixed"

    else:
        quality = "Defective"

    return pd.Series([
        score,
        quality,
    ])


# ------------------------------------------------------------
# Graphs
# ------------------------------------------------------------

def save_plots(
    df,
    output_dir,
    names,
):

    plt.figure(
        figsize=(7, 4.5)
    )

    if not df.empty:

        bins = min(
            12,
            max(
                3,
                int(
                    np.sqrt(
                        len(df)
                    )
                ),
            ),
        )

        plt.hist(
            df["Area"],
            bins=bins,
        )

    plt.xlabel(
        "Grain Area (pixels)"
    )

    plt.ylabel(
        "Number of Grains"
    )

    plt.title(
        "Grain Area Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            names["area"],
        ),
        dpi=150,
    )

    plt.close()

    plt.figure(
        figsize=(7, 4.5)
    )

    if not df.empty:

        order = [
            "Good",
            "Mixed",
            "Defective",
        ]

        counts = (
            df["Quality"]
            .value_counts()
            .reindex(
                order,
                fill_value=0,
            )
        )

        counts.plot(
            kind="bar"
        )

    plt.xlabel(
        "Quality Category"
    )

    plt.ylabel(
        "Number of Grains"
    )

    plt.title(
        "Grain Quality Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            names["quality"],
        ),
        dpi=150,
    )

    plt.close()


# ============================================================
# Main analysis function
# ============================================================

def analyze_grain_image(
    input_path,
    output_dir,
    job_id,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load image
    # --------------------------------------------------------

    image = cv2.imread(
        input_path
    )

    if image is None:
        raise ValueError(
            "OpenCV could not read the uploaded image."
        )

    # --------------------------------------------------------
    # 2. Adaptive segmentation
    # --------------------------------------------------------

    initial_mask = create_grain_mask(
        image
    )

    # --------------------------------------------------------
    # 3. Separate touching grains
    # --------------------------------------------------------

    clean_mask, contours = split_touching_grains(
        initial_mask
    )

    # --------------------------------------------------------
    # 4. Sort
    # --------------------------------------------------------

    contours.sort(
        key=lambda c: (
            cv2.boundingRect(c)[1],
            cv2.boundingRect(c)[0],
        )
    )

    # --------------------------------------------------------
    # 5. Features
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    rows = []

    for grain_id, contour in enumerate(
        contours,
        start=1,
    ):

        row = extract_features(
            contour,
            gray,
        )

        row["Grain_ID"] = grain_id

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    if not df.empty:

        ordered = [
            "Grain_ID",
            "Area",
            "Perimeter",
            "Length",
            "Width",
            "Aspect_Ratio",
            "Circularity",
            "Equivalent_Diameter",
            "Solidity",
            "Mean_Intensity",
            "Intensity_Std",
            "X",
            "Y",
            "W",
            "H",
        ]

        df = df[
            ordered
        ]

        median_area = float(
            df["Area"].median()
        )

        df[
            [
                "Quality_Score",
                "Quality",
            ]
        ] = df.apply(
            lambda row: classify_grain(
                row,
                median_area,
            ),
            axis=1,
        )

    else:

        median_area = None

        df["Quality_Score"] = pd.Series(
            dtype=float
        )

        df["Quality"] = pd.Series(
            dtype=str
        )

    # --------------------------------------------------------
    # 6. Annotated image
    # --------------------------------------------------------

    annotated = image.copy()

    for index, contour in enumerate(
        contours
    ):

        grain_id = index + 1

        if df.empty:
            quality = "Unknown"
        else:
            quality = str(
                df.iloc[index]["Quality"]
            )

        if quality == "Good":
            color = (
                60,
                190,
                80,
            )

        elif quality == "Mixed":
            color = (
                0,
                180,
                255,
            )

        elif quality == "Defective":
            color = (
                40,
                60,
                220,
            )

        else:
            color = (
                255,
                255,
                255,
            )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.drawContours(
            annotated,
            [contour],
            -1,
            color,
            2,
        )

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            1,
        )

        cv2.putText(
            annotated,
            f"{grain_id}",
            (
                x,
                max(
                    y - 8,
                    20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # 7. Output filenames
    # --------------------------------------------------------

    names = {
        "annotated":
            f"{job_id}_annotated.jpg",

        "mask":
            f"{job_id}_mask.png",

        "features":
            f"{job_id}_features.csv",

        "summary":
            f"{job_id}_summary.csv",

        "area":
            f"{job_id}_area.png",

        "quality":
            f"{job_id}_quality.png",
    }

    # --------------------------------------------------------
    # 8. Save files
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            output_dir,
            names["annotated"],
        ),
        annotated,
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            names["mask"],
        ),
        clean_mask,
    )

    df.to_csv(
        os.path.join(
            output_dir,
            names["features"],
        ),
        index=False,
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    total = len(df)

    if total:

        good = int(
            (
                df["Quality"]
                == "Good"
            ).sum()
        )

        mixed = int(
            (
                df["Quality"]
                == "Mixed"
            ).sum()
        )

        defective = int(
            (
                df["Quality"]
                == "Defective"
            ).sum()
        )

    else:

        good = 0
        mixed = 0
        defective = 0

    quality_percentage = (
        good / total * 100
        if total
        else 0.0
    )

    summary = pd.DataFrame([
        {
            "Total Grains":
                total,

            "Good Grains":
                good,

            "Mixed Grains":
                mixed,

            "Defective Grains":
                defective,

            "Quality Percentage":
                round(
                    quality_percentage,
                    2,
                ),
        }
    ])

    summary.to_csv(
        os.path.join(
            output_dir,
            names["summary"],
        ),
        index=False,
    )

    # --------------------------------------------------------
    # 10. Graphs
    # --------------------------------------------------------

    save_plots(
        df,
        output_dir,
        names,
    )

    # --------------------------------------------------------
    # 11. Flask result
    # --------------------------------------------------------

    return {
        "total":
            total,

        "good":
            good,

        # Compatibility with the existing template.
        "average":
            mixed,

        "mixed":
            mixed,

        "defective":
            defective,

        "quality_percentage":
            round(
                quality_percentage,
                2,
            ),

        "annotated_image":
            names["annotated"],

        "mask_image":
            names["mask"],

        "features_csv":
            names["features"],

        "summary_csv":
            names["summary"],

        "area_plot":
            names["area"],

        "quality_plot":
            names["quality"],

        "features":
            df.to_dict(
                orient="records"
            ),
    }
