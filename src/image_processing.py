import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# GrainVision - Robust Grain Detection
# ============================================================
#
# Designed for:
#   1. light grains on dark backgrounds
#   2. dark/brown grains on light backgrounds
#
# The detector DOES NOT assume a fixed threshold or fixed
# foreground polarity.
#
# Pipeline:
#   image
#      -> automatic polarity selection
#      -> Otsu / adaptive threshold
#      -> morphology
#      -> component filtering
#      -> optional watershed separation
#      -> feature extraction
#      -> preliminary Good / Mixed / Defective classification
#
# NOTE:
# The quality classifier is a preliminary demonstration model.
# It is not a scientifically validated agricultural grading standard.
# ============================================================


MIN_AREA = 60
MAX_AREA = 200000

# Grain-like objects should not be extremely thin or circular.
MIN_ASPECT = 1.25
MAX_ASPECT = 8.0

MIN_SOLIDITY = 0.55
MIN_CIRCULARITY = 0.10

# A large connected component is considered a possible group
# of touching grains and may be processed with watershed.
LARGE_COMPONENT_FACTOR = 2.5


def ratio(a, b):
    return float(a / b) if b else 0.0


# ============================================================
# Basic contour plausibility
# ============================================================

def plausible_grain(contour, shape):
    area = cv2.contourArea(contour)

    if area < MIN_AREA or area > MAX_AREA:
        return False

    x, y, w, h = cv2.boundingRect(contour)

    if w < 3 or h < 3:
        return False

    aspect = ratio(max(w, h), min(w, h))

    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False

    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return False

    circularity = ratio(
        4.0 * math.pi * area,
        perimeter * perimeter
    )

    if circularity < MIN_CIRCULARITY:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = ratio(area, hull_area)

    if solidity < MIN_SOLIDITY:
        return False

    image_area = shape[0] * shape[1]

    # Reject a region occupying a huge fraction of the image.
    if area > image_area * 0.20:
        return False

    return True


# ============================================================
# Build a mask for ONE assumed foreground polarity
# ============================================================

def build_mask(gray, foreground):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu
    if foreground == "bright":
        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Adaptive threshold gives a useful fallback when illumination
    # is uneven.
    if foreground == "bright":
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5
        )
    else:
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5
        )

    # Otsu is the primary mask. Adaptive threshold is used only
    # where it overlaps Otsu, which avoids turning text/noise into
    # a giant foreground region.
    mask = cv2.bitwise_and(
        otsu,
        adaptive
    )

    # If the overlap is too restrictive, retain Otsu.
    if cv2.countNonZero(mask) < max(
        100,
        int(0.001 * gray.size)
    ):
        mask = otsu

    # Morphological cleanup.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        small_kernel,
        iterations=1
    )

    medium_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        medium_kernel,
        iterations=2
    )

    return mask


# ============================================================
# Remove obviously invalid connected components
# ============================================================

def clean_components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, count):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Keep objects touching the border out of the final result.
        # They are usually incomplete grains or background.
        if x <= 0 or y <= 0:
            continue

        if x + w >= mask.shape[1] - 1:
            continue

        if y + h >= mask.shape[0] - 1:
            continue

        component = np.uint8(labels == label) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        if plausible_grain(
            contour,
            mask.shape
        ):
            cleaned[labels == label] = 255

    return cleaned


# ============================================================
# Automatically choose bright or dark foreground
# ============================================================

def choose_best_mask(gray):

    h, w = gray.shape

    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    border_median = float(
        np.median(border)
    )

    border_mean = float(
        np.mean(border)
    )

    # Strongly dark border -> likely bright grains.
    if border_median < 60 and border_mean < 70:
        candidates = ["bright"]

    # Strongly bright border -> likely dark grains.
    elif border_median > 195 and border_mean > 185:
        candidates = ["dark"]

    else:
        # Unknown lighting: test BOTH polarities.
        candidates = ["bright", "dark"]

    best_mask = None
    best_contours = []
    best_score = -1e9
    best_polarity = None

    for polarity in candidates:

        raw = build_mask(
            gray,
            polarity
        )

        cleaned = clean_components(
            raw
        )

        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = [
            c for c in contours
            if plausible_grain(
                c,
                gray.shape
            )
        ]

        # Score by number of plausible grains, but penalize a
        # foreground area that is suspiciously large.
        count_score = min(
            len(contours),
            300
        )

        foreground_fraction = (
            cv2.countNonZero(cleaned)
            / float(gray.size)
        )

        penalty = 0

        if foreground_fraction > 0.35:
            penalty += 150

        elif foreground_fraction > 0.20:
            penalty += 40

        # Very tiny foreground is also suspicious.
        if foreground_fraction < 0.00005:
            penalty += 20

        score = (
            count_score * 10
            - penalty
        )

        if score > best_score:
            best_score = score
            best_mask = cleaned
            best_contours = contours
            best_polarity = polarity

    if best_mask is None:
        best_mask = np.zeros_like(gray)
        best_contours = []

    return (
        best_mask,
        best_contours,
        best_polarity
    )


# ============================================================
# Optional watershed separation
# ============================================================

def split_large_components(mask, contours):

    if not contours:
        return mask, contours

    areas = [
        cv2.contourArea(c)
        for c in contours
    ]

    median_area = float(
        np.median(areas)
    )

    if median_area <= 0:
        return mask, contours

    output_mask = np.zeros_like(mask)
    final_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Normal isolated grain: retain it directly.
        if area <= median_area * LARGE_COMPONENT_FACTOR:

            cv2.drawContours(
                output_mask,
                [contour],
                -1,
                255,
                -1
            )

            final_contours.append(contour)
            continue

        # Large component: try to separate touching grains.
        component = np.zeros_like(mask)

        cv2.drawContours(
            component,
            [contour],
            -1,
            255,
            -1
        )

        distance = cv2.distanceTransform(
            component,
            cv2.DIST_L2,
            5
        )

        maximum = float(
            distance.max()
        )

        if maximum <= 0:
            cv2.drawContours(
                output_mask,
                [contour],
                -1,
                255,
                -1
            )
            final_contours.append(contour)
            continue

        _, sure_fg = cv2.threshold(
            distance,
            0.38 * maximum,
            255,
            cv2.THRESH_BINARY
        )

        sure_fg = np.uint8(
            sure_fg
        )

        sure_bg = cv2.dilate(
            component,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3)
            ),
            iterations=2
        )

        unknown = cv2.subtract(
            sure_bg,
            sure_fg
        )

        n_markers, markers = cv2.connectedComponents(
            sure_fg
        )

        # If there is only one foreground marker, watershed
        # cannot meaningfully split the component.
        if n_markers <= 2:

            cv2.drawContours(
                output_mask,
                [contour],
                -1,
                255,
                -1
            )
            final_contours.append(contour)
            continue

        markers = markers + 1
        markers[unknown == 255] = 0

        temp = cv2.cvtColor(
            component,
            cv2.COLOR_GRAY2BGR
        )

        cv2.watershed(
            temp,
            markers
        )

        for marker_id in range(
            2,
            n_markers + 1
        ):

            region = (
                np.uint8(
                    markers == marker_id
                ) * 255
            )

            region_contours, _ = cv2.findContours(
                region,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not region_contours:
                continue

            part = max(
                region_contours,
                key=cv2.contourArea
            )

            if plausible_grain(
                part,
                mask.shape
            ):
                cv2.drawContours(
                    output_mask,
                    [part],
                    -1,
                    255,
                    -1
                )
                final_contours.append(part)

    return (
        output_mask,
        final_contours
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_features(contour, gray):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(
        contour
    )

    length = max(
        w,
        h
    )

    width = min(
        w,
        h
    )

    aspect = ratio(
        length,
        width
    )

    circularity = ratio(
        4 * math.pi * area,
        perimeter * perimeter
    )

    equivalent_diameter = math.sqrt(
        ratio(
            4 * area,
            math.pi
        )
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    solidity = ratio(
        area,
        hull_area
    )

    rect = cv2.minAreaRect(
        contour
    )

    rw, rh = rect[1]

    if rw > 0 and rh > 0:
        oriented_length = max(
            rw,
            rh
        )
        oriented_width = min(
            rw,
            rh
        )
    else:
        oriented_length = length
        oriented_width = width

    grain_mask = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    cv2.drawContours(
        grain_mask,
        [contour],
        -1,
        255,
        -1
    )

    mean, std = cv2.meanStdDev(
        gray,
        mask=grain_mask
    )

    return {
        "Area": round(float(area), 2),
        "Perimeter": round(float(perimeter), 2),
        "Length": round(float(oriented_length), 2),
        "Width": round(float(oriented_width), 2),
        "Aspect_Ratio": round(float(aspect), 3),
        "Circularity": round(float(circularity), 3),
        "Equivalent_Diameter": round(
            float(equivalent_diameter),
            2
        ),
        "Solidity": round(float(solidity), 3),
        "Mean_Intensity": round(
            float(mean[0][0]),
            2
        ),
        "Intensity_Std": round(
            float(std[0][0]),
            2
        ),
        "X": int(x),
        "Y": int(y),
        "W": int(w),
        "H": int(h)
    }


# ============================================================
# Preliminary quality classification
# ============================================================

def classify_grain(row, median_area):

    score = 100.0

    area = float(row["Area"])
    aspect = float(row["Aspect_Ratio"])
    solidity = float(row["Solidity"])
    circularity = float(row["Circularity"])
    intensity_std = float(row["Intensity_Std"])

    # Relative size instead of a universal fixed grain area.
    if median_area > 0:

        size_ratio = area / median_area

        if size_ratio < 0.55 or size_ratio > 1.75:
            score -= 25

        elif size_ratio < 0.70 or size_ratio > 1.45:
            score -= 10

    if aspect < 1.25 or aspect > 6.5:
        score -= 12

    if solidity < 0.72:
        score -= 20

    elif solidity < 0.85:
        score -= 8

    if circularity < 0.18:
        score -= 10

    elif circularity < 0.28:
        score -= 5

    if intensity_std > 50:
        score -= 15

    elif intensity_std > 35:
        score -= 7

    score = max(
        0,
        min(
            100,
            round(score)
        )
    )

    if score >= 78:
        quality = "Good"

    elif score >= 55:
        quality = "Mixed"

    else:
        quality = "Defective"

    return pd.Series([
        score,
        quality
    ])


# ============================================================
# Graphs
# ============================================================

def save_plots(df, output_dir, names):

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
                )
            )
        )

        plt.hist(
            df["Area"],
            bins=bins
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
            names["area"]
        ),
        dpi=150
    )

    plt.close()

    plt.figure(
        figsize=(7, 4.5)
    )

    if not df.empty:

        order = [
            "Good",
            "Mixed",
            "Defective"
        ]

        (
            df["Quality"]
            .value_counts()
            .reindex(
                order,
                fill_value=0
            )
            .plot(
                kind="bar"
            )
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
            names["quality"]
        ),
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN FUNCTION USED BY app.py
# ============================================================

def analyze_grain_image(
    input_path,
    output_dir,
    job_id
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    image = cv2.imread(
        input_path
    )

    if image is None:
        raise ValueError(
            "OpenCV could not read the uploaded image."
        )

    # --------------------------------------------------------
    # 2. Detect foreground automatically
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mask, contours, polarity = choose_best_mask(
        gray
    )

    # --------------------------------------------------------
    # 3. Separate touching grains when necessary
    # --------------------------------------------------------

    mask, contours = split_large_components(
        mask,
        contours
    )

    # Final validation.
    contours = [
        c for c in contours
        if plausible_grain(
            c,
            image.shape
        )
    ]

    # Sort top-to-bottom / left-to-right.
    contours.sort(
        key=lambda c: (
            cv2.boundingRect(c)[1],
            cv2.boundingRect(c)[0]
        )
    )

    # --------------------------------------------------------
    # 4. Features
    # --------------------------------------------------------

    rows = []

    for grain_id, contour in enumerate(
        contours,
        start=1
    ):

        row = extract_features(
            contour,
            gray
        )

        row["Grain_ID"] = grain_id

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    if not df.empty:

        columns = [
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
            "H"
        ]

        df = df[columns]

        median_area = float(
            df["Area"].median()
        )

        df[
            [
                "Quality_Score",
                "Quality"
            ]
        ] = df.apply(
            lambda row:
                classify_grain(
                    row,
                    median_area
                ),
            axis=1
        )

    else:

        df["Quality_Score"] = pd.Series(
            dtype=float
        )

        df["Quality"] = pd.Series(
            dtype=str
        )

    # --------------------------------------------------------
    # 5. Annotated image
    # --------------------------------------------------------

    annotated = image.copy()

    for index, contour in enumerate(
        contours
    ):

        grain_id = index + 1

        quality = (
            str(
                df.iloc[index]["Quality"]
            )
            if not df.empty
            else "Unknown"
        )

        if quality == "Good":
            color = (
                60,
                190,
                80
            )

        elif quality == "Mixed":
            color = (
                0,
                165,
                255
            )

        elif quality == "Defective":
            color = (
                40,
                60,
                220
            )

        else:
            color = (
                255,
                255,
                255
            )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.drawContours(
            annotated,
            [contour],
            -1,
            color,
            2
        )

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            color,
            1
        )

        cv2.putText(
            annotated,
            str(grain_id),
            (
                x,
                max(
                    y - 8,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # 6. Files
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
            f"{job_id}_quality.png"
    }

    cv2.imwrite(
        os.path.join(
            output_dir,
            names["annotated"]
        ),
        annotated
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            names["mask"]
        ),
        mask
    )

    df.to_csv(
        os.path.join(
            output_dir,
            names["features"]
        ),
        index=False
    )

    # --------------------------------------------------------
    # 7. Summary
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
                    2
                )
        }
    ])

    summary.to_csv(
        os.path.join(
            output_dir,
            names["summary"]
        ),
        index=False
    )

    # --------------------------------------------------------
    # 8. Graphs
    # --------------------------------------------------------

    save_plots(
        df,
        output_dir,
        names
    )

    # --------------------------------------------------------
    # 9. Return to Flask
    # --------------------------------------------------------

    return {
        "total":
            total,

        "good":
            good,

        # Existing results.html may still use "average".
        # Keep it as an alias for Mixed.
        "average":
            mixed,

        "mixed":
            mixed,

        "defective":
            defective,

        "quality_percentage":
            round(
                quality_percentage,
                2
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
            )
    }
