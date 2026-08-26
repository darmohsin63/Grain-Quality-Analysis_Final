import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# GrainVision - Grain Detection & Quality Analysis
#
# Detection strategy:
# 1. Build foreground mask from intensity + saturation
# 2. Remove small noise
# 3. Fill/close grain regions
# 4. Reject obvious text/background components using shape
# 5. Use watershed to separate touching grains
# 6. Extract geometric + intensity features
# 7. Apply preliminary Good/Mixed/Defective rules
#
# NOTE:
# Quality labels are preliminary image-processing rules.
# They are NOT scientifically validated grades until calibrated
# with a labelled Good/Mixed/Defective dataset.
# ============================================================


# Conservative limits for candidate filtering.
MIN_CANDIDATE_AREA = 120
MAX_CANDIDATE_AREA = 200000

# Shape limits. These are deliberately broad enough for rice/wheat/
# similar grains while rejecting most text fragments and thin objects.
MIN_SOLIDITY = 0.72
MIN_CIRCULARITY = 0.18
MIN_ASPECT = 1.25
MAX_ASPECT = 7.0

# Touching-grain watershed settings.
DISTANCE_RATIO = 0.38


def safe_ratio(a, b):
    return float(a / b) if b else 0.0


def contour_features(contour, gray):
    """Extract geometric and intensity features from one contour."""

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    length = max(w, h)
    width = min(w, h)

    aspect_ratio = safe_ratio(length, width)

    circularity = (
        safe_ratio(4.0 * math.pi * area, perimeter * perimeter)
        if perimeter > 0
        else 0.0
    )

    equivalent_diameter = math.sqrt(
        safe_ratio(4.0 * area, math.pi)
    )

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = safe_ratio(area, hull_area)

    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]

    if rw > 0 and rh > 0:
        oriented_length = max(rw, rh)
        oriented_width = min(rw, rh)
    else:
        oriented_length = length
        oriented_width = width

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    mean_intensity = cv2.mean(gray, mask=mask)[0]

    # Standard deviation is useful for detecting uneven/defective
    # surfaces without pretending it is a laboratory measurement.
    intensity_std = cv2.meanStdDev(gray, mask=mask)[1][0][0]

    return {
        "Area": round(float(area), 2),
        "Perimeter": round(float(perimeter), 2),
        "Length": round(float(oriented_length), 2),
        "Width": round(float(oriented_width), 2),
        "Aspect_Ratio": round(float(aspect_ratio), 3),
        "Circularity": round(float(circularity), 3),
        "Equivalent_Diameter": round(float(equivalent_diameter), 2),
        "Solidity": round(float(solidity), 3),
        "Mean_Intensity": round(float(mean_intensity), 2),
        "Intensity_Std": round(float(intensity_std), 2),
        "X": int(x),
        "Y": int(y),
        "W": int(w),
        "H": int(h),
    }


def create_grain_mask(image):
    """
    Create a foreground mask.

    The previous implementation used a broad OR between Otsu and
    saturation masks. That can classify dark text/logos as grains.

    Here we:
      - estimate foreground from intensity,
      - optionally use saturation as supporting evidence,
      - perform stronger morphology,
      - remove components that are clearly tiny noise.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mild denoising while preserving grain boundaries.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Darker-than-background foreground.
    _, dark_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Saturation can help when grains are brown/yellow, but it is NOT
    # allowed to dominate because colored text can also be saturated.
    sat_threshold = max(25, int(np.percentile(saturation, 65)))
    saturation_mask = cv2.inRange(
        saturation,
        sat_threshold,
        255,
    )

    # Prefer intensity segmentation. Add saturation only where the
    # intensity mask already indicates a plausible darker foreground.
    combined = cv2.bitwise_and(
        dark_mask,
        cv2.bitwise_or(
            dark_mask,
            saturation_mask,
        ),
    )

    # Remove thin strokes / tiny fragments.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )

    # Close small gaps inside grains.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    # Fill holes.
    flood = combined.copy()
    h, w = combined.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(
        flood,
        flood_mask,
        (0, 0),
        255,
    )

    flood_inv = cv2.bitwise_not(flood)
    filled = combined | flood_inv

    # Connected-component cleanup.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled,
        connectivity=8,
    )

    cleaned = np.zeros_like(filled)

    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]

        if MIN_CANDIDATE_AREA <= area <= MAX_CANDIDATE_AREA:
            cleaned[labels == label] = 255

    return cleaned


def is_plausible_grain(contour, image_shape):
    """
    Reject obvious non-grain contours.

    This is the critical step that the old implementation lacked.
    It prevents arbitrary contours such as text fragments from being
    counted as grains.
    """

    area = cv2.contourArea(contour)

    if area < MIN_CANDIDATE_AREA or area > MAX_CANDIDATE_AREA:
        return False

    h_img, w_img = image_shape[:2]

    x, y, w, h = cv2.boundingRect(contour)

    if w <= 2 or h <= 2:
        return False

    # Extremely thin objects are usually text strokes, lines or noise.
    aspect = safe_ratio(max(w, h), min(w, h))

    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False

    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return False

    circularity = safe_ratio(
        4.0 * math.pi * area,
        perimeter * perimeter,
    )

    if circularity < MIN_CIRCULARITY:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = safe_ratio(area, hull_area)

    if solidity < MIN_SOLIDITY:
        return False

    # A contour occupying nearly the entire image is almost certainly
    # background/foreground leakage rather than one grain.
    image_area = h_img * w_img

    if area > image_area * 0.35:
        return False

    # Very long, very thin components are rejected even if their
    # aspect ratio happens to pass the broad limits above.
    if max(w, h) > min(w_img, h_img) * 0.35 and aspect > 4.5:
        return False

    return True


def split_touching_grains(mask):
    """
    Watershed segmentation.

    If two or more grains touch, a simple external contour returns one
    large contour. Distance-transform watershed separates those objects.
    """

    if cv2.countNonZero(mask) == 0:
        return np.zeros_like(mask), []

    # Sure foreground from distance transform.
    distance = cv2.distanceTransform(
        mask,
        cv2.DIST_L2,
        5,
    )

    max_distance = float(distance.max())

    if max_distance <= 0:
        return mask.copy(), []

    _, sure_fg = cv2.threshold(
        distance,
        DISTANCE_RATIO * max_distance,
        255,
        cv2.THRESH_BINARY,
    )

    sure_fg = np.uint8(sure_fg)

    # Sure background.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
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

    n_markers, markers = cv2.connectedComponents(
        sure_fg,
    )

    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed requires a 3-channel image.
    watershed_input = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.watershed(
        watershed_input,
        markers,
    )

    separated = np.zeros_like(mask)

    contours = []

    # Each positive marker is one separated region.
    for marker_id in range(2, n_markers + 1):

        region = np.uint8(markers == marker_id) * 255

        area = cv2.countNonZero(region)

        if area < MIN_CANDIDATE_AREA:
            continue

        contours_region, _ = cv2.findContours(
            region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours_region:
            continue

        contour = max(
            contours_region,
            key=cv2.contourArea,
        )

        if cv2.contourArea(contour) < MIN_CANDIDATE_AREA:
            continue

        if is_plausible_grain(contour, mask.shape):
            cv2.drawContours(
                separated,
                [contour],
                -1,
                255,
                -1,
            )
            contours.append(contour)

    # If watershed did not produce useful markers, fall back to the
    # cleaned connected components.
    if not contours:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = [
            c
            for c in contours
            if is_plausible_grain(c, mask.shape)
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


def classify_grain(row, population_median_area=None):
    """
    Preliminary Good / Mixed / Defective classification.

    This is intentionally a demonstration classifier.
    It should be calibrated against labelled examples before being
    described as a validated agricultural quality standard.
    """

    score = 100.0

    area = float(row["Area"])
    aspect = float(row["Aspect_Ratio"])
    solidity = float(row["Solidity"])
    circularity = float(row["Circularity"])
    intensity_std = float(row["Intensity_Std"])
    mean_intensity = float(row["Mean_Intensity"])

    # Population-relative size is more useful than a hardcoded
    # universal grain size.
    if population_median_area:
        area_ratio = area / population_median_area

        if area_ratio < 0.55 or area_ratio > 1.75:
            score -= 25
        elif area_ratio < 0.70 or area_ratio > 1.45:
            score -= 10

    # Shape irregularity.
    if aspect < 1.25 or aspect > 6.0:
        score -= 15

    if solidity < 0.82:
        score -= 20
    elif solidity < 0.88:
        score -= 8

    if circularity < 0.25:
        score -= 15
    elif circularity < 0.35:
        score -= 7

    # High within-grain intensity variation can indicate discoloration,
    # cracks, dark spots or surface irregularity.
    if intensity_std > 45:
        score -= 15
    elif intensity_std > 32:
        score -= 7

    # Extremely dark objects are suspicious, but this is not a
    # laboratory defect measurement.
    if mean_intensity < 45:
        score -= 15
    elif mean_intensity < 65:
        score -= 7

    score = max(0, min(100, round(score)))

    if score >= 78:
        quality = "Good"
    elif score >= 55:
        quality = "Mixed"
    else:
        quality = "Defective"

    return pd.Series([score, quality])


def save_plots(df, output_dir, names):
    """Create analysis graphs."""

    # Area distribution.
    plt.figure(figsize=(7, 4.5))

    if not df.empty:
        bins = min(12, max(3, int(np.sqrt(len(df)))))
        plt.hist(
            df["Area"],
            bins=bins,
        )

    plt.xlabel("Grain Area (pixels)")
    plt.ylabel("Number of Grains")
    plt.title("Grain Area Distribution")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, names["area"]),
        dpi=150,
    )

    plt.close()

    # Quality distribution.
    plt.figure(figsize=(7, 4.5))

    if not df.empty:
        order = ["Good", "Mixed", "Defective"]
        counts = (
            df["Quality"]
            .value_counts()
            .reindex(order, fill_value=0)
        )

        counts.plot(kind="bar")

    plt.xlabel("Quality Category")
    plt.ylabel("Number of Grains")
    plt.title("Grain Quality Distribution")
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, names["quality"]),
        dpi=150,
    )

    plt.close()


def analyze_grain_image(input_path, output_dir, job_id):

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Read image
    # ------------------------------------------------------------

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(
            "OpenCV could not read the uploaded image."
        )

    # ------------------------------------------------------------
    # 2. Segmentation
    # ------------------------------------------------------------

    initial_mask = create_grain_mask(image)

    # ------------------------------------------------------------
    # 3. Shape filtering + watershed
    # ------------------------------------------------------------

    clean_mask, valid_contours = split_touching_grains(
        initial_mask
    )

    # ------------------------------------------------------------
    # 4. Sort top-to-bottom, left-to-right
    # ------------------------------------------------------------

    valid_contours.sort(
        key=lambda c: (
            cv2.boundingRect(c)[1],
            cv2.boundingRect(c)[0],
        )
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # ------------------------------------------------------------
    # 5. Feature extraction
    # ------------------------------------------------------------

    results = []

    for grain_id, contour in enumerate(
        valid_contours,
        start=1,
    ):

        features = contour_features(
            contour,
            gray,
        )

        features["Grain_ID"] = grain_id

        results.append(features)

    # ------------------------------------------------------------
    # 6. DataFrame
    # ------------------------------------------------------------

    df = pd.DataFrame(results)

    # Keep useful columns in a clean order.
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
            "H",
        ]

        df = df[columns]

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

    # ------------------------------------------------------------
    # 7. Annotated image
    # ------------------------------------------------------------

    annotated = image.copy()

    for _, row in df.iterrows():

        grain_id = int(row["Grain_ID"])

        x = int(row["X"])
        y = int(row["Y"])
        w = int(row["W"])
        h = int(row["H"])

        quality = row["Quality"]

        # BGR colors:
        # Good      -> green
        # Mixed     -> yellow/orange
        # Defective -> red
        if quality == "Good":
            color = (60, 190, 80)
        elif quality == "Mixed":
            color = (0, 180, 255)
        else:
            color = (40, 60, 220)

        # Find corresponding contour.
        contour = valid_contours[grain_id - 1]

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

        label = f"{grain_id} {quality}"

        cv2.putText(
            annotated,
            label,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # 8. Output names
    # ------------------------------------------------------------

    names = {
        "annotated": f"{job_id}_annotated.jpg",
        "mask": f"{job_id}_mask.png",
        "features": f"{job_id}_features.csv",
        "summary": f"{job_id}_summary.csv",
        "area": f"{job_id}_area.png",
        "quality": f"{job_id}_quality.png",
    }

    # ------------------------------------------------------------
    # 9. Save images
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # 10. Save features
    # ------------------------------------------------------------

    df.to_csv(
        os.path.join(
            output_dir,
            names["features"],
        ),
        index=False,
    )

    # ------------------------------------------------------------
    # 11. Statistics
    # ------------------------------------------------------------

    total = len(df)

    if total:
        good = int(
            (df["Quality"] == "Good").sum()
        )

        mixed = int(
            (df["Quality"] == "Mixed").sum()
        )

        defective = int(
            (df["Quality"] == "Defective").sum()
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

    # ------------------------------------------------------------
    # 12. Save summary
    # ------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "Total Grains": total,
                "Good Grains": good,
                "Mixed Grains": mixed,
                "Defective Grains": defective,
                "Quality Percentage": round(
                    quality_percentage,
                    2,
                ),
            }
        ]
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            names["summary"],
        ),
        index=False,
    )

    # ------------------------------------------------------------
    # 13. Graphs
    # ------------------------------------------------------------

    save_plots(
        df,
        output_dir,
        names,
    )

    # ------------------------------------------------------------
    # 14. Return result to Flask
    # ------------------------------------------------------------

    return {
        "total": total,
        "good": good,
        "average": mixed,  # keeps old template compatible
        "mixed": mixed,
        "defective": defective,
        "quality_percentage": round(
            quality_percentage,
            2,
        ),
        "annotated_image": names["annotated"],
        "mask_image": names["mask"],
        "features_csv": names["features"],
        "summary_csv": names["summary"],
        "area_plot": names["area"],
        "quality_plot": names["quality"],
        "features": df.to_dict(
            orient="records"
        ),
    }

