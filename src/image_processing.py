import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Minimum contour area.
# This removes tiny noise from the image.
MIN_AREA = 300


def classify_grain(row):
    """
    Preliminary rule-based quality classification.

    IMPORTANT:
    These thresholds are demonstration values.
    They should be calibrated using a properly labelled
    grain dataset before being presented as scientifically
    validated quality standards.
    """

    score = 100

    # Very small grain
    if row["Area"] < 500:
        score -= 20

    # Unusual shape
    if row["Aspect_Ratio"] < 1.5:
        score -= 10

    # Irregular boundary
    if row["Solidity"] < 0.75:
        score -= 25

    # Darker grain
    if row["Mean_Intensity"] < 80:
        score -= 20

    if score >= 75:
        quality = "Good"

    elif score >= 50:
        quality = "Average"

    else:
        quality = "Defective"

    return pd.Series([score, quality])


def create_grain_mask(image):
    """
    Creates a more robust grain segmentation mask.

    The previous version used a fixed threshold of 100.
    This version combines:
        1. Otsu grayscale thresholding
        2. Bright-background separation
        3. HSV saturation information
        4. Morphological processing
    """

    # --------------------------------------------------
    # 1. Convert to grayscale
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # 2. Smooth the image
    # --------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------
    # 3. Otsu threshold
    # --------------------------------------------------

    _, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # 4. Bright-background mask
    #
    # Grain pixels are generally darker than a
    # white/light background.
    # --------------------------------------------------

    background_mask = cv2.inRange(
        blurred,
        0,
        235
    )

    # --------------------------------------------------
    # 5. HSV saturation mask
    #
    # Brown/yellow grains generally have more
    # saturation than a white background.
    # --------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    _, saturation_mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------
    # 6. Combine masks
    # --------------------------------------------------

    combined = cv2.bitwise_or(
        otsu_mask,
        saturation_mask
    )

    combined = cv2.bitwise_and(
        combined,
        background_mask
    )

    # --------------------------------------------------
    # 7. Morphological opening
    # Removes small noise
    # --------------------------------------------------

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    opened = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # --------------------------------------------------
    # 8. Morphological closing
    # Fills small gaps inside grains
    # --------------------------------------------------

    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=2
    )

    # --------------------------------------------------
    # 9. Remove small connected components
    # --------------------------------------------------

    number_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8
    )

    cleaned = np.zeros_like(closed)

    for label in range(1, number_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area >= MIN_AREA:

            cleaned[labels == label] = 255

    return cleaned


def analyze_grain_image(
    input_path,
    output_dir,
    job_id
):

    # --------------------------------------------------
    # 1. Read image
    # --------------------------------------------------

    image = cv2.imread(input_path)

    if image is None:
        raise ValueError(
            "OpenCV could not read the uploaded image."
        )

    # --------------------------------------------------
    # 2. Create segmentation mask
    # --------------------------------------------------

    clean_mask = create_grain_mask(image)

    # --------------------------------------------------
    # 3. Find contours
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= MIN_AREA:

            valid_contours.append(contour)

    # --------------------------------------------------
    # 4. Sort contours from top-left to bottom-right
    # --------------------------------------------------

    valid_contours.sort(
        key=lambda contour: (
            cv2.boundingRect(contour)[1],
            cv2.boundingRect(contour)[0]
        )
    )

    # --------------------------------------------------
    # 5. Prepare grayscale image
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    results = []

    # --------------------------------------------------
    # 6. Extract features
    # --------------------------------------------------

    for grain_id, contour in enumerate(
        valid_contours,
        start=1
    ):

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # ----------------------------------------------
        # Bounding rectangle
        # ----------------------------------------------

        x, y, w, h = cv2.boundingRect(
            contour
        )

        length = max(w, h)

        width = min(w, h)

        aspect_ratio = (
            length / width
            if width > 0
            else 0
        )

        # ----------------------------------------------
        # Circularity
        # ----------------------------------------------

        if perimeter > 0:

            circularity = (
                4 * math.pi * area
                / (perimeter ** 2)
            )

        else:

            circularity = 0

        # ----------------------------------------------
        # Equivalent diameter
        # ----------------------------------------------

        equivalent_diameter = math.sqrt(
            4 * area / math.pi
        )

        # ----------------------------------------------
        # Convex hull / solidity
        # ----------------------------------------------

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area > 0:

            solidity = (
                area / hull_area
            )

        else:

            solidity = 0

        # ----------------------------------------------
        # Minimum area rectangle
        # ----------------------------------------------

        rect = cv2.minAreaRect(
            contour
        )

        rect_width = rect[1][0]

        rect_height = rect[1][1]

        if rect_width > 0 and rect_height > 0:

            oriented_length = max(
                rect_width,
                rect_height
            )

            oriented_width = min(
                rect_width,
                rect_height
            )

        else:

            oriented_length = length
            oriented_width = width

        # ----------------------------------------------
        # Grain mask
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Mean intensity
        # ----------------------------------------------

        mean_intensity = cv2.mean(
            gray,
            mask=grain_mask
        )[0]

        # ----------------------------------------------
        # Store features
        # ----------------------------------------------

        results.append({

            "Grain_ID": grain_id,

            "Area": round(
                area,
                2
            ),

            "Perimeter": round(
                perimeter,
                2
            ),

            "Length": round(
                oriented_length,
                2
            ),

            "Width": round(
                oriented_width,
                2
            ),

            "Aspect_Ratio": round(
                aspect_ratio,
                3
            ),

            "Circularity": round(
                circularity,
                3
            ),

            "Equivalent_Diameter": round(
                equivalent_diameter,
                2
            ),

            "Solidity": round(
                solidity,
                3
            ),

            "Mean_Intensity": round(
                mean_intensity,
                2
            )
        })

    # --------------------------------------------------
    # 7. Create DataFrame
    # --------------------------------------------------

    df = pd.DataFrame(
        results
    )

    if not df.empty:

        df[
            [
                "Quality_Score",
                "Quality"
            ]
        ] = df.apply(
            classify_grain,
            axis=1
        )

    else:

        df["Quality_Score"] = []

        df["Quality"] = []

    # --------------------------------------------------
    # 8. Create annotated image
    # --------------------------------------------------

    annotated = image.copy()

    for grain_id, contour in enumerate(
        valid_contours,
        start=1
    ):

        # Draw complete contour
        cv2.drawContours(
            annotated,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (255, 180, 0),
            1
        )

        # Grain number
        cv2.putText(
            annotated,
            str(grain_id),
            (
                x,
                max(y - 8, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------
    # 9. File names
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 10. Save annotated image
    # --------------------------------------------------

    cv2.imwrite(
        os.path.join(
            output_dir,
            names["annotated"]
        ),
        annotated
    )

    # --------------------------------------------------
    # 11. Save binary mask
    # --------------------------------------------------

    cv2.imwrite(
        os.path.join(
            output_dir,
            names["mask"]
        ),
        clean_mask
    )

    # --------------------------------------------------
    # 12. Save feature CSV
    # --------------------------------------------------

    df.to_csv(
        os.path.join(
            output_dir,
            names["features"]
        ),
        index=False
    )

    # --------------------------------------------------
    # 13. Calculate quality statistics
    # --------------------------------------------------

    total = len(df)

    if total:

        good = int(
            (
                df["Quality"]
                == "Good"
            ).sum()
        )

        average = int(
            (
                df["Quality"]
                == "Average"
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
        average = 0
        defective = 0

    quality_percentage = (
        good / total * 100
        if total
        else 0
    )

    # --------------------------------------------------
    # 14. Save summary CSV
    # --------------------------------------------------

    summary = pd.DataFrame([{

        "Total Grains":
            total,

        "Good Grains":
            good,

        "Average Grains":
            average,

        "Defective Grains":
            defective,

        "Quality Percentage":
            round(
                quality_percentage,
                2
            )
    }])

    summary.to_csv(
        os.path.join(
            output_dir,
            names["summary"]
        ),
        index=False
    )

    # --------------------------------------------------
    # 15. Grain area graph
    # --------------------------------------------------

    plt.figure(
        figsize=(7, 4.5)
    )

    if total:

        plt.hist(
            df["Area"],
            bins=min(
                10,
                max(1, total)
            )
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

    # --------------------------------------------------
    # 16. Quality distribution graph
    # --------------------------------------------------

    plt.figure(
        figsize=(7, 4.5)
    )

    if total:

        df[
            "Quality"
        ].value_counts().plot(
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
            names["quality"]
        ),
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 17. Return results
    # --------------------------------------------------

    return {

        "total":
            total,

        "good":
            good,

        "average":
            average,

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
    pd.DataFrame([{
        "Total Grains": total,
        "Good Grains": good,
        "Average Grains": average,
        "Defective Grains": defective,
        "Quality Percentage": round(quality_pct, 2)
    }]).to_csv(os.path.join(output_dir, names["summary"]), index=False)

    plt.figure(figsize=(7, 4.5))
    if total:
        plt.hist(df["Area"], bins=min(10, max(1, total)))
    plt.xlabel("Grain Area (pixels)")
    plt.ylabel("Number of Grains")
    plt.title("Grain Area Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, names["area"]), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    if total:
        df["Quality"].value_counts().plot(kind="bar")
    plt.xlabel("Quality Category")
    plt.ylabel("Number of Grains")
    plt.title("Grain Quality Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, names["quality"]), dpi=150)
    plt.close()

    return {
        "total": total, "good": good, "average": average,
        "defective": defective,
        "quality_percentage": round(quality_pct, 2),
        "annotated_image": names["annotated"],
        "mask_image": names["mask"],
        "features_csv": names["features"],
        "summary_csv": names["summary"],
        "area_plot": names["area"],
        "quality_plot": names["quality"],
        "features": df.to_dict(orient="records")
    }
