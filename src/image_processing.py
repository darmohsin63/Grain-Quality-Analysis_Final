import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_AREA = 200
THRESHOLD_VALUE = 100

def classify_grain(row):
    score = 100
    if row["Area"] < 500: score -= 25
    if row["Aspect_Ratio"] < 1.5: score -= 10
    if row["Solidity"] < 0.75: score -= 25
    if row["Mean_Intensity"] < 80: score -= 30
    quality = "Good" if score >= 75 else "Average" if score >= 50 else "Defective"
    return pd.Series([score, quality])

def analyze_grain_image(input_path, output_dir, job_id):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("OpenCV could not read the uploaded image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid_contours = [
        c for c in contours if cv2.contourArea(c) >= MIN_AREA
    ]

    results = []
    for grain_id, contour in enumerate(valid_contours, 1):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        length, width = max(w, h), min(w, h)
        aspect_ratio = length / width if width else 0
        circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter else 0
        equivalent_diameter = math.sqrt(4 * area / math.pi)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area else 0

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]

        results.append({
            "Grain_ID": grain_id,
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2),
            "Length": length,
            "Width": width,
            "Aspect_Ratio": round(aspect_ratio, 3),
            "Circularity": round(circularity, 3),
            "Equivalent_Diameter": round(equivalent_diameter, 2),
            "Solidity": round(solidity, 3),
            "Mean_Intensity": round(mean_intensity, 2)
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df[["Quality_Score", "Quality"]] = df.apply(classify_grain, axis=1)
    else:
        for col in ["Quality_Score", "Quality"]:
            df[col] = []

    annotated = image.copy()
    for grain_id, contour in enumerate(valid_contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 220, 120), 2)
        cv2.putText(
            annotated, str(grain_id), (x, max(y-8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 40, 240), 2
        )

    names = {
        "annotated": f"{job_id}_annotated.jpg",
        "mask": f"{job_id}_mask.png",
        "features": f"{job_id}_features.csv",
        "summary": f"{job_id}_summary.csv",
        "area": f"{job_id}_area.png",
        "quality": f"{job_id}_quality.png"
    }

    cv2.imwrite(os.path.join(output_dir, names["annotated"]), annotated)
    cv2.imwrite(os.path.join(output_dir, names["mask"]), clean_mask)
    df.to_csv(os.path.join(output_dir, names["features"]), index=False)

    total = len(df)
    good = int((df["Quality"] == "Good").sum()) if total else 0
    average = int((df["Quality"] == "Average").sum()) if total else 0
    defective = int((df["Quality"] == "Defective").sum()) if total else 0
    quality_pct = good / total * 100 if total else 0

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
