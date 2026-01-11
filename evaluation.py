import os
import cv2
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# Set Matplotlib backend
matplotlib.use('TkAgg')


# =============================================================================
# ⚙️ CONFIGURATION & CONSTANTS
# =============================================================================

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

IMG_DIR = cfg['INPUT_PATHS']['RAW_ZIP_FILE']
TXT_PATH = cfg['INPUT_PATHS']['RAW_LIST_FILE']
MODEL_DIR = cfg['MODEL']['MODEL_DIR']

# IDs specifically selected for detailed analysis
TARGET_IDS = [295, 300, 685, 660, 624, 687, 576, 590, 262, 418, 653, 753,
              598, 614, 654, 717, 473, 480, 96, 108, 642, 657, 337, 554, 476]

# Threshold to skip "tracklet" matches (images from the same camera sequence)
# Used in Step 1 to find "Hard" matches.
SKIP_SIMILARITY_THRESHOLD = 0.92


# =============================================================================
# 🛠️ SHARED UTILITY FUNCTIONS
# =============================================================================

def load_dataset_metadata(txt_path):
    print(f"📄 Reading Dataset List: {txt_path}")
    df = pd.read_csv(txt_path, sep=r'\s+', header=None, names=["image", "label"])
    # Ensure extension exists
    if not str(df['image'].iloc[0]).lower().endswith(('.jpg', '.png', '.jpeg')):
        df['image'] = df['image'].astype(str) + '.jpg'
    return df


def extract_features(model_fn, df, img_dir, desc="Extracting Features"):
    """
    Generic feature extraction function used by all steps.
    """
    feats, labels, paths = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        path = os.path.join(img_dir, row['image'])
        if not os.path.exists(path): continue

        try:
            img = load_img(path, target_size=(224, 224))
            x = preprocess_input(img_to_array(img))
            preds = model_fn(tf.expand_dims(x, axis=0))
            key = list(preds.keys())[0]
            # L2 Normalize
            feat = tf.nn.l2_normalize(preds[key], axis=1).numpy()

            feats.append(feat[0])
            labels.append(row['label'])
            paths.append(path)
        except Exception:
            continue
    return np.array(feats), np.array(labels), np.array(paths)


def prepare_query_gallery_split(df, target_ids=None):
    """
    Splits data into Query and Gallery.
    If target_ids is provided, limits Query to those IDs.
    """
    query_list, gallery_list = [], []
    grouped = df.groupby('label')

    for label, group in grouped:
        imgs = group['image'].tolist()

        # Check if we should use this ID as a query
        use_as_query = True
        if target_ids is not None and label not in target_ids:
            use_as_query = False

        if use_as_query and len(imgs) >= 2:
            # Pick 1 random image for query
            q = np.random.choice(imgs, 1, replace=False)[0]
            query_list.append({'image': q, 'label': label})
            # Add ALL images to gallery (filtering happens at distance calculation)
            for img in imgs:
                gallery_list.append({'image': img, 'label': label})
        else:
            # Just add to gallery
            gallery_list.extend([{'image': i, 'label': label} for i in imgs])

    return pd.DataFrame(query_list), pd.DataFrame(gallery_list)


# =============================================================================
# 1️⃣ STEP 1: HARD RE-ID ANALYSIS (Specific Targets & Thresholding)
# =============================================================================
def run_hard_reid_analysis(infer, df_full):
    print("\n" + "=" * 50)
    print("🚀 STEP 1: Hard Re-ID Analysis (Tracklet Filtering)")
    print("=" * 50)

    df_q, df_g = prepare_query_gallery_split(df_full, target_ids=TARGET_IDS)

    if df_q.empty:
        print("⚠️ Warning: No Target IDs found for Step 1.")
        return

    q_feats, q_lbls, q_paths = extract_features(infer, df_q, IMG_DIR, desc="Step 1 Feat. Extraction")
    g_feats, g_lbls, g_paths = extract_features(infer, df_g, IMG_DIR, desc="Step 1 Feat. Extraction")

    dists = cosine_distances(q_feats, g_feats)
    results = []

    print(f"🔍 Analyzing matches (Skipping similarity > {SKIP_SIMILARITY_THRESHOLD})...")

    for i in range(len(q_lbls)):
        sorted_idx = np.argsort(dists[i])
        q_filename = os.path.basename(q_paths[i])

        # --- Find Hardest Positive ---
        top1_idx = -1
        for idx in sorted_idx:
            g_filename = os.path.basename(g_paths[idx])
            similarity = 1 - dists[i, idx]
            is_same_id = (g_lbls[idx] == q_lbls[i])

            if q_filename == g_filename: continue  # Skip self
            if is_same_id and similarity > SKIP_SIMILARITY_THRESHOLD: continue  # Skip easy tracklets

            top1_idx = idx
            break

        # Fallback if everything was filtered
        if top1_idx == -1:
            top1_idx = sorted_idx[0]
            if os.path.basename(g_paths[top1_idx]) == q_filename:
                top1_idx = sorted_idx[1]

        # --- Find Best Ground Truth ---
        true_matches = np.where(g_lbls == q_lbls[i])[0]
        best_true_idx = top1_idx

        if len(true_matches) > 0:
            valid_indices = []
            for tm in true_matches:
                t_name = os.path.basename(g_paths[tm])
                t_sim = 1 - dists[i, tm]
                if t_name != q_filename and t_sim <= SKIP_SIMILARITY_THRESHOLD:
                    valid_indices.append(tm)

            if valid_indices:
                sub_dists = dists[i, valid_indices]
                best_true_idx = valid_indices[np.argmin(sub_dists)]
            else:
                # Fallback logic for ground truth
                best_true_idx = true_matches[np.argmin(dists[i, true_matches])]
                if os.path.basename(g_paths[best_true_idx]) == q_filename and len(true_matches) > 1:
                    best_true_idx = true_matches[np.argsort(dists[i, true_matches])[1]]

        results.append({
            'q_path': q_paths[i], 'q_id': q_lbls[i],
            'p_path': g_paths[top1_idx], 'p_id': g_lbls[top1_idx], 'p_sim': 1 - dists[i, top1_idx],
            't_path': g_paths[best_true_idx], 't_sim': 1 - dists[i, best_true_idx],
            'correct': (g_lbls[top1_idx] == q_lbls[i])
        })

    # Visualization
    if results:
        fig, axes = plt.subplots(len(results), 3, figsize=(18, 5 * len(results)))
        if len(results) == 1: axes = np.expand_dims(axes, axis=0)

        cols = ["Query", f"Rank-1 (Sim < {SKIP_SIMILARITY_THRESHOLD})", "Hardest Ground Truth"]
        for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=16, fontweight='bold', pad=20)

        for i, res in enumerate(results):
            # 1. Query
            axes[i, 0].imshow(cv2.cvtColor(cv2.imread(res['q_path']), cv2.COLOR_BGR2RGB))
            axes[i, 0].set_xlabel(f"ID: {res['q_id']}", fontweight='bold', fontsize=12)

            # 2. Prediction
            color = 'green' if res['correct'] else 'red'
            status = "CORRECT" if res['correct'] else "WRONG"
            img_p = cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(res['p_path']), cv2.COLOR_BGR2RGB),
                                       20, 20, 20, 20, cv2.BORDER_CONSTANT,
                                       value=[0, 255, 0] if res['correct'] else [255, 0, 0])
            axes[i, 1].imshow(img_p)
            axes[i, 1].set_xlabel(f"ID: {res['p_id']}\nScore: {res['p_sim']:.3f}\n[{status}]", color=color,
                                  fontweight='bold', fontsize=12)

            # 3. Ground Truth
            axes[i, 2].imshow(cv2.cvtColor(cv2.imread(res['t_path']), cv2.COLOR_BGR2RGB))
            axes[i, 2].set_xlabel(f"True ID: {res['q_id']}\nScore: {res['t_sim']:.3f}", color='green',
                                  fontweight='bold', fontsize=12)

            for j in range(3):
                axes[i, j].set_xticks([]);
                axes[i, j].set_yticks([])

        plt.tight_layout()
        plt.savefig("Hard_ReID_Analysis.png", dpi=150)
        print("✅ Saved: 'Hard_ReID_Analysis.png'")
        # plt.show() # Uncomment to see popup


# =============================================================================
# 2️⃣ STEP 2: VISUAL RANK LIST (Target IDs Rank-1 to 5)
# =============================================================================
def run_rank_list_visualization(infer, df_full):
    print("\n" + "=" * 50)
    print("🚀 STEP 2: Visual Rank List Analysis (Top-5)")
    print("=" * 50)

    df_q, df_g = prepare_query_gallery_split(df_full, target_ids=TARGET_IDS)  # Using reuse logic
    # Note: df_q here has one random query per target ID

    g_feats, g_lbls, g_paths = extract_features(infer, df_g, IMG_DIR, desc="Step 2 Feat. Extraction")

    # We need to find the index of our specific target queries within the gallery features
    # since we are comparing gallery-to-gallery basically

    valid_indices = []
    for tid in TARGET_IDS:
        indices = np.where(g_lbls == tid)[0]
        if len(indices) > 0:
            valid_indices.append(indices[0])  # Take first occurrence as query

    if not valid_indices:
        print("❌ No valid targets found for Step 2.")
        return

    num_rows = len(valid_indices)
    top_k = 5
    fig_height = max(5, num_rows * 2.5)
    fig, axes = plt.subplots(num_rows, top_k + 1, figsize=(15, fig_height))
    if num_rows == 1: axes = np.expand_dims(axes, axis=0)

    # Titles
    cols = ["Query"] + [f"Rank-{i + 1}" for i in range(top_k)]
    for ax, col in zip(axes[0], cols): ax.set_title(col, fontweight='bold')

    print("📊 Plotting Rank Lists...")
    for i in range(num_rows):
        q_idx = valid_indices[i]
        q_feat = g_feats[q_idx]
        q_id = g_lbls[q_idx]

        # Calculate distance
        dists = cosine_distances(q_feat.reshape(1, -1), g_feats)[0]
        sorted_indices = np.argsort(dists)

        # Filter self
        filtered_indices = [idx for idx in sorted_indices if idx != q_idx]

        # Plot Query
        axes[i, 0].imshow(cv2.cvtColor(cv2.imread(g_paths[q_idx]), cv2.COLOR_BGR2RGB))
        axes[i, 0].set_ylabel(f"Q_ID: {q_id}", fontweight='bold', rotation=90)
        axes[i, 0].set_xticks([]);
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values(): spine.set_edgecolor('blue'); spine.set_linewidth(3)

        # Plot Top-K
        count = min(top_k, len(filtered_indices))
        for r in range(count):
            idx = filtered_indices[r]
            match_id = g_lbls[idx]
            score = dists[idx]

            axes[i, r + 1].imshow(cv2.cvtColor(cv2.imread(g_paths[idx]), cv2.COLOR_BGR2RGB))
            color = 'green' if match_id == q_id else 'red'
            axes[i, r + 1].set_xlabel(f"{score:.3f}", color=color, fontweight='bold')
            axes[i, r + 1].set_xticks([]);
            axes[i, r + 1].set_yticks([])
            for spine in axes[i, r + 1].spines.values(): spine.set_edgecolor(color); spine.set_linewidth(3)

    plt.tight_layout()
    plt.savefig("All_Targets_Analysis.png", dpi=150)
    print("✅ Saved: 'All_Targets_Analysis.png'")


# =============================================================================
# 3️⃣ STEP 3: HARDEST PAIRS (Global Similarity Matrix)
# =============================================================================
def run_hardest_pairs_analysis(infer, df_full):
    print("\n" + "=" * 50)
    print("🚀 STEP 3: Global Hardest Pairs Analysis (Top-50)")
    print("=" * 50)

    # We use the full dataset for this
    all_feats, all_labels, all_paths = extract_features(infer, df_full, IMG_DIR, desc="Step 3 Feat. Extraction")

    print(f"🔍 Calculating Similarity Matrix for {len(all_feats)} images...")
    sim_matrix = cosine_similarity(all_feats)

    results = []
    processed_pairs = set()

    print("📊 Finding hardest negative pairs (Different ID, High Similarity)...")
    # Optimize loop: iterate upper triangle only
    num_imgs = len(all_labels)
    for i in range(num_imgs):
        for j in range(i + 1, num_imgs):
            id1, id2 = all_labels[i], all_labels[j]

            # We are looking for DIFFERENT IDs that look SIMILAR
            if id1 != id2:
                pair = tuple(sorted((id1, id2)))  # Unique pair check
                if pair not in processed_pairs:
                    val = sim_matrix[i, j]
                    # Threshold optimization: only keep very high similarities to save memory
                    if val > 0.5:
                        results.append({
                            "ID_1": id1, "ID_2": id2,
                            "Similarity": val,
                            "Cosine_Distance": 1 - val,
                            "File_1": os.path.basename(all_paths[i]),
                            "File_2": os.path.basename(all_paths[j])
                        })
                        processed_pairs.add(pair)

    if results:
        df_final = pd.DataFrame(results).sort_values(by="Similarity", ascending=False).head(50)

        print("\n" + "-" * 70)
        print("🔥 THE 50 MOST SIMILAR PAIRS (DIFFERENT IDs) 🔥")
        print("-" * 70)
        pd.options.display.float_format = '{:.6f}'.format
        print(df_final.to_string(index=False))

        output_file = "hardest_pairs.csv"
        df_final.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to '{output_file}'")
    else:
        print("No pairs found matching criteria.")


# =============================================================================
# 4️⃣ STEP 4: FAILURE ANALYSIS (Random Samples)
# =============================================================================
def run_random_failure_analysis(infer, df_full):
    print("\n" + "=" * 50)
    print("🚀 STEP 4: Random Failure Case Analysis")
    print("=" * 50)

    num_failures_needed = 4
    # Split randomly (no target ID restriction)
    df_q, df_g = prepare_query_gallery_split(df_full, target_ids=None)

    q_feats, q_lbls, q_paths = extract_features(infer, df_q, IMG_DIR, desc="Step 4 Feat. Extraction")
    g_feats, g_lbls, g_paths = extract_features(infer, df_g, IMG_DIR, desc="Step 4 Feat. Extraction")

    dists = cosine_distances(q_feats, g_feats)
    failure_cases = []

    for i in range(len(q_lbls)):
        sorted_idx = np.argsort(dists[i])
        top1_idx = sorted_idx[0]

        # Check for failure (Rank-1 ID != Query ID)
        if g_lbls[top1_idx] != q_lbls[i]:

            # Find ground truth for comparison
            true_matches = np.where(g_lbls == q_lbls[i])[0]
            if len(true_matches) > 0:
                best_true_idx = true_matches[np.argmin(dists[i, true_matches])]

                failure_cases.append({
                    'q_path': q_paths[i], 'q_id': q_lbls[i],
                    'f_path': g_paths[top1_idx], 'f_id': g_lbls[top1_idx], 'f_sim': 1 - dists[i, top1_idx],
                    't_path': g_paths[best_true_idx], 't_sim': 1 - dists[i, best_true_idx]
                })

        if len(failure_cases) >= num_failures_needed: break

    if failure_cases:
        fig, axes = plt.subplots(len(failure_cases), 3, figsize=(15, 5 * len(failure_cases)))
        if len(failure_cases) == 1: axes = np.expand_dims(axes, axis=0)

        cols = ["Query", "False Rank-1", "Ground Truth"]
        for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=14, fontweight='bold')

        for i, case in enumerate(failure_cases):
            # Query
            axes[i, 0].imshow(cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(case['q_path']), cv2.COLOR_BGR2RGB),
                                                 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[0, 0, 255]))
            axes[i, 0].set_xlabel(f"True ID: {case['q_id']}", fontweight='bold')

            # False Match
            axes[i, 1].imshow(cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(case['f_path']), cv2.COLOR_BGR2RGB),
                                                 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 0, 0]))
            axes[i, 1].set_xlabel(f"False ID: {case['f_id']}\nSim: {case['f_sim']:.3f}", color='red', fontweight='bold')

            # True Match
            axes[i, 2].imshow(cv2.copyMakeBorder(cv2.cvtColor(cv2.imread(case['t_path']), cv2.COLOR_BGR2RGB),
                                                 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[0, 255, 0]))
            axes[i, 2].set_xlabel(f"True ID: {case['q_id']}\nSim: {case['t_sim']:.3f}", color='green',
                                  fontweight='bold')

            for j in range(3): axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

        plt.tight_layout()
        plt.savefig("Failure_Analysis_Results.png", dpi=300)
        print("✅ Saved: 'Failure_Analysis_Results.png'")
    else:
        print("🎉 No failures detected in random sampling!")


# =============================================================================
# 🚀 MAIN EXECUTION
# =============================================================================
def main():
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    print("🏗️ Loading Model Once (Shared)...")
    model = tf.saved_model.load(MODEL_DIR)
    infer = model.signatures["serving_default"]

    df_full = load_dataset_metadata(TXT_PATH)

    # --- EXECUTE ALL STEPS ---
    run_hard_reid_analysis(infer, df_full)
    run_rank_list_visualization(infer, df_full)
    run_hardest_pairs_analysis(infer, df_full)
    run_random_failure_analysis(infer, df_full) 

    print("\n✅✅ ALL ANALYSES COMPLETED SUCCESSFULLY ✅✅")
    plt.show()


if __name__ == "__main__":
    main()

