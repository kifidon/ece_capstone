# ECE Capstone — Pose Classification

## Steps

1. Download MoveNet Thunder, set up for inference in production, and set up test data. ✅
2. *(In progress)*
3. 
4. 
5. 

## Dataset

We recorded original video clips across 4 pose classes:

| Class | Videos | Total Frames |
|-------|--------|-------------|
| Lying | 5 | 898 |
| Reaching | 5 (incl. 04.1 & 04.2) | 790 |
| Sitting | 7 (incl. 2 dupes) | 1,244 |
| Standing | 7 (incl. 2 dupes) | 1,230 |

Sitting and Standing had significantly more frames than Lying and Reaching. To balance the dataset, we duplicated select clips (e.g. `Lying01dupe.mp4`, `Reaching02dupe.mp4`), bringing the total to **30 videos** with roughly equal frame counts per class:

| Class | Videos (incl. dupes) | Balanced Frames |
|-------|----------------------|-----------------|
| Lying | 7 | 1,267 |
| Reaching | 9 | 1,261 |
| Sitting | 7 | 1,244 |
| Standing | 7 | 1,230 |

## Gotchas

### macOS Python SSL certificates

The macOS Python installer doesn't include SSL certificates, so downloads (e.g. MoveNet from TF Hub) fail with `SSL: CERTIFICATE_VERIFY_FAILED`. Fix by running:

```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```

### `setuptools` / `pkg_resources`

`tensorflow_hub` depends on `pkg_resources`, which was removed in `setuptools` v82+. Pin to an older version:

```bash
pip install "setuptools<81"
```

## Model Training

| Version | Validation | Mean Accuracy | Notes |
|---------|-----------|---------------|-------|
| Baseline | Fixed 70/15/15 | ~49% | Severe overfitting |
| V1.0 | K-Fold (k=5) | 92.6% | Noise augmentation + dropout |
| V1.01 pre-mirror | LOVO (n=30) | 96.7% | Identified Lying04 as the only failure (0%) |
| **V1.01** | **LOVO (n=30)** | **100%** | + mirroring + confidence scores |

See [src/README.md](src/README.md) for full training documentation, architecture details, and usage.

## Design Decisions

### Pose normalization: centering only, no scaling

**Decision:** We center pose keypoints by the hip midpoint but do not apply scale normalization.

**Rationale:**
- **Centering** makes poses translation-invariant: the model sees the same pose whether the person is on the left or right of the frame. This matters because we care about pose shape (squat vs jump vs stand), not camera framing. The hip midpoint is used because hips are usually visible and stable.
- **Scaling** is skipped because it is more fragile. Scale normalization (e.g., dividing by torso length or keypoint spread) can fail with occlusion, side views, or bad keypoints. Centering alone gives most of the benefit with fewer edge cases.
