# ECE Capstone — Pose Classification

## Steps

1. Download MoveNet Thunder, set up for inference in production, and set up test data. ✅
2. *(In progress)*
3. 
4. 
5. 

## Design Decisions

### Pose normalization: centering only, no scaling

**Decision:** We center pose keypoints by the hip midpoint but do not apply scale normalization.

**Rationale:**
- **Centering** makes poses translation-invariant: the model sees the same pose whether the person is on the left or right of the frame. This matters because we care about pose shape (squat vs jump vs stand), not camera framing. The hip midpoint is used because hips are usually visible and stable.
- **Scaling** is skipped because it is more fragile. Scale normalization (e.g., dividing by torso length or keypoint spread) can fail with occlusion, side views, or bad keypoints. Centering alone gives most of the benefit with fewer edge cases.
