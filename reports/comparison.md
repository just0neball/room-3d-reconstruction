# VGGT vs GLOMAP — Reconstruction Benchmark

Both methods used the same selected video frame identities. VGGT internally resizes frames, whereas GLOMAP uses the extracted resolution. VGGT camera predictions are not equivalent to geometrically verified registration.

| Metric | VGGT benchmark | GLOMAP benchmark |
|---|---:|---:|
| Input images | 32 | 32 |
| Predicted / registered cameras | 32 predicted | 32 registered |
| 3D geometry | Dense predicted point map | 4247 triangulated points |
| Total stage runtime (s) | 88.976 | 52.881 |
| Allocated GPU VRAM (GB) | 7.066 | 0 for CPU SIFT/matching; mapper unmeasured |
| Processing resolution | Internal input [294, 518] (H×W) | Extracted images, original 960×540 in reference sequence |

**Full sequence:** VGGT predicted 180 poses in 11 overlapping chunks, mean alignment residual 0.006422252819277097; GLOMAP registered 180 cameras and 23586 points. Full GLOMAP runtime was 1626.324 s, including 1465.753 s for feature matching.

**Selection:** GLOMAP was chosen as the reconstruction source because it supplies SfM-verified cameras and a COLMAP sparse scene compatible with camera undistortion and Gaussian Splatting. VGGT is substantially faster during inference, but chunk-merging residuals are only internal-consistency measurements and cannot prove absolute pose accuracy. Sequential and exhaustive matching provide strong loop-closure coverage in the selected room sequence, at substantial CPU cost; a reduced loop-closure strategy is recommended for production scaling.

**Visual evidence:** `pose_comparison/vggt/vggt_pose_pointcloud.png`, `pose_comparison/glomap/sparse_preview.png`, `full_reconstruction/vggt/vggt_pose_pointcloud.png`, and `full_reconstruction/glomap/sparse_preview.png`.
