"""Geometry + tracking tests on the synthetic room (CPU only, no models)."""
import numpy as np
import pytest

from gpu_server import geometry
from gpu_server.mock import synthetic as syn
from gpu_server.vision import tracking
from gpu_server.vision.depth import SplatDepth
from gpu_server.vision.segmenters import MockSegmenter, mask_iou


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    out = tmp_path_factory.mktemp("proj")
    pts, col, cams = syn.build_project(out, frames=36)
    from gpu_server import datasets, settings
    settings.DATA_DIR = out / "data"
    datasets.register_dir("t", out, "v1")
    ds = datasets.Dataset("t")
    return pts, col, ds, SplatDepth(ds.cams, pts)


def sofa_points(pts):
    a, d, size, _ = syn.OBJECTS["sofa"]
    c = np.array([d * np.sin(np.deg2rad(a)), 1.3 - size[1] / 2, d * np.cos(np.deg2rad(a))])
    return pts[np.all(np.abs(pts - c) <= np.array(size) / 2 + 1e-6, axis=1)]


def test_colmap_roundtrip(world):
    pts, _, ds, _ = world
    ref = syn.make_cameras(36)
    for a, b in zip(ds.cams, ref):
        np.testing.assert_allclose(a.w2c, b.w2c, atol=1e-5)


def test_project_backproject_roundtrip(world):
    pts, _, ds, depth = world
    cam = ds.cams[0]
    mask = np.zeros((cam.height, cam.width), bool)
    mask[150:210, 280:360] = True
    lifted = geometry.backproject(mask & (depth(0) > 0), depth(0), cam)
    uv, z = geometry.project(lifted, cam)
    assert (uv[:, 0] > 270).all() and (uv[:, 0] < 370).all() and (z > 0).all()


def test_lift3d_tracks_sofa_and_skips_frames_without_it(world):
    pts, _, ds, depth = world
    seg = MockSegmenter()
    ref = 0  # camera 0 looks straight at the sofa (angle 0)
    mask, _ = seg.segment(ds.image(ref), [(320, 250, 1)])
    assert mask.mean() > 0.02
    res, info = tracking.track_lift3d(ds, depth, seg, ref, mask, lambda a, b: None)
    included = [r.index for r in res if r.included]
    assert 0 in included and 1 in included and 35 in included  # neighbours on both sides
    assert 18 not in included  # opposite direction: lamp, no sofa -> frustum filter drops it
    # accuracy vs ground truth on included auto frames
    ious = []
    for r in res:
        if r.source == "auto":
            ious.append(mask_iou(r.mask, syn.object_mask_truth("sofa", ds.cams[r.index])))
    assert len(ious) >= 4 and np.mean(ious) > 0.6, ious


def test_neighbor_tracks_adjacent_frames(world):
    pts, _, ds, depth = world
    seg = MockSegmenter()
    mask, _ = seg.segment(ds.image(1), [(320, 250, 1)])
    res, _ = tracking.track_neighbor(ds, seg, 1, mask, lambda a, b: None)
    assert res[0].included and res[2].included and res[3].included  # both directions in time order
    assert not res[18].included


def test_manual_uses_projected_hints(world):
    pts, _, ds, depth = world
    seg = MockSegmenter()
    mask, _ = seg.segment(ds.image(0), [(320, 250, 1)])
    res, _ = tracking.track_manual(ds, depth, seg, 0, mask, [2, 34], {}, [], lambda a, b: None)
    assert all(r.included and r.source == "manual" for r in res), res
