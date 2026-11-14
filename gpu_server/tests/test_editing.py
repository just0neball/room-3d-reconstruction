import numpy as np

from gpu_server.editing import compose
from gpu_server.editing.editors import MockEditor


def _scene():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), bool)
    mask[150:300, 200:420] = True
    mask[180:200, 250:270] = False  # a hole inside the mask must stay untouched too
    return img, mask


def test_crop_is_multiple_of_16_and_covers_mask():
    img, mask = _scene()
    crop = compose.crop_box(mask)
    assert crop.size[0] % 16 == 0 and crop.size[1] % 16 == 0
    assert crop.x0 <= 200 and crop.x1 >= 420 and crop.y0 <= 150 and crop.y1 >= 300
    assert compose.cut(img, crop).shape[:2][::-1] == crop.size


def test_nothing_changes_outside_mask():
    img, mask = _scene()
    crop = compose.crop_box(mask)
    edited = 255 - compose.cut(img, crop)  # a drastic "edit" over the whole crop
    out = compose.composite(img, edited, mask, crop)
    assert compose.outside_mask_max_diff(img, out, mask) == 0
    assert np.abs(out[mask].astype(int) - img[mask].astype(int)).mean() > 50  # but the inside did change


def test_mock_editor_follows_prompt_and_reference():
    img = np.full((128, 128, 3), (150, 105, 70), np.uint8)
    m = np.ones((128, 128), bool)
    ed = MockEditor()
    first = ed.edit([img], "make the sofa dark green leather", 1, [m])
    hue = np.median(__import__("cv2").cvtColor(first, __import__("cv2").COLOR_RGB2HSV)[..., 0])
    assert 50 <= hue <= 70
    second = ed.edit([img, first], "make the sofa dark green leather", 2, [m, m])
    h2 = np.median(__import__("cv2").cvtColor(second, __import__("cv2").COLOR_RGB2HSV)[..., 0])
    assert abs(h2 - hue) <= 3
