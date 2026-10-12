import numpy as np
import pytest

from matting.pipeline_vlm import inside_fraction, select_instances
from matting.qwen import QwenError, parse_subjects
from matting.sam3 import Instances


def test_parse_subjects_object_form_scales_0_1000_to_pixels():
    text = '```json\n{"subjects": [{"label": "Cat", "box_2d": [100, 200, 500, 1000]}]}\n```'
    (s,) = parse_subjects(text, 640, 480)
    assert s.label == "cat"
    assert s.box == pytest.approx((64, 96, 320, 480))


def test_parse_subjects_bare_list_and_bbox_2d_key():
    text = 'here: [{"bbox_2d": [13, 115, 501, 986], "label": "cat"}, {"bbox_2d": [542, 57, 999, 771], "label": "cat"}]'
    subjects = parse_subjects(text, 1000, 1000)
    assert [s.box for s in subjects] == [(13, 115, 501, 986), (542, 57, 999, 771)]


def test_parse_subjects_drops_degenerate_and_fixes_swapped_corners():
    text = '{"subjects": [{"label": "a", "box_2d": [500, 500, 500, 600]}, {"label": "b", "box_2d": [900, 900, 100, 100]}]}'
    (s,) = parse_subjects(text, 1000, 1000)
    assert s.label == "b" and s.box == (100, 100, 900, 900)


def test_parse_subjects_empty_and_garbage():
    assert parse_subjects('{"subjects": []}', 10, 10) == []
    with pytest.raises(QwenError):
        parse_subjects("sorry, I can't see a subject", 10, 10)


def rect_mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), bool)
    m[y1:y2, x1:x2] = True
    return m


def instances(*rects, h=100, w=200):
    masks = np.stack([rect_mask(h, w, *r) for r in rects])
    return Instances(masks, masks.astype(np.float32), np.array(rects, float), np.ones(len(rects)))


def test_inside_fraction_with_margin():
    m = rect_mask(100, 200, 0, 0, 100, 100)
    assert inside_fraction(m, (0, 0, 50, 100), margin=0) == pytest.approx(0.5)
    assert inside_fraction(m, (0, 0, 100, 100)) == 1.0


def test_select_keeps_the_instance_in_the_box_not_its_twin():
    # Two cats; Qwen boxed the right one.
    inst = instances((5, 10, 90, 90), (110, 10, 195, 90))
    assert select_instances(inst, [(105, 5, 199, 95)]) == [1]


def test_select_keeps_object_extending_past_a_tight_box():
    # The plate sticks out below Qwen's box: SAM3's mask wins, the box only picks it.
    inst = instances((50, 10, 150, 95))
    assert select_instances(inst, [(50, 10, 150, 60)]) == [0]


def test_select_keeps_several_parts_inside_one_box():
    # "grocery bags": SAM3 returns each bag, all inside the one box.
    inst = instances((20, 40, 40, 90), (45, 40, 65, 90), (70, 40, 90, 90), (150, 10, 190, 50))
    assert select_instances(inst, [(15, 30, 95, 95)]) == [0, 1, 2]


def test_select_falls_back_to_best_overlap_or_nothing():
    inst = instances((0, 0, 100, 100))
    # Box covers only the left 40%: no instance is >=50% inside, but IoU 0.4 >= 0.3 -> fallback keeps it.
    assert select_instances(inst, [(0, 0, 40, 100)], min_inside=0.9) == [0]
    # Box elsewhere: nothing.
    assert select_instances(inst, [(150, 0, 199, 99)]) == []


def test_complete_instance_fills_unsure_part_touching_the_core_and_drops_specks():
    from matting.pipeline_vlm import complete_instance

    prob = np.zeros((100, 200), np.float32)
    prob[20:80, 20:100] = 0.9  # confident body
    prob[30:50, 100:130] = 0.4  # unsure beak attached to the body
    prob[85:95, 150:160] = 0.4  # unsure blob elsewhere
    prob[5:7, 5:7] = 0.9  # confident speck
    m = complete_instance(prob, (15, 15, 135, 85))
    assert m[40, 120]  # beak completed
    assert not m[90, 155]  # unattached blob dropped
    assert not m[6, 6]  # speck dropped
    assert m[50, 50]


def test_drop_duplicates_keeps_disjoint_objects_and_drops_contained_ones():
    from matting.pipeline_vlm import drop_duplicates

    inst = instances((10, 10, 60, 90), (12, 12, 70, 88), (100, 10, 150, 90))
    inst.scores = np.array([0.9, 0.3, 0.8])
    # #1 is mostly inside #0 (cat + a strip of blanket); #2 is a separate bag.
    assert drop_duplicates(inst, [0, 1, 2]) == [0, 2]


def test_plain_background_keys_out_everything_else_and_rejects_busy_borders():
    from PIL import Image as PILImage

    from matting.plain_background import key_mask, plain_background

    a = np.full((200, 300, 3), 250, np.uint8)
    a[60:150, 80:200] = (120, 90, 60)  # the object
    a[150:170, 70:210] = (90, 50, 30)  # its base, a different colour
    a[5, 5] = (0, 0, 0)  # a speck
    img = PILImage.fromarray(a)
    bg = plain_background(img)
    assert bg is not None and list(bg) == [250, 250, 250]
    m = key_mask(img, bg)
    assert m[100, 100] and m[160, 100]  # object and base both kept
    assert not m[5, 5] and not m[10, 280]
    busy = np.random.default_rng(0).integers(0, 255, (200, 300, 3), np.uint8)
    assert plain_background(PILImage.fromarray(busy)) is None


def test_matte_cropped_runs_vitmatte_on_the_subject_area_only():
    from PIL import Image as PILImage

    from matting.pipeline_vlm import matte_box, matte_cropped

    trimap = np.zeros((300, 400), np.uint8)
    trimap[100:150, 200:260] = 128
    trimap[110:140, 210:250] = 255
    assert matte_box(trimap, margin=32) == (168, 68, 292, 182)
    seen = []

    def fake_matte(img, t):
        seen.append(img.size)
        return (t / 255.0).astype(np.float32)

    alpha = matte_cropped(fake_matte, PILImage.new("RGB", (400, 300)), trimap)
    assert seen == [(188, 178)]  # default margin 64
    assert alpha[120, 230] == 1.0 and alpha[0, 0] == 0.0 and alpha.shape == (300, 400)
