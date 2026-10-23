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


# --- candidates, judge, remove regions ---

from matting.pipeline_vlm import grow_like, needs_judge  # noqa: E402
from matting.plain_background import fill_holes, key_candidates  # noqa: E402
from matting.qwen import parse_detection, parse_keep  # noqa: E402
from PIL import Image  # noqa: E402


def test_parse_detection_reads_remove_regions():
    text = '{"subjects": [{"label": "girl", "box_2d": [100, 100, 500, 900]}], "remove": [{"label": "checkerboard", "box_2d": [600, 600, 700, 800]}]}'
    subjects, remove = parse_detection(text, 1000, 1000)
    assert [s.label for s in subjects] == ["girl"]
    assert remove[0].label == "checkerboard" and remove[0].box == (600, 600, 700, 800)


def test_parse_detection_repairs_box_without_brackets_and_trailing_commas():
    text = '{"subjects": [{"label": "umpire", "box_2d": 37, 421, 251, 969},], "remove": []}'
    (s,), remove = parse_detection(text, 1000, 1000)
    assert s.box == (37, 421, 251, 969) and remove == []


def test_parse_keep_takes_the_last_answer_and_drops_bad_numbers():
    assert parse_keep('candidate 1 looks like a chair... {"keep": [3, 1, 9]}', 3) == [0, 2]
    with pytest.raises(QwenError):
        parse_keep("no idea", 3)


def test_needs_judge_only_when_a_subject_has_several_candidates():
    assert not needs_judge([0, 1])
    assert needs_judge([0, 0, 1])
    assert not needs_judge([None])
    assert needs_judge([None, None])


def test_fill_holes_fills_enclosed_but_not_open_notches():
    m = np.zeros((20, 20), bool)
    m[2:18, 2:18] = True
    m[6:9, 6:9] = False  # enclosed hole (teeth)
    m[10:13, 2:8] = False  # notch open to the outside
    f = fill_holes(m)
    assert f[7, 7] and not f[11, 3]


def test_key_candidates_keeps_white_teeth_in_a_face_on_white():
    a = np.full((60, 60, 3), 255, np.uint8)
    a[10:50, 10:50] = (230, 170, 140)  # face
    a[30:35, 20:40] = 255  # white teeth
    (c,) = key_candidates(Image.fromarray(a), np.array([255, 255, 255]))
    assert c[32, 30] and not c[2, 2]


def test_grow_like_takes_connected_look_alikes_but_not_distant_ones():
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:] = (200, 30, 30)  # balloon
    rgb[5:15, 5:25] = (230, 230, 230)  # checkerboard notch, SAM3 found only part of it
    rgb[30:34, 30:34] = (230, 230, 230)  # white teeth elsewhere
    within = np.ones((40, 40), bool)
    seed = np.zeros((40, 40), bool)
    seed[5:15, 15:25] = True
    g = grow_like(seed, within, rgb, (15, 5, 25, 15))
    assert g[10, 6] and not g[32, 32] and not g[20, 20]
