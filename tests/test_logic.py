import numpy as np

from matting.common import make_trimap
from matting.pipeline_a import complete_mask, fuse_masks, pad_box, prompts_from_saliency, sample_points


def disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def test_trimap_three_values_and_band_direction():
    m = disk(200, 200, 100, 100, 50)
    t = make_trimap(m, k=5)
    assert set(np.unique(t)) == {0, 128, 255}
    assert t[100, 100] == 255  # deep inside -> fg
    assert t[0, 0] == 0  # far outside -> bg
    assert t[100, 150] == 128  # on the boundary -> unknown
    assert (t == 255).sum() < m.sum() < (t > 0).sum()


def test_trimap_thin_component_keeps_fg_core():
    # Text strokes thinner than the erosion band must not become 100% "unknown" (ViTMatte may erase them).
    m = np.zeros((200, 300), bool)
    m[100:106, 20:120] = True  # 6 px stroke
    m |= disk(200, 300, 100, 220, 50)
    t = make_trimap(m, k=8)
    assert (t[100:106, 20:120] == 255).any()  # stroke keeps a definite-foreground core
    assert (t[100:106, 20:120] == 255).sum() < m[100:106, 20:120].sum()  # ...but only its centre
    big = make_trimap(disk(200, 300, 100, 220, 50), k=8)
    assert ((t == 255) & disk(200, 300, 100, 220, 50)).sum() == (big == 255).sum()  # thick parts unchanged


def test_prompts_multi_component_box_spans_all_and_points_inside():
    sal = np.zeros((200, 300), np.float32)
    sal[disk(200, 300, 100, 70, 40)] = 1.0
    sal[disk(200, 300, 100, 220, 30)] = 1.0
    sal[disk(200, 300, 10, 10, 2)] = 1.0  # speck, < 10% of largest -> ignored
    p = prompts_from_saliency(sal)
    x1, y1, x2, y2 = p.box
    assert x1 <= 31 and x2 >= 249 and y1 >= 50  # spans both blobs, speck excluded
    for x, y in p.points:
        assert p.sal_mask[int(y), int(x)]  # never on the speck or the background
    assert any(disk(200, 300, 100, 70, 40)[int(y), int(x)] for x, y in p.points)
    assert any(disk(200, 300, 100, 220, 30)[int(y), int(x)] for x, y in p.points)


def test_prompts_empty_saliency_falls_back_to_center_box():
    p = prompts_from_saliency(np.zeros((100, 200), np.float32))
    assert p.box == [20.0, 10.0, 180.0, 90.0]
    assert p.points == [[100.0, 50.0]]
    assert not p.sal_mask.any()


def test_prompts_drop_secondary_component_touching_border():
    # Jar in the middle + a leaf sprig cut by the frame corner (test case 11_风格化).
    sal = np.zeros((200, 300), np.float32)
    sal[disk(200, 300, 120, 160, 50)] = 1.0
    sal[disk(200, 300, 0, 0, 30)] = 1.0  # ~18% of main, touches top/left border
    p = prompts_from_saliency(sal)
    assert all(p.sal_mask[int(y), int(x)] for x, y in p.points)
    assert not p.sal_mask[5, 5]
    assert p.box[0] >= 100 and p.box[1] >= 60


def test_prompts_keep_main_component_touching_border_and_inner_secondaries():
    sal = np.zeros((200, 300), np.float32)
    sal[disk(200, 300, 190, 150, 60)] = 1.0  # main subject cut by the bottom border
    sal[disk(200, 300, 60, 60, 25)] = 1.0  # second subject fully inside the frame
    p = prompts_from_saliency(sal)
    assert p.sal_mask[195, 150] and p.sal_mask[60, 60]
    assert any(disk(200, 300, 60, 60, 25)[int(y), int(x)] for x, y in p.points)
    assert any(disk(200, 300, 190, 150, 60)[int(y), int(x)] for x, y in p.points)


def test_fuse_drops_sam_islands_keeps_connected_extension_and_clips_to_box():
    h, w = 200, 300
    sal = np.zeros((h, w), bool)
    sal[60:140, 100:200] = True  # saliency stops at row 139 (plate rim)
    sam = np.zeros((h, w), bool)
    sam[60:160, 100:200] = True  # SAM continues down to row 159 (plate underside) -> keep
    sam[20:30, 20:30] = True  # island not touching saliency (window speck) -> drop
    sam[60:140, 260:299] = True  # touches nothing salient, outside box -> drop
    out = fuse_masks(sam, sal, box=[90, 50, 250, 170])
    assert out[150, 150]
    assert not out[25, 25]
    assert not out[100, 280]
    assert out[sal].all()  # saliency itself is always kept


def test_fuse_keeps_saliency_parts_sam_missed():
    sal = np.zeros((100, 100), bool)
    sal[20:80, 20:80] = True
    sam = np.zeros((100, 100), bool)
    sam[20:80, 20:50] = True
    out = fuse_masks(sam, sal, box=[0, 0, 99, 99])
    assert out[50, 70]


def test_sample_points_cover_every_component_and_stay_inside():
    m = np.zeros((300, 400), bool)
    m |= disk(300, 400, 150, 120, 70)
    m |= disk(300, 400, 150, 320, 40)
    pts = sample_points(m, max_points=25)
    assert 2 <= len(pts) <= 25
    assert all(m[int(y), int(x)] for x, y in pts)
    assert any(disk(300, 400, 150, 120, 70)[int(y), int(x)] for x, y in pts)
    assert any(disk(300, 400, 150, 320, 40)[int(y), int(x)] for x, y in pts)


def test_sample_points_thin_component_still_gets_a_point():
    m = np.zeros((300, 400), bool)
    m |= disk(300, 400, 150, 120, 70)
    m[200:203, 250:390] = True  # 3 px stroke: erodes away entirely, must still get its peak point
    pts = sample_points(m, max_points=25)
    assert any(200 <= y <= 202 and 250 <= x <= 389 for x, y in pts)


def test_sample_points_respects_budget_with_large_mask():
    m = np.ones((1000, 1000), bool)
    assert len(sample_points(m, max_points=25)) <= 25


def test_complete_mask_accepts_object_level_rejects_scene_level():
    h, w = 300, 400
    sal = np.zeros((h, w), bool)
    sal[100:141, 50:251] = True  # plate top surface (what saliency sees)
    part = np.zeros((h, w), bool)
    part[110:131, 100:200] = True  # tart only
    obj = np.zeros((h, w), bool)
    obj[100:161, 50:251] = True  # whole plate incl. underside rows 141-160 -> 2/3 inside
    scene = np.zeros((h, w), bool)
    scene[100:300, 0:400] = True  # table -> mostly outside
    cands = np.stack([np.stack([part, obj, scene])])  # (1 point, 3 levels, H, W)
    out = complete_mask(cands, sal, min_inside=0.6)
    assert out[150, 150]  # underside recovered
    assert not out[250, 10]  # table rejected
    assert out.sum() == obj.sum()


def test_complete_mask_nothing_accepted_gives_empty():
    sal = np.zeros((50, 50), bool)
    sal[20:30, 20:30] = True
    far = np.zeros((50, 50), bool)
    far[0:10, 0:10] = True
    out = complete_mask(np.stack([np.stack([far, far, far])]), sal, min_inside=0.6)
    assert not out.any()


def test_pad_box_clamps():
    assert pad_box([50, 40, 250, 160], 30, w=300, h=180) == [20, 10, 280, 179]


def test_complete_mask_rejects_unstable_mask_even_if_mostly_inside():
    sal = np.zeros((100, 100), bool)
    sal[20:61, 20:61] = True
    obj = sal.copy()
    obj[61:70, 20:61] = True  # inside 0.82: passes the inside rule
    cands = np.stack([np.stack([obj, obj, obj])])
    assert complete_mask(cands, sal, stability=np.array([[0.9, 0.9, 0.9]]))[65, 40]
    assert not complete_mask(cands, sal, stability=np.array([[0.3, 0.3, 0.3]])).any()  # soft glow -> rejected
    assert complete_mask(cands, sal, stability=None)[65, 40]  # no stability info -> inside rule only
