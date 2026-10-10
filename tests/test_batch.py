from pathlib import Path

import pytest
from PIL import Image

from matting.batch import default_output_dir, list_images, output_names, plan_jobs, run_batch


def make_img(path: Path, color=(200, 50, 50)):
    Image.new("RGB", (32, 24), color).save(path)


def test_list_images_non_recursive_case_insensitive_sorted(tmp_path):
    make_img(tmp_path / "b.JPG")
    make_img(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    make_img(tmp_path / "sub" / "c.jpg")
    assert [p.name for p in list_images(tmp_path)] == ["a.png", "b.JPG"]


def test_default_output_dir_is_sibling(tmp_path):
    src = tmp_path / "photos"
    assert default_output_dir(src) == tmp_path / "photos_matting"


def test_output_names_disambiguate_same_stem(tmp_path):
    files = [tmp_path / "a.jpg", tmp_path / "a.png", tmp_path / "b.webp"]
    assert output_names(files) == ["a_jpg.png", "a_png.png", "b.png"]


class FakePipe:
    def run(self, img):
        if img.getpixel((0, 0)) == (0, 0, 0):
            raise RuntimeError("boom")
        return img.convert("RGBA"), {"total": 0.1, "stages": {"x": 0.1}}


def test_run_batch_continues_after_failure_and_writes_outputs(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    make_img(src / "1.jpg")
    make_img(src / "2.jpg", color=(0, 0, 0))  # FakePipe fails on this one
    make_img(src / "3.png")
    events = list(run_batch(FakePipe(), src, out))
    assert [e.index for e in events] == [1, 2, 3] and all(e.total == 3 for e in events)
    assert [e.error is None for e in events] == [True, False, True]
    assert "boom" in events[1].error
    assert sorted(p.name for p in out.iterdir()) == ["1.png", "3.png"]
    assert Image.open(out / "1.png").mode == "RGBA"


def test_run_batch_refuses_output_equal_to_input(tmp_path):
    make_img(tmp_path / "a.png")
    with pytest.raises(ValueError):
        list(run_batch(FakePipe(), tmp_path, tmp_path))


def test_run_batch_empty_folder_creates_nothing(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    assert list(run_batch(FakePipe(), src, None)) == []
    assert not (tmp_path / "empty_matting").exists()


def test_run_batch_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(run_batch(FakePipe(), tmp_path / "nope", None))


def test_plan_folder_goes_to_sibling_dir(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    make_img(src / "a.jpg")
    make_img(src / "a.png")
    make_img(src / "b.webp")
    jobs, skipped = plan_jobs([src])
    out = tmp_path / "photos_matting"
    assert [(j.src.name, j.dst) for j in jobs] == [
        ("a.jpg", out / "a_jpg.png"),
        ("a.png", out / "a_png.png"),
        ("b.webp", out / "b.png"),
    ]
    assert skipped == []


def test_plan_loose_files_grouped_by_parent_and_deduplicated(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    make_img(tmp_path / "x" / "a.jpg")
    make_img(tmp_path / "y" / "b.jpg")
    jobs, _ = plan_jobs([tmp_path / "x" / "a.jpg", tmp_path / "y" / "b.jpg", tmp_path / "x" / "a.jpg", tmp_path / "x"])
    assert [j.dst for j in jobs] == [tmp_path / "x_matting" / "a.png", tmp_path / "y_matting" / "b.png"]


def test_plan_shared_out_dir_disambiguates_across_folders(tmp_path):
    for d in ("x", "y"):
        (tmp_path / d).mkdir()
        make_img(tmp_path / d / "a.jpg")
    out = tmp_path / "out"
    jobs, _ = plan_jobs([tmp_path / "x", tmp_path / "y"], out)
    assert [j.dst for j in jobs] == [out / "a.png", out / "a_2.png"]


def test_plan_skips_bad_inputs_and_output_equal_to_input(tmp_path):
    make_img(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "empty").mkdir()
    jobs, skipped = plan_jobs([tmp_path / "notes.txt", tmp_path / "nope.jpg", tmp_path / "empty", tmp_path / "a.png"], tmp_path)
    assert jobs == []
    assert [(p.name, r) for p, r in skipped] == [
        ("notes.txt", "不支持的格式"),
        ("nope.jpg", "文件不存在"),
        ("empty", "文件夹里没有图片"),
        ("a.png", "输出文件夹不能和输入文件夹相同"),
    ]
