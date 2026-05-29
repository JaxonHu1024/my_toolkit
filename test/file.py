"""test/file.py

对 `my_toolkit.file` 的最小可运行测试脚本。

运行方式：
    - `python test/file.py`
    - 或在仓库根目录使用 `pytest -q`（若已安装 pytest）
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
import sys


def _import_module():
    # 让 `import my_toolkit.xxx` 在直接运行脚本时也能生效
    root = Path(__file__).resolve().parents[1]          # .../my_toolkit
    sys.path.insert(0, str(root.parent))                # .../code/my
    try:
        return importlib.import_module("my_toolkit.file"), None
    except Exception as exc:
        return None, exc


file_mod, _IMPORT_ERR = _import_module()


@unittest.skipIf(file_mod is None, f"my_toolkit.file 导入失败: {_IMPORT_ERR}")
class _Base(unittest.TestCase):
    """统一的跳过条件基类。"""
    pass


class TestFileTxt(_Base):
    def test_txt_roundtrip_lines_and_string(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a" / "b.txt"

            file_mod.write_txt(["  hello ", "world"], p)
            self.assertEqual(file_mod.read_txt(p, as_lines=True), ["  hello ", "world"])

            # 覆盖写入字符串
            file_mod.write_txt("raw\ntext\n", p)
            self.assertEqual(file_mod.read_txt(p, as_lines=False), "raw\ntext\n")

    def test_txt_append(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "append.txt"
            file_mod.write_txt(["a"], p)
            file_mod.write_txt(["b"], p, append=True)
            self.assertEqual(file_mod.read_txt(p, as_lines=True), ["a", "b"])


class TestFileJson(_Base):
    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.json"
            obj = {"a": 1, "b": [1, 2], "c": {"x": "y"}}
            file_mod.write_json(obj, p)
            self.assertEqual(file_mod.read_json(p), obj)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "rows.jsonl"
            rows = [{"i": 0}, {"i": 1}, {"i": 2, "s": "中文"}]
            file_mod.write_jsonl(rows, p)
            self.assertEqual(file_mod.read_jsonl(p), rows)


class TestFilePickle(_Base):
    def test_pickle_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.pkl"
            obj = {"k": (1, 2, 3), "v": {"nested": True}}
            file_mod.write_pickle(obj, p)
            self.assertEqual(file_mod.read_pickle(p), obj)


class TestFileCsvParquetDispatcher(_Base):
    def test_csv_roundtrip_dataframe_and_list(self):
        try:
            import pandas as pd
        except Exception as exc:
            self.skipTest(f"pandas 不可用，跳过 CSV 测试: {exc}")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.csv"

            df = pd.DataFrame({"a": [1, 2], "b": ["x", None]})
            file_mod.write_csv(df, p)
            read_df = file_mod.read_csv(p, format="dataframe")
            self.assertEqual(read_df.shape, (2, 2))
            # replace_na=True 会把 NaN 替换为 None
            self.assertIsNone(read_df.loc[1, "b"])

            # list 模式
            p2 = Path(td) / "rows.tsv"
            rows = [["1", "x"], ["2", "y"]]
            file_mod.write_csv(rows, p2, sep="\t", header=["a", "b"])
            read_rows = file_mod.read_csv(p2, sep="\t", format="list", skip_header=True)
            self.assertEqual(read_rows, rows)

    def test_parquet_roundtrip_if_available(self):
        try:
            import pandas as pd
        except Exception as exc:
            self.skipTest(f"pandas 不可用，跳过 Parquet 测试: {exc}")

        # 需要 pyarrow 或 fastparquet
        try:
            import pyarrow  # noqa: F401
        except Exception:
            # 尝试 fastparquet
            try:
                import fastparquet  # noqa: F401
            except Exception as exc:
                self.skipTest(f"缺少 Parquet 引擎（pyarrow/fastparquet），跳过: {exc}")

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.parquet"
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            file_mod.write_parquet(df, p)
            read_df = file_mod.read_parquet(p)
            self.assertEqual(read_df.shape, df.shape)

    def test_dispatcher_read_write(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)

            # txt
            p_txt = base / "a.txt"
            file_mod.write_file(["x", "y"], p_txt)
            self.assertEqual(file_mod.read_file(p_txt, as_lines=True), ["x", "y"])

            # json
            p_json = base / "a.json"
            obj = {"x": 1}
            file_mod.write_file(obj, p_json)
            self.assertEqual(file_mod.read_file(p_json), obj)

            # pickle
            p_pkl = base / "a.pkl"
            obj2 = [1, 2, 3]
            file_mod.write_file(obj2, p_pkl)
            self.assertEqual(file_mod.read_file(p_pkl), obj2)

    def test_dispatcher_unsupported_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.unsupported"
            with self.assertRaises(ValueError):
                file_mod.read_file(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
