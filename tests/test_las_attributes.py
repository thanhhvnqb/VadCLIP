import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from las_data import UCF_NAMES, XD_NAMES, attribute_descriptions
from las_import_attributes import parse_table


class AttributeTests(unittest.TestCase):
    def test_official_table_parser_preserves_dataset_specific_text(self):
        rows = []
        for dataset, names in [('xd', XD_NAMES), ('ucf', UCF_NAMES)]:
            for name in names:
                key = 'roadAccidents' if name == 'road accident' else name
                rows.append(f'“{key}”: “{dataset} synthetic\n description for {name}”')
        parsed = parse_table('\n'.join(rows))
        self.assertEqual(parsed['ucf']['road accident'], 'ucf synthetic description for road accident')
        self.assertNotEqual(parsed['xd']['fighting'], parsed['ucf']['fighting'])
        for broken in (rows[:-1], rows + [rows[-1]], rows[:7]):
            with self.assertRaises(ValueError):
                parse_table('\n'.join(broken))

    def test_flat_and_dataset_specific_files_keep_class_order(self):
        nested = {d: {name: f'{d} {name}' for name in reversed(names)}
                  for d, names in [('ucf', UCF_NAMES), ('xd', XD_NAMES)]}
        nested['_source'] = {'sha256': 'test'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'attributes.json'
            path.write_text(json.dumps(nested))
            self.assertEqual(attribute_descriptions('ucf', path), [f'ucf {n}' for n in UCF_NAMES])
            path.write_text(json.dumps(nested['xd']))
            self.assertEqual(attribute_descriptions('xd', path), [f'xd {n}' for n in XD_NAMES])
            path.write_text(json.dumps({'xd': {'normal': ''}}))
            with self.assertRaises(ValueError):
                attribute_descriptions('xd', path)

    def test_repository_paper_attributes_cover_both_datasets(self):
        path = Path(__file__).resolve().parents[1]/'configs/las_attributes_paper.json'
        for dataset, names in [('ucf', UCF_NAMES), ('xd', XD_NAMES)]:
            descriptions = attribute_descriptions(dataset, path)
            self.assertEqual(len(descriptions), len(names))
            self.assertTrue(all(descriptions))


if __name__ == '__main__':
    unittest.main()
