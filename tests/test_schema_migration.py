import unittest
from schemas.migration import normalize_manifest

class SchemaMigrationTests(unittest.TestCase):
    def test_v11_normalizes_visible_full_geometry_and_status(self):
        data=normalize_manifest({'schema_version':'1.1','objects':[{'id':'x','status':'pass','asset_path':'assets/x.png','mask_path':'masks/x.png','qa':{'status':'pass','reason':'ok'},'bbox':{'x':1,'y':2,'w':3,'h':4}}]})
        self.assertEqual(data['schema_version'],'1.2'); self.assertEqual(data['objects'][0]['geometry']['visible_bbox'],data['objects'][0]['bbox'])
        self.assertEqual(data['objects'][0]['base_asset_status'],'ready')
