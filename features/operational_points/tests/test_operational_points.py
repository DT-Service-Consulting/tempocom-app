import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from features.operational_points.OperationalPoints import OperationalPoints
from tempocom.services import DBConnector

class TestOperationalPoints:
    dbc = DBConnector()
    oper_points = OperationalPoints(dbc)

    def test_get_all_points(self):
        df = self.oper_points.df
        assert not df.empty, "Operational points DataFrame should not be empty."
    
    def test_get_type_by_id(self):
        test_id = self.oper_points.df['ID'].sample(1).iloc[0]
        point_type = self.oper_points.get_type_by_id(test_id)
        assert point_type is not None, "Point type should not be None."
    
    def test_get_geopoint_by_id(self):
        test_id = self.oper_points.df['ID'].sample(1).iloc[0]
        geopoint = self.oper_points.get_geopoint_by_id(test_id)
        assert geopoint is not None, "Geopoint should not be None."
        assert isinstance(geopoint, tuple) and len(geopoint) == 2, "Geopoint should be a tuple of two elements."
        assert all(isinstance(coord, float) for coord in geopoint), "Both elements of the geopoint tuple should be floats."