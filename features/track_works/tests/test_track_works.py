import pytest
import sys
import os
import pandas as pd
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from features.track_works.TrackWorks import TrackWorks
from tempocom.services import DBConnector

class TestTrackWorksModel:
    
    @pytest.fixture
    def mock_dbc(self):
        dbc = Mock(spec=DBConnector)
        dbc.conn = Mock()
        return dbc
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'id': [1, 2, 3],
            'cou_id': [100, 101, 102],
            'section_from_id': [10, 20, 30],
            'section_to_id': [15, 20, 35],
            'impact': ['CTL', 'SAVU', 'Keep Free']
        })
    
    @pytest.fixture
    def track_works(self, mock_dbc, sample_data):
        with patch('pandas.read_sql_query', return_value=sample_data):
            return TrackWorks(mock_dbc)
    
    def test_get_active_works(self, track_works, sample_data):
        result = track_works.get_active_works()
        pd.testing.assert_frame_equal(result, sample_data)
    
    def test_get_work_by_id_exists(self, track_works):
        result = track_works.get_work_by_id(2)
        assert result['id'] == 2
        assert result['cou_id'] == 101
    
    def test_get_work_by_id_not_exists(self, track_works):
        result = track_works.get_work_by_id(999)
        assert result is None
    
    def test_has_valid_sections_true(self, track_works):
        row = track_works.df.iloc[0]
        assert track_works.has_valid_sections(row) is True
    
    def test_has_valid_sections_false(self, track_works):
        row = pd.Series({'section_from_id': pd.NA, 'section_to_id': 15})
        assert track_works.has_valid_sections(row) is False
    
    def test_is_point_work_true(self, track_works):
        row = track_works.df.iloc[1]
        assert track_works.is_point_work(row) is True
    
    def test_is_point_work_false(self, track_works):
        row = track_works.df.iloc[0]
        assert track_works.is_point_work(row) is False
    
    def test_get_description_with_data(self, track_works):
        desc_data = pd.DataFrame({'description_of_works': ['Test description']})
        with patch('pandas.read_sql_query', return_value=desc_data):
            result = track_works.get_description(100)
            assert result == 'Test description'
    
    def test_get_description_no_data(self, track_works):
        with patch('pandas.read_sql_query', return_value=pd.DataFrame()):
            result = track_works.get_description(999)
            assert result == "No description available"





