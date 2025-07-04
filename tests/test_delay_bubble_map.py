"""
Unit tests for the DelayBubbleMap class in objects.Delay_network.

Tests:
- Data preparation and merging of station and delay data.
- Correct parsing of geolocation points.
- Calculation of total delay in minutes.
- Rendering of Folium map object.

Author: Mohamad Hussain
Date: [2025-06-20]
"""
# ...existing code...



import pytest
import pandas as pd
import ast

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from objects.Delay_network import DelayBubbleMap

@pytest.fixture
def mock_station_data(tmp_path):
    # Create temporary stations CSV
    data = {
        'Name_FR': ['Bruxelles-Central', 'Charleroi-Central'],
        'Geo_Point': ['(50.8467, 4.3525)', '(50.4113, 4.4447)']
    }
    df = pd.DataFrame(data)
    path = tmp_path / "stations.csv"
    df.to_csv(path, index=False)
    return path

@pytest.fixture
def mock_delay_data(tmp_path):
    # Create temporary delays CSV
    data = {
        'Stopping place (FR)': ['Bruxelles-Central', 'Charleroi-Central', 'Charleroi-Central'],
        'Delay at arrival': [300, 120, 240]  # in seconds
    }
    df = pd.DataFrame(data)
    path = tmp_path / "delays.csv"
    df.to_csv(path, index=False)
    return path

def test_prepare_data(mock_station_data, mock_delay_data):
    bubble_map = DelayBubbleMap(stations_path=mock_station_data, delay_data_path=mock_delay_data)

    # Check merged data
    merged = bubble_map.merged
    assert not merged.empty
    assert 'Geo_Point' in merged.columns
    assert all(isinstance(pt, tuple) for pt in merged['Geo_Point'])
    assert merged.loc[merged['Name_FR'] == 'Charleroi-Central', 'Total_Delay_Minutes'].iloc[0] == pytest.approx((120 + 240) / 60)

def test_render_map_returns_folium_map(mock_station_data, mock_delay_data):
    bubble_map = DelayBubbleMap(stations_path=mock_station_data, delay_data_path=mock_delay_data)
    fmap = bubble_map.render_map()

    import folium
    assert isinstance(fmap, folium.Map)
