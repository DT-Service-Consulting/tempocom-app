"""
Unit tests for the MacroNetwork class in objects.MacroNetwork.

Tests:
- Shortest path calculation between stations.
- Handling of no-path scenarios after closing links.
- Shortest path when start and end stations are the same.
- Effect of closing links on shortest path calculation.

Setup:
- Loads environment variables from '../../tempocom_config/.env'.
- Initializes a MacroNetwork instance for each test.

Author: Mohamd Hussain  
Date: [2025-06-20]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from features.macro_network.MacroNetworkController import MacroNetwork
from tempocom.services import DBConnector


class TestNetwork:
    dbc = DBConnector()
    Charleroi_Central_ID = 259
    Bruxelles_Central_ID = 215
    Couvin_ID = 291
    Walcourt_ID = 1207
    Pry_ID = 976
    Luttre_ID = 768
    Courcelles_Motte_ID = 286

    def setup_method(self):
        print("Setting up test network...")
        self.network = MacroNetwork(self.dbc)
        print("Test network setup complete")

    def test_get_shortest_path(self):
        print("Testing shortest path between Charleroi-Central and Bruxelles-Central")
        distance = self.network.macro_links.get_shortest_path_distance(self.Charleroi_Central_ID, self.Bruxelles_Central_ID)
        print(f"Distance found: {distance}")
        print(f"Rounded distance: {round(distance, 2)}")
        assert round(distance, 2) == 57.0
    
    def test_get_shortest_path_no_path(self):
        print("Testing no path scenario")
        print("Closing link between Walcourt and Pry")
        self.network.macro_links.close_links([(self.Walcourt_ID, self.Pry_ID)])
        print("Searching path between Couvin and Charleroi-Central")
        distance = self.network.macro_links.get_shortest_path_distance(self.Couvin_ID, self.Charleroi_Central_ID)
        print(f"Distance result: {distance}")
        assert distance is None
    
    def test_get_shortest_path_same_station(self):
        print("Testing same station path")
        distance = self.network.macro_links.get_shortest_path_distance(self.Charleroi_Central_ID, self.Charleroi_Central_ID)
        print(f"Same station distance: {distance}")
        assert distance == 0

    def test_cut_and_get_shortest_path(self):
        print("Testing path change after cutting link")
        print("Getting initial path from Charleroi-Central to Bruxelles-Central")
        distance1 = self.network.macro_links.get_shortest_path_distance(self.Charleroi_Central_ID, self.Bruxelles_Central_ID)
        print(f"Initial distance: {distance1}")
        print("Closing link between Luttre and Courcelles-Motte")
        self.network.macro_links.close_links([(self.Luttre_ID, self.Courcelles_Motte_ID)])
        print("Getting new path after closing link")
        distance2 = self.network.macro_links.get_shortest_path_distance(self.Charleroi_Central_ID, self.Bruxelles_Central_ID)
        print(f"New distance: {distance2}")
        print(f"Distances comparison - Before: {round(distance1, 2)}, After: {round(distance2, 2)}")
        assert round(distance1, 2) != round(distance2, 2)