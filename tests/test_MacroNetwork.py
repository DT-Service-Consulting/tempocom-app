import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.MacroNetwork import MacroNetwork

class TestNetwork:

    def setup_method(self):
        print("Setting up test network...")
        self.network = MacroNetwork()
        print("Test network setup complete")

    def test_get_shortest_path(self):
        print("Testing shortest path between Charleroi-Central and Bruxelles-Central")
        _,distance = self.network.get_shortest_path('Charleroi-Central', 'Bruxelles-Central')
        print(f"Distance found: {distance}")
        print(f"Rounded distance: {round(distance, 2)}")
        assert round(distance, 2) == 57.1
    
    def test_get_shortest_path_no_path(self):
        print("Testing no path scenario")
        print("Closing link between Walcourt and Pry")
        self.network.close_links([('Walcourt', 'Pry')])
        print("Searching path between Couvin and Charleroi-Central")
        _,distance = self.network.get_shortest_path('Couvin', 'Charleroi-Central')
        print(f"Distance result: {distance}")
        assert distance is None
    
    def test_get_shortest_path_same_station(self):
        print("Testing same station path")
        _,distance = self.network.get_shortest_path('Charleroi-Central', 'Charleroi-Central')
        print(f"Same station distance: {distance}")
        assert distance == 0

    def test_cut_and_get_shortest_path(self):
        print("Testing path change after cutting link")
        print("Getting initial path from Charleroi-Central to Bruxelles-Central")
        _,distance1 = self.network.get_shortest_path('Charleroi-Central', 'Bruxelles-Central')
        print(f"Initial distance: {distance1}")
        print("Closing link between Luttre and Courcelles-Motte")
        self.network.close_links([('Luttre', 'Courcelles-Motte')])
        print("Getting new path after closing link")
        _,distance2 = self.network.get_shortest_path('Charleroi-Central', 'Bruxelles-Central')
        print(f"New distance: {distance2}")
        print(f"Distances comparison - Before: {round(distance1, 2)}, After: {round(distance2, 2)}")
        assert round(distance1, 2) != round(distance2, 2)