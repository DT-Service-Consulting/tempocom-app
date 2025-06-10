# Filename: test_macro_network.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.MacroNetwork import MacroNetwork

class TestNetwork:

    def setup_method(self):
        self.network = MacroNetwork("/home/learner/Desktop/internship/Dealy_analysis/tempocom-app/mart")

    def test_get_shortest_path(self):
        _, distance = self.network.get_shortest_path('Charleroi-Central', 'Bruxelles-Central')
        assert round(distance, 2) == 57.1

    def test_get_shortest_path_no_path(self):
        self.network.close_links([('Walcourt', 'Pry')])
        _, distance = self.network.get_shortest_path('Couvin', 'Charleroi-Central')
        assert distance is None

    def test_get_shortest_path_same_station(self):
        _, distance = self.network.get_shortest_path('Charleroi-Central', 'Charleroi-Central')
        assert distance == 0

    def test_cut_and_get_shortest_path(self):
        _, distance1 = self.network.get_shortest_path('Charleroi-Central', 'Bruxelles-Central')
        self.network.close_links([('Luttre', 'Courcelles-Motte')])
        _, distance2 = self.network.get_shortest_path('Charleroi-Central', 'Bruxelles-Central')
        assert round(distance1, 2) != round(distance2, 2)
