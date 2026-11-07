# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for el_farol.py."""

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import el_farol
import pyspiel

STAY = el_farol.Action.STAY
GO = el_farol.Action.GO


class ElFarolTest(absltest.TestCase):

  def test_default_params(self):
    game = pyspiel.load_game("python_el_farol")
    self.assertEqual(game.num_players(), 5)
    self.assertEqual(game.capacity, 3)
    self.assertEqual(game.max_game_length(), 10)

  def test_params_from_string(self):
    game = pyspiel.load_game(
        "python_el_farol(players=7,capacity=4,num_rounds=3)")
    self.assertEqual(game.num_players(), 7)
    self.assertEqual(game.capacity, 4)
    self.assertEqual(game.max_game_length(), 3)

  def test_invalid_params(self):
    with self.assertRaises(ValueError):
      el_farol.ElFarolGame({"players": 3, "capacity": 3, "num_rounds": 10})
    with self.assertRaises(ValueError):
      el_farol.ElFarolGame({"players": 5, "capacity": 3, "num_rounds": 0})

  def test_uncrowded_bar_rewards_goers(self):
    state = pyspiel.load_game("python_el_farol").new_initial_state()
    state.apply_actions([GO, GO, STAY, GO, STAY])
    np.testing.assert_array_equal(state.rewards(), [1, 1, 0, 1, 0])

  def test_crowded_bar_punishes_goers(self):
    state = pyspiel.load_game("python_el_farol").new_initial_state()
    state.apply_actions([GO, GO, GO, GO, STAY])
    np.testing.assert_array_equal(state.rewards(), [-1, -1, -1, -1, 0])

  def test_minority_game_setting(self):
    game = pyspiel.load_game("python_el_farol(players=5,capacity=2)")
    state = game.new_initial_state()
    state.apply_actions([GO, GO, STAY, STAY, STAY])
    np.testing.assert_array_equal(state.rewards(), [1, 1, 0, 0, 0])
    state.apply_actions([GO, GO, GO, STAY, STAY])
    np.testing.assert_array_equal(state.rewards(), [-1, -1, -1, 0, 0])

  def test_game_ends_after_num_rounds(self):
    game = pyspiel.load_game("python_el_farol(num_rounds=2)")
    state = game.new_initial_state()
    state.apply_actions([GO, STAY, STAY, STAY, STAY])
    self.assertFalse(state.is_terminal())
    state.apply_actions([GO, GO, GO, GO, GO])
    self.assertTrue(state.is_terminal())
    np.testing.assert_array_equal(state.returns(), [0, -1, -1, -1, -1])

  def test_players_see_attendance_not_who_went(self):
    game = pyspiel.load_game("python_el_farol")
    a = game.new_initial_state()
    a.apply_actions([GO, GO, STAY, STAY, STAY])
    b = game.new_initial_state()
    b.apply_actions([STAY, STAY, GO, GO, STAY])
    self.assertEqual(a.observation_string(4), b.observation_string(4))
    self.assertEqual(a.observation_tensor(4), b.observation_tensor(4))
    self.assertNotEqual(a.observation_string(0), b.observation_string(0))
    self.assertEqual(a.observation_string(0), "attendance:2 went:G")

  def test_information_state_keeps_full_history(self):
    game = pyspiel.load_game("python_el_farol")
    state = game.new_initial_state()
    state.apply_actions([GO, GO, STAY, STAY, STAY])
    state.apply_actions([GO, GO, GO, GO, STAY])
    self.assertEqual(
        state.information_state_string(0), "attendance:2,4 went:GG")
    self.assertEqual(state.observation_string(4), "attendance:4 went:S")
    self.assertLen(state.information_state_tensor(0), 20)

  def test_random_sim(self):
    game = pyspiel.load_game("python_el_farol")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)

  def test_game_as_turn_based(self):
    game = pyspiel.convert_to_turn_based(
        pyspiel.load_game("python_el_farol(players=3,capacity=1,num_rounds=4)"))
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=True)


if __name__ == "__main__":
  absltest.main()
