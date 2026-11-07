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

"""Python implementation of the El Farol Bar problem.

Every round, each player decides at the same time whether to go to the bar or
stay at home. Going is worth +1 if at most `capacity` players go and -1 if the
bar is crowded; staying home is worth 0. Players only see how many people went
in each round, not who.

With an odd number of players and `capacity = (players - 1) // 2`, a player who
goes wins exactly when going is the minority choice. This is the threshold game
that motivated the Minority Game; unlike the Minority Game, staying home always
pays 0.

References:
  W. Brian Arthur, "Inductive Reasoning and Bounded Rationality", American
  Economic Review 84(2), 1994.
  D. Challet and Y.-C. Zhang, "Emergence of cooperation and organization in an
  evolutionary game", Physica A 246, 1997.
"""

import enum

import numpy as np

import pyspiel

_MAX_PLAYERS = 100
_DEFAULT_PARAMS = {"players": 5, "capacity": 3, "num_rounds": 10}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_el_farol",
    long_name="Python El Farol Bar",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_MAX_PLAYERS,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class Action(enum.IntEnum):
  STAY = 0
  GO = 1


class ElFarolGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    num_players = params["players"]
    capacity = params["capacity"]
    num_rounds = params["num_rounds"]
    if not 2 <= num_players <= _MAX_PLAYERS:
      raise ValueError(f"players must be in [2, {_MAX_PLAYERS}]")
    if not 0 <= capacity < num_players:
      raise ValueError("capacity must be in [0, players)")
    if num_rounds < 1:
      raise ValueError("num_rounds must be at least 1")
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=len(Action),
            max_chance_outcomes=0,
            num_players=num_players,
            min_utility=-num_rounds,
            max_utility=num_rounds,
            utility_sum=None,
            max_game_length=num_rounds,
        ),
        params,
    )
    self.capacity = capacity
    self.num_rounds = num_rounds

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return ElFarolState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return ElFarolObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        self.num_rounds,
        params)


class ElFarolState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._capacity = game.capacity
    self._num_rounds = game.num_rounds
    self.round_actions = []
    self._rewards = np.zeros(game.num_players())
    self._returns = np.zeros(game.num_players())

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    return pyspiel.PlayerId.SIMULTANEOUS

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    return [Action.STAY, Action.GO]

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self.is_terminal()
    attendance = sum(actions)
    payoff = 1 if attendance <= self._capacity else -1
    self._rewards = np.array(
        [payoff if a == Action.GO else 0 for a in actions], dtype=float)
    self._returns += self._rewards
    self.round_actions.append(list(actions))

  def _action_to_string(self, player, action):
    """Action -> string."""
    return Action(action).name

  def is_terminal(self):
    """Returns True if the game is over."""
    return len(self.round_actions) >= self._num_rounds

  def rewards(self):
    """Reward at the previous step."""
    return self._rewards

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns

  def attendance(self):
    """Number of players who went to the bar in each round so far."""
    return [sum(actions) for actions in self.round_actions]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return " ".join(["attendance:"] + [str(a) for a in self.attendance()])


class ElFarolObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, num_rounds, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self._perfect_recall = iig_obs_type.perfect_recall
    rounds = num_rounds if self._perfect_recall else 1

    pieces = []
    if iig_obs_type.public_info:
      pieces.append(("attendance", rounds))
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("went", rounds))

    self.tensor = np.zeros(sum(size for _, size in pieces), np.float32)
    self.dict = {}
    index = 0
    for name, size in pieces:
      self.dict[name] = self.tensor[index:index + size]
      index += size

  def _rounds(self, state):
    if self._perfect_recall:
      return state.round_actions
    return state.round_actions[-1:]

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    for i, actions in enumerate(self._rounds(state)):
      if "attendance" in self.dict:
        self.dict["attendance"][i] = sum(actions) / len(actions)
      if "went" in self.dict:
        self.dict["went"][i] = actions[player]

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    rounds = self._rounds(state)
    pieces = []
    if "attendance" in self.dict:
      pieces.append("attendance:" + ",".join(str(sum(a)) for a in rounds))
    if "went" in self.dict:
      pieces.append("went:" + "".join("SG"[a[player]] for a in rounds))
    return " ".join(pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, ElFarolGame)
