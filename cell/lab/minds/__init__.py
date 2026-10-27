"""The minds: what decides what the arm does. Each has reset(), act(obs) -> rows, done, status(), and optionally
score(truth meshes) for the simulators' truth check. v1's sort policy is archived (git checkout v1-sort)."""
from lab.minds.mapper import MapperMind
from lab.minds.sorter import SorterMind
from lab.minds.twostream import TwoStreamMind
from lab.minds.focus import FocusMind

MINDS = {"map": MapperMind, "sort": SorterMind, "streams": TwoStreamMind, "focus": FocusMind}
