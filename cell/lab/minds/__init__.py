"""The minds: what decides what the arm does. Each has reset(), act(obs) -> rows, done, status(), and optionally
score(truth meshes) for the simulators' truth check. v1's sort policy is archived (git checkout v1-sort)."""
from lab.minds.mapper import MapperMind

MINDS = {"map": MapperMind}
