# Codynamic Head IMU to UR5e Simulator

**A functional simulation architecture for real-time causal reconstruction in embodied robotics.**

---

## Overview

This project implements a modular, real-time simulator that translates **head-mounted IMU data** into motion commands for a **UR5e robotic arm**, using a **causally consistent**, **rewindable**, and **logarithmically-integrated** simulation framework.

This is not just another control stack. It’s a **codynamic simulator**—one that models **becoming** rather than merely computing state updates.

---

## What Makes This Novel?

Most simulators evolve state forward in time using a fixed structure. This simulator doesn’t.

It:

- **Rewinds and re-evaluates** past states when new data arrives  
- **Samples trajectories** not just forward but across temporal ambiguity  
- **Chooses the most causally consistent path** among alternatives  
- **Maintains modularity** at the level of sensors, estimators, state representations, and effectors  

---

## Architecture

The codebase is organized for maximum modularity and extensibility.

```bash
codynamic_simulator/
├── core/                # Abstract interfaces & reusable codynamic logic
│   ├── codynamic_simulator.py
│   └── data_provider.py
│
├── implementations/     # Concrete implementations (Kalman filters, state models)
│   └── ...
│
├── inputs/              # IMU and other sensor feeds (live or simulated)
│   └── ...
│
├── systems/             # Full sensor-to-effector bindings (e.g. head-to-UR5e)
│   └── ...
│
├── tests/               # Unit and integration tests
│   └── test_simulator.py
```
Each layer has one job:
	- core/: Defines what a codynamic system is — not how it’s implemented
	- implementations/: Algorithmic primitives for filtering and estimation
	- inputs/: Time-stamped sensor sources
	- systems/: Compose a full inference-actuation loop

⸻

## The Codynamic Protocol

All simulators follow the same fundamental interface:

class CodynamicSimulator:
    def rewind_and_update(self, t_groundtruth, provider):
        """Reconstruct causal state at t_groundtruth using latest sensor history."""

    def get_state_at_time(self, t_query):
        """Access simulator state at an arbitrary point in the past or present."""

    def sample_state(self, now, dt):
        """Emit current predicted output for actuation, aware of uncertainty."""

This protocol allows the simulator to:
	- Deal with asynchronous, delayed, or contradictory sensor data
	- Adapt its structural understanding over time
	- Maintain multiple hypotheses and collapse based on causal consistency

⸻

## Application: Head-Controlled UR5e Arm

The primary system implemented here uses:
	- IMU orientation data from a headset
	- To control the TCP (Tool Center Point) of a UR5e arm
	- With spring-back behavior for yaw and spherical radial control
	- Ensuring the TCP’s Z-axis remains normal to the control sphere

This enables intuitive teleoperation with physically grounded logic, suitable for real-world embodied agents.

⸻

## Testability

Every module is fully testable in isolation. The architecture was designed to support:
	- Sensor stubbing
	- Historical replay
	- Deterministic consistency checks
	- Plug-and-play estimator swaps

Run tests with:

pytest tests/


⸻
## Vision

This simulator is a substrate for modeling process, not just physics.

It’s designed to grow into:
	- A general-purpose framework for recursive perception-action cycles
	- A platform for modeling systems where structure is not fixed, but emerges through interaction
	- A tool for embedding duronic logic—the idea that decisions arise in time’s extended now, not a discrete tick

⸻

## Author

Arthur Petron — darthur.space
MIT x3 · Robotics · AI · Mathematics · Codynamic Systems

⸻

## License

MIT License. See LICENSE file.