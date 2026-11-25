# Design Document: Smart-Traffic-RL (v1)

**Status:** Draft 1. Sections marked **Decided** are settled. Anything marked **Open** is deliberately undecided and listed again in [Open decisions](#16-open-decisions).
**Numbers:** Every number in this document is a *starter value*. Starter values are tuned after the baselines have been measured, and each change is recorded.

---

## 1. Problem and goal

Fixed-time traffic lights give each direction the same green time whatever the traffic is doing. When one road is congested and the others are nearly empty, cars wait at a red light that serves nobody.

**Goal:** a controller that observes one intersection and decides which direction gets green, so that cars wait less than under a fixed timer, **without any group of cars being starved**. A busy road may keep a long green, but no car should wait indefinitely.

## 2. Scope

**In v1 (Decided)**
- One intersection.
- Four approaches (N, E, S, W). Each has straight, right and left movements.
- U-turns are allowed and have no signal of their own.
- All lanes have equal size.
- Light physics (see section 4).
- Traffic that changes over the day.
- Vehicles only.

**Out of v1, planned for v2 (Decided)**
- Pedestrians.

**Out entirely for now**
- Multiple intersections (multi-agent RL).
- Emergency vehicles.
- Sensor noise.
- Unequal lane widths.
- Real traffic data.

## 3. Intersection model: four phases (Decided)

| Phase | Movements that get green |
|---|---|
| **P1** | North-South straight and right |
| **P2** | North-South left |
| **P3** | East-West straight and right |
| **P4** | East-West left |

Each phase is a **group**. The agent chooses among these four. Phases are stored as **data in a config file**, listing which movements each phase allows. They are not hard-coded, because v2 adds a pedestrian phase.

## 4. Simulator (Decided)

- **Own simulator** in pure Python and NumPy. It is fast, fully testable, and cheap enough to run in CI. Moving to SUMO is a possible later step.
- **Gymnasium-compatible** (`reset` / `step`).
- **Time:** 1 simulated second per tick. The agent decides every **5 seconds**.
- **Arrivals:** random (Poisson) arrivals per movement. The rate changes over the simulated day according to the scenario.
- **Light physics:**
  - Cars travel at a fixed speed and join the queue at the stop line.
  - When their phase is green, cars leave at a fixed rate (starter: about 1 car per 2 s per lane) after a start-up delay.
  - There are no acceleration curves.
- **Deterministic:** the same seed and the same config give the same run.

## 5. Metric: what "better" means (Decided; thresholds Open)

Every controller is judged on the same numbers, measured per car over a whole test run.

| Number | Meaning | Role |
|---|---|---|
| **Average wait** | Mean seconds a car spends stopped before crossing | Headline number; lower is better |
| **95th-percentile wait** | The wait that 95% of cars stay under | Fairness guard against "1 car waits 15 minutes" |
| **Cars served / cars still waiting at the end** | Throughput and leftover queue | Sanity check; a controller can't look good while a huge queue builds |

The maximum wait is also reported but is not used for pass/fail, because a single outlier can set it.

**Pass rule (shape):**
- Average wait must beat the **best baseline** by at least **X%**.
- 95th-percentile wait must not be more than **Y%** worse than that baseline.

**X and Y are Open.** They are set after the baselines are run, from their real numbers, not guessed in advance.

**Metric vs. reward:** the metric is the final report card and is what CI judges. The reward (section 6.3) is the per-step score the agent learns from. They are related but not the same.

## 6. Problem formulation (MDP)

### 6.1 State (Decided)

| Feature | Values | Why |
|---|---|---|
| Queue level per phase group (4 features) | low / medium / high (starter bins: 0–3, 4–8, 9+ cars) | Where cars are waiting |
| Current phase | P1–P4 | Needed so the agent knows what is green now |
| Time in current phase | short / medium / long (starter: 0–10 s, 10–40 s, 40 s+) | Whether a switch is allowed yet; whether the phase has run long |
| Starvation flag | yes / no: is any car waiting more than 90 s? | Cheap view of unfairness |

**Size:** 3⁴ × 4 × 3 × 2 = **1,944 states**, small enough for a Q-table.

**Markov check:** the state includes the current phase and the time in it. Without the current phase the agent could not tell which light to change, and unsafe behaviour would follow. Arrival *trends* are not in the state, which is an accepted simplification.

**Deliberately left out**
- Exact car counts: too many states for a table. A deep-RL version could use them.
- Time of day: the agent would memorize the training patterns instead of reacting to what it sees.
- Car speeds and positions: not needed with light physics.
- Oldest-car wait per group: it would multiply the state space by 81. A single flag gives most of the benefit.

The road, lane and intersection layout is fixed, so it belongs to the environment and not to the state.

### 6.2 Actions and hard constraints (Decided)

Every 5 simulated seconds the agent chooses **one of the 4 phases**. Choosing the phase that is already green means "extend it".

The **simulator enforces** these rules. The agent cannot break them, even while exploring:

| Rule | Starter value |
|---|---|
| Minimum green before a phase can change | 10 s |
| Clearing between phases (yellow, then all-red) | 3 s + 2 s |
| Maximum green, then a forced change | 120 s |

Constraints are enforced in the environment, not through reward penalties. Hard rules cannot be violated, they don't have to be learned, and they are safer for something that controls real traffic.

**Alternative considered:** only "keep" or "advance to the next phase in fixed order". It is simpler, but it cannot skip an empty phase, and skipping is where adaptive control gains the most.

### 6.3 Reward (Decided; expected to be revised)

Each 5-second step:

> **reward = −(total waiting cost of all stopped cars during the step)**

A stopped car costs points every second, and the price rises with how long it has waited:

> **cost per second = 1 + (seconds waited so far ÷ 60)**

| Car has waited | Cost per second |
|---|---|
| 0 s | 1 |
| 60 s | 2 |
| 120 s | 3 |
| 300 s | 6 |

**Why this shape**
- With "minus the number of stopped cars", a lone car in a left-turn lane would cost 1 point per second forever. Nine cars elsewhere would always win, and that car would be starved. With a rising price, a car that has waited 5 minutes costs as much as 6 fresh cars, so the agent goes to serve it.
- Total car-seconds of waiting divided by the number of cars is the average wait, so the reward lines up with the headline metric. The rising price lines up with the 95th-percentile guard.
- **No separate switching penalty.** Cars sit stopped during yellow and all-red, so switching already costs points. One less setting means one less thing to tune.

**Reward-hacking checks**
After training, compare against the baselines and look at:
- the 95th-percentile and maximum wait
- cars left waiting at the end of a run
- any phase the agent never chooses

If the reward improves while any of these get worse, the reward is being gamed and it needs to change.

**The reward is implemented as one function made of weighted terms**, so v2 can add a pedestrian term.

## 7. Baselines (Decided)

All baselines obey exactly the same rules as the learned agent, so the comparison is fair.

| Baseline | Behaviour | Purpose |
|---|---|---|
| **Fixed-time** | Cycle P1→P2→P3→P4, fixed green each (starter: 25 s) | What most real intersections do; the minimum bar |
| **Random** | Random phase at each decision | Sanity check of the simulator and evaluation |
| **Greedy** | Choose the group with the most waiting cars | The honest benchmark: a sensible hand-written rule |

## 8. Algorithms (Decided)

1. **Tabular Q-learning, hand-written.** This is the core of v1. It is small, understandable and fast to train.
2. **DQN and PPO via Stable-Baselines3.** Optional add-on after v1, once the core is done.

Q-learning hyperparameters (learning rate, discount factor, exploration schedule) are **Open**.

## 9. Scenarios and evaluation (Decided; one scenario Open)

A **scenario** is a recipe: arrival rates per movement across the simulated day. Scenarios live in config files.

| Set | Scenarios |
|---|---|
| **Train** | Morning rush (N-S heavy), evening rush (heavy the other way), balanced midday, quiet night |
| **Test (held out, never used for training)** | East-West heavy, sudden event surge, plus one more (**Open**, candidate: left-turn heavy) |

- During training, arrival rates are randomly varied by about ±30% each episode so the agent doesn't memorize exact numbers.
- A new random seed on a training recipe is **not** a held-out test. A held-out test needs a recipe the agent has never seen.
- Results are reported over **several seeds** as mean and spread.
- Every learned agent is always reported next to all three baselines.

## 10. Architecture

**Components**
1. Simulation environment
2. Traffic scenarios (config)
3. Agents / policies (baselines and learners)
4. Training
5. Evaluation (the heart of the CI quality gate)
6. Model registry (saved policies plus their metrics)
7. Serving API (FastAPI)
8. Demo front-end (React, built with Vite)
9. Configuration

**Principles (Decided)**
- **One shared agent interface.** Every controller, from fixed-time to PPO, maps state to action the same way, so any two can be compared directly.
- **One-way dependencies.** The environment does not import agents. The serving API does not import training code.
- **Configuration over code.** Scenarios, phases and hyperparameters live in config files.
- **Determinism by design.** Same seed and config always give the same result.

Detailed interfaces (what `reset` and `step` return, how an agent is saved and loaded, what an evaluation report contains, what the API accepts and returns) are **Open** and written during Stage 1.

## 11. Testing

**Unit tests: simulator invariants** (starting list)
- Queues are never negative.
- Cars are conserved: arrived = departed + still waiting.
- No movement discharges on red or during clearing.
- The minimum green time is respected.
- Reaching the maximum green forces a change.
- The same seed produces the same episode.
- Every agent returns a valid action for a valid state.
- The fixed-time baseline changes phase exactly when configured to.

**Smoke test:** train for a few steps, save the model, load it back, and check that nothing crashes.

**Quality gate:** train with a small budget, evaluate on the CI scenarios across several seeds, and fail the build if the pass rule in section 5 is not met. Averages over seeds, plus the margins X and Y, keep a lucky or unlucky single run from deciding the result.

**CI jobs**
- **Python:** Ruff, mypy, pytest.
- **Front-end:** ESLint, Prettier, Vitest, Vite build.

## 12. Delivery pipeline

1. Pull request: lint, type check, unit tests, smoke training.
2. Merge to `main`: train, evaluate, apply the quality gate.
3. **Champion vs. challenger:** compare the new model's metrics file against the current champion's metrics file. Promote only if it is better under the same evaluation version.
4. Build the Docker image.
5. Deploy to **staging**, then wait for **manual approval**, then deploy to **production** (Render).

The environment and evaluation suite are **versioned** (`env v1`). Scores are only compared within the same version.

Rollback mechanism: **Open**.

## 13. Planned v2: pedestrians

To keep v2 mostly additive, v1 follows these habits:
1. Phases are data in config, not hard-coded.
2. The state is built in one function.
3. The reward is one function of weighted terms.
4. Scenario files have an optional pedestrian rate that defaults to 0.
5. The environment is versioned. A v2 environment starts a fresh champion, and old models are retrained.

**What stays the same:** CI/CD workflows, evaluation code, the agent interface, the API and React shell, the Q-learning code, and config loading.

**What changes:** the simulator, the phases (a walk phase), the state, the reward, the scenarios, the metrics, and the trained models.

## 14. Time budget: about 14 hours (Decided)

| Step | Hours |
|---|---|
| Learn the minimum (Spinning Up Part 1, Sutton & Barto §6.5) | 1.5 |
| This design document | 1.0 |
| Simulator, fixed and random baselines, invariant tests | 2.5 |
| CI: Ruff and pytest, plus one deliberate red build | 1.0 |
| Evaluation, greedy baseline, scenarios | 1.5 |
| Q-learning agent, training, save and load | 2.0 |
| Quality gate in CI and the champion check | 1.5 |
| FastAPI and Dockerfile | 1.0 |
| Deploy to Render: staging, approval, production | 1.0 |
| React demo page and README results | 1.0 |
| **Total** | **14.0** |

**If over budget, cut in this order:** (1) the React demo, (2) the champion/challenger step (the quality gate stays), (3) the extra scenarios.
The biggest risks are simulator bugs and reward tuning, so each attempt is time-boxed.

## 15. Known limitations

- **Sim-to-real gap:** the agent learns from a simplified simulation. Real deployment would need a realistic simulator and real sensor data.
- **Light physics:** no acceleration curves or driver variety.
- **Single intersection:** coordination across intersections is out of scope.
- **Vehicles only** in v1.
- **Starter values:** all thresholds and timings are untuned starting points until measured.

## 16. Open decisions

- [ ] Metric thresholds **X** and **Y** (set after running the baselines)
- [ ] Tuning of all starter values (bins, timings, green times, thresholds)
- [ ] The third held-out test scenario
- [ ] Interface definitions (environment, agent, evaluation report, API)
- [ ] Q-learning hyperparameters
- [ ] Rollback mechanism for the champion model
- [ ] Exact contents of the API and the demo page

## 17. Reward revision log

| Version | Reward | Reason for change |
|---|---|---|
| v1 | −Σ over stopped cars of (1 + wait so far ÷ 60), per second | Initial design: penalize waiting, penalize long waits more, no switching penalty |

*Each future change to the reward is added here with the reason it was made.*
