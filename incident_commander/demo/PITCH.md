# Incident Commander — 2-minute pitch script

A timed walkthrough for the submission video. Read at a normal pace it lands
just under two minutes; cuts marked `[CUT]` are safe drops if you need to
trim further. Each section is keyed to what should be on-screen.

---

## [0:00 – 0:20] Problem

> "Incident Commander is a long-horizon, multi-actor RL environment that puts
> an LLM in the seat of an on-call engineer during a simulated outage. Three
> tasks, deterministic faults, six-component programmatic rubric — no LLM
> judge. It hits both the multi-actor and long-horizon themes and clears the
> 'real-world task' bar: SRE incident response is a multi-billion-dollar
> profession at PagerDuty, Datadog, Grafana."

**On screen:** the FastAPI `/web` UI mid-incident, alerts firing, dashboard
red, NPC chat feed scrolling.

---

## [0:20 – 0:40] Three qualitatively different tasks

> "Three tasks, picked so they exercise *different* reasoning shapes — not
> just different difficulty knobs."

**On screen:** `openenv.yaml`, three tasks block visible.

> "Easy: deductive — a canary deploy regressed payments, you rollback. Medium:
> attributional — payment webhooks are failing, but is it Stripe down, our
> integration broken, or our deploy bad? Three variants picked by seed.
> Hard: silent data corruption — no alerts, dashboard stays green, the only
> way to find it is to query the audit log because a customer reported a
> wrong account balance. Different mitigation kind, different comms
> channel, different telemetry surface."

---

## [0:40 – 1:05] The rubric is structural, not heuristic

> "Six components sum to 1.0: containment, MTTR, RCA, mitigation, comms,
> post-mortem. Every component is a deterministic pure function of
> simulator state. No LLM-as-judge anywhere."

**On screen:** `graders/rubric.py`, scroll past `_update_comms` and
`_update_mitigation`.

> "The anti-gaming guards are structural, not threshold heuristics. The
> `hold` mitigation only credits when ground-truth is `hold`. A second
> status-page update inside 60 seconds zeros the comms component. On the
> hard task the comms grader is task-conditional — `status_page` on a
> silent corruption earns *zero*, you have to send a targeted
> customer_email with a cohort field. And the data-corruption fault
> matcher rejects a full `rollback`; only `partial_rollback` fixes it."

---

## [1:05 – 1:30] Headline demo: same agent, +0.4 score

**On screen:** `python -m incident_commander.demo.run_episodes` running
live in a terminal.

> "Three episodes, all deterministic, replayable bit-identically. Easy
> oracle: 0.872. Hard oracle: 0.855 — same ceiling, even though the agent
> had to read audit events instead of metrics, send customer emails
> instead of status updates, and call partial_rollback instead of
> rollback."

> "Now watch episode three. Same agent, same hard task, same six rubric
> components, same level of effort. The diagnose is correct, the
> post-mortem is correct, the timeline is correct. The agent just used
> the easy-task playbook — status_page broadcast, full rollback. Score
> drops to **0.479**. The 0.4 delta lives entirely in the rubric. The
> agent earned it by recognising the task *type*, not by being smarter."

---

## [1:30 – 1:50] Training evidence

> "We trained Llama-3.2-3B with TRL GRPO + Unsloth 4-bit LoRA. Pre-training
> baseline: 0.736. Post-training: 0.380 — **regressed**. Parse failures
> dominated the advantage signal, no KL penalty, mode collapse at low
> temperature. We archived that as a failed ablation in the README."

**On screen:** README baseline table.

> "We pivoted to inference-time best-of-N sampling at temperature 0.9 —
> the rubric is a verifier, so reward-guided sampling is a valid
> post-training lever. Three samples were enough to match the oracle
> ceiling at 0.872 on the first draw, mean 0.855. The HF free tier ran
> out at sample three, but the signal is clean: the env supports both
> trained-weight and inference-time improvement strategies, and we
> reported the failed run honestly."

---

## [1:50 – 2:00] Close

> "Three tasks, structural anti-gaming, deterministic replay, the env
> ships at openenv build with a Docker image and 12 green smoke tests.
> Built solo in 48 hours. Thank you."

**On screen:** the HF Space URL.

---

## Recording notes

- **Terminal:** run `python -m incident_commander.demo.run_episodes`. The
  script prints headers in box-drawing characters and is laid out for a
  single-pane recording at 100×40. The whole run is well under 5 seconds.
- **Web UI:** `IC_TASK_ID=hard_silent_data_corruption uv run server`,
  open `http://localhost:8000/web`. Take ~10 seconds of B-roll showing
  the green dashboard during the silent-corruption scenario — the
  visual silence is rhetorically stronger than describing it.
- **Code shots:** `graders/rubric.py` lines 189-260 (the
  task-conditional comms branch) and `simulator/faults.py` lines 386-420
  (the strict `partial_rollback` matcher) are the two five-second cuts
  that prove the structural anti-gaming claim.
- **JSON artifact:** `demo/episodes.json` is produced by the same script
  and contains every action + per-component breakdown for all three
  episodes. Reviewers can verify the headline numbers without running
  anything.
