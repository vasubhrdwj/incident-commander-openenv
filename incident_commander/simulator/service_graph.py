"""
Service topology used by the Incident Commander simulator.

We model a small fintech microservices product with six services and a fixed
dependency DAG. Every service carries a *baseline* health profile (latency,
error rate, traffic) that the simulator perturbs when a fault is active.

The topology is small but has enough structure to exercise all three task
scenarios:

* ``api_gw`` fans out to the three "front-door" services (``auth``,
  ``payments``, ``orders``).
* ``orders`` depends on ``inventory`` and ``payments`` (so payments outages
  cascade).
* ``payments`` depends on an external provider (used by the third-party
  attribution task, not modelled as an in-graph service).
* ``notifications`` is a fire-and-forget leaf dependency.

All numbers are deterministic and carry no units beyond "reasonable for a
production microservice" — we only need them to be consistent so alerts and
graders can reason about thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServiceSpec:
    """Static config for one service in the topology."""

    name: str
    depends_on: tuple[str, ...] = ()
    baseline_latency_p99_ms: float = 50.0
    baseline_error_rate: float = 0.002
    baseline_rps: float = 100.0
    slo_latency_p99_ms: float = 200.0
    slo_error_rate: float = 0.01
    external_provider: str | None = None


@dataclass
class ServiceState:
    """Mutable per-service state carried inside the simulator."""

    spec: ServiceSpec
    latency_p99_ms: float
    error_rate: float
    requests_per_sec: float
    has_active_deploy: bool = False
    active_deploy_tag: str | None = None
    is_rolled_back: bool = False

    def reset_to_baseline(self) -> None:
        self.latency_p99_ms = self.spec.baseline_latency_p99_ms
        self.error_rate = self.spec.baseline_error_rate
        self.requests_per_sec = self.spec.baseline_rps
        self.has_active_deploy = False
        self.active_deploy_tag = None
        self.is_rolled_back = False

    def is_healthy(self) -> bool:
        return (
            self.latency_p99_ms <= self.spec.slo_latency_p99_ms
            and self.error_rate <= self.spec.slo_error_rate
        )


@dataclass
class ServiceGraph:
    """Topology + current per-service state.

    The graph is passed to every simulator component that needs to know about
    services; it owns the mutable health state.
    """

    services: dict[str, ServiceState] = field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.services.keys())

    def get(self, name: str) -> ServiceState | None:
        return self.services.get(name)

    def set_health(
        self,
        name: str,
        *,
        latency_p99_ms: float | None = None,
        error_rate: float | None = None,
        requests_per_sec: float | None = None,
    ) -> None:
        state = self.services[name]
        if latency_p99_ms is not None:
            state.latency_p99_ms = latency_p99_ms
        if error_rate is not None:
            state.error_rate = error_rate
        if requests_per_sec is not None:
            state.requests_per_sec = requests_per_sec

    def reset_all(self) -> None:
        for state in self.services.values():
            state.reset_to_baseline()

    def upstream_of(self, service: str) -> list[str]:
        """Services that directly depend on ``service`` (``service`` is upstream to them)."""
        return [
            name
            for name, state in self.services.items()
            if service in state.spec.depends_on
        ]


DEFAULT_SPECS: tuple[ServiceSpec, ...] = (
    ServiceSpec(
        name="api_gw",
        depends_on=("auth", "payments", "orders"),
        baseline_latency_p99_ms=40.0,
        baseline_rps=800.0,
    ),
    ServiceSpec(
        name="auth",
        baseline_latency_p99_ms=25.0,
        baseline_rps=400.0,
    ),
    ServiceSpec(
        name="payments",
        external_provider="stripe",
        baseline_latency_p99_ms=80.0,
        baseline_error_rate=0.003,
        baseline_rps=150.0,
    ),
    ServiceSpec(
        name="orders",
        depends_on=("inventory", "payments"),
        baseline_latency_p99_ms=90.0,
        baseline_rps=180.0,
    ),
    ServiceSpec(
        name="inventory",
        baseline_latency_p99_ms=30.0,
        baseline_rps=220.0,
    ),
    ServiceSpec(
        name="notifications",
        baseline_latency_p99_ms=60.0,
        baseline_error_rate=0.005,
        baseline_rps=90.0,
    ),
)


def build_default_topology() -> ServiceGraph:
    """Return a fresh, healthy :class:`ServiceGraph` with the default 6-service topology."""
    services: dict[str, ServiceState] = {}
    for spec in DEFAULT_SPECS:
        services[spec.name] = ServiceState(
            spec=spec,
            latency_p99_ms=spec.baseline_latency_p99_ms,
            error_rate=spec.baseline_error_rate,
            requests_per_sec=spec.baseline_rps,
        )
    return ServiceGraph(services=services)
