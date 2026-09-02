# Sampling Strategies

Sampling controls the fraction of traces (and associated spans) that are collected, stored, and queryable. A good sampling policy keeps costs and storage predictable without hiding the traces that matter most — errors, slow paths, and production incidents.

## Table of Contents

- [Head sampling](#head-sampling)
- [Tail sampling](#tail-sampling)
- [Reconstructing truth from sampled events](#reconstructing-truth-from-sampled-events)
- [Percentiles under sampling](#percentiles-under-sampling)
- [Consistent sampling: one decision per trace](#consistent-sampling-one-decision-per-trace)
- [Target-rate sampling](#target-rate-sampling)
- [OTel Collector tail-sampling processor](#otel-collector-tail-sampling-processor)
- [Sampling bias — known trap](#sampling-bias--known-trap)
- [Exemplars: wiring sampled traces to Prometheus metrics](#exemplars-wiring-sampled-traces-to-prometheus-metrics)

## Head sampling

Head sampling makes a keep/drop decision at the root span, before any child spans are created. The decision propagates via the `traceflags` bit in W3C `traceparent`, so every downstream service respects the same decision and the trace is either complete or absent.

**When to use:** Low-volume services, development, staging, and any situation where operational simplicity outweighs the cost of keeping all traces. A parent-based ratio sampler (e.g. keep 10% of roots, always propagate the sampling decision downstream) is the lowest-complexity production option.

**Trade-off:** Head sampling is blind to outcome. A 1%-sampled request that later errors is dropped with 99% probability. For low-traffic services this probability is acceptable; for high-traffic services it hides rare failures.

OpenTelemetry SDK configuration (environment variable):

```bash
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1   # keep 10%
```

## Tail sampling

Tail sampling makes the keep/drop decision after all spans in a trace have been collected, allowing the decision to be based on the actual outcome: was there an error? was the request slow? did it touch a high-value path?

**When to use:** High-traffic services where head sampling would hide low-frequency failures. Tail sampling is more complex to operate because the collector must buffer spans long enough to see the full trace before deciding.

**Trade-off:** Requires stateful buffering in the collector. All spans for the same trace must be routed to the same collector instance (use load-balancing exporter for multi-instance collectors). Adds memory and latency overhead to the pipeline.

## Reconstructing truth from sampled events

> **Read this as the reasoning behind what your SDK already does, not as a build-it-yourself spec.** The source itself notes that "Increasingly, it's common for open source instrumentation libraries — such as OpenTelemetry — to implement that type of sampling logic for you", and that as those libraries become standard "it should become less likely that you would need to re-implement these sampling strategies in your own code." It also insists that even when you delegate, "it is essential that you understand the underpinnings of how sampling is implemented." That is what this section is for. Prefer the OTel SDK samplers and the Collector's processors; use the math below to reason about their output, to audit a backend's numbers, and to know what breaks when you configure them wrongly.
>
> Source: *Observability Engineering* (Early Release ch. 13; final ed. ch. 18), "Translating sampling strategies into code".

The mitigation earlier in this file still stands: **derive rates and percentiles from metrics, not from sampled traces.** This section covers the other case — when the sampled events are the only record you have, and you must compute from them anyway.

### Record the sample rate inside the event

The naive fixed-rate approach requires the receiving end to remember which rate was in force. That breaks the moment the rate changes, because "the instrumentation collector wouldn't know exactly when the value changed."

The fix is to make each event self-describing: **pass the current sample rate as a field on the event itself**, indicating that this event "statistically represents `sampleRate` similar events". This matters more than it first appears, because sample rates "can not only vary between services, but also vary within a single service as well" — one global constant is wrong on both axes.

Practically: treat the per-event sample rate as a required attribute on any pipeline that samples, the same way you treat a trace ID. An event without its own rate cannot be weighted correctly after the fact.

### Reconstruction math

With the rate recorded per event, two operations become well defined:

| Quantity | How to reconstruct |
| --- | --- |
| **Count** (e.g. events matching `err != nil`) | Multiply the count of each seen matching event by **its own** recorded `sampleRate`, then sum. Add together the number of *represented* events, not the number of *collected* events. |
| **Sum** (e.g. total `durationMs`) | Weight each sampled event's value by its own `sampleRate` before adding the weighted figures. |

Both are per-event, not per-batch: an event representing 1,000 similar events "should not be directly averaged with another event that represents 100 similar events."

Why this matters for cross-checking a backend: a service instrumented with both events and metrics will show metric counters incremented for every request, while only a fraction of requests were sampled as events. Reporting 100 events for 100,000 requests is misleading if each event represents ~1,000. The reconstruction is what makes the two agree.

## Percentiles under sampling

Percentiles split into two cases, and conflating them is the common error.

**Constant-probability sampling — no adjustment needed.** Quoting directly:

> Scalar distribution properties such as the p99 and median do not need to be adjusted for a constant probability sampling, as they are not distorted by the sampling process.

Every event had the same chance of selection, so the sampled distribution's shape matches the population's shape. Multiplying out would be wrong, not merely unnecessary.

**Dynamic / variable rates — expansion is mandatory.** Once rates vary (adjusted by traffic volume, by key, or by outlier status), "you can no longer multiply out each event by a constant factor when reconstructing the distribution of your data." The telemetry system must use a weighted algorithm accounting for the probability in effect when each event was collected: for aggregating median or p99, **expand each event out into many** when computing the total number of events and where the percentile values fall.

The failure mode if you skip this: an aggressively sampled bucket (high-volume, low-rate) is under-represented in the sorted distribution relative to a lightly sampled one, and the resulting p99 reflects the sampling policy as much as the service.

## Consistent sampling: one decision per trace

Spans of one trace are collected across multiple services, "with each service, potentially, employing its own unique sample strategy and rate." If each service rolls its own dice, "the probability that every span necessary to complete a trace will be the event that each service chooses to sample is relatively low" — you get orphaned fragments, and specifically the failure of "sampl[ing] an error far downstream for which the upstream context is missing."

The fix: derive the sampling decision from a **centrally generated sampling/tracing ID propagated to all downstream handlers**, instead of independently generating a decision inside each one. Each service compares that propagated value against its own local rate rather than drawing a fresh random number.

The property this buys — quoted, because the composition is the whole point:

> Consistent sampling guarantees that if a 1:100 sampling occurs, a 1:99, 1:98, etc. sampling preceding or following it also preserves the execution context. And half of the events chosen by a 1:100 sampling will be present under a 1:200 sampling.

In other words, nested rates compose: services with *different* rates still yield coherent traces, and a rate change is a monotone subset relationship rather than a reshuffle. In OpenTelemetry terms this is what `parentbased_traceidratio` and trace-ID-hash-based samplers implement — the ratio is evaluated against the trace ID, not against a fresh random draw. Prefer those over any per-service independent sampler.

## Target-rate sampling

Rather than flag-adjusting each service's rate as "traffic swells and sags", compute the rate from observed traffic. The book's formula, recomputed on an interval (a minute, in its example):

```text
sampleRate = requestsInPastMinute / (60 × targetEventsPerSec)
```

Clamped so that `sampleRate < 1` becomes `1.0` — i.e. never sample *up*; below the target volume, keep everything. The effect is a predictable resource cost: the sampled event stream targets a fixed events-per-second budget regardless of incoming volume.

**Per-key extension.** A single global target still lets one population drown out another, and still misses long-tail events, "because the chance that a 99.9th percentile outlier event will be chosen for random sampling is slim." Two refinements, in order:

1. **Multiple static rates by key** — sample baseline (non-outlier) events at one rate and errors or slow queries at a much more generous one. Still vulnerable to an error spike: "If the application experiences a spike in the rate of errors, every single error gets sampled," and the instrumentation traffic spikes with it.
2. **Per-key target rates** — maintain a separate counter and separate computed rate per key (e.g. `outliersInPastMinute / (60 × outlierEventsPerSec)` alongside the ordinary-request rate). Anomalous requests get their own guaranteed budget while ordinary requests are rate-limited into theirs. Keys can be simple (HTTP method) or composite (method + request size + user-agent, or `[customer ID, dataset ID, error code]`).

A key seen many times in the recent window "is less interesting than combinations that were seen less often" — per-key dynamic rates are how you keep low-volume sources visible without letting high-volume ones set the bill.

**Do not hand-roll this.** The book's own Go examples are pedagogical and flag their own defect — the counter swap is annotated "Real production code would do something less prone to race conditions". Configure the equivalent in the OTel SDK sampler or a Collector processor (`probabilistic_sampler`, `tail_sampling` with `rate_limiting` and `composite` policies) and use the formulas above to set and audit the parameters.

## OTel Collector tail-sampling processor

The OpenTelemetry Collector `tailsampling` processor implements tail sampling without changes to application code. Configure it in the collector pipeline:

```yaml
processors:
  tail_sampling:
    decision_wait: 10s          # buffer window to collect all spans
    num_traces: 50000           # max traces held in memory
    expected_new_traces_per_sec: 10
    policies:
      - name: errors-policy
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: slow-traces-policy
        type: latency
        latency: { threshold_ms: 500 }
      - name: probabilistic-policy
        type: probabilistic
        probabilistic: { sampling_percentage: 5 }

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling]
      exporters: [otlp/backend]
```

Key policy types: `status_code`, `latency`, `probabilistic`, `rate_limiting`, `string_attribute`, `composite`. Combine policies in a `composite` policy with `and_sub_policy` / `or_sub_policy` for fine-grained control.

Reference: [opentelemetry.io/docs/collector/](https://opentelemetry.io/docs/collector/) and the `tailsampling` processor source in [github.com/open-telemetry/opentelemetry-collector-contrib](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor).

## Sampling bias — known trap

Any sampling strategy that drops requests non-uniformly introduces bias into metric calculations derived from traces.

**Common forms:**

- **Survivor bias from head sampling:** Error rates computed from sampled traces undercount errors when errors are rarer than the sampling ratio. A 1% sampler on a service with a 0.01% error rate will drop most errors.
- **Cardinality-driven bias:** If sampling rules key on an attribute such as `user.tier`, premium-tier traces may be fully retained while free-tier traces are aggressively sampled, making error rates across tiers incomparable.
- **Warm/cold bias:** Services sampled by probabilistic ratio produce uneven coverage when traffic spikes — a sudden burst keeps fewer traces in absolute terms even at a constant ratio.
- **Cascade bias:** When a downstream service applies its own head sampling independently of the upstream decision, traces appear truncated and the sampling rate in the stored trace is a product of multiple independent decisions, not the intended single policy.

**Mitigations:**

- Use parent-based sampling so the upstream decision is respected downstream (no independent re-sampling).
- Derive error rates and latency percentiles from metrics (Prometheus counters and histograms), not from trace data. Metrics are not sampled; traces are.
- Use tail sampling to ensure errors and slow traces are always retained regardless of volume.
- Document and version the sampling policy so engineers know what the expected blind spots are.

## Exemplars: wiring sampled traces to Prometheus metrics

Exemplars are sample data points attached to Prometheus histogram and counter observations. Each exemplar carries a trace ID (and optionally a span ID), linking a specific metric observation to the trace that produced it.

**Why this matters:** Metrics give you aggregated truth; traces give you causal detail. Exemplars bridge the gap: from a latency spike on a Prometheus histogram, a single click navigates to the sampled trace that best represents that spike.

**Requirements:**

- Prometheus must be configured to scrape and store exemplars (enabled by default in Prometheus 2.27+; set `--enable-feature=exemplar-storage`).
- The OpenTelemetry SDK must be configured to emit exemplars. For the metrics SDK, set the exemplar filter:
  ```bash
  OTEL_METRICS_EXEMPLAR_FILTER=TRACE_BASED   # only attach when a sampled trace is active
  ```
- The application must have an active sampled span when the metric observation is recorded. Exemplars are only attached when the current trace is sampled.
- Grafana reads exemplars from Prometheus natively; enable the exemplar toggle on histogram panels.

**Sampling interaction:** `TRACE_BASED` exemplar filter only attaches a trace ID when the current span is sampled. If head sampling drops 99% of traces, 99% of metric observations carry no exemplar. For high-traffic services, consider always-sampling a small trace-exemplar path (e.g. use a sampler that keeps traces sampled solely for exemplar purposes but does not export the full span data).

**PromQL exemplar query example:**

```promql
# Show p99 latency with exemplars enabled in Grafana
histogram_quantile(0.99, rate(http_server_request_duration_seconds_bucket[5m]))
```

Navigate from the histogram panel to the trace: select any exemplar point on the panel to open the linked trace in Tempo or Jaeger.

Reference: [prometheus.io/docs/practices/exemplars](https://prometheus.io/docs/practices/exemplars/) and [opentelemetry.io/docs/specs/otel/metrics/sdk/#exemplar](https://opentelemetry.io/docs/specs/otel/metrics/sdk/#exemplar).
