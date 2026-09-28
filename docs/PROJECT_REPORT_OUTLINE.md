# Project report outline

1. **Abstract and problem statement:** failure modes of deployed models and consequences of monitoring blind spots.
2. **Literature review:** feature drift, label drift, concept drift, performance degradation, PSI and limitations of distributional tests.
3. **Problem specification:** measurement plan, reliability requirements, threat model, and acceptance criteria.
4. **System design:** architecture, data flow, event contract, time windows, reference provenance, storage schema and failure behavior.
5. **Model reference implementation:** Iris dataset provenance, train/holdout methodology, preprocessing, class schema and approved performance baseline.
6. **Implementation:** FastAPI auth and ingestion, immutable references, Evidently snapshots, worker, alert state machine, Streamlit history, Docker.
7. **Validation:** unit/integration tests, explicitly simulated feature shifts, statistical threshold discussion, webhook receipt testing, error cases and reset procedure.
8. **Results:** measured scores and delivery acknowledgements, with event counts, labels, confidence and reference versions.
9. **Threats to validity:** sample size, realistic workload unknown, Iris not production-like, label selection, synthetic transforms, detector sensitivity.
10. **Limitations and future work:** Postgres/queue, multi-tenant auth/TLS, realistic production labels, calibration, model availability probes and concept-drift investigation.
11. **Conclusion:** summarize accomplishments and note remaining engineering work.
12. **Appendices:** API schema, reproducibility commands, snapshots, full tests and references.
