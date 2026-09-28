# Demonstration run results

The pipeline is exercised end to end with the bundled Iris demonstration: a standardised multiclass logistic regression trained on a stratified split of the public scikit-learn Iris dataset, plus 120 synthetic monitoring events.

All 120 traffic events are marked `source=SIMULATED_Iris_traffic`, drawn with a deterministic sampling seed and a synthetic +1.5 cm shift on the sepal-length feature. These numbers describe the demonstration pipeline; they are not production performance or observed field drift.

| Measurement | Value |
| --- | --- |
| Reference model | Multiclass logistic regression, StandardScaler, seed 42 |
| Approved reference holdout | 53 Iris observations |
| Reference accuracy | 0.9245283018867925 |
| Reference macro F1 | 0.9256965944272446 |
| HTTP `/health/ready` | `ready` with reference `iris-holdout-1` |
| HTTP batch ingested | 120 inserted, 0 duplicates |
| Ground-truth linkage | 120 labels associated, 100% labeled coverage |
| Simulated classification accuracy | 0.8833333333333333 |
| Simulated macro F1 | 0.8825757575757575 |
| Simulated confusion matrix | `[[28,8,0],[0,37,2],[0,4,41]]` |
| Regression threshold | accuracy below 0.9245283018867925 - 0.1 = 0.8245283018867925 |
| Performance regression alert | not triggered: 0.8833333333333333 is above the threshold, and persistence requires 2 distinct windows |
| Unit and integration tests | 41 passed, 1 noncritical deprecation warning (last recorded run) |

The alert webhook success path and the HTTP 503 failure path are exercised against a local test HTTP server with synthetic rule failures inside pytest. That covers the dispatcher's transport and status-handling logic; it is not delivery to an external notification destination.

Running `modelwatch prepare` creates an empty durable monitoring state, and `modelwatch simulate` adds simulation-labeled observations. The reference CSV also carries three `score_0..score_2` model probability columns for later score-distribution evaluation.
