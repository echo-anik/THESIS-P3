# SCADA Anomaly Detection System - Thesis Project

**Thesis Phase 3 - Deployment & Production-Ready Implementation**

## Quick Summary

Advanced anomaly detection system for SCADA networks using hybrid LSTM-GDN architecture deployed on edge devices (Jetson Nano) and centralized servers.

- **F1 Score:** 0.876 (vs SOTA 0.62, +41.3%)
- **Hardware:** Jetson Nano ($99, 0.21ms latency)
- **Model Size:** 82 KB TensorFlow Lite
- **Deployment:** 6 weeks to production

## Documentation

📖 **Start Here:** [DEPLOYMENT_AND_NOVELTY.md](DEPLOYMENT_AND_NOVELTY.md)
- 5-phase deployment plan
- Novelty analysis & SOTA comparison
- Shadow IDS architecture
- Retrofitability with water treatment plant integration

📝 **Defense Prep:** [THESIS_EXPLANATION_AND_DEFENSE.txt](THESIS_EXPLANATION_AND_DEFENSE.txt)
- Complete system explanation
- Pre-written defense responses to panel criticism
- Real attack scenarios & timing analysis

🗺️ **Navigation:** [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- Topic-based guide to all documentation
- Q&A for common questions
- Audience-specific reading paths

## Repository Structure

```
├── experiments/           # Model training & validation code
├── deployment/            # Edge deployment & inference scripts
├── optimizers/            # Metaheuristic optimization (GWO)
├── Final P2/              # Thesis document (LaTeX)
├── DEPLOYMENT_AND_NOVELTY.md
├── THESIS_EXPLANATION_AND_DEFENSE.txt
├── DOCUMENTATION_INDEX.md
└── .gitignore
```

## Key Contributions

1. **Hybrid LSTM-GDN Architecture** - Combines temporal and relationship patterns
2. **Pareto Frontier Achievement** - High accuracy on $99 edge hardware
3. **Data Leakage-Free** - 7-point audit passed
4. **Production-Ready** - Complete SCADA integration with OPC-UA
5. **Shadow IDS** - Jetson Nano-based redundancy & validation

## Technology Stack

- **Framework:** TensorFlow Lite
- **Edge Device:** Jetson Nano (NVIDIA)
- **SCADA Protocol:** OPC-UA
- **Containerization:** Docker
- **Dataset:** WADI (962K training samples)

## Performance

| Metric | Value | Comparison |
|--------|-------|-----------|
| F1 Score | 0.876 | +41.3% vs SOTA |
| Precision | 93.8% | Fewer false alarms |
| Recall | 82.2% | More attacks detected |
| Edge Latency | 0.21ms | Real-time capable |

## Cost-Benefit Analysis

**Centralized Option:**
- Initial: $3-5K
- Annual: $9K
- 5-year ROI: 15.7×

**Edge Option:**
- Initial: $2-4K
- Annual: $1.5K
- 5-year ROI: 82×

## Deployment Timeline

- **Phase 1:** Preparation (1 week)
- **Phase 2:** Training (2 weeks)
- **Phase 3:** Optimization (1 week)
- **Phase 4:** Integration (2 weeks)
- **Phase 5:** Production (2 weeks)
- **Total:** 6 weeks to full deployment

## Defense Resources

Use these files for your defense:

1. **System Explanation:** [THESIS_EXPLANATION_AND_DEFENSE.txt](THESIS_EXPLANATION_AND_DEFENSE.txt) Part 1
2. **Novelty Justification:** [DEPLOYMENT_AND_NOVELTY.md](DEPLOYMENT_AND_NOVELTY.md) Section 4
3. **Retrofitability:** [DEPLOYMENT_AND_NOVELTY.md](DEPLOYMENT_AND_NOVELTY.md) Section 6
4. **Common Questions:** [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) "How to Answer Common Questions"

## Citation

```
@thesis{thesis2025scada,
  author={echo-anik},
  title={Advanced Anomaly Detection for SCADA Systems},
  year={2025},
  school={University}
}
```

---

**Status:** Ready for thesis submission and defense presentation
