Now let me run calibration searches in parallel to anchor the score.Now I have sufficient calibration data. Let me write the final review.

---

## Summary

The paper addresses silent data corruption (SDC) errors during gradient aggregation in distributed training, which cause gradient inconsistency across workers. The authors formalize gradient inconsistency as additive i.i.d. Gaussian noise on the aggregated gradient, prove that it leads to unbounded model divergence and non-convergence of vanilla DSGD (Theorem 3.2), and propose PAFT — a system combining periodic parameter synchronization (PAFT-Sync) with dynamic frequency scheduling and asynchronous communication overlap (PAFT-Dyn). PAFT-Sync provably restores SGD-equivalent convergence rates (Corollary 4.3), and experiments are conducted on ResNet-18, ResNet-50, GPT-2, and LLaMA-2.

---

## Strengths

- **Formally grounded problem motivation:** Lemma 3.1 (Eq. 6) provides a clean closed-form for how model divergence accumulates as $\frac{(M+1)\sigma^2}{M}\sum_{s=0}^t \eta_s^2$, giving a precise, interpretable diagnosis of why noised DSGD fails. This grounds the design intuition for parameter synchronization in a non-trivial way.

- **Provable convergence restoration:** Theorem 4.2 and Corollary 4.3 (Eq. 10–11) show that periodic synchronization with gap $H$ converts the divergent $T_3$ term from Theorem 3.2 into $\mathcal{O}(\sigma^2 H \kappa / (\mu M T^2))$, achieving the same $\mathcal{O}(1/T)$ rate as vanilla SGD. This is the central theoretical result and it is correctly derived given its assumptions.

- **Multi-model empirical coverage:** Experiments span ResNet-18/50, GPT-2 pre-training, and LLaMA-2 fine-tuning across 4–32 GPUs with four noise levels and burst-noise patterns. For the small-noise regime ($\sigma^2 \in \{0.0001, 0.001\}$), PAFT substantially recovers accuracy (e.g., Table 1: ResNet-18 recovers from 60.5% → 85.2% at $\sigma^2=0.01$).

- **Principled SNR-based scheduling:** PAFT-Dyn sets $H = \|g_t\| / \sigma$ (Algorithm 2, Line 11), directly balancing the second and third terms in Theorem 4.2's convergence bound. This explicit connection between theory and the scheduling heuristic is a genuine design insight.

- **Practical system implementation:** Section 4.3 and Table 3 demonstrate asynchronous overlap reducing synchronization overhead from PAFT-Sync to PAFT, with only ~18.9% wall-clock overhead versus vanilla DSGD at 32 workers.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Core algorithm is equivalent to Local SGD / FedAvg without acknowledgment.** Algorithm 1 / Eq. 8 — train for $H$ steps under noisy gradients, then all-reduce parameters — is exactly the Local SGD step studied extensively under the names Local SGD (Stich, 2018), FedAvg (McMahan et al., 2017), and SlowMo. The structure of Theorem 4.2 and Corollary 4.3, producing $\mathcal{O}(1/T)$ convergence with a divergence term proportional to $H\sigma^2$, replicates the structure of known Local SGD convergence analyses. The paper's Related Work (Section 6) discusses Byzantine fault tolerance and checkpointing, but the most directly relevant literature — periodic averaging for communication-efficient/noise-robust distributed optimization — is entirely absent. This is not merely a citation gap: it fundamentally affects how novel the paper's contribution is. The authors need to explicitly state what is new relative to Local SGD convergence theory, or the contribution reduces to applying a known algorithm to a new noise model.

- **No comparison to any noise-robust training baseline.** The experiments compare PAFT only against vanilla DSGD (with and without injected noise). Gradient norm clipping — already standard in LLM training and directly applicable to the large-noise regime — is not included. Byzantine-robust aggregators (e.g., coordinate-wise median, Krum, trimmed mean) are cited in Section 6 as related work yet excluded from the experimental comparison. Without such baselines, it is impossible to assess whether PAFT represents a meaningful advance over existing practice, particularly since the paper frames itself as "the first effort to improve system reliability against GA errors at scale."

- **Overclaiming in abstract and conclusion relative to large-noise experimental results.** The abstract states PAFT "efficiently defends against gradient aggregation error degrees." Table 2 shows that for $\sigma^2=0.1$ with 4 workers: Oracle=75.0%, Noised DSGD=1.3%, PAFT=1.4%. The 0.1 percentage-point improvement is not meaningful mitigation. The paper itself equates large $\sigma^2$ with bit corruptions (Section 2.1: "the large $\sigma^2$ can represent the larger noise like bit corruptions"), meaning the large-noise regime is precisely the stated motivating use case. The paper acknowledges the gap in Section 5.1 and Section 7 (Limitations), but the abstract and conclusion make unqualified success claims that are inconsistent with these results. The method demonstrably works well only for small communication noise ($\sigma^2 \leq 0.001$).

### Minor

- **Theorem 3.2's non-convergence claim is overstated.** The remark states that $T_3$ "only converges when setting $\eta_t = 0$." However, this holds under constant or non-summable learning rates. With a summable learning rate schedule ($\sum_t \eta_t^2 < \infty$, e.g., $\eta_t = c/t$), $T_3$ is bounded, and noised DSGD would converge. The claim of *fundamental* non-convergence is an artifact of the learning rate assumption, not an inherent property of the noised problem. This should be stated more carefully.

- **PAFT-Dyn underperforms PAFT-Sync H=5 in the 32-worker, $\sigma^2=0.01$ regime** (Table 2: PAFT=40.9% vs PAFT-Sync H=5=44.4%). The dynamic scheduling is supposed to adapt optimally, yet it underperforms the simplest fixed-frequency variant in a high-noise, many-worker setting. This failure case is not discussed, suggesting the PAFT-Dyn SNR heuristic may misfire under certain conditions.

- **Inconsistent use of "failed convergence."** Section 3 claims "even the small noise 0.001 also leads to failed training convergence," but Table 1 shows Noised DSGD achieves 91.1% at $\sigma^2=0.001$ versus Oracle 94.0%. That is a 2.9-point degradation, not failure to converge. The term is used loosely, conflating accuracy degradation with divergence.

### Trivial

- The estimation formula $H_{\text{new}} = \|g_t^m\| / \sigma_{\text{est}}$ (Algorithm 2, Line 11) is stated but not derived from first principles. The paper asserts it balances the second and third terms in Theorem 4.2, but this approximation is informal and a brief derivation showing the balance condition would strengthen the argument.

---

## Nice-to-Haves

- A comparison to simple gradient norm clipping as a baseline would be the minimal credible experiment for the large-noise regime.
- A brief ablation of when PAFT-Dyn's dynamic scheduling helps versus PAFT-Sync with optimal fixed H would clarify the conditions where the adaptive heuristic provides value.
- Positioning the convergence analysis explicitly against Local SGD convergence results (e.g., Stich 2018) would clarify what is novel: whether the noise model, the tighter divergence bound, or just the motivating framing.
- A burst-noise experiment on LLMs (not only ResNet-18) would make the accidental corruption scenario more relevant to the paper's LLM-training motivation.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Figure 8 caption referencing "Do=0.001, Do=0.01..."**: The harsh reviewer flagged this as an inconsistency with the paper's noise parameterization. This is a parser artifact from the image description extraction, not an error in the actual paper. **Removed per hard rule on formatting artifacts.**

- **Noise model mismatch with real IEEE 754 bit corruptions**: The reviewer argued that bit flips in floating-point produce catastrophic values, not Gaussian noise. While this is physically accurate, the paper explicitly scopes large σ² as approximating bit corruption effects, and includes burst-noise experiments. More importantly, the model's failure to perfectly match hardware bit-flip statistics is a scope limitation, not a flaw — the paper studies a mathematically tractable generalization of SDC effects. The claim "this requires a different problem formulation" is overreach. **Weakened to being subsumed by the overclaiming weakness.**

- **LLM fine-tuning scale being unrepresentative**: The harsh reviewer criticized LoRA fine-tuning with 1 epoch on 52K examples as "not representative of pre-training." While the scale is small, this is a reasonable scope choice for validation purposes in a systems paper, and the paper explicitly presents it as fine-tuning validation, not pre-training simulation. **Removed as scope creep.**

- **18.9% wall-clock overhead being "substantial"**: This is a subjective judgment call. 18.9% overhead for fault tolerance is generally considered acceptable in the distributed systems community. **Removed.**

- **Adam extension being insufficiently covered in the main text**: The harsh reviewer complained that Adam extension is deferred to appendix. Parser strips appendices; this criticism is invalid. **Removed per hard rule.**

---

## Novel Insights

The paper's most valuable observation — worth preserving regardless of eventual disposition — is that gradient inconsistency from SDC errors is fundamentally different from ordinary stochastic gradient noise: the per-worker noise terms are independent across workers, so they do *not* average out during gradient aggregation (unlike mini-batch sampling variance). This means the noise effect scales adversely with the number of workers (the $(M+1)/M$ factor in Lemma 3.1 and Theorem 4.2), while ordinary stochastic noise benefits from more workers. This distinction is clearly stated and motivates why standard DSGD robustness intuitions fail for SDC. The connection to learning rate decay naturally suppressing noise sensitivity (Section 5.1, accidental noise experiments) is also a practically useful empirical insight consistent with the theory.

---

## Suggestions

1. **Cite and differentiate from Local SGD**: Add a formal comparison to Stich (2018) or McMahan et al. (2017). The novelty claim should be: "PAFT applies periodic averaging to the SDC noise model (which differs from data heterogeneity), and our convergence analysis captures a noise-specific divergence term." This is a legitimate distinction; just state it explicitly.

2. **Add gradient clipping as a baseline**: Run the same noise injection experiments with gradient norm clipping (as used in standard LLM training) and show where PAFT provides additional benefit. This single experiment would substantially strengthen the empirical contribution.

3. **Revise the abstract and conclusion** to accurately characterize the operating regime: PAFT reliably mitigates small-to-moderate communication noise ($\sigma^2 \leq 0.001$) and provides partial mitigation at moderate noise ($\sigma^2 = 0.01$), but does not address the large-noise regime associated with bit corruptions.

4. **Correct the convergence remark in Theorem 3.2** to acknowledge that non-convergence is learning-rate-schedule-dependent, not absolute.

---

## Score and Decision

**Calibration anchors used:**

- *Flag Aggregator (7avlrpzWqo)* — Byzantine-robust distributed training with formal theory and experiments, scored **6,6,6** (Accept poster). This paper has both theory and strong empirical comparison against multiple baselines; that's why it got 6s.
- *Delayed Local-SGD (jw8EoY1FvF)* — Local SGD variant with convergence analysis, scored **5,3,5,3** (Reject). Similar structural issues (incremental Local SGD analysis) and missing baselines led to rejection.
- *LoCoDL (PpYy0dR3Qw)* — Local SGD + compression with novel doubly-accelerated analysis, scored **6,8,8,8** (Accept Spotlight). This demonstrates that Local SGD analysis papers can score high when the contribution is genuinely new and clearly differentiated.
- *Robust Decentralized VFL (ddNZLAWPdT)* — fault-tolerant federated learning with theory, scored **5,3,3,6** (Reject). Rejected partly due to missing novelty justification and limited baselines.

**Assessment relative to anchors:** The paper under review is closest to *Delayed Local-SGD* (rejected at 4.0 average) and *Flag Aggregator* (accepted at 6.0 average). It is better than Delayed Local-SGD in problem motivation and experimental scope (LLMs included), but worse than Flag Aggregator in novelty (algorithm is known), baseline comparison (none), and accuracy of claims (overclaiming at large noise). The paper falls between these two clusters.

The three major weaknesses — unacknowledged equivalence to Local SGD, missing baselines, and overclaiming — are all addressable in a revision but cannot be resolved in a rebuttal alone. This aligns with a marginally-below-threshold score.

**Originality:** Low-to-medium. The noise model and SDC framing are novel; the algorithm is not.
**Importance:** High. SDC in LLM training is a real and growing problem.
**Claim support:** Weak. The central "efficient defense" claim is directly contradicted for the large-noise regime.
**Experimental soundness:** Below par. Missing all relevant baselines.
**Clarity:** Adequate, with some inconsistent terminology.
**Value to community:** Moderate. The formal framework and diagnosis are useful even if the algorithmic solution is known.

**Final Score: 4.5 (Reject, revise and resubmit)**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>