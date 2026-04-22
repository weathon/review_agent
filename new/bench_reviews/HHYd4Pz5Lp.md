Now I have all the information I need. Let me compile the final review.

## Summary

DelRec introduces the first surrogate gradient learning (SGL)-based method for learning transmission delays in recurrent connections of spiking neural networks. Using a differentiable triangle interpolation function with annealed width (σ) to provide gradients for real-valued delay parameters that converge to integers at inference, the method achieves new state-of-the-art on SSC (82.58%) and PS-MNIST (96.21%) using only vanilla LIF neurons, outperforming prior methods that rely on more complex neuron models.

## Strengths

- **First SGL-based method for recurrent delay learning**: Prior delay-learning methods in SNNs were restricted to feedforward connections (DCLS, Sun et al.), and the only prior recurrent delay method (Meszáros et al., 2025) relied on EventProp with scalability limitations. DelRec integrates recurrent delay optimization into the dominant SGL/backpropagation paradigm (Section 1), filling a genuine methodological gap.

- **New SOTA on SSC and PS-MNIST with vanilla LIF neurons**: Table 1 shows DelRec (recurrent delays only) achieves 82.58±0.08% on SSC and 96.21% on PS-MNIST, surpassing methods using substantially more complex neuron models (SE-adLIF at 80.44%, RadLIF at 77.40%) while maintaining competitive parameter counts (0.37M for SSC, 0.16M for PS-MNIST).

- **Clean and efficient algorithmic design**: The scheduling buffer with circular pointer (Algorithm 1) and the approximate buffer range Ê(σ,D) (Eq. 13) implement delay learning without predefining a maximum delay range, and the σ annealing naturally controls memory overhead. The method is compatible with any spiking neuron model fitting the standard formalism (Eqs. 1–3).

- **Thorough functional study on SHD**: The three-phase study (validation → simplification → comparative, Section 3.2) with controlled parameter counts and firing rate constraints is more systematic than typical SNN ablation studies. The finding that even random fixed recurrent delays improve training (Fig 3B) is practically useful, and the parameter-scaling analysis (Fig 3C) provides the strongest evidence for recurrent delays' parameter efficiency.

- **Honest assessment of SHD saturation**: The paper correctly identifies that SHD is saturated for benchmarking (confidence intervals overlap above 93%) and recommends using it only for proof-of-concept (Section 3.2), setting a higher standard for the community.

## Weaknesses

### Fatal
None.

### Major

- **The core claim "trainable recurrent delays outperform feedforward ones" is overgeneralized.** The abstract states this as an unqualified fact, but the evidence only supports it under specific conditions. In the SHD large-model regime (Table 2), DCLS (feedforward only) achieves 93.77±0.68% while DelRec (recurrent only) achieves 93.39±0.45%. On SSC (Table 1), combining both delay types *hurts* performance (82.19±0.16% for rec+ff vs 82.58±0.08% for rec-only). The supportive evidence comes primarily from the small-model regime on SHD (Fig 3C). The paper does moderate its language in the body ("may yield greater benefits," "under low parameters constraints"), but the abstract claim remains misleadingly strong. Furthermore, the paper acknowledges but does not fully address that the comparison confounds delay type with parameterization (axonal recurrent: 1 param/neuron vs synaptic feedforward: 1 param/synapse) — though this confound actually makes the parameter-efficiency argument meaningful, it means the abstract's unqualified statement is imprecise.

- **The SSC anomaly where combining recurrent and feedforward delays hurts performance is not discussed.** Table 1 shows DelRec (rec+ff) = 82.19±0.16% vs DelRec (rec-only) = 82.58±0.08%. The paper mentions that combining delay types offers no advantage in small SHD configurations (Section 3.2), but this same pattern appearing in the large-model regime on SSC — the paper's primary competitive benchmark — deserves explicit investigation. This could indicate optimization interference between delay types, which would be critical for practitioners.

### Minor

- **PS-MNIST reported from a single random seed.** The paper justifies this by noting "we only test one seed as all the previous state-of-the-art models on the dataset," but following prior poor experimental practice does not excuse continuing it, especially for a headline SOTA claim. The 0.44 pp improvement over ASRC-SNN (95.77%) cannot be assessed for significance without variance estimates.

- **No empirical validation of the gradient mitigation motivation.** Section 1 and Fig 1B argue that recurrent delays act as temporal skip connections mitigating vanishing/exploding gradients. The paper uses tentative language ("may mitigate"), and Fig 3B shows that random fixed recurrent delays help training, which is consistent with but does not validate this hypothesis. A simple gradient-norm comparison during training with/without delays would substantially strengthen this motivation.

- **Dataset-specific p parameter (Eq. 15) lacks motivation.** The per-neuron p parameter that modulates σ decay rate is added only for SSC, described as enabling "a quicker decay of σ_epoch for specific neurons." Its necessity suggests the base method requires dataset-specific tuning, but no ablation or analysis is provided to explain why SSC needs this and SHD/PS-MNIST do not.

### Trivial
None.

## Nice-to-Haves

- Distribution/histogram of learned delay values across neurons for each dataset — this would reveal whether delays converge to interpretable temporal patterns or degenerate solutions.
- A fair comparison of axonal recurrent delays vs axonal feedforward delays (same delay parameterization, same number of delay parameters) to disentangle mechanism type from parameterization density.
- Statistical significance tests on SSC results, though this is not standard practice in the SNN benchmarking community.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"SSC improvement over SiLIF may not be statistically significant"** (Harsh Critic, Critical Issue 3): The claimed overlap is incorrect if ± represents standard deviation (82.58±0.08 vs 82.03±0.25 — the intervals [82.50, 82.66] and [81.78, 82.28] do not overlap). Even under the most conservative interpretation, the point-estimate gap (0.55 pp) with low variance in DelRec's results makes this a reasonable SOTA claim. Statistical testing is also not standard in this community for these benchmarks.

- **"Fair comparison would pit axonal-against-axonal or synaptic-against-synaptic"** (Harsh Critic, Critical Issue 1): The paper explicitly acknowledges the axonal vs synaptic confound (Section 3.2: "It is worth noting that we are comparing synaptic feedforward delays...with axonal recurrent delays"). The comparison under *equal total parameter budgets* (Fig 3C) is the relevant one for practitioners deciding where to allocate parameters. The comparison is fair in that framing — it shows which approach uses a given parameter budget more efficiently.

- **"DCLS outperforms DelRec on SHD large models"** (Harsh Critic, Critical Issue 1): The confidence intervals overlap substantially (93.77±0.68 vs 93.39±0.45), and the paper's own argument that SHD is saturated above ~93% undermines using this difference as evidence against DelRec.

- **"Request for missing related works"**: Cannot verify existence of suggested missing references without external sources.

- **"σ annealing schedule is heuristic with no sensitivity analysis"**: This is a standard hyperparameter choice in SNN methods; sensitivity analysis would strengthen but is not a core flaw.

- **"Per-neuron p parameter feels like ad-hoc patch"**: While it is dataset-specific, it is documented and its role is explainable. Moved to Minor weakness as it deserves mention but is not disqualifying.

- **"Formatting/style nitpicks"** (convention that d=0 means delay=1 could confuse): This is a notation choice, not a substantive issue.

## Novel Insights

The functional study reveals an interesting asymmetry: recurrent delays provide superior parameter efficiency (Fig 3C), but feedforward delays achieve better energy efficiency via lower firing rates (Fig 3D-E). This suggests a previously unarticulated tradeoff between representational efficiency and energy efficiency in delay-augmented SNNs, which could inform architectural decisions for neuromorphic deployment depending on whether the constraint is memory/parameters or power/energy.

## Suggestions

- Soften the abstract claim from "trainable recurrent delays outperform feedforward ones" to something like "trainable recurrent delays offer superior parameter efficiency compared to feedforward delays, particularly under constrained model capacity."
- Run ≥3 seeds on PS-MNIST and report mean±std to properly support the SOTA claim.
- Add a brief discussion of the SSC rec+ff anomaly — even a hypothesis (e.g., optimization interference, overfitting with additional parameters) would be valuable.
- Add a simple gradient-norm plot comparing training with vs without recurrent delays to validate the gradient mitigation motivation from Fig 1B.

## Score and Decision

**Calibration anchors:**

- **DCLS (Hammouamri et al., 2024)** — `/home/wg25r/review_agent/human_reviews/4r2ybzJnmN.md`, avg score 7.0. Directly comparable prior work on learning feedforward delays in SNNs. DelRec extends the approach to recurrent connections with similar technical quality, but overclaims more strongly than DCLS did.

- **Online Stabilization of SNNs** — `/home/wg25r/review_agent/human_reviews/CIj1CVbkpr.md`, avg score 7.0. Engineering contribution to SNN training with clean algorithmic design. Similar level of novelty and experimental depth. DelRec has comparable algorithmic cleanliness but less thorough ablation.

- **LMUFormer** — `/home/wg25r/review_agent/human_reviews/oEF7qExD9F.md`, avg score 5.75. SOTA on speech commands with SNN, some novelty concerns and mixed reviewer opinions. DelRec has clearer novelty and stronger results.

- **Composing RSNNs** — `/home/wg25r/review_agent/human_reviews/uxYye6i2Xi.md`, avg score 5.25. RSNN architecture optimization with genuine contribution but limited experimental scope and incomplete validation. DelRec has a more focused contribution with cleaner evidence.

- **Self-Supervised Pseudodata Filtering** — `/home/wg25r/review_agent/human_reviews/i3DyRNgCey.md`, avg score 2.5. Overclaimed results with trivial methodology and no statistical significance. DelRec is far above this — it has genuine novelty and real SOTA improvements.

DelRec sits above the borderline-reject papers (5.0-5.75) due to its clear methodological contribution, genuine SOTA improvements on SSC, and thorough functional study. It sits below DCLS and Online Stabilization (7.0) primarily because the abstract overclaims relative to the evidence and the PS-MNIST result lacks multi-seed validation. The paper's primary contribution — the first SGL-based method for recurrent delay learning — is real and well-executed, but the gap between the abstract's claims and the actual evidence prevents a higher score.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>