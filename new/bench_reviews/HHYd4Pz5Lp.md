Now I have all the information needed to write the final review. Let me compile everything carefully.

## Summary

DelRec introduces the first surrogate gradient learning (SGL) method to optimize delays in recurrent connections of spiking neural networks. The method relaxes integer delays to real values during training via a differentiable triangle-spreading interpolation function (with annealed width σ), then rounds to the nearest integer at inference. Using only vanilla LIF neurons with instantaneous synapses, DelRec achieves new state-of-the-art accuracy on SSC (82.58%) and PS-MNIST (96.21%), and includes a functional study comparing recurrent delays against feedforward delays under parameter and sparsity constraints on SHD.

## Strengths

- **First SGL-based method for learning recurrent delays**: DelRec is genuinely the first method that integrates learnable delays in recurrent connections into the surrogate gradient learning / backpropagation pipeline, a gap explicitly noted in prior work (Mészáros et al., 2025 addressed only EventProp). The scheduling matrix formulation (Eqs. 8–13) and σ-annealed triangle interpolation are clean, practical, and well-described (Algorithm 1).

- **SOTA on SSC with simple LIF neurons**: Table 1 shows DelRec achieving 82.58 ± 0.08% on SSC using only vanilla LIF neurons with instantaneous synapses, surpassing methods that require more complex neuron models (SiLIF at 82.03%, SE-adLIF at 80.44%, DCLS at 80.69%). This demonstrates that learned recurrent delays can substitute for complex neuronal dynamics.

- **Honest assessment of SHD saturation**: The paper correctly identifies that SHD has become saturated (Section 3.2, line 234: "further improvements in performance are likely not statistically significant given the small size of the test set") and recommends it only for proof-of-concept studies—a useful community contribution.

- **Compatible with any neuron model**: The method operates on the general Eq. 1–3 formalism and does not tie itself to a particular neuron model (Section 2.1, line 114), making it broadly applicable.

- **Efficient scheduling with bounded support**: The finite support property of h_{σ,d} (Eq. 12) and the practical approximation of the scheduling range (Eq. 13) ensure the method scales tractably without predefining a maximum delay range—a practical advantage over methods requiring a priori delay range specification.

## Weaknesses

### Fatal
None.

### Major

- **The "recurrent delays outperform feedforward delays" comparison is confounded by concurrent changes in connectivity structure**: The central comparative finding in Section 3.2 (Fig. 3B–C) compares a model with learned recurrent delays (which has recurrent weights + recurrent delays) against a model with learned feedforward delays via DCLS (which has no recurrent connections at all). The performance gap could stem from the recurrent weights themselves rather than from delays being in recurrent connections specifically—Fig. 3B already shows that adding recurrent connections alone (Vanilla RNN vs Vanilla SNN) has a massive effect, and fixed random recurrent delays (~78%) already outperform learned feedforward delays (~80%) by a smaller margin that further obscures the attribution. The paper partially acknowledges this by noting the axonal vs synaptic delay distinction (Section 3.2, line 228: "It is worth noting that we are comparing synaptic feedforward delays (one delay per synapse), with axonal recurrent delays (one delay per neuron)"), but does not run the critical missing control: an RSNN (with recurrent weights) equipped with learned *feedforward* delays but no recurrent delays. Without this control, the conclusion in the conclusion ("we present a study suggesting that recurrent delays can achieve better performance than feedforward delays") is not properly isolated. This matters because it is the paper's central comparative finding beyond the SOTA results themselves.

- **Combining feedforward and recurrent delays degrades SSC performance, undermining the generality narrative**: Table 1 shows DelRec with only recurrent delays achieves 82.58 ± 0.08% on SSC, but adding feedforward delays (DCLS) drops performance to 82.19 ± 0.16%. If both delay types provide orthogonal, complementary benefits—as the paper argues—their combination should not degrade results. The paper acknowledges this briefly (conclusion: "further improvements could be obtained by better combining DelRec with feedforward delays") but offers no explanation or investigation. This unexplained degradation suggests either optimization interference between delay types or that the recurrent delay advantage is more fragile than claimed, and it directly challenges the narrative that delays provide complementary benefits.

### Minor

- **PS-MNIST SOTA claim based on a single seed with a marginal improvement**: DelRec reports 96.21% on PS-MNIST from one seed (line 190: "we only test one seed as all the previous state-of-the-art models on the dataset"), improving over ASRC-SNN (95.77%) by only 0.44 percentage points. While this follows field convention on this benchmark, the improvement is well within typical run-to-run variation, making the SOTA claim on PS-MNIST less robust than the SSC result (3 seeds). The SSC improvement (82.58% vs 82.03%, with variance estimates) is more credible.

- **The "recurrent delays as temporal skip connections" motivation (Fig. 1B) lacks direct empirical validation**: The paper motivates recurrent delays as mitigating gradient issues by implementing temporal skip connections (Fig. 1B). While the empirical observation that fixed random recurrent delays improve accuracy over vanilla RSNN (~78% vs ~40% in Fig. 3B) is consistent with this theory, no experiment directly measures gradient magnitudes, training dynamics, or loss landscapes with vs. without recurrent delays. The claim (line 271) that "the simple introduction of delays in recurrent connections mitigates the training difficulties of RSNNs due to gradient issues" remains plausible but empirically unverified.

### Trivial
None.

## Nice-to-Haves

- Report the distribution of learned recurrent delays after training (histograms per layer) to reveal whether the method discovers meaningful temporal structure or converges to arbitrary values.
- Run the missing control experiment: an RSNN with learned feedforward delays (no recurrent delays) to properly isolate whether the benefit comes from delays being in recurrent connections specifically.
- Investigate and explain why combining both delay types hurts SSC performance—this could reveal important insights about optimization dynamics.
- Provide computational cost analysis (memory and runtime overhead of DelRec vs. vanilla RSNN), especially given the claim about "efficient deployment on neuromorphic hardware."

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"σ schedule described only vaguely"**: The paper states "we decrease the parameter σ throughout training down to 0" (line 146) and Figure 2C shows the initial value σ=5. Full hyperparameters are in Appendix A.2.5 (stripped from our view but available in the original submission). This is standard practice for a methods paper—the σ annealing mechanism is well-described conceptually, and exact schedules are implementation details.

- **"SiLIF inclusion in Table 1 is inconsistent"**: The paper draws a reasonable line at "substantially more complex neuron models" (multi-compartment, attention/GRU-based) while including SiLIF which still operates within the LIF-derived family. This is a judgment call, not an inconsistency.

- **"Simplification phase changes too many variables simultaneously"**: Table 3 documents all changes clearly. The key comparison is within the small-model configurations (Fig. 3B–C) where all models share the same base architecture, so the simplification is appropriately controlled for the actual comparative claims.

- **"Footnote reporting Wang et al. (83.69%) makes SOTA less impressive"**: The paper explicitly excludes attention/distillation-based methods from the comparison table for fairness, and discloses their results in a footnote. This is transparent and appropriate—the methods are architecturally too different for meaningful comparison.

- **"Code availability as a strength"**: Generic strength without specific evidence of what the code enables beyond what's standard.

- **Strength claim "demonstrates recurrent delays outperform feedforward delays under parameter constraints" (Fig. 3C top)**: This is confounded by the same issue as the Major weakness above—the comparison does not isolate delay location from connectivity structure. Downgraded from a strength since it conflicts with a verified Major weakness.

- **"Missing σ schedule ablation"**: Demanding ablations on hyperparameter schedules (initial σ, decay rate) is a nice-to-have, not a core flaw. The method uses the same annealing strategy as DCLS (Hammouamri et al., 2024), which is precedent.

## Novel Insights

The paper's observation that combining feedforward and recurrent delays degrades performance on SSC (but not on SHD with larger models, Table 2) suggests a scale-dependent interaction between delay types that is not yet understood. In small models (Fig. 3B), the "learned ff + rec delays" model (~75%) actually underperforms even the "learned ff delays" model (~80%), whereas in larger SHD models (Table 2), the combined model matches DCLS. This hints that the benefit of combining delay types may depend on whether the model has sufficient capacity/regularization to avoid interference during joint optimization—a hypothesis the authors do not explore but that could meaningfully advance understanding of delay learning in SNNs.

## Suggestions

- Add the critical control experiment: train an RSNN (with recurrent weights) using DCLS feedforward delays but no recurrent delays. This would cleanly isolate whether the advantage of DelRec comes from delays being in recurrent connections specifically, or simply from having recurrent connections at all.
- Run at least 3-5 seeds on PS-MNIST to strengthen the SOTA claim; even if field convention is single-seed, providing variance estimates would be a meaningful improvement.
- In the conclusion, soften the language from "recurrent delays can achieve better performance than feedforward delays" to acknowledge the confound explicitly, or remove the claim until the control experiment is run.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| DCLS (feedforward delays in SNNs) | 4r2ybzJnmN | 7.00 (Accept poster) | Most directly related prior work. Scored higher due to cleaner experimental design (simpler comparison: with vs without delays in same architecture). DelRec extends to harder problem (recurrent delays) but has confounded comparison. |
| Meta-SpikeFormer | 1SIBN5Xyw7 | 5.67 (Accept poster) | Similar pattern: SOTA claims with confounded ablation. DelRec has a more focused and novel contribution but similar experimental weaknesses. |
| SC-ML (RSNN architecture) | uxYye6i2Xi | 5.25 (Reject) | RSNN work with weak experiments/ablations. DelRec is stronger: cleaner method, SOTA results, more focused contribution. |
| BC-SNN | fIKRJeLH7W | 4.33 (Reject) | Confounded ablation with limited improvements. DelRec has a clearer contribution and stronger empirical results. |
| STFormer | wPK65O4pqS | 3.00 (Reject) | Overclaimed SOTA without technical rigor. DelRec is substantially better. |

DelRec makes a genuine and important methodological contribution—it is the first SGL-based method for learning delays in recurrent connections, and achieves meaningful SOTA results on SSC using only simple LIF neurons. However, the central comparative claim that "recurrent delays outperform feedforward delays" is undermined by a confounded experimental design that changes both connectivity structure and delay type simultaneously, without the critical control of an RSNN with feedforward delays. This is a Major issue that the authors could address with one additional experiment. Compared to DCLS (7.00), which had a cleaner experimental setup but a simpler contribution, DelRec scores lower due to this confound. Compared to SC-ML (5.25), DelRec scores higher due to its clearer and more novel contribution. The paper sits in the borderline range: the contribution is real and important, but the evidence for the secondary claim (recurrent > feedforward delays) is not properly isolated.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>