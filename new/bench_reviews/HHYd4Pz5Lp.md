Now I have all the information needed to compose the final review. Let me synthesize my findings.

## Summary

The paper introduces DelRec, the first surrogate gradient learning (SGL) method to learn per-neuron (axonal) or per-synapse delays in recurrent connections of spiking neural networks. DelRec uses a differentiable triangle interpolation function with annealed width parameter σ to provide gradients for non-integer delays during training, converging to integer delays for inference. It achieves new state-of-the-art results on SSC (82.58±0.08%) and PS-MNIST (96.21%) using only vanilla LIF neurons with instantaneous synapses, and performs competitively on SHD.

## Strengths

- **Well-designed and clearly specified method**: The differentiable interpolation technique (triangle function h_{σ,d} with annealed σ, Eq. 10–14) is a principled extension of prior feedforward delay learning (DCLS/Khalfaoui-Hassani et al., 2023) to recurrent connections. The circular buffer scheduling mechanism (Algorithm 1) with support set analysis (Eq. 12–13) makes the method computationally tractable and reproducible.

- **SOTA results with simple neurons**: Achieving 82.58±0.08% on SSC (Table 1) surpasses prior methods that use more complex neuron models (SE-adLIF at 80.44%, SiLIF at 82.03%) while using competitive parameter counts (0.37M). This demonstrates that temporal processing capacity can come from network-level delays rather than neuron-level complexity—a meaningful finding for the SNN community.

- **Methodologically sound SHD evaluation**: The paper transparently acknowledges SHD saturation, uses a proper train/validation/test split (unlike much prior work), and recommends against relying on SHD as a primary benchmark—commendable community guidance.

- **Useful functional study insights**: The parameter-constrained comparison (Fig. 3C) and energy-accuracy tradeoff analysis (Fig. 3D) provide practical guidance for neuromorphic deployment, and the finding that even fixed random recurrent delays help training (Fig. 3B) supports the gradient-flow hypothesis.

## Weaknesses

### Fatal
None.

### Major

- **Confounded comparison between recurrent and feedforward delays**: The paper's central functional claim—"trainable recurrent delays outperform feedforward ones" (abstract, Section 3.2)—rests on a comparison that simultaneously varies two factors: (a) recurrent vs. feedforward connectivity, and (b) axonal (1 delay per neuron) vs. synaptic (1 delay per synapse) parameterization. The paper acknowledges this: "It is worth noting that we are comparing synaptic feedforward delays (one delay per synapse), with axonal recurrent delays (one delay per neuron)" (Section 3.2). This acknowledgment does not resolve the confound: the observed advantage could stem from the axonal parameterization (which enforces weight-sharing across targets, acting as a regularizer) rather than from recurrence. To cleanly support the stated claim, the paper needs at minimum an ablation comparing axonal feedforward vs. axonal recurrent delays (or synaptic feedforward vs. synaptic recurrent delays). As presented, Fig. 3C cannot distinguish whether the benefit comes from recurrence or from parameterization, yet the conclusion is stated unambiguously throughout the abstract, results, and conclusion.

- **PS-MNIST SOTA claim rests on a single seed**: The 96.21% result on PS-MNIST (Table 1) comes from one seed, with the justification that "all the previous state-of-the-art models on the dataset" also used single seeds. The 0.44% gap over ASRC-SNN (95.77%) could easily be noise: SSC results show standard deviations of 0.08–0.25% across just 3 seeds, and PS-MNIST's sequential nature may yield even higher variance. A single-seed SOTA claim on one of two headline datasets undermines the statistical credibility of a core result.

### Minor

- **Combined feedforward+recurrent delays underperform recurrent-only on SSC without explanation**: DelRec with both delay types achieves 82.19±0.16% on SSC vs. 82.58±0.08% for recurrent-only (Table 1), despite having more parameters (0.55M vs. 0.37M). The paper notes a similar finding for small SHD models (Section 3.2: "we found no advantage in using both types of delays in these small configurations") but does not analyze the SSC result directly. This contradicts the paper's stated goal of combining delay types and deserves investigation (e.g., overfitting analysis, delay value distributions).

- **"First SGL-based method" claim is imprecise**: The abstract claims DelRec is "the first SGL-based method to train axonal or synaptic delays in recurrent spiking layers," but ASRC-SNN (Xu et al.) also learns recurrent delays with SGL—albeit one per layer via softmax rather than per-neuron via continuous interpolation. The paper acknowledges Xu et al. in the introduction but the "first" claim as stated overstates the distinction. The actual contribution is the first per-neuron/per-synapse recurrent delay learning with continuous interpolation under SGL—a narrower but still valuable claim.

- **Gradient-flow mechanism not empirically validated**: The theoretical motivation that recurrent delays act as temporal skip connections improving gradient propagation (Fig. 1B) is appealing but untested via direct evidence (gradient norms, effective path length, or gradient variance comparisons). Fig. 3B provides only indirect support (random fixed delays help vs. vanilla RSNN). Direct gradient analysis would substantially strengthen this mechanism claim.

### Trivial
None.

## Nice-to-Haves

- Ablation on SSC or PS-MNIST (not just SHD) to bridge the gap between where SOTA claims originate and where mechanistic understanding comes from.

- Learned delay distribution visualizations (histograms of learned delay values across neurons) to reveal what temporal structure the network exploits.

- Multi-seed PS-MNIST evaluation (3–5 seeds) to properly ground the SOTA claim.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic's demand for gradient flow validation as a mandatory experiment**: Demanding empirical gradient norm analysis goes beyond what is standard for an empirical SNN methods paper. The paper provides indirect evidence (Fig. 3B) and a theoretical argument (Fig. 1B). This is a nice-to-have, not a core flaw. Moved to Nice-to-Haves.

- **Harsh Critic's concern about cherry-picking the comparison set in Table 1**: The paper explicitly justifies excluding models with "substantially more complex neuron models" (multi-compartment, attention, GRU-based). This is a reasonable scope restriction—the paper's thesis is that delays can replace neuron-level complexity, so comparing within the LIF-derived family is appropriate.

- **Strength Finder's claim about "first surrogate gradient learning method for recurrent delays" as a core strength**: This is partially undermined by the existence of ASRC-SNN (Xu et al.), which also uses SGL for recurrent delays (per-layer). The novelty is real but narrower than claimed—moved to minor weakness instead.

- **Harsh Critic's suggestion that the paper needs ablation on SSC/PS-MNIST to be acceptable**: This is a reasonable suggestion for strengthening the paper but is not a requirement for acceptance. The SHD functional study provides the mechanistic insights; requiring replication on every dataset is scope creep. Moved to Nice-to-Haves.

## Novel Insights

The paper's most insightful finding is the tension between the two types of delay parameterization: axonal delays (one per neuron, weight-sharing regularizer) vs. synaptic delays (one per synapse, more expressive but more parameters). This distinction, while acknowledged by the authors, is underexplored and may be more important than the recurrent/feedforward distinction they emphasize. The observation that adding feedforward delays to recurrent delays can hurt performance (Table 1, SSC) suggests that delay types may compete rather than complement in certain regimes—a finding that challenges the intuitive assumption that more temporal flexibility is always better.

## Suggestions

- Run the functional study comparison with axonal feedforward delays vs. axonal recurrent delays on SHD to cleanly isolate the effect of recurrence from parameterization. This single ablation would significantly strengthen the paper's central claim.

- Add 3–5 seed runs on PS-MNIST and report mean ± std to properly ground the SOTA claim.

- Soften the "recurrent delays outperform feedforward delays" claim to acknowledge the axonal/synaptic confound, e.g., "the specific recurrent-delay configuration (axonal) outperforms the standard feedforward-delay configuration (synaptic) under parameter constraints."

## Evaluation

**Originality**: The method is a clear and natural extension of DCLS to recurrent connections, with the differentiable interpolation and buffer scheduling being well-designed contributions. The per-neuron recurrent delay learning is novel relative to prior per-layer approaches, though the overall approach builds closely on existing techniques.

**Importance of research question**: Understanding the role of delays in recurrent SNNs is timely and relevant—delays are biologically motivated and neuromorphic hardware supports them. The question of whether network-level delays can substitute for neuron-level complexity is important for the field.

**Claim support**: The SOTA claims on SSC are well-supported (3 seeds, clear margins). The PS-MNIST claim is weakly supported (1 seed). The "recurrent outperforms feedforward" claim is confounded and not rigorously established.

**Experimental soundness**: Results are generally solid but the functional study has the confound issue and the PS-MNIST evaluation lacks statistical rigor. The SHD methodology is commendable.

**Clarity**: The paper is well-written with clear method description and honest discussion of limitations (SHD saturation, validation methodology).

**Community value**: The method is practical (SpikingJelly implementation, compatible with any neuron model), the code is available, and the benchmarking discussion provides useful guidance.

## Calibration

**Anchors used**:
- `/home/wg25r/review_agent/human_reviews_2026/yQ7ssakeKM.md` (avg 6.0, Accept Poster): Cascading eligibility traces for delays in spiking/plasticity. Topically close—also about delays in SNNs. DelRec has similar clarity of method and competitive results, but has the confounded comparison issue that yQ7ssakeKM avoids. **Below this anchor.**
- `/home/wg25r/review_agent/human_reviews_2026/6ZietpbPoB.md` (avg 6.0, Accept Poster): Online pseudo-zeroth-order SNN training. Novel method with competitive results. DelRec has similar profile but with the confound and single-seed issues. **Below this anchor.**
- `/home/wg25r/review_agent/human_reviews_2026/ARDsBYnarO.md` (avg 4.0, Reject): Gamma-memory delays for SNNs. Weaker novelty, limited baselines. DelRec has clearer novelty and stronger results. **Above this anchor.**
- `/home/wg25r/review_agent/human_reviews_2026/GLPmZhhCAE.md` (avg 5.5, Accept Poster): Confounded prior comparisons in debiasing benchmark. Similar weakness pattern (confounded comparisons) but real contribution. **Comparable.**
- `/home/wg25r/review_agent/human_reviews_2026/69mFD2J9rg.md` (avg 2.67, Reject): Seed instability in KG embeddings. Much weaker paper—purely negative results with less constructive contribution. **Well above this anchor.**
- `/home/wg25r/review_agent/human_reviews_2026/zEJd3JXVxb.md` (avg 5.0, Reject): Overclaimed dataset distillation results. Similar weakness pattern but DelRec has a clearer methodological contribution. **Above this anchor.**

The paper sits between the 5.0–5.5 range: above papers with overclaimed results but below papers with clean SNN contributions (6.0). The confounded comparison and single-seed PS-MNIST are real issues that weaken the central claims, but the method itself is solid and the SSC results are well-grounded.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>