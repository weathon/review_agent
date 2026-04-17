## Summary

DelRec introduces the first surrogate gradient learning (SGL)-based method for learning axonal/synaptic delays in recurrent spiking neural network layers. The core technique adapts the differentiable triangular-interpolation scheme from DCLS (feedforward delay learning) to the recurrent setting, using a progressive scheduling buffer with an annealed width parameter σ that transitions from broad temporal spreading to precise integer delays. Using vanilla LIF neurons, DelRec achieves claimed new SOTA on SSC (82.58%) and PS-MNIST (96.21%), and competitive results on SHD.

## Strengths

- **Well-motivated and genuinely novel contribution**: Recurrent delays are theoretically important (Izhikevich's polychronization, temporal skip connections for gradient flow) but have been practically inaccessible to SGL training. DelRec fills this gap—the first SGL-compatible method for recurrent delay learning, complementing feedforward-only approaches (DCLS) and the non-SGL EventProp-based method of Meszáros et al.

- **Strong empirical results with simple neurons**: Achieving SOTA on SSC and PS-MNIST using vanilla LIF neurons (no adaptation, no attention, no GRU gating) is practically significant, as it demonstrates that learned temporal structure via delays can substitute for more complex neuron dynamics. This has clear implications for neuromorphic hardware deployment.

- **Thoughtful functional study on SHD**: The multi-model ablation (vanilla SNN, uniform-delay RSNN, random fixed delays, learned feedforward, learned recurrent, both) systematically demonstrates that: (a) any form of delay helps, (b) random delays partially mitigate training difficulties, and (c) learned recurrent delays show particular robustness under parameter constraints (Fig. 3C). The SHD methodology also correctly identifies the dataset's saturation problem and adopts proper validation splits.

- **Clean algorithmic formulation**: The scheduling-matrix-with-circular-buffer approach (Algorithm 1) is well-designed—the observation that only time steps in Ẽ(σ, D) need storage is an effective complexity-reduction insight, and the progressive narrowing of σ is a principled curricular strategy.

## Weaknesses

### Major:

- **Core mechanism novelty is incremental**: The differentiable interpolation technique (triangle kernel h_{σ,d} with annealed σ) is directly adopted from prior work on feedforward delays (Khalfaoui-Hassani et al., 2023; Hammouamri et al., 2024). The paper's primary novelty lies in applying this to recurrent connections and implementing it via a scheduling buffer. While this extension is meaningful and non-trivial (recurrent delays require scheduling spikes into future time steps of the same layer, fundamentally different from the feedforward case where they are just convolutions across time), the paper does not acknowledge this relationship clearly enough, and the algorithmic contribution beyond "DCLS for recurrent connections" is limited.

- **No controlled ablation isolating DelRec's interpolation mechanism from generic learned delays**: The experiments demonstrate that *having learned recurrent delays* improves over no delays or random delays, but they do not compare DelRec's continuous-relaxation-with-σ-annealing against simpler alternatives (e.g., straight-through estimator for integer delays, the softmax-over-discrete-delays approach of ASRC-SNN, or direct gradient through rounded delays). Without this ablation, it is unclear whether the performance gains come from the specific DelRec mechanism or simply from having optimizable recurrent delays at all. This directly undermines claims about DelRec *as a method* rather than merely the concept of learnable recurrent delays.

- **PS-MNIST SOTA claim rests on a single seed**: The paper explicitly states "we only test one seed as all the previous state-of-the-art models on the dataset" for PS-MNIST. The improvement over ASRC-SNN (96.21% vs 95.77%) is modest and entirely unreplicated. SSC also uses only 3 seeds with tight but small margins over SiLIF (82.58% ± 0.08 vs 82.03% ± 0.25). The SOTA narrative—central to the paper's framing—would be substantially strengthened by multi-seed evaluation on PS-MNIST and more seeds on SSC.

- **Apples-to-oranges in "recurrent delays outperform feedforward delays" claim**: In the SHD functional study, recurrent delays are axonal (one per neuron) while feedforward delays are synaptic (one per synapse via DCLS). These differ not only in where delays appear (recurrent vs feedforward) but also in parameterization, granularity, and how they interact with the network. The paper acknowledges this in passing but still draws the strong conclusion that "recurrent delays can achieve better performance than feedforward delays." This conclusion is suggestive but not definitively supported by the current experimental design.

### Minor:

- **Inconsistency in combining feedforward and recurrent delays**: On SSC, the recurrent-delays-only model (82.58%) *outperforms* the combined model (82.19%). On SHD small models (Fig. 3C), combining both provides no advantage. This is acknowledged but not explained, and it raises questions about whether the two delay types interfere during joint optimization or whether the σ annealing schedule creates optimization conflicts.

- **No analysis of learned delay distributions**: The paper never visualizes or analyzes what delays are actually learned—whether they cluster, spread, match task-relevant timescales, or collapse to trivial values. This is a missed opportunity to provide mechanistic insight into *why* recurrent delays help, which would strengthen both the biological interpretation and the practical understanding of the method.

- **No computational overhead or memory analysis**: The scheduling buffer X_rec has dimension N × dim(Ẽ(σ,D)), and early in training σ_init = 10 implies a wide scheduling window. The paper provides no quantitative comparison of training time, memory consumption, or inference cost relative to baseline RSNNs or DCLS-based models, which matters for practical adoption and neuromorphic deployment claims.

- **Only LIF neurons tested despite claimed generality**: The paper repeatedly states the method is "compatible with any spiking neuron model" (Section 2.1), but all experiments use LIF neurons. Even one experiment with a different neuron type would validate this claim and assess complementarity with adaptive mechanisms.

### Trivial:

- The per-neuron learnable parameter p (Eq. 15) is used only on SSC without explanation or ablation of its contribution.

- The claim about neuromorphic hardware deployment ("paving the way for efficient deployment on neuromorphic hardware with programmable delays") is speculative—no hardware experiment or hardware-oriented complexity analysis is provided.

## Nice-to-Haves

- Ablation comparing DelRec's interpolation against simpler delay-learning baselines (straight-through, softmax-over-discrete, etc.)
- Multi-seed evaluation on PS-MNIST (even 3–5 seeds would substantially strengthen the claim)
- Analysis of learned delay distributions and their evolution during training
- Experiments combining DelRec with more expressive neuron models (e.g., AdLIF) to assess complementarity
- Long-sequence task (e.g., Long Sequential MNIST, sCIFAR-10/100) to better test the gradient skip-connection argument
- Computational cost and memory overhead comparisons

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Missing comparison with non-spiking baselines (LSTMs, GRUs, temporal convolutions)"**: The paper is explicitly scoped to the SNN community and compares against SNN methods. Demanding non-SNN baselines is scope creep—this is standard in SNN benchmark papers.

- **"PS-MNIST is not challenging enough; should test on sCIFAR-10/100"**: The paper evaluates on three standard SNN benchmarks (SSC, PS-MNIST, SHD), which is comparable to other accepted work in this space (e.g., DCLS). Broader evaluation would strengthen but is not a core flaw.

- **"The scheduling matrix memory could be prohibitively large for long sequences"**: This concern about computational cost is valid in principle, but the paper does address it via the Ẽ(σ,D) approximation that bounds the scheduling window. The actual memory overhead should be quantified (see minor weakness above), but blanket claims of prohibitive cost without evidence are not justified.

- **"The modified spread function with per-neuron p parameter (Eq. 15) is only used on SSC with no justification"**: This is a minor observation, not a substantive weakness—it's a dataset-specific engineering choice documented in the appendix.

- **"SHD is saturated and shouldn't count as a major benchmark"**: The paper *itself* identifies this problem and explicitly excludes SHD from Table 1, recommending it only as a "proof-of-concept" study. This is good practice, not a weakness.

## Novel Insights

The most interesting empirical finding is the inconsistency where adding feedforward delays on top of recurrent delays sometimes *hurts* performance (SSC: 82.58% → 82.19%). This suggests that the two delay types may create redundant or conflicting temporal representations during joint optimization, and that recurrent delays may subsume much of the temporal modeling capability that feedforward delays provide—particularly in networks where recurrence already enables temporal re-processing. Understanding this interaction (and whether it stems from optimization dynamics or representational redundancy) would be a meaningful direction for future work.

## Suggestions

- **Add a straight-through estimator or discrete-delay ablation** on SHD (small models) to isolate the contribution of the DelRec interpolation mechanism versus naive learned delays.
- **Run PS-MNIST on 3–5 seeds** to make the SOTA claim defensible; report mean ± std.
- **Show histograms of learned delay values** per layer, and ideally their evolution during training, to provide mechanistic insight.
- **Soften the "recurrent delays outperform feedforward delays" claim** to acknowledge the confounding of axonal-vs-synaptic parameterization, or run a controlled experiment with synaptic recurrent delays.

## Score and Decision

**Calibration references:**
- **DCLS (Hammouamri et al., 2024)** — accepted ICLR poster with scores 6,8,8,6. The core novelty critique ("the key method is not new, only its application is") applies similarly here. DCLS adapted an existing interpolation technique to feedforward SNN delays; DelRec adapts it to recurrent delays. Both achieve SOTA results on similar benchmarks.
- **Co-learning delays (Deckers et al.)** — withdrawn/rejected with scores 5,3,3,3. Much weaker novelty and evaluation.
- **P-SPIKESSM** — accepted poster with scores 8,5,8,6. Stronger novelty (new architecture combining SSMs with spiking) but similar benchmark scope concerns.

DelRec makes a real but incremental contribution: extending feedforward delay learning to the recurrent setting with a practical SGL-compatible mechanism. The results are solid, the functional study is informative, and the problem is well-motivated. However, the paper overclaims in two respects (SOTA robustness on limited seeds; recurrent vs feedforward superiority from confounded comparisons), and the lack of a controlled ablation against simpler delay-learning alternatives leaves the specific contribution of the DelRec mechanism somewhat unresolved. These are addressable concerns but they meaningfully weaken the current submission.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>