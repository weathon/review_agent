Now I have a thorough understanding of the paper. Let me synthesize the review.

## Summary

DelRec introduces the first surrogate gradient learning (SGL)-compatible method for training delays in recurrent connections of spiking neural networks (RSNNs). Using a differentiable triangular interpolation kernel with annealed width (σ→0) and a circular-buffer-based spike scheduling mechanism, the method relaxes integer delays to real-valued ones during training and rounds to integers at inference. DelRec achieves new SOTA on SSC (82.58%) and PS-MNIST (96.21%) using simple LIF neurons, and matches SOTA on SHD.

## Strengths

- **Novel and well-motivated contribution**: Learning delays in recurrent SNN connections within the SGL/backprop framework is a natural and important extension of prior feedforward delay work (DCLS). The theoretical motivation (Fig. 1) clearly illustrates how recurrent delays create self-sustaining patterns and temporal skip connections.
- **Strong empirical results with simple neuron models**: Achieving new SOTA on SSC and PS-MNIST using only vanilla LIF neurons (without adaptive mechanisms or multi-compartment structures) is a notable result, demonstrating that learned delays can substantially improve temporal processing without complex neuron models.
- **Methodologically sound functional study**: The SHD ablation (Fig. 3) comparing recurrent vs. feedforward delays under parameter constraints, including random-delay baselines, provides useful mechanistic insight beyond just benchmark numbers. The authors' candor about SHD saturation is commendable.
- **Elegant technical design**: The future-oriented scheduling with a circular buffer and annealed triangular kernel is a clean and practical solution. The paper explicitly acknowledges the relationship to DCLS's similar strategy (Section 2.2, "A similar strategy was used in Hammouamri et al. (2024)").

## Weaknesses

### Major:

- **Limited novelty beyond applying DCLS to recurrent connections**: The core technical idea—differentiable interpolation with annealed σ for delay learning—is directly adapted from the feedforward DCLS method (Hammouamri et al., 2024). The paper itself acknowledges this ("A similar strategy was used in (Hammouamri et al., 2024)"). The main new contribution is the future-oriented scheduling matrix/circular buffer for recurrent connections, which is a reasonable but incremental architectural change. The closest prior work on recurrent delays (Mészáros et al., 2025; Xu et al.) is discussed but differs mainly in using EventProp vs. SGL, rather than fundamentally different algorithmic ideas. This mirrors concerns raised about DCLS itself: "The novel contribution of this paper over the DCLS paper is not clear. Is it just the evaluation on multiple tasks?" (DCLS Reviewer 4, score 6).

- **The claim that recurrent delays "outperform" feedforward delays is not well-supported on the primary benchmarks**: On SSC, "DelRec (only Rec. delays)" achieves 82.58% while "DelRec (Rec. and Ff. delays)" achieves only 82.19%—adding feedforward delays *hurts*. This counterintuitive result is not explained. On SHD (Table 2), DCLS (feedforward only) at 93.77% outperforms DelRec (only Rec. delays) at 93.39%. The claim that recurrent delays outperform feedforward ones rests entirely on the SHD parameter-constrained ablation (Section 3.2), which the authors themselves characterize as using a saturated dataset with small models. The abstract's statement "trainable recurrent delays outperform feedforward ones" is thus stronger than the evidence warrants—it is accurate only in the constrained SHD regime, not on the headline SOTA benchmarks.

- **PS-MNIST result lacks statistical confidence**: The PS-MNIST SOTA claim is based on a single seed (the paper states: "we only test one seed as all the previous state-of-the-art models on the dataset"). This makes it impossible to assess statistical significance of the 96.21% vs. 95.77% (ASRC-SNN) improvement. For a SOTA claim on a benchmark, multi-seed evaluation is essential.

- **Insufficient methodological detail on key design choices**: (a) The paper claims it "eliminates the need to predefine a maximum delay range" (Introduction), but Eq. 13 shows that the buffer dimension depends on max_j(d_j), which requires bounds on delay parameters. How delays are bounded during training (softplus? clamp?) is not specified. (b) The per-neuron annealing parameter p_i (Eq. 15) used on SSC adds complexity without discussion of its effect in the main text. (c) Algorithm 1 is referenced but not included in the paper text (only in the appendix). These omissions matter because they directly affect what delays can be learned and how the method scales.

### Minor:

- **The gradient-stability narrative is asserted without direct evidence**: The Introduction and Fig. 1B motivate recurrent delays as mitigating vanishing/exploding gradients via temporal skip connections, but the paper provides no quantitative gradient norm measurements, training loss curves, or convergence speed comparisons to substantiate this claim. The only evidence is better test accuracy on SHD, which could arise from many factors.

- **No analysis of learned delay values**: The paper never visualizes what delays are actually learned—their distributions, magnitudes, or temporal structure. This makes it impossible to assess whether DelRec discovers meaningful temporal representations or whether delays function merely as additional learnable parameters.

- **No computational overhead analysis**: The scheduling matrix X_rec of size N × dim(Ẽ(σ,D)) introduces memory overhead that grows with max delay. While this is the price of the method, no quantification of training time or memory is provided, despite neuromorphic deployment being a stated motivation.

### Trivial:

- The "Rec. and Ff." vs "only Rec." column marking in Table 1 is slightly confusing since "DelRec (only Rec. delays)" still has Rec. Delays checked (which is correct, but could be clearer).

## Nice-to-Haves

- Ablation of recurrent vs. feedforward delays on SSC/PS-MNIST (not just SHD), to support the "recurrent delays outperform feedforward" claim where it matters most.
- Run PS-MNIST with multiple seeds to establish statistical significance.
- Report learned delay distributions (histograms, statistics) to reveal what temporal structure is captured.
- At least one experiment combining DelRec with a more complex neuron model (e.g., AdLIF) to demonstrate whether gains are complementary.
- Report wall-clock training time and memory overhead compared to a baseline RSNN.

## Removed Points

- **"Cannot be independently verified" / model availability concerns**: The paper references DCLS, ASRC-SNN (Xu et al.), and other methods. These are cited works that exist; availability concerns are not valid criticisms.
- **"Not yet released" concerns about Xu et al.** (ASRC-SNN): Cited and compared against; treat as real.
- **Unfair comparison with other methods because baselines use more complex neurons**: The paper deliberately excludes multi-compartment/attention/GRU models from comparison (with clear justification in the footnote), and this actually makes the comparison *more* controlled since DelRec uses simple LIF neurons. The comparison favors fairness, not the authors.
- **Strawman claim that the comparison with ASRC-SNN (0.37M params, 81.54%) is unfair because DelRec uses same param count**: This is a controlled, same-parameter comparison—which is exactly what one wants. The improvement is real.
- **Demanding comparison with non-spiking networks**: Outside the stated scope of the paper, which is an SNN contribution evaluated on SNN benchmarks.
- **Format nitpicks about parser artifacts, equation numbering, LaTeX issues**: These are not substantive criticisms.

## Novel Insights

The ablation on SHD revealing that random fixed delays already improve RSNN training (Fig. 3B) is an interesting finding with practical implications: even non-learned delays provide gradient benefits, suggesting the expressivity gain from temporal skip connections may be more important than the optimized delay values themselves. The counterintuitive result that combining recurrent and feedforward delays hurts on SSC (82.19%) vs. recurrent-only (82.58%) suggests potential interference between the two delay optimization pathways that merits investigation.

## Suggestions

- Soften the claim "trainable recurrent delays outperform feedforward ones" in the abstract to reflect that this finding is specifically supported in the low-parameter SHD regime, not uniformly across all benchmarks.
- Run PS-MNIST with ≥3 seeds and report mean ± std.
- Add a paragraph to the main text analyzing learned delay distributions (min, max, mean, histogram) and discussing the buffer memory overhead.
- Explain how delay values are bounded during training and clarify the relationship between the "no need to predefine a maximum delay" claim and the buffer dimension in Eq. 13.

## Evaluation Axes

- **Originality**: Moderate. Extending DCLS's differentiable interpolation to recurrent connections is a natural and well-executed step, but the core technique is adapted from prior work. The future-oriented scheduling buffer is the main algorithmic novelty.
- **Importance of research question**: High. Learning delays in recurrent SNNs is genuinely underexplored and could unlock better temporal processing.
- **Claims well-supported**: Partially. SOTA results are solid, but the "recurrent delays outperform feedforward" claim is overstrengthened relative to evidence, and the gradient-stability claim is unsubstantiated.
- **Soundness of experiments**: Adequate for the main benchmarks, but the SHD-only ablation for the feedforward/recurrent comparison and the single-seed PS-MNIST result are weaknesses.
- **Clarity**: Generally good; the method description is clear though some implementation details are deferred to the appendix.
- **Value to community**: High within the SNN field. DelRec fills a clear gap (learning recurrent delays with SGL) and the code release enhances reproducibility.

## Score and Decision Calibration

Comparing to calibration anchors:
- **DCLS (ICLR 2024, scores 6,8,8,6, Accept poster)**: The direct predecessor. DCLS extended delay learning to feedforward SNNs using DCLS, scored ~7 average. DelRec applies similar ideas to recurrent connections with new scheduling mechanism + stronger benchmarks. Similar novelty profile (method adaptation rather than fundamental novelty). DCLS had comparable concerns about novelty ("just applying DCLS").
- **Co-learning delays (scores 5,3,3,3, Withdrawn)**: Much weaker paper—limited datasets, insufficient novelty, no convincing SOTA.
- **SHD-SNN papers receiving 5-6**: Generally have limited novelty or limited benchmark scope.

DelRec is above the Co-learning paper due to stronger results and more thorough evaluation. It is likely below DCLS's best scores (~8) because it lacks DCLS's novelty of introducing the delay learning mechanism itself. The key question is whether extending an existing method to recurrent connections, with competitive but not dramatically superior results, warrants the same scores. The SHD-only functional study and overclaiming about recurrent vs. feedforward delay superiority pull this down from DCLS-level.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>