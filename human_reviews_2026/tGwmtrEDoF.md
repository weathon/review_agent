# Architectural Plasticity for Continual Learning

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Neural networks for continual reinforcement learning (CRL) often suffer from plasticity loss, i.e., a progressive decline in their ability to learn new tasks arising from increased churn and Neural Tangent Kernel (NTK) rank collapse. We propose InterpLayers, a drop-in architectural solution that combines a fixed, parameter-free reference pathway with a learnable projection pathway using input-dependent interpolation weights. Without requiring algorithmic adaptation, InterpLayers conserve gradient diversity and constrain output variability by integrating stable and adaptive computations. We provide theoretical guarantees for bounded churn and show that, under mild assumptions, InterpLayers prevent NTK rank collapse through a non-zero rank contribution from the interpolation weights. Across environments with distributional shifts including permutation, windowing, and expansion, InterpLayer variants (convonly, fullinterp) consistently mitigate performance degradation compared to parameter-matched baselines. Furthermore, lightweight modifications such as dropout improve performance, especially under gradual shifts. These results position InterpLayers as a simple, complementary solution for maintaining plasticity in CRL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes **InterpLayer**, a new neural network layer designed to address the issue of plasticity loss in continual reinforcement learning. The authors provide theoretical analysis of two key properties of the layer: bounded representational drift and preservation of gradient diversity—which offer insight into its stability and adaptability. Finally, empirical studies on the ProcGen benchmark demonstrate the effectiveness of the proposed layer, showing improved plasticity, reduced churn, and better preservation of NTK rank.

### Strengths
- InterpLayer can directly replace standard MLP and convolutional layers without further modification or additional hyperparameters, offering a simple enhancement to existing architectures.
- The theoretical properties of InterpLayer are promising and well-grounded.

### Weaknesses
- Only one environment (*coinrun*) from the ProcGen benchmark is used in the experiments. This seems insufficient for a comprehensive empirical study. I highly recommend that the authors include additional environments to better support the claimed effectiveness of the proposed layer.
- Although Figure 2 shows that InterpLayer achieves good performance relative to the initial task, the raw returns presented in Figures 6–8 indicate that *Standard+SSP+LN* consistently achieves the best overall performance and outperforms all variants of InterpLayer by a substantial margin. If this is not an mistake, does InterpLayer obtain worse initial task performance when compared to *Standard+SSP+LN*? I also recommend moving the raw return plots from the appendix to the main paper to make this comparison more transparent.

### Questions
- Would it be possible for the authors to run additional experiments with a **Standard+SSP+LN+dropout** configuration, as it seems to represent a strong and relevant baseline for comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces InterpLayers, an architectural modification aimed at mitigating plasticity loss in continual reinforcement learning (CRL). Instead of relying on algorithmic resets or regularization methods, InterpLayers embed adaptability directly into the network structure. Each layer consists of two paths: a fixed, parameter-free reference pathway that preserves stable representations, and
a learnable projection pathway that adapts through gradient updates. The two pathways are combined via input-dependent interpolation weights (gates) that dynamically control the mix between stability and adaptability.

The authors provide theoretical analysis showing that InterpLayers (1) bound representational drift (churn) and (2) maintain Neural Tangent Kernel (NTK) rank under mild assumptions. Empirically, experiments on ProcGen CoinRun across several distributional shifts (permutation, windowing, and expansion) demonstrate that InterpLayers outperform parameter-matched baselines and perform competitively with Soft Shrink–Perturb + LayerNorm (SSP+LN). The addition of dropout further improves performance under gradual shifts. Overall, the work proposes a simple yet effective architectural alternative to algorithmic interventions for maintaining plasticity in CRL.

### Strengths
- The work reframes plasticity preservation as an architectural rather than an optimization problem. By introducing a fixed reference pathway, it provides an elegant, orthogonal solution to methods like resets, normalization, or regularization.
- The paper connects architectural design to measurable stability properties, bounded churn and non-collapsing NTK rank, supported by derivations and intuitive proofs.
- Experiments across multiple ProcGen shifts, with 10 random seeds and diagnostics such as NTK rank and churn, show consistent improvements.
- Results are aligned with the theoretical claims and display robustness.
- InterpLayers introduce negligible computational overhead, require no hyperparameter tuning, and can be easily integrated into existing architectures or combined with other methods.
-This direction, architectural plasticity, is promising and opens up pathways for scalable continual learning systems where task resets or algorithmic perturbations are infeasible.

### Weaknesses
- All results are on ProcGen CoinRun, a single-task environment. Other ProcGen games (e.g., CaveFlyer, Heist) or domains such as DMControl or Atari are missing. This makes it unclear whether the gains generalize across input modalities or reward structures.

- The paper mentions “sparse selection or zero-padding” when input and output dimensions differ but does not specify how these projections are chosen. Without an ablation (identity vs. random vs. padded), one cannot tell if the stability comes from architecture or from implicit regularization.
- Methods like Highway Networks, Gated ResNets, FiLM, or LoRA adapters all interpolate between fixed and learned paths. None are compared, which weakens the claim of architectural novelty.
- While dropout improves performance, it’s unclear why? is it due to decorrelation of gates, or mitigation of reference-path bias? An analysis of gate variance or activation entropy under dropout would help.
- Theorem 1 and Theorem 2 bound churn and NTK rank but are non-tight and do not quantify the degree of stability gained. Without empirical correlation (e.g., churn vs performance curve), the theoretical component remains mostly illustrative.
-Plots of gate activations, reference-vs-projection contribution ratios, or NTK spectra evolution could provide intuition about how stability emerges.
- Since one path is fixed, initialization may strongly influence representational geometry. No experiment examines robustness to initialization variance.
- A few formatting and typographical problems (“ae shown”, inconsistent NTK logging frequency—50 vs 100 epochs) slightly reduce polish and precision.

### Questions
- Please clarify how P(MLP) and Pr(Conv) are defined when input and output sizes differ. Are they fixed random matrices, identity selections, or sparse maps? How sensitive is performance to these definitions?
- Could you include baselines such as a Gated-ResNet, Highway Network, or Squeeze-and-Excitation module with learnable skip coefficients? These seem architecturally close to InterpLayers and would sharpen the novelty claim.
- Since one branch is frozen from initialization, does the choice of initialization scheme (Xavier, Kaiming, orthogonal) affect stability or gate distribution?
- Have you tested InterpLayers on larger backbones (e.g., deeper CNNs or transformers)? what would be the intuition here?
- Does the fixed path introduce representational bottlenecks as depth increases?
- Do sigmoid gates saturate toward 0 or 1 during long training?
- Beyond FLOPs/parameter count, what is the actual wall-clock increase in training time and memory usage?
- Since LayerNorm affects gradient variance and NTK rank, how does its presence (or absence) interact with InterpLayers’ stability claims?
- If the task sequence is reversed or randomized, does the observed stability persist, or does the architecture implicitly overfit to a specific shift pattern?
- You mention orthogonality to SSP+LN and ReDo. Could you include quantitative results combining InterpLayers with these methods to test additive benefits?
- Have you tried using InterpLayers in supervised continual learning (e.g., permuted MNIST) or offline RL?
- Consider plotting feature-space trajectories or NTK spectra over time to visualize representational stability versus baselines.
- Could you correlate measured churn or NTK-rank changes with actual performance degradation to show causal alignment between theory and outcomes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work studies Continual RL, especially focusing on the issue of plasticity loss. This paper proposes InterpLayers, a drop-in architectural solution to plasticity loss, to improve the learning performance of CRL. The key idea of InterpLayers is to build a fixed reference pathway and a learnable projection for each layer, which are further interpolated by a state-dependent learnable parameter. The experiments are conducted in CoinRun with three CRL settings. The results include performance comparison and the analysis of churn, NTK, and the interpolation weights learned.

### Strengths
- The proposed method is clearly presented.
- The experiments provide multi-faceted results, including performance comparison and the analysis of churn, NTK, and the interpolation weights learned.
- The related works are satisfactorily discussed. Some more related works should be included, like [1,2] for plasticity loss study, and [3,4] for network architecture study.

---
### Reference

[1] Directions of Curvature as an Explanation for Loss of Plasticity. https://arxiv.org/abs/2312.00246

[2] Deep Reinforcement Learning with Plasticity Injection. https://arxiv.org/abs/2305.15555

[3] Hyperspherical Normalization for Scalable Deep Reinforcement Learning. https://arxiv.org/abs/2502.15280

[4] Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control. https://arxiv.org/abs/2405.16158

### Weaknesses
- I feel that the proposed method is not well motivated in this draft. I did not get the point of the proposed method until I finished reading Section 3.2. I think using some illustrative examples will help convey the motivation and key idea.
- The InterpLayer has a similar architecture with Residual layer. It seems that the major difference is the frozen reference pathway and a learnable interpolation weight $z$. The significance of the difference should be elaborated and empirically examined.
- The experiments should be strengthened:
    - My major concern is that only one environment is used (i.e., CoinRun), which is insufficient to provide a convincing evaluation.
    - The readability of Figure 2 and Figure 3 should be improved, e.g., fontsize, color (red and pink/purple are hard to tell).
    - InterpLayer by itself does not work well according to the green and yellow lines in Figure 2.
    - The authors mentioned that the proposed method is a complementary solution to existing methods. However, InterpLayer is not combined with existing methods to examine its orthogonal benefit.

### Minors

- “(Abbas et al., 2023) investigated weight clipping to provide an upper bound to parameter growth”. Incorrect reference.

### Questions
1. Since the InterpLayer has a similar architecture with Residual layer (the major difference seems to be the frozen reference pathway and a learnable parameter $z$), I recommend that the authors include residual layer as a baseline in the experiments. 
2. It seems that InterpLayer by itself does not work well as shown in Figure 2. Does it mean dropout is necessary to be used together with InterpLayer?
3. Could the authors provide the results for InterpLayer + SSP + LN, as the authors mention that the proposed method is a complementary solution to existing methods?
4. In my opinion, only using CoinRun is not enough to obtain convincing experimental results. Could the author provide (at least) 3 more Procgen environments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes InterpLayers, an architectural solution for plasticity loss in continual reinforcement learning. Each InterpLayer combines a fixed reference pathway with a learnable projection pathway using input-dependent interpolation weights. The authors provide theoretical analysis showing bounded churn and NTK rank preservation, and evaluate on three ProcGen distribution shifts (permute, window, expand), demonstrating that InterpLayers with dropout achieve comparable or better performance than baselines including SSP+LN.

### Strengths
- The architectural approach is clean and task-agnostic, requiring no task boundaries, reset schedules, or additional hyperparameters beyond standard training, making it easy to integrate into existing systems.
- Theoretical analysis provides formal guarantees for bounded representational drift (Theorem 1) and NTK rank preservation under variance assumptions (Theorem 2), connecting architectural design to plasticity metrics.
- Empirical evaluation includes comprehensive ablations examining visual uncertainty, entropy bonuses, annealed sampling, and dropout interactions, with analysis of theoretical metrics (churn, NTK rank) validating the theoretical predictions.

### Weaknesses
- Experimental evaluation is severely limited in scope, testing only on ProcGen coinrun with three distribution shift types. The paper lacks evaluation on standard CRL benchmarks (Atari suite, other ProcGen games, MuJoCo control) that would demonstrate generalizability and enable fair comparison with the broader continual learning literature.
- Performance improvements are modest and inconsistent across conditions. SSP+LN sometimes outperforms InterpLayers (e.g., permute task), and the best results require combining InterpLayers with dropout, suggesting the core mechanism alone is insufficient. The paper provides no clear guidance on when conv-only versus fullinterp should be used.

### Questions
- Can you provide results on standard CRL benchmarks (Atari, other ProcGen environments, MuJoCo continuous control) to demonstrate the method's generalizability and enable comparison with the broader literature?
- The best results require combining InterpLayers with dropout, which suggests the core mechanism may be insufficient. Can you provide deeper analysis of why dropout is necessary and whether InterpLayers provide benefits beyond standard networks with dropout?

### Soundness
2

### Presentation
2

### Contribution
2
