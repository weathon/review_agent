# Global Neural Inhibition Improves Mixture of Experts in Multi-task Reinforcement Learning

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 2

## Abstract
Title: DYNAMICALLY GATED MIXTURE OF EXPERTS FOR MULTI-TASK REINFORCEMENT LEARNING

Multi-task reinforcement learning (MTRL) promises unique strengths against single-task RL because of generalization across tasks through parameter sharing and composition. Existing methods rely on local routing or static compositional weights in Mixture-of-Experts (MoE) without the ability to adapt to evolving temporal context. To improve the inherently temporal and dynamic systems in MTRL, we introduce a global recurrent inhibition network (GRIN) that performs dynamic gating across time, selectively modulating expert activations based on temporally accumulated context. Our formulation propagates information across time steps to preserve global activation information across the model. Notably, using the gating approach, we found statistically significant improvements over state-of-the-art MTRL methods, with an empirical +3.7% improvement on the Metaworld MT50 benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose an extension of MoEs by which they modulate features to route data more effectively through the network. They validate this approach through experiments on MT10 and MT50 from Meta-World, alongside experiments in vision and language modelling.

### Strengths
Effective architecture design that can be applied across multiple problem settings is a useful idea which should be more widely explored.

### Weaknesses
1. The largest weakness are the MTRL results. Recent work has found that the results produced by the PaCo and MOORE papers was inaccurate [1]. Due to the re-using of the results from MOORE, I ask the authors to run the relevant baselines under the same benchmark settings (reward version, specifically) to enable effective comparison between results. From my understanding, [1] produced a set of benchmark algorithms that should be easily run. 

2. The need of specialized architectures, especially in homogeneous MTRL (i.e. tasks share same action & state spaces), has been found to be inaccurate while the driver of performance is in fact the parameter count [2]. The results of MOORE, for example, can be matched/outperformed by a similarly scaled feed-forward neural network as found in [2]. How does the proposed method compare to the findings in [2], that suggests that specialized architectures are not needed in MTRL?

3. The authors need to update their results to use a measure such as IQM [3] to ensure that their results are statistically significant. The current mean & std formulation shows large overlaps in performance which clouds statistical significance. 

4. This is a common misconception, but there is no such thing as 'mt10-rand' or 'mt10-fixed.' Early works used Meta-World incorrectly and introduced this odd nomenclature. Due to it's goal conditioned nature, Meta-World should only be 'mt10-rand' (but you don't need to refer to it as such) where new goals are sampled upon resetting. 


Citations
1. https://arxiv.org/abs/2505.11289

2. https://arxiv.org/abs/2503.05126

3. Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare (2021). Deep RL at the Edge of the Statistical Precipice. NeurIPS

### Questions
How could the method be extended to be multi-modal? Could the method be extended to the heterogeneous MTRL setting?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GRIN (Global Recurrent Inhibition Networks), a mechanism for dynamically modulating mixture-of-experts (MoE) architectures using global network activations. The approach introduces an "inhibition head" that observes activation patterns across the network and generates sigmoid gating masks applied at three locations: input representations (IR), mixture weights (MW), and output representations (OR). The paper presents a recurrent formulation that iteratively refines these masks over T timesteps using backpropagation through time. Empirical evaluation focuses on MetaWorld multi-task reinforcement learning benchmarks (MT10 and MT50), with additional demonstrations on a synthetic vision dataset and limited-scale language modeling.

### Strengths
**Clear and Modular Architectural Design.** The paper proposes a modular inhibition head that can be integrated with existing MoE architectures, targeting three distinct modulation points (IR, MW, OR). 

**Competitive MT10 Performance.** The reported 89.4 ± 1.0% success rate on MetaWorld MT10 at 20M steps represents good empirical performance. 

**Cross-Domain Evaluation Attempt.** The paper evaluates on RL, vision, and language tasks, demonstrating an effort to show generality beyond a single domain. This attempt to demonstrate broader applicability is valuable even if the non-RL experiments are limited in scale.

### Weaknesses
**1. Central Contradiction: Recurrence Motivation Not Supported by Evidence.**

The paper's title (*pdf title mentions "recurrent", openreview title omits it*), abstract, and introduction strongly emphasize "Recurrent Inhibition Networks" as the core contribution. Lines 42-44 explicitly motivate recurrence by stating that feedforward architectures "*make routing decisions based on current input alone, without information from later layers.*" The proposed solution is described as a recurrent formulation that "*iteratively refines modulation decisions*" (line 45).

However, Table 3 directly contradicts this central claim through the ablation on recurrence depth:
- 1 step (T=1): 91.3 ± 8.4%
- 2 steps (T=2): 76.7 (15 percentage point drop)
- 3 steps (T=3): 78.0 (13 percentage point drop)

These performance degradations are substantial and fall well outside the baseline's error bars when leveraging the central mechanism the paper proposes. The paper kind of tacitly acknowledges this with the brief explanation in L311 "a single recurrence step is typically sufficient for less complex MTRL tasks" (line 311). So in essence it means that without recurrence is how we get best results?

- **At T=1, there is no recurrence.** The system performs a single forward pass where an RNN module processes one timestep. The recurrent connections, the defining feature of RNNs that enables integration over multiple steps, are never utilized. What remains is effectively a feedforward network with a particular parameterization.

- **Almost all main results use T=1**. The paper explicitly uses T=1 for MT50 (line 266, 'recurrence depth of 1, as determined optimal in our ablation studies'). For MT10, while the recurrence depth is not explicitly stated, it would be inconsistent to use T>1 given that: (a) the ablation shows T=1 is optimal and T>1 severely degrades performance, and (b) MT50 explicitly uses T=1 based on these ablations. If the paper did use T>1 for MT10, this would contradict their own ablation findings and should be clarified.

- **The paper never demonstrates where T>1 helps.** The explanation references "less complex" tasks, yet MT50 (50 tasks, substantially more complex) also uses T=1. No experiments test whether harder tasks benefit from multiple recurrence steps.

- **The recurrence formulation is both unused *and* conceptually flawed.** On top of the paper not leveraging recurrence (T>1) in any main results the formulation itself is mismatched with its motivation. The paper motivates recurrence as accessing "information from later layers" (lines 42-44), yet states "*we use the same x to propagate multiple time-steps*" (line 169). With repeated processing of the same input, the system at best observes how layers responded in previous iterations, not how downstream layers respond to current gating decisions in the architectural sense. The paper never specifies whether $a_{global}$​ includes later-layer activations.

Overall, all of these seem to represent a fundamental misalignment between what the paper claims to contribute (recurrent inhibition that accesses later-layer information) and what actually works (global feedforward inhibition at T=1). 


**2. Mathematical specification insufficient for reproduction**
- **Undefined $a_{\text{global}}$.** The paper never specifies which layers feed $a_{\text{global}}$, how heterogeneous tensors are pooled/aggregated, or the resulting shape.

- **Mask dimension mismatch.** Eq. 1 sets $s^{l_{\text{IR}}}, s^{l_{\text{MW}}}, s^{l_{\text{OR}}}=\sigma(G(a_{\text{global}}))$, but these masks require *different sizes* ($\mathbb{R}^d$ for IR/OR, $\mathbb{R}^N$ for MW). No multi-head decoder, per-site projections, or broadcasting rules are described.
    
- **Objective over steps unclear.** Text suggests per-step losses are summed ($L=\sum_t L_t$), which scales gradients with $T$; if only $L_T$ is used, it isn’t stated.

In general key specs are missing, so correctness can’t be verified and reproduction requires guesswork.

**Further issues**
- **MT50 incomplete & MT10 not statistically validated.** MT50 is reported only at 10M/16M (one case **1 seed**), although authors mention preliminary so the results cant be considered with much weight. The MT10 SOTA gap ($89.4 \pm 1.0$ vs. $88.4 \pm 3.4$) lacks is too narrow to claim SOTA imo without more rigorous statistical validations.
  
- **CV/LM results are marginal and under-contextualized.** Vision shows a ~0.6 pp gain with overlapping variance; LM reports losses on unusual $10^{-6}$–$10^{-10}$ scales without calibration.
    
    
- **Ablation inconsistencies & OPE mislabeling.** 
- Table 3 favors **3 experts** yet main runs use **4/6** without rationale; OR-only ≈ OR+MW (MW-only lags), implying gains come mainly from output modulation.
- Section 4.3’s “FQE” seems to be Gaussian Q-fit/correlation instead, can the authors comment on this?

### Questions
1. **Recurrence:** Since $T>1$ degrades performance and main results use $T=1$, where (if anywhere) does recurrence help, and by what mechanism?    
2. Precisely define $a_{\text{global}}$ (layers and aggregation), explain how $G$ produces IR/MW/OR masks of different sizes (e.g., separate heads), and state the objective over steps ($L_T$ vs $\sum_t L_t$) and any normalization/clipping.
3. **CIGI:** How is deterministic single-sample inference performed given dependence on prior-batch activations, and what do ablations show about CIGI’s contribution?
4. **Design choices:** Why use 4/6 experts when ablations favor 3, and is MW gating necessary given OR-only $\approx$ OR+MW?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Global Recurrent Inhibition Networks (GRIN), a biologically inspired mechanism for dynamic modulation in Mixture-of-Experts (MoE) architectures. GRIN adds a global inhibitory head that aggregates activations from across the network and produces gating masks applied at multiple points (input representation, mixture weights, and output representation). The inhibition is recurrent, iteratively refining the modulation over several time steps, akin to LSTM-style gating. 

The authors evaluate GRIN primarily in multi-task reinforcement learning (MTRL) using MetaWorld (MT10, MT50), achieving strong results compared to prior MoE-based MTRL methods (PaCo, MOORE). They also test GRIN on a small vision dataset (mixed MNIST/squares) and a language modeling task (WMT-En) to show generalization. Ablations investigate recurrence depth, gating locations, number of experts, and the impact of inhibition masks on Q-values. The results suggest that global inhibitory modulation can stabilize and improve multi-task learning dynamics.

### Strengths
- The idea of using recurrent global inhibition for adaptive gating in MoE architectures is interesting. The paper connects biological inhibition mechanisms with modern sparse neural architectures in a plausible way.
- The recurrent formulation (Eq. 2) and the integration of inhibition masks at different architectural locations are coherent. It could be attached to existing MoE architectures with minimal changes.
- The use of MetaWorld and ablations provides credible empirical grounding.
- Improving stability and specialization in MTRL MoEs is relevant. GRIN could inspire new designs for adaptive routing or modulation in large-scale RL and multi-domain systems.
- GRIN achieves consistent improvements on MT10 and MT50 over strong baselines (PaCo, MOORE).

### Weaknesses
- The paper lacks a solid theoretical explanation or variance analysis for why recurrent inhibition improves stability. The analogy to biological inhibition is interesting but remains qualitative.
- The work’s main claim of “general applicability” is under-supported. The vision and LM results are limited to small synthetic or low-scale datasets, leaving unclear whether GRIN scales to large-scale LLMs or complex visual tasks.
- Although several ablations are presented, they omit key comparisons such as ablations without recurrence but with global inputs, or with local recurrent gating only. This makes it difficult to isolate which component drives the gains.
- Confidence intervals and significance tests are missing for some results (especially MT50). Some tables include partial seeds or intermediate checkpoints, which weakens reproducibility.
- Some design and training details are underspecified, for example, hidden dimension of the inhibition head, update frequency of global activations, and precise backprop-through-time length. This may limit re-implementation.
- The relation to prior MoE gating approaches (e.g., Expert Choice, Switch Transformers, or ST-MoE) is discussed only superficially. It is unclear whether GRIN could integrate or conflict with these systems.
- The exposition is dense and occasionally ambiguous. Some notation and definitions are introduced without sufficient explanation, making sections hard to follow. Figures and tables could be more self-contained and clearly labeled, as captions often lack essential context. Overall, the low clarity and incomplete descriptions reduce the paper’s accessibility and reproducibility.

### Questions
- Can you provide variance reduction or gradient-norm analyses to support the claim that GRIN stabilizes optimization compared to local gating or MOORE?
- What is the computational overhead (in FLOPs or wall-time) introduced by the recurrent inhibition step?
- Did you test GRIN in single-task RL or standard supervised settings to check whether the inhibition mechanism harms specialization when no task diversity is present?
- How sensitive is GRIN to the number of recurrence steps T and the update frequency of global activations?
- Could GRIN be combined with routing regularizers (like load-balancing or entropy penalties) used in large-scale MoEs?
In the language modeling results, what is the model scale (parameter count) and training compute? Are improvements statistically significant across seeds?
- What is the interpretation of negative correlations between mixture-weight inhibition and Q-value improvement (Table 4)?
- How is backpropagation-through-time truncated or regularized to avoid exploding gradients during recurrent inhibition?
- How does the recurrent inhibition affect gradient flow and credit assignment across experts? Have you visualized gradient magnitudes through time?
- Is the recurrent head shared across layers or instantiated per MoE layer? Clarify parameter sharing and scaling implications.
- I was wondering how does GRIN behave as the number of experts N grows (e.g., 8→64)? Is the global inhibition signal still efficient?
- Have you tested inference latency when the recurrent inhibition loop is unrolled?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Dynamic neural architectures, such as Mixture-of-Experts (MoE) models, are designed for efficiency and expressivity by using specialized pathways, but their core routing mechanism is challenging to design. To enhance this, the authors introduce Global Recurrent Inhibition Networks (GRIN), which are inspired by biological inhibitory neurons and apply an inhibitory gating function to the mixture weights, dynamic routers, and output representations. GRIN employs a recurrent algorithm that integrates signals from all neurons to maximize connection diversity, allowing the network to dynamically modulate its paths, suppress redundant signals, and improve pathway selectivity. The research tests a cascaded architecture where connection diversity progressively increases, demonstrating a direct correlation between this diversity and performance improvements. Empirical evaluations across simulated vision, language modeling, and MTRL show that GRIN consistently outperforms baseline MoE architectures, particularly on the challenging MetaWorld MT10 and MT50 benchmarks.

### Strengths
- I believe the introduced solution is interesting and the idea of considering a global state for routing sounds.
- The methodology section is well-presented as well as the literature discussion.

### Weaknesses
- The paper needs a couple more iterations over the writing and formatting. For instance, there are many unused spaces (which created noticeable white spaces at every page). In addition, subsection heading 5.2 is placed in a wrong position.
- However, the main and the most crucial limitation of this work is the motivation of the presented problem and solution to MTRL. I think this point is not well stated in the work. I was also confused because of the cross domain experiments. It seems like the solution is for many domains (which is totally fine), why should the paper target MTRL as the main problem?
- The authors choose to not run the baselines (including MOORE) on MT10 while rerunning only MOORE on MT50. Why is that? I think it is better to either borrow the results or rerun everything.
- In addition, there is inconsistency in training steps across baselines in MT50, the authors acknowledged that. But this is an indication that this submission is incomplete.
- I believe the performance gain of using the method in MTRL setting is not significant.
- I think the discussion in Section 4.2 needs writing adjustment since I could not understand everything.

I think this work has a great potential. However, the main issue with this submission is that the motivation for the problem and solution in the context of MTRL, which is the main problem setting, is not well highlighted and written. In general, the submission needs some writing improvements. In addition, the main results in the paper, on Metaworld, should be completed. Considering all of that in the submission, the work would be ready. However, I understand that the rebuttal period could be not enough for that.

### Questions
- What is the motivation behind proposing this solution for MTRL? Why is it important to consider a global state to create pathways in MTRL?
- What is the reason for rerunning only MOORE in MT50  while taking results from the MOORE paper for others? I would recommend rerunning all baselines or at least becoming more consistent in terms of results reporting. This should improve the quality of the paper.
- Since it is a MTRL paper, why not focusing on more MTRL benchmarks, for example, by reporting more results on the MiniGrid benchmark instead of the cross domain studies? Maybe moving them to the appendix or highlighting only one. Yet, I believe there is no real space issue since there are white spaces at the end of each page.
- How important is to use MOORE, in terms of orthogonal features, as a base algorithm?

### Soundness
2

### Presentation
1

### Contribution
2
