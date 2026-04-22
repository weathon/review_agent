# Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Sequential firing of hippocampal place cells is often attributed to sequential sensory drive along a trajectory, and has also been attributed to planning and other cognitive functions. Here, we propose a mechanistic and parsimonious interpretation to complement these ideas: hippocampal sequences arise from intrinsic recurrent circuitry that propagates transient input over long horizons, acting as a temporal memory buffer that is especially useful when reliable sensory evidence is sparse.
We implement this idea with a minimal sequence generator inspired by neurobiology and pair it with an actor–critic learner for egocentric visual navigation. Our agent reliably solves a continuous maze without explicit geometric cues, with performance depending on the length of the recurrent sequence. Crucially, the model outperforms LSTM cores under sparse input conditions (16 channels, $\sim2.5\%$ activity), but not under dense input, revealing a strong interaction between representational sparsity and memory architecture.
Through learning, units develop localized place fields, distance-dependent spatial kernels, and task-dependent remapping, while inputs to the sequence generator orthogonalize and spatial information increases across layers. These phenomena align with neurobiological data and are causal to performance. Together, our results show that sparse input synergizes with sequence-generating dynamics, providing both a mechanistic account of place cell sequences in the mammalian hippocampus and a simple inductive bias for reinforcement learning based on sparse egocentric inputs in navigation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This submission studies the computational role of hippocampal sequences by constructing a neural network model with recurrence and examining how it performs on spatial navigation tasks. The authors find that sequence length is short, the model does not learn the task well (and struggles to generalize). Increasing the sequence length improves performance. The authors' model outperformed several other ML models when the inputs were sparse, but not dense. Investigation of the model revealed that place cell like responses emerged, as did the decorrelation of inputs.

### Strengths
1. Understanding hippocampal replay is an important open question and the authors tackle it with an interesting computational approach. 

2. The results demonstrating the interplay between sequence length and sparsity are interesting and will be of broad interest to computational neuroscientists. 

3. The authors have a detailed discussion section where they emphasize the potential impact their results could have on improving existing ML models. This will help broaden the impact of the work and help it fit more broadly in ICLR. 

4. The experimental design, with obstacles and repeated visual inputs, was nicely done and makes the work more aligned with actual ethologically relevant environments. 

5. The SI ablation study, where the weights were shuffled as opposed to just removed, was nice and convincingly shows spatial information is used.

### Weaknesses
1. The biggest weakness in my mind is that the explanation regarding the CA3 shift register was not so clear. In particular, I did not really understand: a) what it meant that each feature had a ``dedicated prewired sequence''; b) how the sequence was input to the decoder layers; c) what was learnable in the architecture. Later, in the Discussion, the authors mention that the sequence was a ``linear reservoir'', which I think clarifies some of these questions. But providing more details on this, in the section on the model, would be helpful. Perhaps including a figure demonstrating example sequences and a schematic highlighting how the sequence is passed to the next layer? There is also a minor typo in Eq. 2, where the number of 0s in J should be $L - R$, not $L - 1$. 

2. The authors mention in the abstract, introduction, and discussion section the remapping between different environments. I never saw that results mentioned in the results (the figure being alluded to only in the discussion). If that's an important result, it should be highlighted earlier in the results section. 

3. The writing in general felt terse. Not much detail was provided and many new paragraphs were used. This led to jumping between ideas/results and taking up space which then limited how much could be discussed. This was especially apparent in the Introduction, which did not feel very coherent or informative. 

4. Bats and some birds do not exhibit theta oscillations. Despite this, these animals are extraordinary spatial navigators. I don't think this diminishes the authors' work, but I do think it is: a) worth commenting on; b) worth discussing how and why a lack of theta can be reconciled with their model/results.

### Questions
1. How exactly is the CA3 recurrent model being implemented and used by the rest of the model?

2. How do results showing a lack of theta oscillations in bats and birds line-up with the results the authors have shown?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors introduce a biologically inspired model of the hippocampus focused on sequential connectivity from dentate gyrus (DG) to CA3, implemented within a reinforcement learning (RL) agent for visual navigation tasks. The model is designed with theta-sequence firing, while using sparse DG inputs. Navigation experiment demonstrates the model reproduces place field formation, remapping with reward changes, and orthogonalization of trajectories. The CA3 sequence generator helps the agent navigate more efficiently under sparse input conditions compared to standard RNN models.

### Strengths
- The architecture is constrained to reflect biological hippocampal anatomy, specifically the DG-to-CA3 connectivity and sparse input regimes.
- The model captures some experimental observations including place field formation, population-level spatial encoding, and remapping to new rewards.
- The baseline comparative analysis across different recurrent architectures suggests the functional significance of the sequence-generating module.

### Weaknesses
- Inconsistent performance: The CA3-based agent does not outperform LSTM or random RNNs in the dense input regime, and in some benchmarks, random RNNs do better than the CA3 module. I wonder about the generality and robustness of the sequence generation module. Is sequence generation necessarily a built in feature of the hippocampus or is it a consequence of something else? The current results do not support sequence generation in the hippocampus for faster learning.

- Ambiguous role of RL signals: The connection between reinforcement learning algorithms (e.g., A2C/TD error) and the emergence of hippocampal representations is not clearly explored. How does learning a policy or value function specifically influence place fields emergence and remapping (Kumar et al. 2025 ICML)? Could we get the same phenomena if we do behavior cloning or other supervised learning regimes? I do not see the point of using RL in this study other than to model navigation behavior. 

- Insufficient clarity on planning: The authors do not analyze how the CA3 sequence generator supports planning, forward sweeps (Ito et al. 2015 Nature; Pfeiffer & Foster 2013 Nature), or predictive representations (Gardner et al. 2018). E.g. how do reward or environmental changes translate to different planning behaviors. Is your definition of a plan (rapid adaptation) different from policy learning (gradual learning)? 

- Comparative analysis is limited: There is limited discussion on the contribution of sequence generation versus input thresholding in the observed performance with sparse inputs. Why does the proposed model only perform better with input thresholding and worse with dense inputs? More rigorous ablation studies might be relevant to disentangle these factors.

### Questions
1. Is it not expected that a CA3 module with high-thresholded, 2.5% sparsity will outperform other RNNs with sparse input (as in Fig. 2C)? Previous work (Kumar et al. 2022 Cerebral Cortex) has shown improved learning with higher thresholding but only to a point; how does your contribution differ?

2. Similarly, is it not trivial that different trajectories post-training will result in different CA3 sequences?

3. Could a chaotic attractor or standard reservoir network (e.g. Kumar et al. 2022 Cerebral Cortex) achieve similar benefits as your sequence generator?

6. To what extent do your representations align with the successor representation framework (Stachenfeld et al. 2017 Nature Neuro.), or are they more directly tied to reward/value learning signals (as in Kumar et al. 2025 ICML)?

References:
- Kumar et al. 2025 (https://icml.cc/virtual/2025/poster/46112)
- Ito et al. 2015 (https://www.nature.com/articles/nature14396)
- Pfeiffer & Foster 2013 (https://www.nature.com/articles/nature12112)
- Gardner et al. 2018 (https://royalsocietypublishing.org/doi/full/10.1098/rspb.2018.1645)
- Kumar et al. 2022 (https://doi.org/10.1093/cercor/bhab456)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a neuroscience-inspired architecture that combines sparse input - mimiking dentate gyrus (DG) - with a recurrent CA3 sequence generator to model spatial navigation. The authors demonstrate that under sparse input conditions, the CA3 architecture — implemented as a fixed shift-register that propagates theta sequences — outperforms LSTM cores of comparable size on an egocentric visual navigation task in a maze environment. The model develops place cell-like spatial representations and generalizes well to new reward locations, obstacles, and maps after training. Critically, the performance advantage disappears under dense input conditions, suggesting a specific synergy between sparse coding and sequence dynamics that may account for the emergence of hippocampal theta sequences.

### Strengths
**Neuroscience-AI integration:** The work effectively bridges computational neuroscience, AI, and reinforcement learning to account for the neuroscience-related problem of DG and CA3 synergy, theta sequences, and spatial representations emergence.

**Task validity:** The navigation task appropriately mirrors real-world neuroscience experimental paradigms, with the agent navigating using vision to locate rewards.

**Architectural interpretability:** The relatively simple architecture facilitates interpretation of individual components and minimizes confounding variables, allowing clear attribution of the effects observed.

**Generalization capacity:** The trained model demonstrates robust transfer learning capabilities when adapted to new reward locations, novel obstacle configurations, and new maps (Figure 2).

**Regime-dependent performance:** The demonstrated input-regime specificity provides evidence for the functional role of hippocampal sequences: CA3 outperforms LSTMs and alternative architectures under sparse input (~2.5% activity), while underperforming under dense input conditions. This specificity offers mechanistic insight into why theta sequences may have emerged under sparse input conditions.

### Weaknesses
**Incomplete parametric analysis:** The relationship between input density and CA3 performance (also w.r.t. LSTM or alternatives) requires systematic investigation. The paper tests R=1 only with L=1, leaving unexplored critical control conditions such as L=64 with R=4 (more sparsity) or L=64 with R=16 (less sparsity). A thorough parametric sweep varying R while holding L fixed would strengthen conclusions about the sparse-input CA3-DG synergy.

**Missing entorhinal cortex input:** The model considers only DG input to CA3, omitting entorhinal cortex projections entirely. The absence of discussion regarding this major hippocampal input limits biological completeness.

**Insufficient quantitative support for spatial tuning claims:** The claim of line 261 "Units positioned later in CA3 sequences (e.g. columns no.48 and no. 64 in Fig. 4) show broader spatial tuning as reported experimentally" lacks clear visual support in Figure 4 and requires quantitative validation.

**Unclear novelty and utility of spatial kernel measure:** Section 3.5's spatial kernel analysis does not clarify whether this measure is newly introduced or standard in spatial representation studies. The section primarily describes how the measure behaves across architectural modules and training epochs without drawing clear conclusions from these observations.

**(minor) Figure readability:** Font sizes in figures, particularly Figure 2, are too small for comfortable reading while going through the text.

### Questions
I believe the CA3 architecture (central to the paper) explanation of Section 2.3 is not entirely clear and I would have preferred an expansion in the appendix, expecially of the "Multiple Features" case. In my opinion, the meaning of "Each feature is assigned a dedicated prewired sequence of length l" (line 134) requires clarification, as does the definition of $I_F$, $S$, and $J$ in the multi-feature scenario (line 157).

### Soundness
3

### Presentation
2

### Contribution
3
