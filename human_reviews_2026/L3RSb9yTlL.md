# Scalable Multi-Task Low-Rank Model Adaptation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Scaling multi-task low-rank adaptation (LoRA) to a large number of tasks induces catastrophic performance degradation, such as an accuracy drop from 88.2% to 2.0% on DOTA when scaling from 5 to 15 tasks. This failure is due to parameter and representation misalignment. We find that existing solutions, like regularization and dynamic routing, fail at scale because they are constrained by a fundamental trade-off: strengthening regularization to reduce inter-task conflict inadvertently suppresses the essential feature discrimination required for effective routing. In this work, we identify two root causes for this trade-off. First, uniform regularization disrupts inter-task knowledge sharing: shared underlying knowledge concentrates in high-SV components (89% alignment on Flanv2→BBH). Uniform regularization forces high-SV components to update in orthogonal directions, directly disrupting the shared knowledge. Second, Conflict Amplification: Applying LoRA at the component-level (*e.g.*, $W_q, W_v$) amplifies gradient conflicts; we show block-level adaptation reduces this conflict with only 50% parameters. Based on these insights, we propose mtLoRA, a scalable solution with three novel designs: 1) Spectral-Aware Regularization to selectively orthogonalize low-SV components while preserving high-SV shared knowledge, 2) Block-Level Adaptation to mitigate conflict amplification and largely improve parameter efficiency, and 3) Fine-Grained Routing using dimension-specific weights for superior expressive power. On four large-scale (15-25 tasks) vision (DOTA and iNat2018) and NLP (Dolly-15k and BBH) benchmarks, mtLoRA achieves 91.7%, 81.5%, 44.5% and 38.5% accuracy on DOTA, iNat2018, Dolly-15k and BBH respectively, outperforming the state-of-the-art by 2.3% on average while using 47% fewer parameters and 24% less training time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses performance degradation in multi-task LoRA fine-tuning caused by parameter and representation misalignment across tasks. It observes that high singular-value (SV) components are discriminative yet conflict-prone, while low-SV ones are mostly noise, and that inserting LoRA at fine-grained attention components amplifies gradient conflicts. To tackle this, the authors propose mtLoRA, which introduces three modules: spectral-aware regularization to orthogonalize low-SV components, fine-grained routing to weight LoRA dimensions adaptively, and block-level adaptation to mitigate attention coupling.

### Strengths
**Simple Plug-and-play Module:** The three changes can be plugged into existing multi-task LoRA systems; the block-level residual adapter is especially practical.

**Strong Empirical Results:** The full configuration (orthogonality + block-level scope + channel-wise routing) achieves the best results.

### Weaknesses
**Incremental Technical Novelty:** Each component (orthogonal regularization, routing, adapter placement) is known; the contribution is largely in diagnostics + integration. The spectral re-weighting uses SVD on LoRA factors with a monotone mask, and fine-grained routing is a natural extension of MoE-style gating—useful, but methodologically incremental.

**More Analysis in Routing/regularization Interaction:** The work notes that stronger orthogonalization can harm routing discriminability (entropy rises), but practical guidance is thin (e.g., how to pick λacross datasets; robustness to mis-specification).

**Compute and Memory Costs:** Fine-grained (dimension/group-wise) routing and per-epoch SVD add overhead; block-level adapters may increase parameters. There is no wall-clock/FLOPs/memory report to substantiate the claimed practicality at scale.

**Ablation Depth:** Useful analyses exist (e.g., routing granularity; attach scope), but further breakdowns would strengthen claims: (i) how much each component contributes under very high N (e.g., 50–100 tasks) in both domains; (ii) failure modes where block-level adapters underperform (Table 7 suggests fine-grained routing can hurt in vision).

**Routing Entropy vs. Accuracy:** Can you provide a practical recipe for tuning λthat balances conflict reduction and routing discriminability across datasets/backbones? Any automatic schedule?

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In the paper 'Multi-Task Low Rank Model Adaptation' the authors identify two problems.
Parameter and representation misalignmen. Which lead to conflicting weight updates, which cause different Lora updates to pushe the model in oposite directions.

As a remedy the paper proposes
spectrum aware regularization,
block level lora, and the use of a a router network

The proposed methods ar e tested on the
dolly-15k, Dota and iNaturalist datasets.

### Strengths
- The mask formulation is very elegant. It reminded me a lot of Thikonov regularization, which is a good idea.
- The experiments agree with related work, orthogonalization, for example, has been shown to work well for fine-tuning in concurrent related work like https://arxiv.org/pdf/2507.13260 .
- Experimentally the authors observe competitive or improved results.

### Weaknesses
- The variables appearing in the equations throughout the paper are often not properly defined. In equation (1) for example, dimensions and the domain, are missing.
- The dataset choice is not well explained. Why did the authors choose the Dota dataset instead of VTAB-1k for example? I don't think this choice is explained in a convincing way. 
- While the paper presents improved numbers, it does not present evidence for the mechanism that the paper presents as an explanation for the efficiency of the presented methods. Which could have included plots of spectra or correlation plots of LORA adapters. 

#### Code
On my machine the anonymous code repository link did not work.

### Questions
- Is it possible to plot singular value spectra, before and after tuning, with and without tuning, as well as pairwise correlations of adapter updates to support the claims made regarding the mechanisms which underpin the presented methods?

- Is it possible to report standard deviation across across multiple seeds, for all experiments? This would make the paper more credible statistically.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the misalignment problem in multi-task LoRA models. The authors observe that when multiple LoRA modules are trained independently for different tasks, feature subspaces become misaligned, causing interference and performance degradation. To overcome this, the paper proposes mtLoRA, which introduces three key innovations: spectral-aware regularization, fine-grained routing and block-level adaptation. Comprehensive experiments across vision (DOTA, iNat2018) and NLP (Dolly-15K) benchmarks show consistent gains, achieving up to 4.4% improvement over HydraLoRA.

### Strengths
1. The paper identifies and formalizes spectral heterogeneity in multi-task LoRA modules—an overlooked cause of task conflict. The introduction of spectral-aware regularization and fine-grained routing is conceptually fresh and well-motivated by empirical analysis (e.g., SVD studies in Table 1).

2. The methodology is well-grounded and clearly explained, combining theoretical motivation (singular value analysis, gradient interference) with solid empirical validation. The proposed masking and weighting schemes in spectral regularization are intuitive yet effective.

3. The writing is structured and readable, with comprehensive ablations (Tables 4–9) that isolate each design’s contribution. The visualization of routing entropy and singular-value conflicts provides helpful interpretability.

4. The method significantly improves multi-task adaptation robustness, especially in extreme task settings (up to 25–100 tasks). The generality across both vision and NLP domains enhances its impact.

### Weaknesses
1. While the empirical spectral analysis is convincing, the paper lacks a formal theoretical justification of why spectral-aware regularization specifically balances discrimination and conflict. A more rigorous connection between singular value magnitude and information content could strengthen the conceptual contribution.

2. Although Experiments span domains, all are relatively moderate in scale and may not fully test scalability to large LLM adaptation or multi-domain real-world tasks.

3. The paper does not cite or discuss related efforts that also attempt to align task representations, such as [1]. That work similarly mitigates task conflicts by aligning inter-task representation distributions (through variance regularization). A comparative discussion would help clarify how mtLoRA differs or complements such representation-alignment perspectives.
- [1] Impartial Multi-Task Representation Learning via Variance-invariant Probabilistic Decoding. ACL 2025.


4. The ablation on routing granularity (Table 5) could be more thorough by exploring computational trade-offs and visualizing per-dimension routing patterns.

5. Since mtLoRA builds upon HydraLoRA with added modules, the parameter increase and training cost should be analyzed more explicitly.

### Questions
1. How does mtLoRA differ from distribution-alignment approaches (e.g., task covariance alignment)? Could spectral-aware regularization be interpreted as an implicit distribution alignment mechanism?

2. The paper empirically splits singular values into top-10%, 10–50%, and 50–100% bands. Is this partition fixed or adaptive? Would learning the spectral mask end-to-end yield further improvement?

3. What is the computational and memory cost of channel-wise routing for large models (e.g., LLaMA2-13B)? How does it scale compared with standard HydraLoRA?

4. Have the authors tested the contribution of each design on distinct architectures? Does block-level adaptation generalize beyond ViTs and LLMs?

5. Does mtLoRA preserve balanced performance across heterogeneous tasks (e.g., easy vs. hard or high-data vs. low-data tasks)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes mtLoRA, a "plug-and-play" solution to address catastrophic performance degradation in multi-task Low-Rank Adaptation. The authors argue that naively combining task-specific LoRA modules leads to "misaligned feature subspaces", and that existing solutions (regularization or routing) fail because they are applied independently and uniformly. The paper identifies two primary causes for this failure: 1) Spectral Heterogeneity, where LoRA's high singular value (SV) components encode both critical task-discriminative information and the majority of parameter conflicts, making uniform regularization suboptimal; and 2) Gradient Amplification, where attaching LoRA to individual $W_q/W_k$ matrices amplifies gradient conflicts through the attention mechanism's Softmax. Based on this diagnosis, mtLoRA introduces three components: 1) Spectral-Aware Regularization, which selectively applies strong orthogonality to low-SV (noise) components while preserving high-SV (discriminative) components; 2) Fine-Grained Routing, which uses a router network to produce dimension-specific (vector) weights for combining LoRA modules, rather than a single scalar weight; and 3) Block-Level Adaptation, which applies the LoRA module as a residual to an entire Transformer block, bypassing the internal Softmax conflicts. Empirically, mtLoRA is shown to provide significant gains over a strong baseline (HydraLoRA) on vision (DOTA, iNat2018) and language (Dolly-15k) benchmarks .

### Strengths
1. The primary strength of the paper is its clear and empirically-supported diagnosis of why multi-task LoRA fails. The analyses in Table 1, showing the failure of uniform regularization (1A) , the spectral heterogeneity of LoRA (1B) , and the weakness of component-level attachment (1C), are persuasive and form a strong foundation for the method.

2. The three proposed components of mtLoRA map directly and logically to the identified problems. Spectral-aware regularization is a clever fix for the issue identified in Table 1B. Block-level adaptation is an intuitive solution to the gradient amplification problem.

3. The paper's component-level analysis is strong. Tables 5, 6, 7, and 8 clearly demonstrate the additive benefits of the components and their individual contributions (e.g., fine-grained routing on NLP and block-level adaptation on vision ).

4. The method is presented as a "plug-and-play" add-on for existing methods like HydraLoRA, which increases its practical relevance. The evaluation across both vision and language domains demonstrates its general applicability.

5. The finding that fine-grained routing is critical for NLP but harms performance in vision (Tables 7 & 8) is a valuable and interesting insight into the differing nature of representations in these domains.

### Weaknesses
# Major Concerns

1. **Ambiguous Baseline Comparison:** The main results in Table 2 compare mtLoRA to "HydraLoRA" and "Baseline". The check-marked rows indicate that mtLoRA is built on top of HydraLoRA. This table is effectively an ablation study, not a SOTA comparison. It is unclear how mtLoRA stacks up against other, different SOTA multi-task methods mentioned in the related work, such as MoLE , TIES-Merging , or DARE. The claim of "state-of-the-art results"  is not fully substantiated against the broader field.

2. **Confusing Implementation of Fine-Grained Routing:** The description of fine-grained routing is contradictory. Section 3.3 states weights are $\Pi_i \in \mathbb{R}^{d/g}$. Section 4.2 states "Larger g means finer grained routing". However, $g$ is defined as "group size". Logically, a larger group size $g$ would mean coarser routing (e.g., $g=d$ would be module-wise). This is directly contradicted by the text. Furthermore, Table 5 lists "Full $g=768$". This implies $g=d=768$, which should be module-wise (the coarsest setting), but it's labeled "Full" (the finest) and performs best. This ambiguity is a major flaw in the paper's presentation and reproducibility.

3. **Un-benchmarked Computational Overhead:** The spectral-aware regularization requires performing SVD on the $B$ matrices "every epoch". For a large number of tasks (e.g., $N=100$) and large models, this is a significant, non-trivial computational cost that is not benchmarked or discussed.

4. **Under-justified SVD Proxy:** The method applies SVD to the $B_i$ matrix as a proxy for the full $\Delta_i = B_i A_i$ update, justified only by cost. The spectral properties of $B_i$ and $B_i A_i$ are not the same, as the $A_i$ matrix performs a rank-bottleneck projection. This simplification is a potential threat to the method's soundness, and it is not theoretically or empirically justified why $B_i$'s singular vectors are a sufficient proxy.

## Minor Concerns

1. **Synthetic Task Creation:** The tasks for the NLP benchmark (Dolly-15k) are created by performing K-Means clustering on instruction embeddings. This is a synthetic setup. The method's performance may be sensitive to the number of clusters (N) or the quality of the clustering.

2. **Missing Definition of HydraLoRA:** The paper states it builds on HydraLoRA and only briefly defines it in related work as having an "asymmetric LoRA structure". It is not clarified in the method section if mtLoRA _requires_ this asymmetric structure, or if it can be applied to standard LoRA modules.

3. **Implementation of Spectral Loss:** Equation (3) defines the conceptual loss based on pairwise singular vector alignment. However, Section 4.2 describes a different implementation: computing weighted matrices $B_i'$ and then a Frobenius norm on their product. The paper should clarify if Eq. 3 is just conceptual and the (different) formulation in Sec 4.2 is the actual loss used.

### Questions
1. Please resolve the contradiction in Section 4.2. Does larger $g$ mean finer or coarser routing? Please provide pseudocode for how the router's $N \times (d/g)$ output is broadcast and multiplied with the LoRA update.

2. Can you provide any empirical or theoretical justification for why the SVD of $B_i$ (the $d \times r$ matrix) is a high-fidelity proxy for the spectral properties of the full $\Delta_i = B_i A_i$ (the $d \times d$ update)?

3. What is the wall-clock training time overhead of enabling spectral-aware regularization (with its per-epoch SVD) and fine-grained routing (with its larger router network ) compared to the HydraLoRA baseline?

4. Does mtLoRA's methodology require the asymmetric (shared $A$, task-specific $B$) structure of HydraLoRA? Or can the three components (spectral-reg, fine-routing, block-level) be applied to a standard multi-task setup with independent $B_i A_i$ modules?

5. How does the full mtLoRA method compare against other SOTA methods from the literature, such as MoLE or TIES-Merging, on the iNat2018 or Dolly-15k benchmarks?

6. In Table 1(C), does the "Block" level LoRA have the same number of trainable parameters as the "Component" level? Please clarify the parameter counts for these different attachment strategies.

### Soundness
2

### Presentation
2

### Contribution
3
