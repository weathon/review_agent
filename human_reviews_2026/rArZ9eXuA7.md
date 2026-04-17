# SyMoFlow: Interaction-Aware Motion Synthesis from Text via Symmetric Flows

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Human-Human Interaction (HHI) generation aims to synthesize plausible and coordinated motion sequences for multiple agents in a shared environment. Existing approaches often struggle to capture reciprocal dependencies, maintain semantic alignment with textual descriptions, or balance realism and diversity. To address these challenges, we propose SyMoFlow, a text-driven motion synthesis framework that leverages an interaction-symmetric decomposition of the joint motion distribution. SyMoFlow generates sequential single-agent motions: it first produces an interaction-aware motion for one agent conditioned on text, then synthesizes the second agent’s motion conditioned on the first, capturing both prior action and reciprocal reaction. By explicitly modeling interdependent dynamics, our approach produces coordinated, causally consistent behaviors while allowing flexible flow-based sampling to enhance multimodality and diversity. Extensive experiments on the InterHuman and InterX benchmarks demonstrate that SyMoFlow achieves state-of-the-art realism and text alignment while significantly improving the diversity of plausible interactions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SyMoFlow, a text-driven motion synthesis framework that decomposes the joint motion distribution with interaction symmetry. SyMoFlow generates sequential, single-agent motions: it first produces an interaction-aware trajectory for one agent conditioned on text, then synthesizes the second agent’s motion conditioned on the first human motion, capturing both prior actions and reciprocal reactions. By explicitly modeling interdependent dynamics, SyMoFlow generates coordinated, causally consistent behaviors, while flexible flow-based sampling enhances multimodality and diversity. Extensive experiments on InterHuman and InterX demonstrate good result in human-human interaction motion generation.

### Strengths
1. They propose a paradigm for generate human-human interactive motion generation. 
2. They propose a flow-matching based framework to generate the second human motion based on the first human motion.
3. The experiments are comprehensive.

### Weaknesses
1. This paper doesn't compare its result with the state-of-the-art model--TIMotion mentioned in the quantitative result table.

### Questions
1. Why do you use a VQ-VAE encoder and build a discrete flow-matching pipeline instead of a continuous formulation? 

2. Would reducing the VQ-VAE codebook size affect the results? If so, in what ways (e.g., fidelity, diversity, stability)?

3. How do you handle relative position and relative orientation in interactive human motion?

4. You did not compare against the best results reported in TIMotion. Why not? TIMotion’s FID is substantially better than yours—please explain.

5. Could it generalize to generate 3 or more human motion sequence?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
**SyMoFlow** proposes two-person motion generation from text by (i) discretizing motions with a VQ-VAE, then (ii) using discrete flow matching (DFM) to generate one agent’s motion and condition the other’s “reaction” on the first—calling this an “interaction-symmetric decomposition”. On InterHuman and InterX, the paper reports competitive R-Precision/MM-Dist and diversity/multimodality, with qualitative figures that look coordinated and prompt-aligned.

### Strengths
* Clean, modular pipeline (VQ-VAE → DFM + transformer with intra/inter-person attention). 
* Targeting interaction diversity is valuable; multimodality scores are strong in some tables. 
* Ablations try to isolate “symmetric path” and a motion-consistency constraint.

### Weaknesses
1. The paper provides no video results, making it difficult to judge temporal coherence, contact fidelity, and overall plausibility. In Figure 2, the upper example appears to show the blue actor’s global position shifting without corresponding foot motion which is commonly observed as drifting problem. In the lower example, the intended pattern of “moving farther away” is visually unclear from the frames shown. More broadly, the qualitative samples feel odd/semantically ambiguous - e.g., “one person walks forward and throws an object to the right; the other turns and waves their right hand” - and the paper includes almost no close-interaction or near-contact scenarios, which are precisely the challenging cases for two-person motion. Without videos and stronger examples, it’s hard to credit the method’s claims about interaction quality and controllability.

2. The authors define an “interaction-symmetric property” as ($p_\phi(x_B\mid x_A)=p_\phi(x_A\mid x_B)$) (Eq. 2) and then argue it “allows either agent to serve as initiator.” But the model trains and infers sequentially (A$\rightarrow$B), with weight sharing and conditioning; that does not imply the conditional distributions are equal in law. Equality of these two conditionals almost never holds. 

3. In section 3.2.2 they set the **source** for B as (p(z_B)=q(z_A)) (“use the first agent’s *data* distribution as the prior for the second”). That is unusual and under-motivated: the DFM source is typically a *fixed prior* independent of the particular sample; here it depends on A’s *empirical* target distribution, which changes semantics and may violate independence assumptions used by the path construction.


4. **Training/inference mismatch & exposure bias risk.**
   Training mixes real partner codes and random noise as ($z_{\text{cond}}$); inference sets ($z_0=z_A$), ($z_{\text{cond}}=z_A$). That is a much sharper conditioning regime than half-noisy training and may cause drift unless scheduled conditioning or teacher-forcing is carefully handled (not shown). 


5. In the ablation study, “Joint distribution” vs “naive decomposition” vs “symmetric flow path” are named but not concretely specified.

### Questions
1. How is Eq. (2) enforced or tested?
2. Can the authors justify this substitution yields a valid DFM path or unbiased sampling.
3. Is the method capable of extending to multi-agent ($\ge 3$) scenario? If so, how would the DFM blocks, intra-person, and inter-person attention change?

### Soundness
1

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
3

### Summary
This paper presents SyMoFlow, a text-driven framework for human-human interaction motion synthesis that models reciprocal dynamics between two agents through an interaction-symmetric decomposition of their joint motion. It sequentially generates motions, one agent acting, the other reacting, capturing causal and conditional dependencies for coordinated, semantically aligned behaviors. Using flow-based modeling, SyMoFlow achieves state-of-the-art realism, text alignment, and interaction diversity on InterHuman and InterX benchmarks

### Strengths
1. The writing is good.
2. The idea of interaction symmetry is intuitive, and the method designed based on this concept is simple yet effective. The proposed approach also demonstrates certain advantages over existing methods in the experimental results.

### Weaknesses
1. The performance improvement is not significant, showing only marginal gains compared with other methods.
2. Figure 1 lacks clarity. The upper and lower parts of the figure are not clearly labeled, making it difficult to interpret. The lower part appears to depict a Transformer, but this is not explicitly indicated.

### Questions
1. Regarding Table 3:
    1. The table lacks results for other ablation combinations (e.g., using only MCC). What are the performance outcomes for these additional configurations?
    2. What is the result of authors’ baseline model that uses neither the symmetric flow path nor MCC?
2. Computational cost:
    - How does the proposed method compare to other approaches in terms of computational cost, such as training time, GPU memory usage, inference speed, and FLOPs?

### Soundness
3

### Presentation
4

### Contribution
2
