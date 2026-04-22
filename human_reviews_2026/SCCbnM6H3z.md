# Bridging the Gap Between Preference Alignment and Machine Unlearning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Despite advances in Preference Alignment (PA) for Large Language Models (LLMs), mainstream methods like Reinforcement Learning with Human Feedback (RLHF) face notable challenges. These approaches require high-quality datasets of positive preference examples, which are costly to obtain and computationally intensive due to training instability, limiting their use in low-resource scenarios. The LLM unlearning technique presents a promising alternative by directly removing the influence of negative examples. However, current research has primarily focused on empirical validation, lacking systematic quantitative analysis. To bridge this gap, we propose a framework to explore the relationship between PA and LLM unlearning. Specifically, we introduce a bi-level optimization-based method to quantify the impact of unlearning specific negative examples on PA performance. Our analysis reveals that not all negative examples contribute equally to alignment improvement when unlearned, and the effect varies significantly across examples. Building on this insight, we pose a crucial question: how can we optimally select and weight negative examples for unlearning to maximize PA performance? To answer this, we propose a framework called Unlearning to Align (U2A), which leverages bi-level optimization to efficiently select and unlearn examples for optimal PA performance. We validate the proposed method through extensive experiments, with results confirming its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a principled connection between Preference Alignment (PA) and Machine Unlearning (MU) for LLMs. Using a bi-level formulation, the authors quantify how unlearning a specific negative example affects PA and then build U2A (Unlearning-to-Align), which selects and reweights forget samples to maximize PA under a sparsity constraint.

### Strengths
A principled bridge between MU and PA with a clean sensitivity view (negative vs positive cosine) and an actionable marginal-gain selector; mirror-descent on simplex gives closed-form weight updates and encourages sparse, high-impact forgetting. 
Clear theoretical intuition for when/why unlearning helps alignment; the cosine-of-gradients view is compelling and actionable.
Bi-level design is reasonable and supported by convergence & complexity analysis.
Good ablations.

### Weaknesses
The theory uses Hessian PSD & diagonal approximations; stress-testing against curvature mis-specification would be valuable.

Results are on three popular PA datasets; more diversity (such as safety red-teaming, multilingual, domain tasks) would help generality.

While U2A is competitive in some settings, it trails DPO/PPO in others (e.g., with weaker base models / noisier negatives). A controlled compute-matched study would clarify trade-offs.

### Questions
see Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the limitations of reinforcement learning from human feedback (RLHF) for preference alignment (PA) in low-resource environments. Standard RLHF requires many high-quality positive preference samples and can be unstable and computationally costly. The authors propose reframing PA as a machine unlearning (MU) problem: instead of learning from positives, the model can align with human preferences by selectively unlearning harmful negative samples.

They introduce U2A, a bi-level optimization framework that estimates the contribution of each negative sample to PA performance, then applies weighted unlearning to maximize alignment efficiency. Their theoretical analysis reveals two key insights. First, the effect of unlearning depends on whether the gradient direction of a sample conflicts with PA objectives. Negative samples with consistently low-reward behavior typically yield positive alignment gains when unlearned. Second, impact magnitude varies by sample and is a function of both inherent gradient norms and the chosen unlearning weight.

Experiments show that U2A significantly improves alignment under constrained resources while maintaining efficiency comparable to standard fine-tuning methods. The work offers a new perspective that connects PA and MU, opening new directions for optimizing alignment algorithms without heavy reliance on costly reward data.

### Strengths
The paper presents a clear line of reasoning from motivation to methodology and results. The insight that negative samples contribute unequally to preference alignment is both novel and practically valuable, especially in low-resource settings. The proposed U2A framework is grounded in bi-level optimization theory, enabling principled analysis of how individual negative samples should be weighted during unlearning. Experimental results demonstrate that selectively unlearning harmful samples leads to meaningful improvements in alignment efficiency and effectiveness, reinforcing the core contribution.

### Weaknesses
Some important elements of the method, including the primary algorithm and several key conceptual details, are placed in the appendix rather than the main paper. This makes the flow feel crowded in some areas while leaving gaps in others, and requires readers to frequently jump to supplementary materials to fully understand the approach. A more balanced layout would improve clarity and accessibility.

### Questions
How well does the proposed unlearning-based alignment hold up on larger, higher-quality models and more diverse preference datasets? A few comparison with the preference alignment using both positive and negative samples would be beneficial?

### Soundness
3

### Presentation
2

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
The paper investigates how machine unlearning can be leveraged to improve preference alignment (PA) in large language models and proposes U2A (Unlearning to Align). It formulates unlearning-for-alignment as a bi-level problem: the inner loop performs unlearning on selected training samples with a chosen weight, while the outer loop measures the resulting change in PA quality. Using implicit differentiation, the paper shows that the PA change from unlearning a sample depends on gradient alignment between the PA objective and the unlearning loss; unlearning can help or harm PA depending on this alignment and the chosen weight, motivating selective, weighted unlearning rather than blanket removal.

U2A then picks a small set of impactful samples and assigns weights, using a greedy selector plus simple weight updates; it exploits efficient Hessian-vector products to keep the cost manageable. On SafeRLHF, UltraFeedback, and HaluEval with LLaMA-2/3, U2A improves preference alignment and unlearning metrics over prior heuristics (GA, GradDiff, NPO) and is competitive with PPO/DPO baselines.

### Strengths
Originality: The paper reframes machine unlearning as a means to improve preference alignment rather than only a privacy safeguard. Casting “unlearning to align” as a bi-level problem with an explicit criterion for whether removing a sample helps or hurts alignment is a fresh, principled alternative to heuristic data filtering.

Quality: The technical pathway from the bi-level formulation to a practical algorithm is coherent. The use of a first-order influence-style approximation to estimate alignment impact, plus a selective procedure (greedy/matching-pursuit) with simple weight updates (mirror descent) and efficient Hessian–vector products, makes the method implementable at scale and links theory to practice.

Clarity: The paper communicates the core intuition clearly: unlearning helps when the unlearning and alignment gradients point in similar directions and can harm otherwise. The pipeline—outer evaluation of alignment, inner weighted unlearning, then selection updates is laid out in a way that readers can follow and reproduce.

Significance: Results across multiple datasets and backbones indicate consistent gains over common unlearning heuristics and competitive performance with strong alignment baselines. The approach offers a data-centric knob that can complement or reduce reliance on heavier RLHF-style training, suggesting practical value for teams maintaining alignment under evolving safety and content policies.

### Weaknesses
The paper’s structure is difficult to follow. Sections frequently interleave intuition, which obscures the main thread of the method. The evaluation is also hard to parse: the metrics used in tables and figures are not defined where they first appear, and captions do not restate what higher/lower means or why a difference is meaningful. Baselines are listed with names only, without a brief description or implementation details, so readers cannot judge comparability.

The reported win rates show little or no strengthening over baselines (especially Table 2). Given the small margins, the evidence is insufficient to claim a meaningful improvement. As LLM-as-judge metrics can be noisy and prompt-sensitive, these results should be complemented with stronger analysis.

### Questions
1. Assumption 3.1 (mildness and applicability). Could the authors precisely restate Assumption 3.1 and explain whether it is a mild assumption in practice? Please discuss concrete conditions under which modern LLMs (and the proposed training setup) satisfy it. Clarify how the algorithm remains stable when the assumption is violated.

2. Model family dependence (beyond Llama): All experiments use Llama backbones. Please either include an additional LLM (e.g., Mistral, Qwen, Gemma) or provide an analysis explaining why the method is expected to transfer. 

3. Computational overhead of selection. What is the additional cost introduced by the dynamic selection and weighting procedure? 
Please report GPU-hours, compared to baselines.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to connect machine unlearning (MU) with preference alignment (PA) for LLMs via a bi-level formulation that measures how unlearning specific negative samples changes a PA objective. From this analysis, the authors design Unlearning to Align (U2A), which selects and weights negative examples to unlearn, aiming to maximize PA. Experiments on SafeRLHF and UltraFeedback with Llama-2/3 variants show that U2A improves over full-set unlearning baselines and is competitive with DPO/PPO in some settings.

### Strengths
- A new viewpoint of PA into unlearning seems interesting
- Consistent improvements of GA/GradDiff/NPO with U2A plugged in supports the importance of sample-wise weighting of the negative samples.

### Weaknesses
- Proposition 3.1 involves $\nabla L_{\text{forget}}(x)$, but there are no concrete properties of the training sample $x$. Conclusion 1-2 become trivial restatements abour cosine similarity and gradient norm. The proposition should derive sample-dependent bounds under further properties of training sample $x$.
- Justification on "negative-only is cheaper": There are no evidence on actual costs, noise, or coverag for user reports/red-team data. As this work's motivation stems from this problem. It should be further justified.
- Although U2A consistently improves the unlearning baseine, it's efficacy hinders when compared with basic PA methods. This result does not seem to strongly support U2A's practicality.

### Questions
- Could you provide a intuitive elaboration as to why U2A helps unlearning methods beneficial to PA?

### Soundness
2

### Presentation
2

### Contribution
2
