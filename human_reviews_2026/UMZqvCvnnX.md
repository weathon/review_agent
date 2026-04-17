# Exploring System-1 and System-2 Communication for Latent Reasoning in LLMs

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Should LLM reasoning live in a separate coprocessor, or within a single model that uses the same forward pass and representational space? We study dual-architecture latent reasoning, where a fluent Base exchanges latent messages with a Coprocessor, and test two hypotheses aimed at improving latent communication over Liu et al. (2024b): (H1) increase channel capacity; (H2) learn communication via joint finetuning. Under matched latent-token budgets on GPT-2 and Qwen-3, H2 is consistently strongest while H1 yields modest gains. A unified soft-embedding baseline—a single model with the same forward pass and shared representations, using the same latent-token budget—nearly matches H2 and surpasses H1, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning. Across GSM8K, ProsQA, and a Countdown stress test with increasing branching factor, scaling the latent-token budget beyond small values fails to improve robustness. Latent analyses show overlapping subspaces with limited specialization, consistent with weak reasoning gains. We conclude dual-model latent reasoning remains promising in principle, but likely requires objectives or training schedules that explicitly shape latent spaces for algorithmic planning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper revisits dual-architecture latent reasoning for large language models (LLMs), where a “Base” model exchanges latent messages with a “Coprocessor.” It proposes two hypotheses to improve cross-module communication: (H1) augmenting KV-cache communication in a frozen Base and (H2) co-finetuning both models jointly.

### Strengths
* **Clear motivation and structure**: The paper clearly frames the problem as evaluating and improving communication between “System-1” (Base) and “System-2” (Coprocessor) reasoning modules.

* **Rigorous experimental baselines and fair token budgeting**: The authors control for latent-token and data budgets and include both continued pretraining and soft-embedding baselines, which exposes when dual architectures add redundant capacity.

* **Comprehensive interpretability analysis**: The cross-capture and silhouette diagnostics offer rare quantitative insight into latent redundancy.

### Weaknesses
* **Limited empirical scope and scale**: Experiments are restricted to GPT-2 (124 M) and Qwen-3 (0.6 B) models with modest token budgets. The findings might not generalize to larger LLMs (≥ 7 B parameters) where representational capacity and KV-cache dynamics differ significantly.

* **Unclear evidence for emergent reasoning**: The paper’s conclusions rest on negative results (flat scaling on GSM8K/Countdown), but it lacks task diversity to confirm that the observed redundancy is a general property rather than a dataset artifact.

* **Theoretical contribution remains descriptive**: While the dual-system framing is conceptually appealing, the paper contributes little formal theory or training objectives to explain why latent specialization fails to emerge.

* **Potential training mismatch**: The authors note that pretraining and curriculum finetuning conflict, but do not analyze how or why this mismatch arises, leaving uncertainty about whether the issue is architectural or procedural.

* **Limited methodological advancement**: This paper primarily builds upon an existing dual-module paradigm and explores a few minor variants of it, using empirical experiments to compare their relative strengths and distinctive properties. While such exploration can offer useful insights for future research, the methodological innovation presented here remains limited.

### Questions
* **Alternative latent objectives**: Have you tried explicit decorrelation or orthogonality regularizers to encourage latent specialization, or contrastive losses to diversify Coprocessor representations?

* **Ablation of communication depth**: For Hypothesis 1, have you examined partial-layer cache concatenation (e.g., top-k layers only) to see if selective coupling can balance compute and communication efficiency?

* **Cross-task generalization**: Could you include additional reasoning benchmarks (e.g., MATH, SVAMP) to test whether the “redundant latents” phenomenon persists across reasoning domains?

* Can you further elaborate on how the findings presented in this work concretely support or inspire the two promising directions mentioned at the end of the paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tests dual-model latent reasoning where a Base LLM exchanges “thought” tokens with a Coprocessor. Of two upgrades over Liu et al. (2024), H2 (joint finetuning) beats H1 (KV-cache concat with frozen Base)—but a single-model soft-embedding baseline with the same latent budget nearly matches H2 and surpasses H1, suggesting the dual setup mostly adds compute, not better reasoning.

### Strengths
1. The paper poses clear, falsifiable hypotheses about cross-module communication in dual-model latent reasoning.

2. The evaluation in the paper matches latent budgets across methods and adds a strong single-model soft-embedding baseline for fair comparison.

3. The proposed three-pass training protocol reduces shortcutting and clarifies where gains originate.

### Weaknesses
1. The paper’s motivation feels underdeveloped. The work is framed as improving the KV-Coprocessor within its own paradigm, but the generality and adoption of KV-Coprocessor remain unsettled, making this study read more like a targeted extension of that framework than a standalone contribution.

2. The dual-model setups introduce additional trainable components relative to the soft-embedding baseline, leaving open whether observed gains stem from architectural choices or simply extra capacity/compute.

3. The ablation of the each component of the proposed structure is missing in the current work.

4. The KV-concat pathway (H1) may be sensitive to sequence length or attention scaling, yet length-generalization tests are absent, leaving robustness claims underexplored.

### Questions
See the weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies dual-architecture latent reasoning, which contains a base model and a coprocessor. The authors tested two hypotheses aimed at improving latent communication compared to a previous work [1]. The evaluation of the two hypotheses suggests that the current dual designs mostly add computation rather than improving reasoning, which is supported by the marginal gain on GSM8k, ProsQA, and the countdown stress test dataset by scaling the latent-token budget beyond small values, and overlapping spaces in latent analysis. 

**References:**

[1] Liu, Luyang, Jonas Pfeiffer, Jiaxing Wu, Jun Xie, and Arthur Szlam. "Deliberation in latent space via differentiable cache augmentation." arXiv preprint arXiv:2412.17747 (2024).

### Strengths
1. The paper studies an important and interesting question: whether the dual model really benefits reasoning by effective communication, or whether it merely adds computation (even in an inefficient way)?

2. The paper conducts a more careful analysis of a previous work, which uncovers interesting results, such as using a single model can achieve similar performance gain as the dual system.

### Weaknesses
1. The major concern is that the scope of this paper might be a bit narrow. The conclusion that the current dual system for latent reasoning is not efficient is based on the analysis of one previous work, which makes the conclusion not fully convincing or solid. The contribution is also limited, since the paper did not propose effective methods to mitigate it.

2. The definition of the whole process can be more precise. Many refer to Liu et al. (2024b), but it would be more helpful for readers unfamiliar with the work to provide the mathematical formulations in greater detail.

3. The flow of the paper could be improved for reading. Some sentences are not very coherent within the context, and some seem broken.

4. “A single forward pass could allow the Base model to shortcut … defeating the purpose of the technique.” Although this might make sense, I think there should be some evidence, even if it is indirect.

### Questions
1. What does the ahead token N_A mean? Is it defined anywhere in the paper? For the latent augmentation, what does M refer to (i.e., what is the difference between multiple augmentations for the same sequence)

2. On page 6, the authors mentioned, “A single LLM with the same aggregate parameter count would almost certainly do better.” But the result for Qwen3 benchmarks shows “Dual models fare better: Hyp. 1 +6.5 pp (47.2 vs. 40.7%); Hyp. 2 +8.7 pp (49.4%).”. Am I missing anything?

3. In diagnostic 1 (line 397), what does $n\_i$ mean?

4. For diagnostic 1, I wonder what the typical dimension of $P\_i$ is. If $P\_i$ is close to the identity matrix, then $H\_{ij}$ should always be very high.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper examines dual-architecture latent reasoning where a Base model exchanges latent messages with a Coprocessor to improve reasoning capabilities. The authors test two hypotheses to strengthen communication over existing work - increasing channel capacity via cache augmentation and learning communication through joint finetuning - but find that a simpler soft-embedding baseline nearly matches the best dual-model variant, suggesting the approach mostly adds compute rather than enabling qualitatively better reasoning.

### Strengths
The paper provides a thorough empirical evaluation across multiple model sizes (GPT-2, Qwen-3) and tasks with proper baselines, which is refreshing. The interpretability analysis using cross-capture heatmaps and silhouette scores to examine latent specialization is insightful and reveals that latents tend to occupy overlapping subspaces rather than specializing. The three-pass training implementation cleanly separates latent computation from next-token prediction, avoiding potential shortcuts.

### Weaknesses
The wrting is bad. 
The negative results, while honestly reported, limit the impact .. the dual-model architecture doesn't deliver on its promise of System-2 reasoning. The soft-embedding baseline performs nearly as well with half the parameters, which undermines the main architectural contribution. The experiments are limited to smaller models due to compute constraints, and it's unclear if findings generalize to larger scales where the original work showed stronger results.

### Questions
Have you tried other communication mechanisms beyond cache augmentation, like attention-based message passing between modules?

### Soundness
2

### Presentation
1

### Contribution
2
