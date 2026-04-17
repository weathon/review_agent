# SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety

- Decision: Accept (Oral)
- Scores: 4, 8, 6, 8

## Abstract
As Large Language Models (LLMs) are increasingly deployed in real-world applications, balancing helpfulness and safety has become a central challenge. A natural approach is to incorporate safety constraints into Reinforcement Learning from Human Feedback (RLHF), where recent studies have shown promising progress. However, these methods often rely on auxiliary networks or multi-stage pipelines, thereby increasing complexity. In this work, we revisit the original safety alignment objective and show that, under mild assumptions, it admits a closed-form optimal policy. We further derive a provably equivalent and tractable objective, enabling direct optimization. Building on this insight, we propose SafeDPO, a lightweight method that preserves the optimal solution of the underlying safety-constrained objective while requiring only one additional hyperparameter and minimal modifications to existing preference-based training methods. SafeDPO eliminates the need for reward models, cost models, and online sampling, relying only on preference data and safety indicators. Despite its simplicity, SafeDPO achieves competitive safety–helpfulness trade-offs compared to existing safety alignment methods. Experiments on the PKU-SafeRLHF-30K benchmark demonstrate that SafeDPO substantially improves safety while maintaining competitive helpfulness. Ablation studies further show that the additional hyperparameter provides a flexible mechanism to enhance safety while preserving the theoretical optimum, and confirm that SafeDPO scales reliably to LLMs with up to 13B parameters. Overall, our results highlight that a simple, theory-driven objective can provide a lightweight yet effective solution for safety alignment in practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SafeDPO, a DPO-based algorithm designed to align large language models (LLMs) to be both helpful and safe. To achieve this, the authors show that the original constrained optimization problem can be reformulated in closed form and solved via a direct optimization objective. Specifically, they derive a modified DPO-style loss function that incorporates safety indicators and prove that optimizing this "SafeDPO" loss is theoretically equivalent to solving the constrained problem under mild assumptions. Empirical results demonstrate that SafeDPO performs competitively with state-of-the-art safety alignment approaches.

### Strengths
1. Similar to other DPO-based methods, the algorithm is indeed simple to implement and does not require auxiliary reward or cost models. The entire training process reduces to single-stage fine-tuning of the policy using a modified loss function.
2. The paper presents experiments leveraging state-of-the-art baselines and prior experimental setups, complemented by theoretical results establishing the equivalence between the SafeDPO objective and the original constrained optimization problem.

### Weaknesses
1. While the original DPO requires only preference data, SafeDPO additionally depends on binary safety labels indicating whether each response is safe or unsafe. These labels typically require specialized datasets such as SafeRLHF, and in practice, collecting such annotations may be more costly than training a separate cost model. Moreover, due to the data processing rules in SafeDPO, any preference pair where both responses are unsafe is discarded, leading to additional data inefficiency.
2. The results rely heavily on the theoretical framework of the original DPO algorithm. Moreover, SafeDPO resembles more of a dataset preprocessing strategy than a fundamental improvement to the underlying optimization algorithm.
3. Another concern not addressed in the paper is the risk of over-refusal, where models excessively prioritize safety by refusing to answer even harmless prompts. As highlighted in the analysis of the IPO algorithm, a key issue with DPO stemming from its Bradley–Terry model formulation is overfitting. In a similar vein, concerns about over-refusal arise when the optimization overly penalizes unsafe responses, potentially leading the model to adopt a conservative strategy of frequent refusals, regardless of the actual prompt content.
4. According to the reason in point 1, the empirical evaluation is limited to a single dataset (PKU-SafeRLHF-30K), which may restrict the generalizability of the conclusions to other domains or safety-critical applications.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes SafeDPO, a lightweight safety-alignment method that embeds binary safety indicators directly into preference optimization. Starting from the safety-constrained objective, the authors show that it admits a closed-form, equivalent reformulation that enables direct, single-stage training, removing the need for reward models, cost models, or online RL sampling. Empirically, on PKU-SafeRLHF-30K, SafeDPO improves safety with minimal loss in helpfulness, and scales to models up to 13B parameters; ablations illustrate the role of the extra hyperparameter. Overall, the work argues for a simple, theory-driven alternative to multi-stage Safe-RLHF pipelines.

### Strengths
This work shows that, the LLM safety alignment can be achieved with a single optimization objective. The rigorous, well-structured proof supports this claim. Compared with prior approaches that require iterative training, this method needs only one training stage, which is more efficient and stable.

### Weaknesses
1. $\textbf{Hyperparameter guidance.}$ The new safety margin parameter is appealing. However, a brief tuning guide (ranges, sensitivity across model sizes) would help practitioners reproduce the safety/helpfulness trade-off.
2. $\textbf{Qualitative analysis.}$ A few case studies would make the improvements more interpretable beyond aggregate metrics.

### Questions
I overall lean to accept the paper. The theoretical simplification and the training path are practical and elegant, and the empirical results seem promising.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SafeDPO, a variant of DPO tailored for safety alignment. This method reformulates the safety alignment problem—which typically involves multiple safety constraints—into an optimization objective with a single constraint, solvable via DPO. This is achieved by constructing a preference hierarchy where: (safe and helpful response) > (safe but unhelpful response) ≫ (unsafe response). Compared to existing approaches such as Safe RLHF, SafeDPO demonstrates greater efficiency in both computational time and memory usage during training.

### Strengths
+ Compared to SafeRLHF, a widely used baseline in safety alignment, SafeDPO offers greater efficiency. It significantly reduces the annotation cost for preference signals and the computational burden during training, while maintaining competitive alignment performance.
+ A fundamental distinction between SafeDPO and SafeRLHF lies in their optimization objectives: SafeDPO directly optimizes the exact objective in Eq. (6), whereas SafeRLHF relies on an approximation in Eq. (7). This direct optimization allows SafeDPO to theoretically provide a stronger safety guarantee.
+ This paper establishes theoretical guarantees for SafeDPO and provides comprehensive experimental evidence demonstrating its efficacy and advantages over existing methods.
+ Owing to its core operation of preference relabeling, SafeDPO exhibits strong generalization capability. This makes it readily adaptable to a broader range of preference modeling frameworks, not just the Bradley-Terry model.

### Weaknesses
+ **​Sample Efficiency and Annotation Cost:​**​ The paper's approach faces challenges in sample efficiency and data requirements. As indicated in Eq. (13), unsafe-unsafe pairs are discarded, which reduces the overall utilization of the available data. More critically, obtaining high-quality, informative safe-unsafe pairs for effective contrastive learning likely incurs significant additional annotation costs. For instance, it may require manually crafting safe responses to unsafe prompts. 
+ **Extension Concern:​**​ While the predefined preference hierarchy—(safe and helpful) > (safe but unhelpful) ≫ (unsafe)—is well-justified for safety alignment, it highlights a potential limitation in the method's generalizability. SafeDPO's core mechanism relies on a strict, linear prioritization of objectives. This approach may not extend well to multi-objective alignment tasks where preferences are not easily rank-ordered. For instance, in a summarization task (e.g., TLDR), qualities such as informativeness, conciseness, and coherence are often competing and lack a clear hierarchy; optimizing for one can compromise another. The inability to handle such non-hierarchical trade-offs may limit SafeDPO's applicability beyond the safety domain.
+ ​**​Potential Over-Refusal:​**​ While the experimental results demonstrate that SafeDPO achieves higher safety, the paper lacks a specific analysis of over-refusal. There is a concern that the method's strong safety performance might come at the cost of increased excessive caution, leading the model to reject reasonable prompts. An evaluation on a dataset containing benign or edge-case queries is needed to rule out this potential drawback.
+ **Clarity of Results Presentation:​**​ The presentation of experimental results could be improved in two aspects: (a) Figure 2 would benefit from the inclusion of specific numerical values to complement the visual depiction of trends. (b) In Appendix D.1, the 1v1 comparison setup using the GPT evaluation protocol from [1, 2] would be more informative if it directly compared SafeDPO (rather than using the SFT model) against the other baseline methods. Based on my practice experiences, the performance gap between SFT models and aligned models can be significant, and a direct comparison among the main aligned models (including SafeDPO) is crucial for a clear and fair assessment of their relative performance.

[1] Dai, Josef, et al. "Safe rlhf: Safe reinforcement learning from human feedback." arXiv preprint arXiv:2310.12773 (2023).

[2] Huang, Xinmeng, et al. "One-shot safety alignment for large language models via optimal dualization." Advances in Neural Information Processing Systems 37 (2024): 84350-84383.

### Questions
Solving the following primary concerns regarding potential weaknesses during the review period might help me make a fairer evaluation of this work:

1. To address concerns about potential over-refusal, I suggest evaluating SafeDPO on the XSTest benchmark [3] to quantify its behavior on prompts where refusal may be unnecessary.

2. I recommend that the 1v1 comparison results be presented to highlight the direct performance difference between SafeDPO and other alignment baselines, as this would more clearly showcase its outstanding performance.

Furthermore, I wish to offer the following suggestions, which fall slightly outside the primary evaluation criteria but may be of interest for discussion or future work：

3. It would be valuable to discuss the limitations on sample efficiency and generalization. Exploring ways to leverage all available data (e.g., reannotating unsafe-unsafe pairs with a reject response like "I can't answer this question.") and extending the SafeDPO principle to multi-objective scenarios present promising directions for future work.

4. To improve the clarity of the results, I suggest presenting the numerical values corresponding to Figure 2 in a table (e.g., in the appendix). This would allow readers to precisely assess the quantitative differences.

[3] Röttger, Paul, et al. "Xstest: A test suite for identifying exaggerated safety behaviours in large language models." arXiv preprint arXiv:2308.01263 (2023).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces SafeDPO, a theoretically grounded yet lightweight method for safety alignment in large language models (LLMs). The authors revisit the constrained safety alignment problem and show that it admits a closed-form reformulation that eliminates the need for auxiliary reward or cost models. Based on this, SafeDPO integrates binary safety indicators directly into preference optimization, requiring only a single additional hyperparameter (the safety margin ∆). The method maintains equivalence with the safety-constrained objective while remaining compatible with existing Direct Preference Optimization (DPO) frameworks. Extensive experiments demonstrate that SafeDPO substantially improves harmlessness with minimal loss in helpfulness, outperforming other baselines.

### Strengths
1. The derivation of a closed-form, constraint-equivalent objective (Eq. 9–12) is elegant and rigorously justified through formal propositions. The proofs (Appendix A) convincingly establish theoretical soundness, equivalence, and unbiasedness.

2. This work only requires a single-stage training compared to previous methods that might need training during muliple stages such as training reward and cost model or iteratively optimizing the objective.

3. The experiment results are comprehensive and show the proposed method's effectiveness.

### Weaknesses
1. Could the authors also show some failure cases of SafeDPO to better understand or conduct failure analysis on those unsafe responses?

2. The figures shown in the paper are relatively hard to read. The authors should consider using larger and thicker dots and lines in the figure.

### Questions
1. Although the experiment is comprehensive, but is only conducted on a single dataset. Could the author extend to some other datasets?

2. Please refer to the weakness section for more questions.

### Soundness
4

### Presentation
3

### Contribution
3
