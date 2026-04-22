# Knowledge-Level Consistency Reinforcement Learning: Dual-Fact Alignment for Long-Form Factuality

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Hallucination and factuality deficits remain key obstacles to the reliability of large language models (LLMs) in long-form generation. Existing reinforcement learning from human feedback (RLHF) frameworks primarily rely on preference rewards, yet they often overlook the model’s internal knowledge boundaries, exacerbating the so-called ``hallucination tax''. To address this challenge, we propose $\textbf{K}$nowledge-$\textbf{L}$evel $\textbf{C}$onsistency Reinforcement Learning $\textbf{F}$ramework ($\textbf{KLCF}$), a novel framework that focuses on the knowledge consistency between the policy model's expressed knowledge and the base model's parametric knowledge, and introduces a Dual-Fact Alignment mechanism to jointly optimize factual recall and precision. Specifically, KLCF leverages pretrained knowledge boundaries to construct fact checklist, guiding online reinforcement learning to improve factual coverage and recall; simultaneously, it trains a self-assessment module based on the base model's internal knowledge to enhance factual precision during generation. Unlike prior methods that rely on external retrieval or heavy verification, our reward design is fully online external-knowledge-free and lightweight, making KLCF efficient and easily scalable to large-scale training. Experimental results demonstrate that KLCF substantially improves factuality metrics across multiple long-form benchmarks and effectively alleviates model hallucinations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes KLCF to address hallucinations in large language models' long-form generation. KLCF aligns the model's expressed knowledge with its internal parametric knowledge using a Dual-Fact Alignment mechanism that jointly optimizes factual recall and precision. KLCF uses lightweight, external-knowledge-free rewards, enabling efficient online training. Experiments across multiple benchmarks and model scales show factuality improvements while maintaining scalability.

### Strengths
1. This paper implements an online RL training algorithm for hallucination mitigation that achieves alignment between the model's internal knowledge and its knowledge utilization strategy.
2. The reward function design effectively balances factual precision and recall, addressing a critical trade-off in hallucination mitigation. As demonstrated in Table 2, different objective emphases lead to opposing model tendencies, revealing that precision and recall are inherently conflicting optimization targets.
3. The method is validated across different model sizes and both thinking/non-thinking architectures, with comprehensive evaluation metrics.

### Weaknesses
1. The claim of being "external-knowledge-free" is inappropriate since both the truthfulness reward and checklist rely on offline Wikipedia verification. The offline and online phases should not be treated separately when making this claim.

2. While claiming to be "lightweight," the offline data preparation and reward model training costs should not be ignored. Moreover, the time ratio between response rollout and reward computation during GRPO training should be provided to assess the significance of the efficiency gains from reward computation improvements.

3. The reward system appears overly complex, with Truthfulness Reward and fact precision reward seemingly optimizing similar objectives. I have concerns about the necessity of Checklist Reward. A simpler approach using shaped rewards combining FActScore (precision) + number of claims (recall) could achieve similar alignment between parametric and expressed knowledge (e.g., https://arxiv.org/abs/2406.12221). The authors should better explain the rationale for eliciting knowledge from pretraining.

4. The paper should include a Related Work section discussing the relationship and distinctions with other hallucination mitigation approaches.

### Questions
1. Training Dynamics and Reward Scaling: Section C.2 shows CoT becomes shorter while response length increases during training. This seems more attributable to General Reward (which prefers longer responses) rather than recall reward. Although the coefficient is 0.1 in Formula 10, could the General Reward have a much larger numerical scale? Could the authors provide the actual value ranges of different reward components to clarify their relative impacts?

2. How are rewards computed in practice? Is it using separately deployed GPUs as in Table 12? If all 16 training GPUs are used for reward scoring instead, would the efficiency gains become much less significant?

### Soundness
3

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
4

### Summary
This paper is dedicated to addressing the hallucination tax in RLHF and proposes the Knowledge-Level Consistency Reinforcement Learning Framework (KLCF), which simultaneously considers the consistency between the output knowledge and the model’s internal knowledge, as well as the truthfulness of the output knowledge. Experimental results demonstrate that KLCF substantially improves factuality metrics across multiple long-form benchmarks and effectively mitigates model hallucinations.

### Strengths
1. This paper focuses on a key issue: how to address the hallucination tax that may arise in reinforcement learning (RL).
2. The paper proposes a new algorithm, KLCF, and demonstrates its effectiveness through extensive experiments. Moreover, the RL process does not rely on any external documents, resulting in higher training efficiency and lower computational overhead.
3. The paper observes that current RL methods primarily reward the precision of the model’s output, which is not conducive to exploring the boundaries of the model’s knowledge. The authors instead incorporate the activation of the model’s internal knowledge as part of the reward signal — a crucial and insightful contribution.

### Weaknesses
1. The novelty of this paper is questionable. The paper proposes two main aspects:
- a) Using a checklist to consider the precision and recall of the model’s internal knowledge. I can understand the goal of encouraging the model to output as much correct knowledge as it truly possesses (recall), but I am not convinced that precision is necessary. b) My understanding is that the authors might worry about the model being rewarded for knowledge it does not actually know, thereby encouraging it to guess. However, in the scenario the paper focuses on, the data used for reward is also generated by the model itself — so should we still avoid giving such rewards?
- Training a reward model to score truthfulness — this is an intuitive and straightforward approach.

2. The writing quality is poor and hard to follow. For example, Section 2.2 starts by describing how to construct training data before clearly explaining the overall objective of the work.

3. In line 171, it seems that only one answer is sampled for each query. Why does the paper sample only one response per query? This way, only part of the model’s internal knowledge can be captured.

4. The description of metrics is important and should be placed in the main text. Since the paper repeatedly discusses the precision and recall of the model’s internal knowledge, having precision and recall metrics here could be confusing. The authors should also include a concrete case to explain why they use the @k metrics.

5. I can understand why, when computing recall, the authors do not use the ground-truth claims annotated in the dataset — because the model may not actually know those claims. Training the model to learn such claims would encourage guessing and increase hallucinations. However, I have a question: under an on-policy setting, wouldn’t we naturally avoid rewarding the model for knowledge it doesn’t have, since all responses are sampled from the model itself? If a correct answer is sampled, that implies the model already knows it.

6. Can the truthfulness reward replace the precision reward for internal knowledge? The checklist effectively ensures that the model’s internal “correct knowledge” can be extracted (recall), but precision might not need to depend on whether the knowledge comes from within the model.

### Questions
See Weaknesses

### Soundness
3

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
The authors propose KLCF, a novel RL framework that aligns a model’s expressed knowledge with its parametric knowledge to improve long-form factuality. It introduces a novel Dual-Fact Alignment mechanism combining a checklist-based consistency reward (for factual recall) and a confidence-based truthfulness reward (for factual precision) without relying on external retrieval. Experiments show that KLCF effectively reduces hallucinations and enhances factual reliability across multiple long-form generation benchmarks.

### Strengths
1. Leveraging the model’s consistency between parametric knowledge and generated knowledge to mitigate hallucinations is both sound and innovative. Incorporating these dual objectives into RL training without relying on external knowledge represents a simple yet elegant paradigm.

2. The authors conduct extensive analyses of their proposed method, examining aspects such as the balance between multiple reward signals and the evolution of intermediate RL variables, which help clarify the underlying mechanisms of their approach. The experimental results further demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The authors conduct main experiments (except from the distilled setting) using the base model rather than a post-trained instruction model (Qwen2.5-14B-Instruct), and they do not provide comparisons with the latter. Intuitively, a post-trained model would likely demonstrate stronger factual accuracy and better instruction-following ability, while the base model typically struggles to adhere to prompts.

### Questions
1. The presentation in Appendix C2, which includes the main results, could be improved. Specifically, the y-axis labels in Figures 4, 5, and 6 are all annotated as “Reward Value,” which is somewhat confusing.

2. Could the authors provide more details about the Claim Extraction Model, such as its model size and some representative extraction results, to better illustrate the effectiveness of this component?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work tackles the hallucination issue in long-form text generation through a novel RL framework called KLCF. Unlike existing RLHF frameworks that primarily rely on preference rewards, KLCF focuses on aligning an LLM's  expressed knowledge with its internal parametric knowledge. To achieve this, it introduces a Dual-Fact Alignment mechanism that optimizes for both factual recall and precision. This mechanism utilizes two complementary rewards: a Checklist-Based Consistency Reward, which enhances factual coverage by identifying consistent, contradictory, or missing information, and a Confidence-Based Truthfulness Reward, which improves factual precision through a self-assessment module trained on the base model's internal knowledge. This approach eliminates the need for costly external knowledge retrieval for real-time fact verification during training, making the reward system lightweight and scalable.

### Strengths
1. The authors propose using a fine-grained checklist for factuality verification, which simultaneously optimizes for both factual recall (comprehensiveness) and precision (correctness). This addresses the common trade-off where models become overly conservative, sacrificing recall to achieve higher precision.
2. By eliminating the reliance on external knowledge sources for verification, the framework is computationally efficient and well-suited for large-scale, online reinforcement learning.
3. The authors conduct extensive experiments to validate the effectiveness of their proposed framework.

### Weaknesses
Thank you for the hard work, but I have a few concerns and suggestions:
1. Definition–Experiment Mismatch for “Hallucination Tax.” The “hallucination tax” is typically defined as the loss in factuality incurred when aligning a model for other desirable traits (e.g., helpfulness). However, the current experiments primarily fine-tune directly for factuality. To more directly support the claim of reducing this tax, it would be stronger to fine-tune on a non-factuality objective and show that factual accuracy is preserved or improved—contrasting against standard RL methods where factuality often degrades.

2. Incremental Contribution. The central idea—using a model’s internal knowledge for factuality verification—has substantial prior art. The main novelty here appears to be a more fine-grained, dual-reward design. This reads as a refinement of established principles rather than a fundamentally new paradigm. Clarifying the conceptual distinction and empirical advantages would help position the contribution.

3. Missing Comparisons to Closely Related Methods. The evaluation would benefit from comparisons to prominent approaches that also leverage internal knowledge for verification, for example, self-consistency, universal self-consistency, and self-evaluation - p(true).

### Questions
Thank you for the efforts. I have two questions and suggestions for clarification:
1. Efficiency Claim. While accessing external knowledge sources can increase training time, the proposed method also incurs nontrivial costs for offline data curation and reward model training. As such, the claim of greater efficiency is not yet fully convincing. Could you provide a more granular cost breakdown (e.g., wall-clock time, GPU hours, data preprocessing time) and an apples-to-apples comparison against retrieval-augmented or tool-using baselines?

 2. Limits of Internal Knowledge and Reward Reliability. The approach relies on the model’s internal knowledge for factuality verification, but a one-off trained model may have incomplete or outdated knowledge. Could you include a more detailed error analysis of the reward model’s failures.

### Soundness
2

### Presentation
2

### Contribution
2
