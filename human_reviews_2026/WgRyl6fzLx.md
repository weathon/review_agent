# SSRL: Self-Search Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
We investigate the potential of large language models (LLMs) to serve as efficient simulators for agentic search tasks in reinforcement learning (RL), thereby reducing dependence on costly interactions with external search engines. To this end, we first quantify the intrinsic search capability of LLMs via structured prompting and repeated sampling, which we term Self-Search. Our results reveal that LLMs exhibit strong scaling behavior with respect to the inference budget, achieving high pass@k on question-answering benchmarks, including the challenging BrowseComp task. Building on these observations, we introduce Self-Search RL (SSRL), which enhances LLMs' Self-Search capability through format-based and rule-based rewards. SSRL enables models to iteratively refine their knowledge utilization internally, without requiring access to external tools. Empirical evaluations demonstrate that SSRL-trained policy models provide a cost-effective and stable environment for search-driven RL training, reducing reliance on external search engines and facilitating robust sim-to-real transfer. We draw the following conclusions:  1) LLMs possess world knowledge that can be effectively elicited to achieve high performance; 2) SSRL demonstrates the potential of leveraging internal knowledge to reduce hallucination; 3) SSRL-trained models integrate seamlessly with external search engines without additional effort. Our findings highlight the potential of LLMs to support more scalable RL agent training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SSRL (Self-Search Reinforcement Learning), a method that trains language models to perform self-generated search before answering questions.
Instead of using a real search engine, the model creates its own “search” and “information” steps inside the output, and then receives a reward based on the answer quality and output format.
The goal is to teach the model to reason and retrieve information more systematically, while reducing the need for real web queries.
The authors evaluate SSRL on several QA benchmarks using models from 3B to 70B parameters, and also test a Sim-to-Real setting where the model interacts with real search results.

### Strengths
1) Novelty:
The paper combines reinforcement learning with self-constructed search reasoning.
This idea of training an LLM in a simulated search environment is novel and interesting.

2) Cost efficiency:
Because it does not depend on real API calls or human feedback, the method is low-cost and scalable for small and medium-sized models.

3) Sim-to-Real transfer:
The results show that a model trained only with self-generated searches can still work reasonably well when connected to a real search engine.
This indicates the learned policy is relatively stable and generalizable.

4) Practical relevance:
The work explores how smaller models can approach the performance of larger ones through structured reasoning and sampling, which is useful for efficient deployment.

### Weaknesses
1) Methodology clarity:
The paper does not clearly define the RL setup — it is unclear what the exact states, actions, and rewards are.
The training pipeline (sampling, reward, update) is not fully described, and the “self-search” mechanism is hard to reproduce.
Although the code is provided, it is difficult to directly examine every detail in code. 

2) Reward design limitations:
The reward is simple and rule-based, combining a binary “outcome reward” (correct / incorrect) and a “format reward” for structural tags.
This design is very discrete and task-specific, which can make learning unstable and limit generalization.
It makes the knowledge of LLM is trapped in its own training environment, and implicitly put a requirement on the training dataset.

3) Claims without quantitative proof:
The paper often claims that SSRL reduces hallucination or improves reasoning quality, but there is no direct measurement (e.g., factual consistency, hallucination rate, or human study).
The evidence is only indirect from QA accuracy, which is not enough.

4) Lack of ablation and analysis:
There is no ablation to test the importance of each reward term or design choice.
It is unclear how much improvement comes from SSRL itself versus the sampling or instruction tuning.

5) Task-specific scope:
The method is only evaluated on QA datasets, and it is uncertain whether the same idea can generalize to open-ended reasoning or dialogue tasks. 
It's necessary to quantify the scope.

### Questions
Can the authors give more details about how the RL process is implemented?
For example, how are advantages computed, how is KL controlled, and how many updates per batch?

How sensitive is SSRL to the reward coefficients (e.g., λf)?
Would smoother or continuous rewards improve stability?

Did the authors try any direct hallucination or factual consistency evaluation?
This would help verify the claimed benefits.

How much of the gain comes from the structured format reward versus the self-search policy itself?
An ablation would clarify this.

Could this method be applied to other tasks beyond QA, such as summarization or dialogue?
If so, what changes would be needed in the reward or structure?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is about training LLMs reinforcement learning (RL) on tasks that require agentic search. The authors propose to approximate costly external search queries with the intrinsic search ability of LLMs (which they call Self-Search) during training. The authors first evaluate the Self-Search ability of some open-source LLMs from three families (Qwen2.5, Qwen3, and Llama3) showing that search result accuracy on six benchmarks (e.g., General QA, Multi-hop QA, Vague QA) increases with repeated sampling (i.e., pass@k). Then, the authors introduce Self-Search Reinforcement Learning (SSRL), a custom RL objective that enables models to progressively enhance their internal use of knowledge. The authors empirically show that SSRL-trained policies are cost-effective and stable for doing RL training with agentic search queries. While SSRL reduces reliance on external search engines during training, it still allows for sim-to-real transfer as evidenced by empirical results.

### Strengths
- The idea of leveraging the intrinsic search ability of LLMs to reduce reliance on external search engines during RL training is interesting and might be novel in this particular setting. I recognize that it addresses a practical challenge in training LLMs for agentic tasks.
- The empirical evaluation is comprehensive, covering multiple LLM families and benchmarks. The results demonstrate the effectiveness of SSRL in improving search accuracy and reducing external search costs. Also, the Appendix contains many ablation studies to validate the effectiveness of each component.

### Weaknesses
- My biggest concern is about the clarity of the proposed approach. While the high-level idea of Self-Search and SSRL is understandable, some details are unclear to me. I believe the paper would benefit from more polishing to improve clarity.
  - Starting from Figure 1 (left), at first glance, it was not clear what the dotted box represented for the full-sim search section.
  - What is the instructions/prompts used. I'm assuming they are different when studying the Self-Search ability (Section 2) versus when training with SSRL (Section 3). The prompt design discussed in the Appendix B.1.1 seems rather important to me to fully understand the proposed approach. I would integrate that in the main text.
  - What does iterative refinement mean? How does that relates to the repeated sampling? Are those the same thing? It is not clear in the main text what iterative refinement means in the context of Self-Search. When reading the Appendix, I understood that it refers to simply continuing the CoT generation alternating <think>, <search> and<information>. Section B.3 mentions 10 as the maximum number of iterations, but it is unclear to me how many iterations are typically needed to get good results.
  - Is the k in pass@k the same as the number of iterations in iterative refinement? Typically pass@k refers to generating k independent full-inferences.
  - Some of this information is only available in the Appendix and the reader has to put all the scattered parts together.

- How do you ensure diversity in the multiple samples generated during SSRL? If the samples are too similar, the benefit of repeated sampling might be limited.
- Sampling temperature can significantly impact the quality of the samples. How sensitive is the Self-Search performance to the choice of sampling temperature? Have you experimented with different temperature settings other than 1 during training?

#### Minor
- This is confusing "The instruction used is shown in Appendix B.1.2. The prompt used is listed in Appendix B.1.1.". When looking at Table 5 and 6 they are very similar. The only different is one is asking for the model to fill in the top search results. Where does "k" comes into play in Table 6?

### Questions
- Sampling temperature can significantly impact the quality of the samples. How sensitive is the Self-Search performance to the choice of sampling temperature? Have you experimented with different temperature settings other than 1 during training?
- In Table 1, where are the results for Self-Search with Search Engine -/G ?
- Are the results in Table 1 represent pass@k? If not how do you aggregate the multiple samples generated during Self-Search to produce a final answer? Are you using majority voting as discussed in Section 2.4?
- In Table 2, which results correspond to SSRL-trained models with external search engines at inference time versus those that retrieve K responses from local corpora?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper investigates an interesting problem of training a self-search agent with reinforcement learning. Existing search agents rely on calling real-time search agents during training and inference, which leads to latency. In this work, the authors propose to use the policy model itself as the search engine and conduct a self-search agent with reasoning and “self-search”. They first conduct experiments to study the test-time scaling performance of the self-search agent and propose a reinforcement learning solution to further improve it. Extensive experiments on several benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is very well written and easy to understand.
2. The inference time scaling experiments are interesting and insightful.
3. The study of “self-search” reinforcement learning is novel, and it is great to see that the model trained with “self-search” RL can generalize to adopting tools during inference.
4. The authors conduct extensive experiments to demonstrate the effectiveness of the proposed method.

### Weaknesses
1. Lack of an ablation study for the format reward. In Section 3.2, the authors propose conducting outcome reward and format reward functions. However, there is no ablation study to verify the effectiveness of the format reward.

2. Is the method only suitable for a specific type of LLM? The main results in Table 1 are based on Llama models. It is questionable whether SSRL can still outperform other methods on other types of LLMs, such as Qwen. From Figure 5, it seems that the performance on Qwen2.5 is not very good with SSRL.

3. It requires further explanation on why information token masking is still useful here. The difference between SSRL and Search-R1 is that here the information tokens are on-policy, thus masking may not be desired.

### Questions
1. What is the performance if we do ablation for the format reward?
2. Is the method only suitable for a specific type of LLM?
3. Why is it still important to conduct information token masking here?

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
This paper introduces self-search reinforcement learning (SSRL), a RL-based framework that trains Large Language Models (LLMs) to perform search tasks by levering their own internal knowledge rather than relying on external search engines. The authors first investigate self-search and found that the performance scale with increasing sample size k, indicating that LLM's intrinsic knowledge may be sufficient for such benchmarks. As such, the authors propose SSRL with format- and outcome-based rewards to enhance such capabilities with RL training, allowing models to refine their internal knowledge utilization through long-form reasoning and self-search. This approach creates a cost-effective training framework compared to search-based ones, and enables the trained models to perform similarly to LLMs trained with real-world search engines.

### Strengths
1. The authors performed extensive experiement on the sample size of search agents and argue that the performance can be improved by scaling sample sizes even without retrieving from external knowledge. Motivated by such observations, the proposed SSRL method eliminates the search API costs and improves the performance of search agents by training models to exploit their own internal knowledge.

2. Through extensive experiemnts, the authors show that models trained with SSRL can both work with integrated knowledge or with real search engines. In addition, the experiment results show that SSRL trained models show comparable performance to the models trained with real search engines.

### Weaknesses
1. The authors did not propose novel insights or new learning framework. Instead, the work seems to be an improvement on the Search-R1  baseline, featuring more refined RL training techniques. Therefore resulting method rather seems to be an improved "R1-Base / Instruct" baseline without searching.

2. Although the experiment results show that LLM performance can match search agents with SSRL training, the model does not access external information or facts that may not exist within the embedded knowledge, which could result in over-confidence or hallucinations, but the authors do not discuss these aspects in detail.

3. As shown in the appendix, when applied to challenging tasks like SimpleQA, models using real search still significantly outperform SSRL. This suggests that SSRL is only effective in scenarios where the model has already internalized the necessary knowledge (e.g., Wikipedia) during pretraining, but it struggles when faced with tasks that require external knowledge it does not possess.

### Questions
1. The authors mention that multi-turn self-search and self-search with reflection hurt performance compared to naive repeated sampling. Does this imply that SSRL is less about improving the model's reasoning process and more about optimizing the simple, one-step extraction of its existing parametric knowledge?

2. Since authors are motivated by strong TTS results, where increasing the sample size at inference time significantly improves performance. However, the ablation in Appendix C.3.8 shows that increasing group size at training time for the GRPO algorithm provides limited to no benefit. Why do the benefits of a larger sample size diminish during training, when TTS proves that a wide range of high-reward trajectories already exist within the model?

### Soundness
3

### Presentation
2

### Contribution
2
