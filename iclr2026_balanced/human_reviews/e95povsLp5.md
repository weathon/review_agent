## Human Reviewer 1

### Summary
This paper presents SSA, a technique that combines parallel and sequential reasoning techniques to help LLMs decide best answer from LLM responses. The key component: the SSA model get parallel responses from the LLM, and reason on top of the concatenated results. The model can be trained in two ways: use RL to reward generated response or use supervised FT based on GPT-4.1 nano. The results show that (1) SSA is a lightweight approach that effectively outperform parallel scoring and getting close to sequential scaling with RL, and (2) using FT to train SSA model performs better than RL based approach.

### Strengths
- simple but very clean problem formulation. The lightweight test-time scaling combines strengths of parallel sampling (cheap) with ability to reason across solutions in sequential.
- comparison of training algorithms (SFT vs RL) for training SSA model
- evaluation showing generalization beyond training from model with different responses

### Weaknesses
- The comparison with reranker model is not quite clear to me, which also reason on top of parallel sampling results from the LLM responses. 
- The paper primarily focuses on smaller/lightweight model. It's ideal to understand whether SSA model performance tops at sequential reasoning, by ablation on SSA model size and SFT reference model responses (e.g., using larger reasoning model). This is particularly interesting especially that smaller model may not be able to leverage large reasoning model's samples to generate highquality results.

### Questions
- (important) Please elaborate more on how reranker / PRM model approach compares with SSA, as both leverage parallel samples for reasoning.
- provide more discussions about how model size / reference model size in SFT affect SSA performance.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes Sample Set Aggregator (SSA) that trains smaller models using RL to aggregate multiple sampled answers from a larger base model. Experiments on 5 math reasoning benchmarks show that SSA outperforms majority votes and re-ranking methods that rely on larger reward models (including ORM and PRM). This paper also offers results on generalization across sample size as well as model families and sizes.

### Strengths
- The core idea is simple yet effective, particularly for in-domain math reasoning tasks.

- Experiments across 5 datasets demonstrate that the RL-trained, smaller SSA model can outperform larger ORM and PRM models on math reasoning benchmarks.

### Weaknesses
- The method is conceptually straightforward and lacks strong technical novelty, and the empirical gains are not substantial enough to offset this limitation.

- Improvements are mostly limited to math reasoning, where prior ORM and PRM methods have shown broader generalization across diverse domains.

- As shown in Figure 2, performance does not generalize well when increasing the number of input samples, and in Table 2, the results using Llama to produce answer input are also only marginally better than baselines.

- The comparison with sequential scaling in Section 5.3 uses a smaller model against much larger models; a fairer comparison with models of similar size would strengthen the claims.

### Questions
- A possible generalization experiment could involve aggregating answers from multiple base models, can be within the same family, to reduce the need for repeated sampling from large models. Have the authors explored this direction?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper proposes an test-time scaling aggregation method that involves training an LLM (Sample Set Aggregator, SSA) to aggregate multiple parallel responses from a base model (concatenated together as input) and output the final response. The SSA is trained via reinforcement learning (GRPO) and has been tested empirically on various math reasoning tasks and two LLM model families. The paper demonstrates empirically how a smaller SSA model (3B model) can surpass the performance fo a much larger 72B process reward model (PRM).

### Strengths
- The paper tackles an important problem of test-time scaling and how search/optimization/aggregation methods could be used to improve model performance, including when the base model is a blackbox/API-only model.
- The paper is relatively well written, with clear structure and presentation
- The paper presents empirical results indicating that their proposed SSA model could match the performance or even outperform larger PRMs, including a 3B SSA model matching/outperforming the Qwen 72B PRM model.
- The authors have provided relatively extensive experimental results, including in the appendix.

### Weaknesses
- The paper should include more detailed comparisons and discussions with the LLM ensemble literature where a base model could be queried in parallel with various different reasoning/role prompts with these responses subsequently aggregated [1-3]. These methods bear similarity in that K parallel candidates are also drawn from a frozen base model before being aggregated, with the aggregation possibly also being an LLM fed with the various candidate responses.

- The paper's claim in proposing a method that is "learning to reason" seems unjustified, especially based on the empirical results provided. As the authors discussed in Sec 6.2, the SSA appears to largely be playing the role of a response selector rather than be a model that actually reasons over the candidate responses -- the model only output placeholder thinking tokens, and the authors noted minor performance degradation without explicit thinking. Appendix Table 9 and 10 also supports this perspective. The authors should consider repositioning the paper to clearly reflect this.

- The paper could benefit from further analysis of how the training data distribution affects the performance and trends of SSA, such as whether removing (or only including) 5/5 or 0/5 (very easy or very tough) questions from the training data would affect the model's 'selection' vs 'synthesis' behavior and overall performance.

- The SSA model seems to not be able to generalize well beyond math tasks, which it has been trained on. Table 11 in the appendix shows inconsistent performance of SSA over majority vote (the only baseline presented with just k=5). The authors could consider providing further analysis to describe the limitations of generalizability of the method.

- The baseline USC, which is the vanilla LLM aggregator approach or SSA without training, seems to have a prompt that is biasing it towards performing majority vote. Fig 10 in the appendix shows that compared to the SSA prompt, the USC prompt has been modified to instruct the model to answer "based on majority consensus". The authors should explain why this is justified.

- The scalability of this approach seems to be a potential issue, especially if there are responses with longer reasoning traces or solutions. While the authors proposed the multi-stage SSA approach, this might lead to increased inference cost and degrade performance for other datasets with longer responses. Appendix A.4 also seems to hint that training performance may also be limited by context window, but more analysis would be needed to check 

[1] Du et al, Improving factuality and Reasoning in Language Models through Multiagent Debate

[2] Hu et al, Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning Tasks

[3] Chen et al, AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors.

### Questions
Please see the weakness section for questions.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper proposes Sample Set Aggregator (SSA): a small LLM reads and concatenates K parallel sampled answers, and uses GRPO reinforcement learning to directly optimize the “final correctness rate.” SSA decouples “generation” and “aggregation,” serving as an efficient answer combiner for black-box large models. On benchmarks such as GSM8K, MATH, AIME24, AMC23, and Olympiad, SSA consistently outperforms majority voting and various PRM/ORM baselines; the 3B version achieves a better efficiency–accuracy trade-off and shows good generalization across model sizes and families.

### Strengths
1. The paper is well written and easy to follow.
2. The experiments are comprehensive and strongly support the authors’ claims.

### Weaknesses
1. The motivation is not sufficiently novel. According to Lines 56–65, the method simply combines two test-time scaling approaches. It neither analyzes which limitations of existing methods this addresses, nor does it offer conceptual insight; the contribution reads more as engineering practice and provides limited guidance for future research.
2. There is a lack of comparison with prior work. Given many studies train verifiers to select best-of-N [1, 2], why is the training approach in Section 3 more effective than these? The experimental section also lacks such comparisons.

Considering the work is overly trivial and provides limited research insight, I recommend rejection.

[1] Mudgal et al., Controlled decoding from language models. ICML 2024.

[2] Huang et al., Is best-of-n the best of them? coverage, scaling, and optimality in inference-time alignment. Arxiv.

### Questions
1. Given that many scenarios lack training data, how can your method be applied in such settings?
2. Additional experiments on long-CoT models (e.g., DeepSeek-R1-Distill-Llama, Qwen3) and broader datasets (e.g., MMLU-Pro, GPQA) would better substantiate the method’s effectiveness.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
0

### Confidence
4