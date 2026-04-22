# Multi-Agent Debate with Memory Masking

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Large language models (LLMs) have recently demonstrated impressive capabilities in reasoning tasks. Currently, mainstream LLM reasoning frameworks predominantly focus on scaling up inference-time sampling to enhance performance. In particular, among all LLM reasoning frameworks, *multi-agent debate* (MAD), which employs multiple LLMs as agents to perform reasoning in the way of multi-round debate, has emerged as a powerful reasoning paradigm since it allows agents to access previous memories to alleviate fallacious content and refine their reasoning iteratively in each debate round. However, although MAD significantly improves the reasoning capabilities of LLMs, in this paper, we observe that there remain erroneous memories, and LLM agents are vulnerable to these erroneous memories. To explore this phenomenon, we provide a theoretical insight that the performance of MAD is highly dependent on the quality of memories derived from the previous debate, indicating that the existence of erroneous memories poses a threat to the performance of MAD. To address this problem, we introduce a simple yet effective multi-agent debate framework, *multi-agent debate with memory masking* (MAD-M$^2$), to improve the robustness of MAD by allowing LLM agents to mask erroneous memories from the previous debate round at the beginning of each debate round. In this way, MAD-M$^2$ can polish the contextual information before each debate round by preserving informative and meaningful memories while discarding the erroneous memories. Extensive experiments and analyses on mainstream mathematical and logical reasoning benchmarks demonstrate that MAD-M$^2$ can identify the erroneous memories and achieve better performance in reasoning than MAD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a fundamental limitation of the Multi-Agent Debate (MAD) framework, demonstrating that LLM agents exhibit vulnerability to erroneous memories from preceding debate rounds, which can propagate incorrect reasoning trajectories. Through rigorous theoretical analysis, the authors establish that MAD performance is intrinsically bounded by e^(-αNₑ), where Nₑ denotes the cardinality of erroneous memories and α characterizes the agent's robustness coefficient. To address this vulnerability, they propose MAD-M² (Multi-Agent Debate with Memory Masking), a framework that incorporates a memory evaluation and masking mechanism, enabling agents to selectively filter low-quality contextual information prior to each reasoning iteration.

### Strengths
1. This paper is the first to demonstrate that Multi-Agent Debate (MAD) frameworks are vulnerable to false memories, providing theoretical proof of this phenomenon.
2. This study offers a thorough analysis of the key characteristics of the proposed MAD-M² framework in relation to existing methods.

### Weaknesses
1. The primary contribution of this paper lies in its theoretical insights rather than methodological innovations.

2. The proposed MAD-M² framework achieves state-of-the-art performance in only a limited subset of experimental scenarios, while underperforming traditional MAD or CoT-SC methods in the majority of cases. Moreover, MAD-M² incurs significantly higher token consumption compared to baseline methods. From a performance-efficiency trade-off perspective, MAD-M² does not demonstrate clear contributions. If additional metrics can substantiate the contributions of MAD-M², please include them in Table 1.

### Questions
Refer to the weaknesses.

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
3

### Summary
This paper investigates the robustness issue in the multi-agent debate (MAD) framework for large language model (LLM) reasoning. The authors theoretically demonstrate that the performance of MAD degrades when agents rely on “wrong memories” from prior debate rounds, since each round’s reasoning depends on previous outputs. To mitigate this, they propose MAD-M2 (Multi-Agent Debate with Memory Masking), which allows each agent to evaluate and selectively mask unreliable memories before generating new responses. The method is conceptually simple yet well-motivated. Experiments across four reasoning benchmarks and diverse models show consistent or improved results compared with vanilla MAD.

### Strengths
1. Clear motivation grounded in a theoretical gap: The authors identify an underexplored limitation in multi-agent debate frameworks — vulnerability to low-quality reasoning memories — and back it up with intuitive examples (Figure 1) and formal analysis.

2. The analysis in Section 2 provides explicit probabilistic bounds linking performance degradation to the number of wrong memories. The inclusion of both hard and easy reasoning settings (HPR/EPR) offers a nuanced interpretation of how memory errors affect debate robustness.

### Weaknesses
1. My biggest concerns are about approach performance. I only see the marginal performance gains in stronger models. While MAD-M2 improves certain benchmarks on easy tasks, the improvements for DeepSeek-V3 are minor or inconsistent. This suggests limited scalability to highly capable LLMs. Could the authors provide more results on diverse and powerful LLM benchmarks?

2. The proposed masking step roughly doubles token consumption in some cases, but the paper provides limited quantitative analysis of efficiency–performance trade-offs. 

3. The masking mechanism is described at a high level but not deeply analyzed. For instance, how different scoring heuristics or thresholds affect the results remains unclear. The authors are suggested to add more fine-grained ablations (e.g., varying mask strictness).

4. Although the paper compares conceptually to S-MAD and S2-MAD, it doesn’t empirically benchmark against them.

### Questions
1. How sensitive is MAD-M2 to the rule for generating the binary mask?

2. Did the authors explore partial masking or soft weighting? Could such approaches improve stability?

3. Can the authors provide more comprehensive experiments on more LLM backbones and compare more baseline MAD approaches?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the robustness of multi-agent debate frameworks to wrong memories from previous debate rounds. The authors provide a theoretical analysis showing that MAD performance is closely related to the number of wrong memories, then propose MAD-M², which allows agents to evaluate and mask undesirable memories before reasoning in the next debate round. Experiments on MATH, MMLU-Pro, AIME24/25 show MAD-M² improves over standard MAD, particularly on easy tasks with weaker models (Qwen2.5-72B, Mixtral-8x7B), though gains on hard tasks and stronger models (DeepSeek-V3) are limited

### Strengths
The paper identifies a real issue in MAD: agents can be misled by wrong memories from previous rounds, as illustrated in Figure 1 where Agent 1 initially answers correctly but is misled in Round 2 after seeing Agent 2's incorrect response. This is intuitive and well-demonstrated.

MAD-M² is straightforward to implement: agents evaluate memories, generate binary masks, and reason with filtered memories. The method doesn't require additional models or complex infrastructure, making it practical for real deployment.


The theoretical analysis in Section 2.2 provides useful insights. Propositions 2.2 and 2.3 formalize the performance bounds for CoT-SC and MAD, showing how wrong memories (Ne) exponentially degrade performance through the term e^(-αNe). The distinction between easy problem reasoning (EPR) and hard problem reasoning (HPR) settings offers guidance for when memory masking helps.

### Weaknesses
MAD-M² shows improvement on MATH (0.90→0.92, +2%) and AIME24 (0.37→0.50), but degrades or maintains performance on MMLU-Pro (0.83→0.83) and AIME25 (0.40→0.40). Meanwhile, MAD-M² shows improvement on MATH (0.90→0.92, +2%) and AIME24 (0.37→0.50), but degrades or maintains performance on MMLU-Pro (0.83→0.83) and AIME25 (0.40→0.40)


Token consumption analysis in Table 1 shows MAD-M² consumes 50-100% more tokens than standard MAD. The cost-benefit ratio is poor, especially when standard MAD already improves factuality and reasoning significantly over single-agent methods

For assumption 2.1, the claim that the probability of correct answers given memories follows e^(-αNe) is asserted without justification. Why exponential decay? Why not linear, polynomial, or other forms?

Hoeffding requires independent samples, but debate rounds are sequential and dependent - each round conditions on previous memories. The analysis doesn't account for this dependency.


Figures 3-4 show that scaling agents help on easy tasks only (consistent with theory), but MAD-M² doesn't consistently outperform MAD even there

### Questions
How often do agents correctly identify wrong memories? What's the precision/recall?


Figure 2 shows one cherry-picked example. Could you also provide the failure cases?

### Soundness
3

### Presentation
3

### Contribution
2
