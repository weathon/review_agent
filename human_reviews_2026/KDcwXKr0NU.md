# DiSRouter: Distributed Self-Routing for LLM Selections

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
The proliferation of Large Language Models (LLMs) has created a diverse ecosystem of models with highly varying performance and costs, necessitating effective query routing to balance performance and expense. Current routing systems often rely on a centralized external router trained on a fixed set of LLMs, making them inflexible and prone to poor performance since the small router can not fully understand the knowledge boundaries of different LLMs. We introduce DiSRouter (Distributed Self-Router), a novel paradigm that shifts from centralized control to distributed routing. In DiSRouter, a query traverses a network of LLM agents, each independently deciding whether to answer or route to other agents based on its own self-awareness—its ability to judge its competence. This distributed design offers superior flexibility, scalability, and generalizability. To enable this, we propose a two-stage Self-Awareness Training pipeline that enhances each LLM's self-awareness. Extensive experiments demonstrate that DiSRouter significantly outperforms existing routing methods in utility across various scenarios, effectively distinguishes between easy and hard queries, and shows strong generalization to out-of-domain tasks. Our work validates that leveraging an LLM's intrinsic self-awareness is more effective than external assessment, paving the way for more modular and efficient multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
A recent strategy to minimize compute involves routing user prompts to an LLM which has just the right size to handle the request, instead of always prompting the most capable model.

Currently, the most popular routing strategy involves using a routing model, which determines which LLM to prompt with the user's request. This paper proposes to instead integrate the routing process directly into the available LLMs. Specifically, the LLMs are trained to defer to the next biggest model when they deem they are not capable of providing a correct answer. The paper shows that this results a better balance between cost and accuracy.

### Strengths
- The problem formulation is clear and the motivation is sound.
- This idea will be of interest to the community.
- The results show that the idea has merit and should be further investigated.

Overall, I believe that this paper should be accepted, although authors should address the weaknesses mentioned below.

### Weaknesses
My main issue is that the authors barely discuss/explore the (potential) limitations of the method. For instance, the method first prompts the smallest model. Because of its size, it might sometimes struggle to properly reason whether it is capable of accurately answering a given question. Although the results seem positive, have the authors observed such phenomena, our found that for certain type of tasks this happened at a greater rate?

Another issue is the computational cost of the method. The most difficult prompts must go through all models before the final largest model answers. This is the worse case scenario, and it is a downside of the method. Authors do not talk about this much. The method does seem more costly than certain router baselines, but performs slightly better. Have the authors considered potentially skipping an LLM when say the first model's confidence is very low, or outputting more nuanced reject answers, such as "I don't now.", "I really don't know.", and depending on such output, skip an LLM or two in the chain?

Another important question is how does internalizing the routing decision degrade a model's capabilities compared to standard instruction-tuning. It is pretty important to know if this creates a gap and if so, how big is it.

Aside from discussing the (potential) limitations of the method, the paper's methodology section would also gain from being clearer. The current description is somewhat high level and leaves a lot of the details unspecified. For instance, during RL fine-tuning, how are the model's answers judged as correct or incorrect?

Finally, it would be interesting to know how the authors designed the prompts for the "Performance First", "Balance", and "Cost First" scenarios in Table 7 of the Appendix. Were multiple prompts tested or did the authors simply use the first prompt that came to mind?

Addressing these aspects would help strengthen the paper.

### Questions
None

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
The paper introduces a distributed routing paradigm, enabled by a concrete self-awareness training pipeline, to address LLM cost-performance trade-offs. However, the core claims are undermined by significant methodological flaws, including the failure to account for cumulative latency and a lack of generalization evidence beyond a single, homogeneous model family for narrow, verifiable tasks.

### Strengths
1. The paper introduces a "Distributed Self-Routing" framework, replacing the external centralized router with intrinsic "self-awareness." 

2. It provides a concrete technical contribution with the two-stage "Self-Awareness Training" pipeline (SFT + RL), enabling models to internally assess their knowledge boundaries and execute a "reject" action.

3. The framework is inherently modular ("plug-and-play").

### Weaknesses
1. The core utility metric (Utility = Performance - α * Cost) is flawed. It completely ignores the significant cumulative latency incurred by the cascade structure. A query rejected multiple times will have a real-world response time far exceeding that of a single-shot centralized router, a critical cost factor this paper overlooks.

2. The experiments are confined to a homogeneous model pool (all Qwen-family models). This is a major threat to validity. The paper provides no evidence that the "self-awareness" training paradigm generalizes to a realistic, heterogeneous ecosystem of diverse model architectures (e.g., Llama, Mistral, Gemma), which is the primary motivation for such a system.

3. The entire methodology (self-awareness training, binary accuracy evaluation) is fundamentally dependent on tasks with clear, verifiable answers (e.g., math, multiple-choice QA). It is unclear how this approach would apply to open-ended generative tasks (e.g., summarization, creative writing) where "correctness" is ill-defined, severely limiting its practical scope.

4. The evidence supporting the superiority of "intrinsic assessment" is marginal. In Table 5, the purpose-trained 7B DiSRouter (0.80 F1) barely outperforms a general-purpose 8B external classifier (0.77 F1). This negligible gap fails to provide a strong justification for the added complexity of the distributed training approach over a simpler, well-trained external router.

### Questions
None

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
2

### Summary
This paper proposes DiSRouter—a distributed self-routing framework that can propagate queries step-by-step through different LLM agents without a central router.
A two-stage Self-Awareness Training (SFT+RL) is designed to teach individual LLMs to "answer if they know the answer and reject if they don't," and the rejection threshold can be adaptively adjusted according to user preferences (performance priority/balance/cost priority).

### Strengths
1.Experiments show that DiSRouter's utility score (performance-α·cost) is significantly superior to both baselines and existing router systems across various performance-cost scenarios, with both in-domain and out-of-domain data. It is more flexible and has "plug-and-play" modular capabilities.

2.The hypothesis that training does not inject new knowledge but only improves self-boundary awareness can be supported by experimental results: ∆Performance <1%, which verifies that the performance improvement mainly comes from better routing rather than the model itself becoming stronger.

### Weaknesses
Although the experimental results have shown that the authors' method is far superior to the baseline and other routing methods, I believe that some additional perspectives are still beneficial for future work:

1.The cost introduced by the two-stage training approach may require further evaluation or comparison with other methods, as each agent needs to perform independent multiple-time inference to prepare its SFT data and conduct RL training.

2.For more open agent systems, data preparation is more challenging, as some questions do not have a single correct answer (e.g., summarizing, creation/generation questions) while still have varying difficulty, making the self-awareness training phase tricky. If only verifiable questions data are synthesized and trained, unprepared tasks may suffer from catastrophic forgetting, and the model's self-awareness on such tasks cannot be guaranteed. The author may provide the performance of the model and the systems on these more complex out-of-domain tasks.

### Questions
1.In Section 3.2.2 and Appendix C, what is the purpose of extracting and reformulating the original model's answer using a stronger model? The intervention of the model may introduce additional errors and information, resulting in unpredictable performance after training.

2.Is the true "routing overhead" underestimated? There may be multiple levels of consecutive rejections on the cascading path (the experiment used a 5-model approach, and the worst case was only 4 rejections to reach 14 bytes). Could you please provide the end-to-end latency of the first token for comparison with other methods?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces **DiSRouter (Distributed Self-Router)**, a framework that replaces centralized query routing in LLM systems with a distributed paradigm where each agent autonomously decides to answer or route queries based on self-awareness. Key contributions include:

- A **distributed routing architecture** that enhances modularity and scalability compared to centralized routers.
- A **Self-Awareness Training pipeline** with supervised fine-tuning (SFT) and reinforcement learning (RL) using a localized reward function, enabling independent agent training.
- **Scenario adaptability** via a preference factor $\alpha$ to balance performance and cost dynamically. Experiments on datasets like GSM8K and MMLU show DiSRouter outperforms baselines (e.g., RouteLLM, FrugalGPT) in utility. The core idea is that intrinsic self-assessment is more effective than external evaluation.

### Strengths
- **Originality**: The shift from centralized to distributed routing is novel, creatively integrating self-assessment concepts (e.g., LLM uncertainty) into a routing framework. The scenario-adaptive reward function is an innovative touch.
- **Quality**: Experimental design is rigorous within its scope, with utility metrics and modularity tests. The pipeline is methodically described.
- **Clarity**: The problem formulation in Section 2 is precise. Writing is concise.
- **Significance**: The "plug-and-play" modularity addresses a key pain point in LLM systems, though real-world impact depends on scalability beyond the tested settings.

### Weaknesses
- **Limited Generalizability**: Experiments use only the Qwen2.5-Instruct series, neglecting other LLM families (e.g. Llama). This casts doubt on DiSRouter's applicability to diverse models. The authors should include cross-architecture validation.
- **Training Efficiency Concerns**: The Self-Awareness Training (SFT + RL) is computationally heavy, but the paper dismisses routing cost as "negligible" without quantifying training overhead. A cost-benefit analysis is missing.
- **Limited Robustness Testing**: There is no evaluation under adversarial conditions (e.g., ambiguous queries) to test the robustness of self-assessment. This is critical for real-world reliability.

### Questions
1. I am still confused that how the the local routing policy $\pi_i$ of agent $m_i$ is trained, it seems that $m_i$ only output "I don't know!" if choosing reject, how does $m_i$ select the next agent?
2. Can DiSRouter's self-awareness training be applied to LLMs with different architectures (e.g., encoder-decoder models)? Please provide results with at least one non-Qwen model family to validate broad applicability.
3. How does DiSRouter perform with large-scale agent pools (e.g., >10 agents)? 
4. What are the practical limitations (e.g., latency, fault tolerance) in deploying DiSRouter? Case studies or simulations with real-world constraints would be valuable.

### Soundness
3

### Presentation
3

### Contribution
3
