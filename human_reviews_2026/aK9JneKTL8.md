# Reverse-Engineered Reasoning for Open-Ended Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
While the "deep reasoning" paradigm has spurred significant advances in verifiable domains like mathematics, its application to open-ended, creative generation remains a critical challenge. The two dominant methods for instilling reasoning—reinforcement learning (RL) and instruction distillation -- falter in this area; RL struggles with the absence of clear reward signals and high-quality reward models, while distillation is prohibitively expensive and capped by the teacher model's capabilities. To overcome these limitations, we introduce REverse-Engineered Reasoning (REER), a new paradigm that fundamentally shifts the approach. Instead of building a reasoning process "forwards" through trial-and-error or imitation, REER works "backwards" from known good solutions to computationally discover the latent, step-by-step deep reasoning process that could have produced them. Using this scalable, gradient-free approach, we curate and open-source DeepWriting-20K, a large-scale dataset of 20,000 deep reasoning trajectories for open-ended tasks. Our model, DeepWriter-8B, trained on this data, not only surpasses strong open-source baselines but also achieves performance competitive with, and at times superior to, leading proprietary models like GPT-4o and Claude 3.5.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Reverse-Engineered Reasoning (REER), a novel paradigm that generates reasoning data for open-ended tasks by working backward from known high-quality solutions instead of using costly RL or distillation. This method frames trajectory discovery as a gradient-free search to find the reasoning path that minimizes the solution's perplexity, resulting in a model trained on this data that competes with top proprietary models.

### Strengths
1. The motivation is clear and important.
2. The paper is well-written and easy to follow.
3. The proposed method has potential for other domains where reasoning paths are rare and hard to obtain.

### Weaknesses
1. The core of the entire method is to assume that minimizing PPL ($y | x, z $) can find a high-quality inference trajectory $ z $. However, the paper does not prove the direct and reliable correlation between PPL and reasoning quality (e.g., efficiency, creativity, and logical depth). PPL often represents the quality and fluency of language, yet it is not equivalent to reasoning quality.
2. The computation of PPL is based on the base model; however, the value of PPL can vary across different LLMs. The paper only conducts experiments on Qwen3-8B, which is insufficient to demonstrate the generalizability in different LLMs.
3. It would be interesting to investigate the performance across different model sizes, which can further verify the ubiquity of the proposed method.
4. The depth and diversity of reasoning paths lack a quantitative evaluation.
5. Although claiming to be more cost-effective than distillation and RL, the paper does not provide a specific cost analysis or comparison of the REER synthesis process, which involves multiple LLM calls and PPL calculations.
6. The evaluation (e.g., PPL) of reasoning path quality is narrow, and it is hard to verify whether the reasoning paths are correct. Wrong reasoning paths can also have low PPL due to other factors (e.g., hallucination), leading to my concern regarding real-world application and applying to other tasks.

### Questions
See weaknesses.

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
This paper introduces Reverse-Engineered Reasoning (REER), a new paradigm for teaching deep reasoning in open-ended, creative tasks. Traditional methods like RL and distillation fail here due to a lack of clear reward signals or high costs. REER works "backwards": it starts with a high-quality solution and computationally discovers the latent reasoning process that could have produced it. This is framed as a gradient-free search to find a thinking trajectory that minimizes the perplexity of the known-good answer. Using this scalable method, the authors created the DeepWriting-20K dataset. The resulting model, DeepWriter-8B, rivals or exceeds top proprietary models like GPT-40 and Claude 3.5 on challenging writing benchmarks.

### Strengths
1. This paper introduces a novel approach that fundamentally shifts how reasoning is taught. Instead of building reasoning forwards through trial-and-error (RL) or imitation (distillation), it works backwards from known good solutions to computationally discover the latent, step-by-step thinking process that could have produced them. This provides a new path for tackling deep reasoning in open-ended, non-verifiable domains.

1. REER is a scalable, gradient-free approach. It successfully bypasses the sample inefficiency of RL and the prohibitively expensive cost of distillation. This creates a cost-effective and automatable pathway to generate vast quantities of high-quality, deep-thinking training data at a scale that was previously impractical.

1. The authors used their method to create and open-source DeepWriting-20K, a large-scale dataset with 20,000 deep reasoning trajectories. This comprehensive resource, spanning 25 categories, is designed to mitigate data scarcity and catalyze future research into planning and structured thought for open-ended generation tasks.

1. The resulting 8B model, DeepWriter-8B, outperforms strong open-source baselines. More significantly, it achieves performance competitive with leading proprietary models like GPT-4o and Claude 3.5. This provides rigorous empirical evidence that human-like deep reasoning can be cultivated without relying on costly distillation or RL.

### Weaknesses
1. The paper's main evaluations rely on using powerful LLMs (like GPT-4o and Claude 3.7) as judges. The authors acknowledge that this method, while scalable, has potential for inherent biases.

1. The core REER algorithm depends on using the perplexity (PPL) of the reference answer as a proxy for the quality of the reasoning trajectory. The assumption that lower PPL, which makes the answer seem maximally probable and logical, directly equates to a higher-quality reasoning process is a proxy and may not perfectly align with human judgments of logic, creativity, or efficiency.

1. The primary experiments were conducted on an 8B base model (Qwen3-8B-Base). The effectiveness of REER on much larger models is not yet validated, as mentioned by the authors in Section 4.

### Questions
1. What if we use a reward model (e.g., a process reward model) as a proxy for the quality of the reasoning trajectory? Will REER yield better performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces REverse-Engineered Reasoning (REER), a new paradigm that infers plausible step-by-step reasoning paths backwards from high-quality outputs, instead of constructing them via reinforcement learning or distillation. By formulating reasoning-path recovery as a gradient-free local search minimizing perplexity, the authors synthesize 20 K structured reasoning trajectories (DeepWriting-20K) and fine-tune an 8B model (DeepWriter-8B). The model achieves near-proprietary-level performance on LongBench-Write, HelloBench, and WritingBench, outperforming all open-source baselines.
Overall, the work provides a cost-effective way to instill deep reasoning for open-ended generation. It offers both a scalable data-generation framework and strong empirical evidence that reverse-engineered reasoning can rival large proprietary models without RL or teacher supervision.

### Strengths
1. Scalable and Cost-Effective Framework: REER eliminates the need for teacher models or reward functions by using a gradient-free local search guided by perplexity, enabling large-scale, low-cost synthesis of deep reasoning trajectories.
2. High-Quality Public Dataset: The released DeepWriting-20K dataset of 20 K structured reasoning trajectories across 25 domains is a valuable open resource for future research on reasoning and creative generation.
3. Strong Empirical Results: The proposed DeepWriter-8B model achieves near-proprietary-level performance on LongBench, HelloBench, and WritingBench, significantly outperforming all open-source baselines.
4. Extending Deep Reasoning to Non-Verifiable Tasks: REER successfully brings deep reasoning to open-ended and creative domains (e.g., writing), bridging the gap between verifiable reasoning (math, code) and subjective generation tasks.

### Weaknesses
1. Subjective Evaluation via LLM Judges: Benchmarks rely heavily on GPT-4o or Claude-3.7 as evaluators, which introduces bias and limits reproducibility; no human or standardized blind evaluation is included.
2. Scalability and Efficiency Trade-offs: The iterative local search for trajectory refinement can be computationally expensive for large datasets, and its efficiency compared to distillation or RL remains under-analyzed.

### Questions
1. Under what conditions does REER fail to find meaningful reasoning trajectories (e.g., nonsensical outputs, repetitive loops), and how are such cases handled?
2. How would REER behave when applied to larger base models or multimodal datasets—would it continue to scale effectively or face diminishing returns?

### Soundness
3

### Presentation
3

### Contribution
3
