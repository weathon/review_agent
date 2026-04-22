# Tagging the Thought: Unlocking Personalization Reasoning via Reinforcement Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Recent advancements have endowed Large Language Models (LLMs) with impressive general reasoning capabilities, yet they often struggle with personalization reasoning—the crucial ability to analyze user history, infer unique preferences, and generate tailored responses. To address this limitation, we introduce \textbf{TagPR}, a novel training framework that significantly enhances an LLM's intrinsic capacity for personalization reasoning through a ''tagging the thought'' approach. Our method first develops a data-driven pipeline to automatically generate and semantically label reasoning chains, creating a structured dataset that fosters interpretable reasoning. We then propose a synergistic training strategy that begins with Supervised Fine-Tuning (SFT) on this tagged data to establish foundational reasoning patterns, followed by a multi-stage reinforcement learning (RL) process. This RL phase is guided by a unique composite reward signal, which integrates tag-based constraints and a novel Personalization Reward Model with User Embeddings (PRMU) to achieve fine-grained alignment with user-specific logic. Extensive experiments on the public LaMP benchmark and a self-constructed dataset demonstrate that our approach achieves state-of-the-art results, delivering an average improvement of 32.65\% over the base model across all tasks. Our work validates that structured, interpretable reasoning is a highly effective pathway to unlocking genuine personalization capabilities in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a personalized reasoning training framework called TagPR to address the limitations of existing LLMs in personalized reasoning. Its core idea is to enable the model to generate explicit and interpretable reasoning steps through a "tagging the thought" approach. This approach involves automatically constructing a semantically labeled reasoning chain dataset, a collaborative training strategy, and a multi-stage reinforcement learning process. Finally, the effectiveness of the proposed method is demonstrated on the LaMP dataset.

### Strengths
1. Unlike implicit reasoning, LLM provides structured and interpretable reasoning, which is valuable.
2. The paper is well-written, clearly describing the complete reasoning chain construction and training process.
3. Extensive experiments are conducted, and the results show that the proposed strategy appears to be very effective. Comprehensive ablation experiments are also provided.

### Weaknesses
1. Equation 5 is strange; the first term is a composite, while the rest are separate. This requires further discussion.
2. The effectiveness of the proposed strategy is only verified on the LaMP dataset (constructing the reasoning chain data and training the model), and its effectiveness on other datasets is unknown (this does not refer to zero-shot generalization).
3. The rationality of the inference process is unclear. Specifically, it is unclear how the distribution of inference labels (Figure 5, right) indicates the rationality of the cognitive process, nor how Figure 6 demonstrates the correctness of the inference model. Furthermore, more case studies are recommended.
4. The proposed training framework has a large number of parameters, and their selection is unclear. Furthermore, sensitivity analysis is lacking.
5. Mathematical notation needs to be standardized. For example, $\theta$ is used in various models, and loss is represented in two ways: $\mathcal{L}$ and $\mathcal{J}$.

### Questions
1. The reward function requires further discussion.
2. Does the proposed framework still work on other datasets?
3. How can we prove the rationality of the reasoning chain, that is, the reasoning that is faithful to the model?
4. How are the numerous hyperparameters selected?

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
2

### Summary
This paper introduces TagPR, a novel framework that enhances LLMs ability for personalization reasoning. TagPR adopts a “tagging the thought” approach that first builds a structured dataset by automatically labeling reasoning chains, then trains the model through Supervised Fine-Tuning and multi-stage Reinforcement Learning. The reinforcement process is guided by a composite reward combining tag-based constraints and a Personalization Reward Model with User Embeddings (PRMU) to align reasoning with user-specific logic. Experiments on public and self-constructed datasets show that TagPR achieves state-of-the-art performance.

### Strengths
1. The authors have developed a new dataset designed to advance future studies on LLM reasoning, providing valuable resources for the research community.
2. The two-stage approach with SFT + RL is technically sound.

### Weaknesses
1. I am not certain about the motivation of the paper which uses tagging to address personalization. Why would this be an effective approach / direction?
2. Evaluation protocols are not clearly mentioned. For example, how is the accuracy and F1 calculated.

### Questions
1. On what basis were the reward weights (α, β, γ) chosen in Eq (5), and how sensitive is the model’s performance to these values?
2. How does the model maintain personalization fidelity and reasoning consistency when the personalization and tag rewards are removed during exploratory RL?
3. I am not very familiar with the benchmark and got quite confused about how the evaluation is done once the LLM is trained. Is it using logprob or generations from LLMs?

### Soundness
3

### Presentation
2

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
This paper addresses the limitation of large language models (LLMs) in personalization reasoning and proposes TagPR, a framework that uses a data-driven pipeline to create tagged reasoning chains and a synergistic training strategy (SFT + multi-stage RL). It demonstrates TagPR achieves state-of-the-art results on the LaMP benchmark and a self-constructed Dianping benchmark, outperforming base models and even larger proprietary LLMs while showing strong generalization.

### Strengths
1. The trained model enhances LLMs’ personalization reasoning, achieving state-of-the-art performance on LaMP benchmark.
2. The proposed training framework (SFT + multi-stage RL) with sophisticated reward design is insightful for relevant tasks.

### Weaknesses
1. The framework relies on high-quality initial reasoning chains generated by Qwen3 Thinking model and GPT-4o, inducing huge cost on constructing data pipelines.
2. The two-stage RL training (especially the guided phase with multiple reward components) involves more tuning, increasing implementation and reproduction difficulties. 
3. The training data and evaluation data are mainly related to the LaMP benchmark, regardless of the private Dianping benchmark, making the generalization capability of this model on other related benchmark questionable.

### Questions
1. Could you elaborate on how the choice of the 9 primary tags in the tagged reasoning chains dataset was determined?
2. Will the method benefit more from more careful choices on the reward composition? 
3. How well does the PRMU improve the model performance?

### Soundness
3

### Presentation
3

### Contribution
2
