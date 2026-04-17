# Learning to Reason for Hallucination Span Detection

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Large language models (LLMs) often generate hallucinations---unsupported content that undermines reliability. While most prior works frame hallucination detection as a binary task, many real-world applications require identifying hallucinated spans, which is a multi-step decision making process.
This naturally raises the question of whether explicit reasoning can help the complex task of detecting hallucination spans. 
To answer this question, we first evaluate pretrained models with and without Chain-of-Thought (CoT) reasoning, and show that  CoT reasoning has the potential to generate at least one correct answer when sampled multiple times. Motivated by this, we propose RL4HS, a reinforcement learning framework that incentivizes reasoning with a span-level reward function. 
RL4HS builds on Group Relative Policy Optimization and introduces Class-Aware Policy Optimization to mitigate reward imbalance issue. Experiments on the RAGTruth benchmark (summarization, question answering, data-to-text) show that RL4HS surpasses pretrained reasoning models and supervised fine-tuning, demonstrating the necessity of reinforcement learning with span-level rewards for detecting hallucination spans.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses hallucination span detection in large language models (LLMs), moving beyond binary detection to identify unsupported spans within generated text. The authors propose RL4HS, a reinforcement learning framework that leverages Chain-of-Thought (CoT) reasoning and introduces Class-Aware Policy Optimization (CAPO) to mitigate reward imbalance. Experiments on the RAGTruth benchmark (summarization, QA, data-to-text) show RL4HS outperforms both supervised fine-tuning and existing reasoning models, demonstrating the necessity of span-level rewards and in-domain reasoning for accurate hallucination detection.

### Strengths
- Novelty: First to train a reasoning-based hallucination span detector using RL with span-level rewards, addressing a gap in prior work focused on binary detection.
- Technical Contribution: CAPO effectively balances precision and recall, overcoming reward hacking issues found in standard GRPO.
- Empirical Rigor: Extensive experiments on RAGTruth across multiple tasks, withcomparisons to strong baselines (SFT, multi-view attention, proprietary reasoning models).
- Insightful Analysis: Ablation studies and case analysis convincingly show the benefits of in-domain reasoning and span-level reward optimization.

### Weaknesses
- Generality: Evaluation is limited to RAGTruth and a few CNLG tasks; broader applicability to other domains or real-world LLM outputs is not demonstrated.
- Model Scale: While RL4HS outperforms larger models, results for very large-scale proprietary models (e.g., GPT-5) are not fully explored.
- Complexity: The RL training setup (GRPO, CAPO) adds complexity and may be challenging to reproduce or deploy in production settings.
- Limited Error Analysis: The paper could benefit from deeper qualitative analysis of failure cases and limitations of RL4HS, especially in ambiguous or noisy contexts.
- Data Requirements: Reliance on span-level annotated data (RAGTruth) may limit adoption, as such datasets are rare.

### Questions
- Generalization: How does RL4HS perform on hallucination detection in domains outside RAGTruth (e.g., medical, legal, conversational AI)?
- Annotation Efficiency: Can RL4HS be adapted to settings with limited or noisy span-level annotations? Is weak supervision feasible?
- Deployment: What are the computational and practical challenges for deploying RL4HS in real-world LLM pipelines?
- Failure Modes: What types of hallucinations or contexts remain challenging for RL4HS? Any observed systematic errors?
- Comparison to Post-hoc Methods: How does RL4HS compare to post-hoc hallucination correction or filtering approaches in terms of accuracy and efficiency?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes RL4HS, a reinforcement learning framework for hallucination span detection in large language models. It builds upon Group Relative Policy Optimization (GRPO) and introduces Class-Aware Policy Optimization (CAPO) to handle class imbalance in rewards. The authors claim that RL4HS enables better reasoning for hallucination localization and outperforms supervised fine-tuning and prior reasoning-based baselines on the RAGTruth dataset. The study attempts to connect reasoning, reinforcement learning, and span-level detection but provides limited novelty beyond adapting existing RL techniques.

### Strengths
The topic—fine-grained hallucination detection—is relevant and timely, given the increasing importance of factual reliability in LLMs. The paper is well-organized, and the experimental setup is systematically described. The inclusion of span-level reward signals and the effort to analyze precision–recall imbalance show some awareness of practical issues in training reasoning-based detection models. The authors also provide qualitative examples to illustrate how reasoning might enhance model behavior.

### Weaknesses
Despite a clear structure, the work suffers from conceptual and methodological shallowness. The proposed RL4HS framework merely repackages existing GRPO methodology with a minor weighting adjustment, which can hardly be considered a significant algorithmic contribution. The claim that RL improves reasoning for hallucination span detection is weakly justified—there is no convincing evidence that the “reasoning” is genuinely learned rather than memorized through reward shaping. Experiments are conducted on a single dataset (RAGTruth), which raises concerns about generalizability. Furthermore, the comparison with GPT and Qwen models seems superficial and lacks control over model size, data exposure, and inference strategies. Many of the reported gains are marginal and could be attributed to overfitting or differences in fine-tuning procedures rather than the proposed RL method. The discussion of “in-domain reasoning” is vague and not theoretically supported. Overall, the paper feels more like an engineering report than a principled research contribution.

### Questions
What specific novelty does RL4HS offer over GRPO beyond the scaling factor (CAPO)? Why is this sufficient for publication in a top-tier conference?

How is “reasoning” objectively measured or verified in this work? Are the CoT traces evaluated for correctness or interpretability?

Given that results rely solely on RAGTruth, can the authors demonstrate performance on unseen domains or datasets?

How are baseline models such as GPT-5 or Qwen3 controlled for fair comparison—are they fine-tuned under identical conditions?

The reported F1 improvements seem small; can the authors provide statistical significance tests or ablations to confirm that these are not due to random variation?

Could similar gains be achieved simply with improved supervised training or calibration, without the complexity of reinforcement learning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors reframe hallucination detection as a reinforcement learning problem instead of a simple classification one. The authors first show that using Chain-of-Thought (CoT) reasoning before predicting hallucinated spans makes the model’s predictions more diverse — so when decoding multiple times (say, K attempts), at least one prediction tends to be correct. Building on this, they design RL4HS, which fine-tunes a model using Group Relative Policy Optimization (GRPO) with a span-level F1 reward to explicitly encourage reasoning that improves hallucination span localization. However, they find that if the input has no hallucination, directly giving reward = 1.0 biases the model toward always predicting “no hallucination.” To fix this, they propose Class-Aware Policy Optimization (CAPO), which scales the non-hallucination advantages (by 0.5) to balance rewards between classes. They also confirm that simple fixes like Dr.GRPO cannot solve this imbalance. Experiments on RAGTruth (covering summarization, QA, and data-to-text) show that RL4HS significantly outperforms supervised fine-tuning and existing reasoning models, proving that reinforcement learning with span-level rewards and in-domain reasoning is essential for robust hallucination detection

### Strengths
- The paper offers a novel reformulation of hallucination span detection as a reinforcement learning problem, which is conceptually original and well-motivated.
- It carefully designs a span-level reward and introduces class-aware scaling to prevent reward hacking, showing thoughtful methodological innovation.
- The motivation analysis with Span-F1@K clearly illustrates the benefit of reasoning diversity and provides strong empirical grounding.
- Experimental results demonstrate substantial improvements over both supervised and reasoning baselines, highlighting the method’s effectiveness and significance.
- The paper is clearly written and logically structured, making complex ideas easy to follow.

### Weaknesses
- The approach mainly focuses on the RAGTruth benchmark; it remains unclear how well it generalizes to other OOD data. But indeed the author shows good transferability among different three subsets under RAGTruth, using and holdout setting.
- It could be beneficial to analyze and categorize what kinds of strategy the RL4HS model uses for producing more accurate hallucination span detection, by some human evaluation on the CoT paths.
- Here are only one qualitative study examples shown in the paper. It would be good if the authors can provide more in the appendix.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose RL4HS, a reinforcement learning framework that designed for hallucination span detection. The authors first conduct experiments comparing the pass@k performance, demonstrating that incorporating reasoning can be beneficial for span detection. Then, they train the model using GRPO with a span-level reward function. To address reward hacking issues caused by the imbalanced advantages between hallucination and non-hallucination cases, this work further introduce Class-Aware Policy Optimization (CAPO), which adjusts the advantages for non-hallucination predictions.

### Strengths
1. The paper presents a systematic scaling analysis, highlighting the potential of reasoning in improving hallucination span detection. These findings and insights could be valuable for guiding future research.

2. The study clearly identifies and analyzes the reward hacking issue caused by the imbalanced reward designs, and the proposed CAPO method offers a effective solutions.

### Weaknesses
1. Experiments are conducted exclusively on RAGTruth. It is unclear whether the proposed method generalizes to other hallucination datasets. There are other hallucination detection benchmarks with span-level annotations, such as FAVA, that could be included for a more comprehensive evaluation.

2. Although the authors compare to several reasoning and proprietary models, hallucination-specific baselines is limited.

3. The paper does not report case-level (or binary-level) results, so it remains unclear whether the span-level reward leads to consistent gains in overall performance.

4. The SFT baseline seems suboptimal relative to the original RAGTruth paper, possibly due to inappropriate learning rates (1e-6 in appendix) or hyperparameters, which may underestimate supervised performance.

### Questions
See above section

### Soundness
2

### Presentation
3

### Contribution
2
