# Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Unlearning in large language models (LLMs) involves precisely removing specific information from a pre-trained model. This is crucial to ensure safety of LLMs by deleting private data or harmful knowledge acquired during pre-training. However, existing unlearning methods often fall short when subjected to thorough evaluation. To overcome this, we introduce JensUn, where we leverage the Jensen-Shannon Divergence as the training objective for both forget and retain sets for more stable and effective unlearning dynamics compared to commonly used loss functions.  
In extensive experiments, JensUn achieves better forget-utility trade-off than competing methods, and even demonstrates strong resilience to benign relearning. Additionally, for a precise unlearning evaluation, we introduce LKF, a curated dataset of lesser-known facts that provides a realistic unlearning scenario. Finally, to comprehensively test unlearning methods, we propose (i) employing an LLM as semantic judge instead of the standard ROUGE score, and (ii) using worst-case unlearning evaluation over various paraphrases and input formats. Our improved evaluation framework reveals that many existing methods are less effective than previously thought.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces JensUn, a new LLM unlearning method that uses Jensen–Shannon Divergence to achieve a better balance between forgetting targeted knowledge and preserving overall model utility. The authors show that JensUn is more stable and resistant to benign relearning, meaning the forgotten information is less likely to re-emerge after additional fine-tuning. To address weaknesses in current evaluation practices, they propose a stronger assessment pipeline that uses LLM-based semantic judging instead of ROUGE, tests multiple paraphrased query versions, and reports worst-case performance. They also introduce a new dataset, LKF (Lesser Known Facts), for more realistic unlearning scenarios. Overall, the paper argues that many existing unlearning methods appear effective only under weak evaluations, while JensUn demonstrates more durable and robust forgetting.

### Strengths
1. The motivation for introducing a new unlearning method is strong.

2. The paper proposes a new evaluation approach using an LLM as a judge.

3. The experimental section is comprehensive and includes various evaluation metrics.

### Weaknesses
1. I do not see the novelty of the proposed unlearning method. The paper introduces JensUn, which uses JSD for the forget and retain losses; however, this design choice appears to be based only on preliminary experiments, and there is no theoretical justification provided.

2. The writing of the paper can be significantly improved. For example, the Introduction lacks a concluding paragraph, making it difficult for readers to clearly grasp the main contributions. Additionally, based on the writing, I assumed that the LKF dataset is one of the primary contributions, yet it is only presented as a subsection in the paper, with significantly less description compared to the introduction of evaluation metrics, which are not core technical contributions.

### Questions
Could you provide theoretical analysis to explain why JSD is a better choice for the unlearning objective?

### Soundness
2

### Presentation
1

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
This paper proposes JensUn, an LLM unlearning method that replaces the existing loss functions with Jensen-Shannon divergence (JSD). Specifically, on the forget set, the authors construct a target answer, such as refusal, and minimize the JSD between the model's output distribution with the target answer distribution. On the retain set, the authors minimize the JSD with the reference model's output. Moreover, the authors propose several improvements in evaluating an unlearned model, such as using the worst-case performance across multiple paraphrases of the question, and adding retain set examples in context. Experiments on two datasets show that the proposed method outperforms strong baselines.

### Strengths
1. The proposed method is simple, and experiments show that it outperforms the strong baselines, achieving a better trade-off between unlearn and retain performance.
2. The authors propose several improvements for evaluating unlearned models. These metrics contribute to a more robust unlearning evaluation.

### Weaknesses
1. My main concern is the lack of a comprehensive ablation study. Particularly, the authors do not show that it is the JSD that improves the performance. For example, since the authors construct a refusal target for the forget data, what's the performance of applying other loss functions (maximize loglikelihood, preference optimization, or minimize KL) on the same refusal target? On the retain set, is JSD necessary, or is KL enough?

2. Although the proposed metrics contribute to a more robust evaluation, these metrics are already introduced and used in prior works. For example, LLM-as-judge to evaluate the answer equivalence is widely used. Particularly for unlearning, paraphrases have been used [1], and worst-case performance under adversarial attacks has also been explored [2-4].

[1] Wang et al., Towards Effective Evaluations and Comparisons for LLM Unlearning Methods.
[2] Lynch et al., Eight Methods to Evaluate Robust Unlearning in LLMs.
[3] Schwinn et al., Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space.
[4] Liu et al., Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective.

### Questions
1. Can the authors provide more explanations for why JSD is better than other loss functions? For example, how's the gradient different and more stable than other methods?

### Soundness
3

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
3

### Summary
This paper proposes a new unlearning loss to improve the utility preserving in LLM unlearning, which leverages Jensen-Shannon Divergence, that optimizes the unlearning LLM to match the distribution of refusing words like "No idea" on forget set. It further introduces a new unlearning evaluation dataset LKF focusing on lesser-known facts in web data. The experiments show that it improves the utility preservation while having matchable forget performance.

### Strengths
* The idea is simple and straightforward to using the Jenson-Shannon Divergence as a distribution match loss for LLM unlearning purpose.
* The writing is clear and easy to follow.
* LKF dataset on lesser-known facts may be a good contribution to the field.

### Weaknesses
* Unconvincing performance gain. While the proposed JS divergence loss is easy to understand, I don't have the intuition that it is better than other unlearning forget loss, like those optimizing the LLM to output refusing words like "I don't know" on forget data. The JS divergence seems yet another loss pushing the LLM to output refusing words. And the performance in Table 1/2 also looks contradicted to me, where the utility preserving seems not so different from other KL divergence method, but much achieves higher forget performance compared to other methods like NPO/SimNPO.
* Inadequate ablation on retain loss selection. The retain loss in the proposed framework is also the JSD loss. However, there is no ablation study on how it performs when combining with other LLM forget loss.

### Questions
* I feel confused about some of the performance result in Table 1. The retain accuracy for SimNPO is 84.2, largely outperforms other baselines and even the original model, why is this case?
* From the example responses in Figure 15, JensUn unlearn response is repeated No idea, which is a bad-quality response, and it seems to contradict with the repetitive performance in main paper.

### Soundness
3

### Presentation
2

### Contribution
2
