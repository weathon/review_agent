# Learn More with Less: Uncertainty Consistency Guided Query Selection for RLVR

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Large Language Models (LLMs) have recently improved mathematical reasoning through Reinforcement Learning with Verifiable Reward (RLVR). However, existing RLVR algorithms require large query budgets, making annotation costly. We investigate whether fewer but more informative queries can yield similar or superior performance, introducing active learning (AL) into RLVR. We identify that classic AL sampling strategies fail to outperform random selection in this setting,  due to ignoring \textbf{objective uncertainty} when only selecting by subjective uncertainty. This work proposes an \textbf{uncertainty consistency} metric to evaluate how well subjective uncertainty aligns with objective uncertainty. In the offline setting, this alignment is measured using the Point-Biserial Correlation Coefficient (PBC). For online training, because of limited sampling and dynamically shifting output distributions, PBC estimation is difficult. Therefore, we introduce a new online variant, computed from normalized advantage and subjective uncertainty.  Theoretically, we prove that the online variant is strictly negatively correlated with offline PBC and supports better sample selection. Experiments show our method consistently outperforms random and classic AL baselines, achieving full-dataset performance while training on only 30\% of the data, effectively reducing the cost of RLVR for reasoning tasks.\footnote{The code is available at \hyperref[https://github.com/yihao-123/uncertainty-consistency]{https://github.com/yihao-123/uncertainty-consistency}.
}

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors are proposing to use Active Learning (AL) to mitigate the large sample requirements of Reinforcement Learning with Verifiable Reward (RLVR) used in the LLM alignment process. 
They find, that samples where the predicted model uncertainty diverges from the actual accuracy on the sample (measured by sampling K outputs and computing the reward) are detremental to model convergence in RLVR.
The authors propose an AL score to find samples with high alignment of uncertainty and accuracy and use this score to sample a subset of the full training data (offline setting).
Since this proposed score is difficult to obtain in the online setting, the authors also propose an approximation of their score and demonstrate its theoretical properties.
Both proposed methods show strong empirical results on 3 models and 2 math datasets.

### Strengths
- Very streamlined description of research gap, related work and proposed method
- Excellent line of reasoning:
	1. Preliminary experiment to show shortcomings of existing solutions to the problem
	2. Starting with a simplyfied setting (offline RL) and showing a principled advantage of the proposed method
	3. Generalizing to realistic settings (online RL) by finding an approximation of their method that has theoretical guarantees
	4. Demonstrating strong performance of the approximation on 3 models and 2 datasets
	5. Providing ablation studies on the additional properties of their method
- Providing believable evidence for the non-standard assumption of "Sample Gradient Orthogonality"
- No additional computational overhead, as sampling K inferences for each x is already part of RLVR

### Weaknesses
- the authors claim "(non-)significant lifts" multiple times (line 78,88,373,381), but do not provide hard evidence for this claim in the form of standard deviations of results, critical difference diagrams or p-tests. We urge the authors to either provide one of these metrics in the appendix, or use a less mathematically loaded term instead of "significant".

### Questions
- Impact of higher gammas (ablation study): Higher gamma values mean a stronger deterrence of predicted negative advantage in Eq. 5. Does this mean, we focus on samples the model knows to improve upon (positive advantage for some output y)? If so, does this not direcetly counteract exploration behaviours in the RL training? Would this be comparable to directly influencing exploration/exploitation ratios in the training?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces the active learning (AL) process with a new acquisition metric into reinforcement learning with verifiable reward (RLVR) for large language models (LLMs). They first confirm the query selection's impact on the stability of the gradients during fine-tuning via RLVL and reveal that the classic AL cannot select the most informative queries as well as random sampling. After assessing the inconsistent between an LLM's output with the lowest probability and a reward model's evaluation of the samples' accuracy effects on the gradient norm, they propose the smallest inconsistent metric $r_{pb}$ by the Point-Biserial Correlation Coefficient (PBC). Moreover, they extend the idea to online training. The experimental results show that the AL with an inconsistent metric could enhance both offline and online training as well as different RL algorithms.

### Strengths
1. The story (writing) is good to easily follow the authors' idea and refresh the utilization of the AL for the emerging field.
2. This paper highlights that the importance of the query selection metric should not only rely on the LLM itself but also require consistency with the reward model's evaluation.

### Weaknesses
1. **My concern about using uncertainty.** After reviewing the Eqs. (2) and (3), IIUC, the definition of the subjective uncertainty is the low average probability of a policy model's responses $\log \pi_\mathrm{ref}(y_{k, t}^{(i)} \vert x^{(i)}, y_{k, <t}^{(i)})$ and the objective uncertainty is the low accuracy of a reward model's evaluation of a model's response, respectively. However, why can we call these two terms uncertainty? For example, if an LLM's response gives a response with lower probability but a 'consistent response' for the same (or similar) prompt $x^{(i)}$, could we still say the sample is uncertain?
2. Follow 1., I feel that the metric of these terms is more like 'difficulty of reasoning the sample $x^{(i)}$', i.e., the degree of the probability that LLM can give the response (reasoning) and get the high reward (correct answer) for a sample. If so, what you check is the consistency between an LLM's output and the reward model's output.
3. **My concern about the comparison with other AL methods.** Follow 2, if the core idea is evaluating a degree of the informative sample requires both LLMs and reward models, the proposed comparison with Entropy, K-center, K-means, and AskLLM might be insufficient, which only considers the LLMs' response and ignores the reward models' evaluation. To strengthen it, I suggest that the authors consider adding an alternative ablation study on the 'uncertainty' of the reward models.
4. **Make a consistent symbol for equations.** For example, Eq. (1) uses $y_i$ but Eq. (3) uses $y_k^{(i)}$, the index of the sample and the index of generations should be differentiated.
5. In Figure 1, the authors present that the inconsistent sample would give high gradient norm dynamics, i.e., these samples would cause gradient instability. However, the (degree of) impact of these gradient instabilities on the final performance is unclear. To highlight the gradient instability would be a significant issue for RLVR, I suggest that the authors provide some illustrations or examples of this issue.

### Questions
1. While you mention that *Because the calculation of $r_{pb}$ relies on a large number of samples $K$, ...* in Sec 4.2, your experimental settings of $K = 8$ in Sec 5.1 seems not large. Could you give stronger motivations for using Online Query Selection? For example, the **Model update** is the key point to address.
2. Following 1., what are the key components in Online Query Selection for addressing sampling distribution shift in **Model update**?

### Soundness
2

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
3

### Summary
This paper investigates query selection strategies for reinforcement learning from vision and reward (RLVR). The authors observe that standard active learning (AL) sampling methods often fail to outperform random selection in this context. To address this, they introduce an uncertainty consistency metric to guide sampling. In the offline setting, they use PBC (policy–behavior consistency) to measure alignment, while for online training—where estimating PBC directly is difficult—they propose a variant based on normalized advantage and subjective uncertainty. The paper also provides a theoretical analysis suggesting a negative correlation between offline and online PBC. Empirically, the proposed approach outperforms both random and classic AL baselines, reaching near–full-dataset performance using only 30% of the data.

### Strengths
1. The problem is well-motivated and relevant to current challenges in RLVR.
2. The paper offers an interesting empirical observation that inconsistent samples can lead to extreme gradients, which explains why standard AL can underperform random sampling.
3. The introduction of two alignment metrics—one for offline and one for online settings—is insightful, and the accompanying theoretical analysis provides some grounding.
4. Experiments are extensive and demonstrate strong results, achieving competitive performance with significantly fewer samples.

### Weaknesses
While the paper is promising, several points could benefit from deeper clarification or justification:

1. The link between sample inconsistency and extreme gradient behavior is intuitively explained but lacks theoretical support or formal analysis.
2. It is unclear why the offline setting cannot also leverage the online metric $r_{pb}^{online}$, which appears to yield stronger performance in experiments.
3. In some cases, training on the full dataset leads to worse results than using only 30% of the data; the paper should provide more discussion or intuition for why this happens.

Typo: In line 451, “Table ??” is not rendering correctly.

### Questions
Please comment/justify the weakness part above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new query selection strategy for Reinforcement Learning with Verifiable Rewards (RLVR) that allows training mathematical reasoning models using far fewer queries, without sacrificing performance. The key insight from the paper is that not all queries are equally informative. Standard active learning methods often pick samples with high subjective uncertainty (e.g., high perplexity), but this fails in RL reasoning because these samples frequently produce unstable or high-variance gradients, which hurts training stability. To address this problem, the paper proposes selecting samples where Subjective uncertainty (model confidence) and Objective uncertainty (whether the answer is correct) are consistent. They define it as “uncertainty-consistent” samples. The Experimental results show that using only 30% of the training data, the model reaches the same or better performance on the reasoning tasks compared to the full datasets RLVR training.

### Strengths
The paper identifies a practical limitation in current RL-based reasoning training pipelines: query selection methods that rely solely on subjective uncertainty (e.g., perplexity) often select examples that are uncertain but uninformative, leading to unstable gradients and inefficient learning. This motivation is clearly articulated and supported by empirical evidence. This insight is both intuitive and impactful—valuable training samples are those where uncertainty meaningfully reflects correctness, rather than those that are merely hard. The theoretical analysis establishing their negative correlation and training benefit is clearly developed and strengthens the contribution.
The method achieves full-dataset performance using only ~30% of the training data, while maintaining or improving generalization on standard math reasoning benchmarks. This is a practically meaningful result, especially given the rising cost of RL-based reasoning training.

### Weaknesses
The consistency metric assumes that the examples used for selection reflect the distributions during RL optimization. If the underlying data distribution shifts over training (which is common in RLVR), the effectiveness of selection may degrade unless the scoring is frequently recomputed.

The evaluation is rather limited to only Math reasoning. For query selection methods, it would be great to draw broader insights on whether the methods can be generalized beyond Math reasoning tasks.

### Questions
Do the uncertainty consistency proposed in this paper still hold in open-ended, multi-step reasoning tasks where correctness is subjective or less binary (e.g., instruction following, safety, dialogue)?

### Soundness
3

### Presentation
3

### Contribution
2
