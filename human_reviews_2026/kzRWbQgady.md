# Rethinking RL Evaluation: Can Benchmarks Truly Reveal Failures of RL Methods?

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Current benchmarks are inadequate for evaluating progress in reinforcement learning (RL) for large language models (LLMs).Despite recent benchmark gains reported for RL, we find that training on these benchmarks' training sets achieves nearly the same performance as training directly on the test sets, suggesting that the benchmarks cannot reliably separate further progress.To study this phenomenon, we introduce a diagnostic suite and the Oracle Performance Gap (OPG) metric that quantifies the performance difference between training on the train split versus the test split of a benchmark. We further analyze this phenomenon with stress tests and find that, despite strong benchmark scores, existing RL methods struggle to generalize across distribution shifts, varying levels of difficulty, and counterfactual scenarios: shortcomings that current benchmarks fail to reveal.We conclude that current benchmarks are insufficient for evaluating generalization and propose three core principles for designing more faithful benchmarks: sufficient difficulty, balanced evaluation, and distributional robustness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers the problem of designing benchmarks for reinforcement learning post-training of LLMs. Using two base models from the Qwen family, it shows experimentally on four datasets that the performance is similar when trained on the training set and test set (characterized by a proposed metric called Oracle Performance Gap), suggesting that current experimental protocols are not sufficient. With more fine-grained experiments, the paper shows that 1) performance should be stratified by task difficulty as average metrics can be misleading, 2) performance on out-of-distribution tasks can become worse after post-training and thus should be reported, and 3) benchmarks should test for counterfactual reasoning, as models can default to reciting memorized knowledge.

### Strengths
Overall, the paper is well-written and provides the reader with motivation and recommends a clear course of action. Both the text and experiments are organized clearly and the results are presented with the hypothesis/conclusion paradigm that made the paper easy to understand. The design of the experiments seem to be sound.

### Weaknesses
To make the experimental results more strongly support the conclusions, standard errors should be included, as they are critical for rigorous statistical testing of hypotheses. 

Another weakness is that the conclusions provided in this paper have already been known elsewhere in the machine learning literature to some extent. The fact that the average performance metric may be misleading is Simpson's Paradox [1], leading to the development of metrics like precision and recall for classification. The importance of testing out-of-distribution generalization is known in reinforcement learning [2]. There is a benchmark for testing LLM's ability for causal inference, which is important in real-world applications [3]. This paper shows that these same phenomena occur in RL-based post-training for LLMs, which is interesting but not really unexpected.

[1] https://en.wikipedia.org/wiki/Simpson%27s_paradox
[2] Packer et al. "Assessing Generalization in Deep Reinforcement Learning", arXiv preprint arXiv:1810.12282.
[3] Du et al. "Ice Cream Doesn’t Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference", arXiv preprint arXiv:2505.13770.

### Questions
- How were the hyperparameters selected?
- Is there a way to interpret the semantic clusters described in section 3.2.1? Do they correspond to certain topics?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to improve the scientific rigour of benchmarks, primarily by measuring how well RL-finetuning of language models generalises out of distribution. It introduces three methods for creating distributional shifts: splitting up existing datasets labeled by difficulty, clustering data points by their vector representations and creating new datasets by e.g. introducing new axioms in maths. The paper also makes a number of recommendations for how future benchmarks should be constructed.

### Strengths
The paper demonstrates a few interesting patterns:

**SFT shows less performant train-test generalisation than RL-finetuning** 
SFTed models show substantial test-set-performance differences when trained on training vs test splits (MATH: 17-24% on test when trained on train set; 40-64% when trained on test set and GSM8K: 17-20% vs 68-79%). In contrast,  RL finetuned models do not show any substantial difference (MATH: 64.2% vs 64.4% (3B) and 74.8% vs 74.4% (7B); GSM8K: 87.5% vs 88.9% (3B) and 91.1% vs 91.8% (7B)).

**RL finetuned models can fail to generalise between semantically cluster data subsets** 
When datasets are split using semantic clustering rather than I.I.D, RL-fine tuning on one cluster can actually slightly decrease performance on another (although this effect size is small and  its unclear if these results are statistically significant).

I believe some of the contributions of this paper, for example the semantic clustering methodology, are somewhat interesting and novel.

### Weaknesses
Core to the contributions of this paper are new results, utilising new methods of analysis on existing benchmarks. However, I am uncertain about the novelty and clarity of the communication of these results; and am particularly concerned about misleading presentation. The experiments also lack statistical rigour.

### “Novel Diagnostic Framework.”
The paper claims a core contribution of the paper is its “Novel Diagnostic Framework”. However, I find the aggregate statistics that are introduced to be not particularly novel, and potentially misleading. Presenting the raw data would arguably have been less confusing and less likely to mislead readers.

#### I think the 'Average Cross-Difficulty Generalization' metric may be misleading.
If I am understanding Figure 3 correctly, the 'Average Cross-Difficulty Generalization' metric excludes each model's own training level from the average. This means the L1-trained model is evaluated on the average of Levels 2-5 (mostly harder problems), while the L5-trained model is evaluated on the average of Levels 1-4 (mostly easier problems). This creates an inherent bias in the comparison - the L5 model appears to 'generalize better' simply because it's being tested on an easier set of problems. The monotonic pattern highlighted in the paper would (as far as I can tell) still appear if the same model was used across all data points. Please could the authors clarify whether I have misunderstood this figure?  

#### The “Oracle Performance Gap” is similar to 
Similarly, the “Oracle Performance Gap” is not particularly helpful relative to just reporting raw numbers on train/test set. Further, I’m not clear on why the “oracle performance gap” is actually different to just reporting the difference between train/test performance: they in fact fine tune on the “test set” and report performance on the test set, but this is effectively just a new train set.

### Missing statistical significance
The paper presents empirical findings without adequate statistical analysis. There are no confidence intervals, significance tests, or multiple seeds reported for key experiments. Statistical tests are essential for a paper aiming to improve evaluation rigour. 

### Additional concerns:
#### Normative framing.
Rather than presenting empirical observations and letting readers evaluate their implications, the authors make prescriptive claims about benchmark design. They introduce "principles" and assert that benchmarks "must" include certain features, without establishing clear criteria for what constitutes good generalization or why their specific recommendations follow from their findings. 
I think Principle 1 does this relatively well: the paper says evaluations “must be balanced and stratified” in order to prevent “masking”. However, it essentially restates the well-established practice of disaggregated reporting. Principle 2 is more problematic, declaring that "faithful" benchmarks must test distributional robustness without justifying why this is necessary. Of course, in many applications, robustness to distributional shifts is important. However, a simple normative claim that distributional robustness should be measured does not add value to the paper.

#### Counterfactual test 
Finally, the counterfactual test reveals an interesting limitation of LLMs but doesn't specifically relate to RL methods or benchmark design.

### Questions
* Have I understood Figure 3 properly? Does the trend persist if you control for the increased difficulty of test sets used for L1 vs L5? 
* Are multiple seeds used for any of the experiments?

### Soundness
2

### Presentation
1

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
This paper challenges the reasonableness of current LLM benchmarks for evaluating RL-based reasoning. The authors find that RL methods achieve near-zero OPG (which is the difference between model trained on test set compared to one trained on train set) and on the other hand, SFT shows large gaps, suggesting that traditional train/test splits may not effectively measure generalization for RL. The paper then proposes 3 tests: difficulty stratification, distribution shift, and counterfactual reasoning.

### Strengths
1. This paper covers an important research direction by critically examining benchmark quality and questioning whether current evaluations truly measure the generalization performance.
2. The paper attempts to assess generalization from multiple angles: difficulty, distribution, counterfactual.
3. The writing is generally clear and the paper is easy to follow.
4. The observation that aggregate scores mask cross-difficulty performance differences (Figure 2) is valuable.

### Weaknesses
1. A major problem that I have with this submission is that this paper assumes low OPG means "benchmark is broken" but never validates this interpretation. Consider: (a) RL might genuinely learn more robust features than SFT, (b) the test set might be easier than assumed, or (c) benchmark saturation is happening for different reasons than claimed. So for me, here's an alternative interpretation:
(1) RL's low OPG might indicate "good" generalization, not benchmark failure
(2) SFT's high OPG might indicate "poor" generalization, not correct behavior
The authors never tested whether RL actually learned robust policies vs memorized patterns. I would like to propose a simple experiment to validate this: train RL on 10% of train set --> evaluate on the remaining 90% of train set vs. test set. If performance on both is similar, that's genuine generalization. The current setup in this submission can't distinguish this.

2. Table 1: DeepScaler shows OPG = -5.07% (oracle *worse* than train) -> this contradicts the hypothesis entirely, yet the authors don't discuss it

3. Lines 183-186: Authors claim to "rule out data leakage" with one sentence, but provide no contamination analysis of Qwen base models


4. Section 3.2.1's entire Distribution Test uses Euclidean distance in t-sne space as "semantic distance" (in appendix D). I think there is a flaw here as t-SNE only preserves local neighborhood structure, not global distances. Hence I feel that conclusions from Table 3 may be unreliable.

5. The authors used a single llm (gemini 2.5 Pro) to annotate difficulty (in appendix B.1) with zero validation, inter-rater agreement, or comparison to existing labels, like MATH benchmark already has difficulty levels 1-5.

6. [minor]: I feel that this paper has limited scope since most of the analysis was done on math domain. Moreover the authors never experimented with a more challenging benchmark (even if it is from math domain, like FrontierMATH).

### Questions
1. How do other RL methods (DPO, PPO, etc.) behave? Is low OPG specific to GRPO or general to offline/online RL? We need to isolate the actual cause i.e. is the algorithm too good that it can easily generalize or is it the benchmark that's quite easy to memorize on?

### Soundness
2

### Presentation
3

### Contribution
2
