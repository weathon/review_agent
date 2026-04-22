# One Sample to Rule Them All: Extreme Data Efficiency in RL Scaling

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 4, 4, 2

## Abstract
The reasoning ability of large language models (LLMs) can be unleashed with reinforcement learning (RL) [OpenAI, 2024, DeepSeek-AI et al., 2025a, Zeng et al.,2025]. The success of existing RL attempts in LLMs usually relies on high-quality samples of thousands or beyond. In this paper, we challenge fundamental assumptions about data requirements in RL for LLMs by demonstrating the remarkable effectiveness of one-shot learning. Specifically, we introduce polymath learning, a framework for designing one training sample that elicits multidisciplinary impact. We present three key findings: (1) A single, strategically selected math reasoning sample can produce significant performance improvements across multiple domains, including physics, chemistry, and biology with RL; (2) The math skills salient to reasoning suggest the characteristics of the optimal polymath sample; and (3) An engineered synthetic sample that integrates elements from multiple subjects outperforms training with individual samples that naturally occur. Our approach achieves superior performance to training with larger datasets across various reasoning benchmarks, demonstrating that sample quality and design, rather than quantity, may be the key to unlock enhanced reasoning capabilities in language models. Our results suggest a shift, dubbed as sample engineering, toward precision engineering of training samples rather than simply increasing data volume.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
This paper proposes a method for generating a single math sample that, when post-trained with RL on, improves performance across other domains like biology, chemistry.

### Strengths
The method proposed by the others does increase performance across other domains as shown by Figure 1.

### Weaknesses
While the work of this paper is not my area of expertise, I find several issues with the current work. First of all, it is quite dense and hard to read. The authors keep referring to LIMR throughout the paper, and claim that they select the samples with the lower LIMR scores (line 162), but this method is never explained or introduced in the current manuscript, hence I am unsure what exactly this means.

Second, as far as I can understand, the authors use "salient skill identification" to construct their samples, but I am not sure how exactly they define the skills. Could the authors explain more here? It seems to be that there is an LLM employed to identify the skills needed to solve a particular problem, but how are these skills defined?

Thirdly, the experimental setup makes no mention of hyperparameter tuning. Have the authors chosen a fixed set of hyperparams without sweeping? If so, how was this decision made?

Lastly, I am not sure I understand exactly what the authors do once they generate the sample. Lines 204-205 mention that they use GRPO and "augment the polymath sample into the batch of 128" - I thought the point of this work was to train on a single sample, generated using their framework. If so, what are the other 127 samples in the batch?

### Questions
I have posed my questions in the weaknesses section.

### Soundness
3

### Presentation
1

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
This paper shows that using one single training example (polymath sample or synthetic data), can improve Qwen2.5-7B on multiple domains beyond only domain, which shows better cross-domain transfer capability than common math dataset like MATH and LIMR.

### Strengths
1. The method is simple and effective; the results are good and reasonable. If RLVR on a single-domain (MATH) dataset brings limited gains or even causes forgetting in other domains, fewer-example RLVR may perform better for cross-domain improvement.
2. The writing and pipeline are clear, and the evaluation covers a comprehensive set of categories.
3. Also mention that training-free in-context learning can yield some improvement.

### Weaknesses
1. As mentioned in [1], the results should not be reported only on the Qwen2.5-7B model. Other one-shot RLVR–related works [2,3,4] also consider other models, like Llama-3 3B or 8B, and maybe other SFT models like OpenThinker3-1.5B. It’s not necessary to beat RLVR with MATH/LIMR on all models, but we should at least see significant improvement from using polymath/synthetic data. I also wonder whether the data transfer to other models, or whether we have to select data for each different model.
2. How does the selected data compare to the training data used in previous work [2,3,4]? I think it’s important to show the advantage of the data-construction pipeline in the paper by comparing with them.
3. How did you get the 1,500 random samples from SuperGPQA? I note that Qwen2.5-7B can get about 25–28% overall performance on the SuperGPQA benchmark, but only 15.7% in your report. Is this mainly affected by the prompt, the selected subset, or are they from the hard part? Similar issues may exist in GPQA Diamond. Although it’s fine to compare under the same evaluation pipeline, the gap is too large and needs explanation.

I think these questions are critical, and would like to increase my score if they are fixed.


[1] Shao, Rulin, et al. "Spurious rewards: Rethinking training signals in rlvr." arXiv preprint arXiv:2506.10947 (2025).
[2] Wang, Yubo, et al. "Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem." arXiv preprint arXiv:2506.03295 (2025).
[3] Wang, Yiping, et al. "Reinforcement learning for reasoning in large language models with one training example." arXiv preprint arXiv:2504.20571 (2025).
[4] Gao Z, Chen L, Luo H, et al. One-shot entropy minimization[J]. arXiv preprint arXiv:2505.20282, 2025.

### Questions
Are you using the same compute for MATH/LIMR training and for one-shot training? Would a larger dataset require more compute to converge? Maybe we should include tables/figures showing accuracy vs. training steps to verify whether the results are converging.

### Soundness
2

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
2

### Summary
This paper introduces "polymath learning," demonstrating that a single, strategically selected math reasoning sample can improve LLM performance across diverse domains (physics, chemistry, biology) through reinforcement learning, often outperforming training on thousands of samples. The authors find that optimal samples exhibit salient algebra and precalculus skills, and that synthetically engineered problems integrating multidisciplinary knowledge achieve the best results, suggesting a shift from data scaling to precision "sample engineering" for more efficient reasoning enhancement.

### Strengths
1. Important research problem.
The paper proposes a meaningful problem to study: the data efficiency in current RL scaling for LLMs. The paper proposes an attemptive method to deal with the problem, which sheds some light on this important direction.

2. Clear Writing.
The writing is easy to follow, and the methods and experiments are clearly presented.

### Weaknesses
1. Unreasonably Low Math500 Performance after GRPO Training. 
After GRPO fine-tuning, the Qwen2.5-7B model achieves only 37.2 accuracy on Math500, which is far below the expected score. This discrepancy raises concerns about the validity of the experiment results.

2. Lack of Robustness Verification for the Proposed Method
The paper does not provide sufficient evidence of the robustness and statistical reliability of the proposed sample selection method. A convincing validation would require multiple repeated experiments (e.g., 100 independent trials) and report the mean and variance of the final performance.

### Questions
1. Unsubstantiated Claim about Low-LIMR Preference
The authors argue that “high LIMR samples lead to over-specialization in mathematics” and therefore choose low-score (≈0.6) LIMR samples, yet provide no controlled experiments to establish a causal relationship or robustness. 

2. Limited Model Diversity and Poor External Validity
Training and evaluation are both performed exclusively on Qwen2.5-7B-Base. The study does not examine how results generalize across different model sizes or architectures. This narrow setup restricts the external validity and robustness of the conclusions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether reinforcement learning on a single training sample can enhance reasoning in large language models across multiple domains. The authors introduce Polymath Learning, a framework for one-shot RL training designed to extract cross-domain reasoning improvements from a single carefully chosen math problem.

### Strengths
1. This paper shows that RL can be extremely efficient on one single example.
2. This paper demonstrates RL on a well-designed cross-domain reasoning problem can improve model's reasoning performance across domains.

### Weaknesses
1. The evaluation is done on a randomly sampled subset (100 problems for each subject) from multiple benchmarks, this seems not a standard evaluation and makes it hard to compare the experimental results with prior works.
2. Only one Qwen base model is tested, it's unclear how the method generalizes to other model families.
3. The sample selection relies on the LIMR score, which actually requires a complete RL training on the full dataset. Therefore, the proposed method is not as efficient as it claims to be "one-shot RL".

### Questions
1. Can the authors provide a principled explanation on why one-shot RL can be better than RL on the whole dataset?
2. Can the authors include the reward curve across training steps (e.g., mean reward) to illustrate how the model’s behavior evolves during one-shot training?
3. Are the results reported in Table 3 statistically significant? For example, are they at least beyond the 2-sigma or 95% confidence level?
4. Can the authors clarify what insight Figure 2 is intended to convey?  
5. Why does RL on a single training example not lead to severe overfitting? Recent works (e.g., [1], [2]) show that reinforcing correct samples can reduce output entropy and lead to overconfidence, and that RL often refines a model’s prior knowledge rather than improving its intrinsic capabilities. Could the observed one-shot improvements arise from implicit refinement of pre-existing knowledge rather than genuine reasoning enhancement?

[1] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning. NeurIPS 2025

[2] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? NeurIPS 2025

### Soundness
1

### Presentation
2

### Contribution
1
