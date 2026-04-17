# Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Masked diffusion language models (MDLMs) promise fast, non-autoregressive text generation, yet existing samplers, which pick tokens to unmask based on model confidence, ignore interactions when unmasking multiple positions in parallel and effectively reduce to slow, autoregressive behavior. 
We propose the Dilated Unmasking Scheduler (DUS), an inference-only, planner-model-free method that partitions sequence positions into non-adjacent dilated groups and unmasked them in parallel so as to minimize an upper bound on joint entropy gain at each denoising step. By explicitly trading off the number of network calls against generation quality, DUS recovers most of the performance lost under traditional parallel unmasking strategies. 
Across math (GSM8K, MATH500), code (HumanEval, MBPP), general‐knowledge (BBH, MMLU-Pro), and instruction following (IFEval) benchmarks, DUS outperforms confidence‐based planners, without modifying the underlying denoiser, and reveals the true speed-quality frontier of MDLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a fixed schedule for sampling masked diffusion models whereby an exponentially increasing number of tokens are revealed at each step. The tokens are spaced maximally apart to reduce the mutual information between the revealed positions, thus incurring less error when sampling versus a token-by-token sampling approach. The authors demonstrate improved performance per NFE versus choosing token positions to reveal via the common self-confidence metric.

### Strengths
The idea is quite simple and easy to implement so could be immediately used by practitioners to improve performance. The experiments show strong results with across the board improvements on practically relevant large scale models on tasks that I believe ICLR attendees will care about e.g. math and coding benchmarks. The authors include comparisons to a self-confidence planner and to a random position planner in the Appendix, both of which are important ablations.

### Weaknesses
The paper's main weakness is in the presentation of the motivation and theoretical analysis of the dilated schedule. I find the description hard to follow and it lacks a lot of detail. Specifically, I have the following doubts:

- In equation 2 it is never justified why the minimization objective is the joint entropy of the full sequence. The joint entropy is not even defined at this point. I assume the authors refer to the entropy of the denoising distribution but this is not clear. It is also not clear why we would want to minimize the entropy of the denoising distribution and why this leads to better samples. I think the author's may be referring to the expected log-likelihood assigned by the model to the datapoints (i.e. the expectation is taken with respect to the data distribution and the probabilities are assigned by the model, a type of cross entropy). You would then want a schedule that maximizes the expected log-likelihood.
- L239 is unclear when referring to the self-confidence planner. This method will select the k-top positions with the highest model likelihood, however the authors simply write argmax p_D(X_{i_j} | S_t) for j \in \{1, \dots, k\} which is not clear. The argmax needs to be taken over positions not already selected as part of the top k and the indeces over which the argmax is taken should be specified.
- Equation (5) is not specific to the self-confidence planner, in fact it doesn't use any property of the self-confidence planner, it is true for any planner, including the schedule the authors propose. It doesn't make sense to use equation (5) as a justification for the limitations of the self-confidence planner.
- In L325 with the use of the markov property, there should be justification that this is valid for any indexing i_j of the markov chain. The markov property is usually specified for a certain ordering, so there should be justification that it is also valid for any other ordering of the chain.
- L331 is very inexact, with the whole proof hinging on removing an epsilon. If L331 was true, then the author's method would have the same accuracy as token-by-token unmasking due to the conditional entropy being equal to the sum of factorized entropies. This is clearly not the case, see Figure 1(a). Therefore Equation (12) is not capturing the effect the authors want it to. I understand the motivation in that picking far away positions should reduce their correlation and so reduce the error incurred when revealing them together. However the current 'proof' of this is not at all up to the required standard for a conference paper.
- The authors rely on the fast-mixing properties of markov chains for their proof so this property should be investigated for the datasets that the authors test on. In fact, I would be very surprised if this assumption held true for complex datasets such as code for which distant tokens can very much have a high mutual information (e.g. variable definitions and uses of the variable in far apart places in code).
- I think the ablation in A.2 where the authors compare to a random planner is important and should be included in the main text. The authors hand craft a schedule which reveals less tokens at the start and more tokens at the end for this baseline, but this is already well known as a 'cosine' style schedule that performs the best in prior work [1]. The authors should have just used the cosine schedule as the baseline.

Overall, I think the experimental results are strong but the entirety of section 3 is a poor explanation of the method so much so that I don't think I can recommend this paper for acceptance.

References
[1] Simplified and Generalized Masked Diffusion for Discrete Data, Shi et al. https://arxiv.org/pdf/2406.04329

### Questions
If you have a fixed revealing schedule that is always used at inference time, why not train the model using only that schedule as well? There is no point training the model with all of these different possible revealing orderings when at inference time you only ever choose one of them.

Where does the k-1 term come from in equation (11), it doesn't seem to appear anywhere else in Lemma 1.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Dilated Unmasking Scheduler (DUS), an inference-only decoding strategy for masked diffusion language models (MDLMs). The core idea is to decode tokens in dilated groups—nonadjacent positions unmasked in parallel—to balance speed and quality. The authors provide a theoretical justification under a Markov assumption that DUS minimizes joint entropy, and empirically evaluate it on reasoning (GSM8K, MATH500), coding (HumanEval, MBPP), and general-knowledge (BBH, MMLU-Pro) benchmarks using diffusion-based LLMs such as LLaDA, Dream, and DiffuCoder.

### Strengths
The topic—efficient decoding for diffusion-based LLMs—is timely and relevant, especially as diffusion LMs mature.

The proposed method is inference-only, model-agnostic, and easy to integrate, which has potential practical utility.

The paper is well-organized and clearly written, with formal derivations and visualizations supporting the intuition of DUS.

### Weaknesses
- Fundamental limitation unaddressed: token independence in parallel decoding. The paper’s core problem is conceptual. When decoding multiple tokens simultaneously, those positions are inherently independent given the model formulation. Regardless of scheduling or dilation, unmasking multiple tokens at once means the model predicts each without access to the others’ values. DUS does not fundamentally overcome this limitation—it merely spreads out the unmasked tokens in a deterministic pattern.
Thus, the claim that it “minimizes joint entropy” or captures inter-token dependencies is theoretically weak and, in practice, illusory. The independence assumption is the bottleneck of all parallel decoding methods, and this paper does not offer a real mechanism to break it.

- Overstated novelty relative to prior blockwise decoding. The proposed “dilated schedule” is essentially a fixed blockwise unmasking pattern, which has been explored extensively (e.g., LLaDA’s semi-AR decoding, and block diffusion). The difference from prior work is largely heuristic rather than conceptual. Calling it a “new planner” seems overstated; it is better viewed as a deterministic variant of blockwise decoding with pre-specified indices.


Experimental evaluation is superficial

- The experiments are limited to basic benchmarks (GSM8K, MBPP, etc.) 
- Crucially, the authors omit a baseline comparison with vanilla token-by-token diffusion sampling. Without showing the raw performance degradation when applying DUS versus standard sequential decoding, it is impossible to quantify the real trade-off between speed and accuracy.
- Missing ablations: results with different diffusion step counts (NFEs), qualitative outputs, or comparisons to autoregressive baselines (e.g., LLaMA-3 or Qwen) are incomplete.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Dilated Unmasking Scheduler (DUS), an inference-only method for accelerating masked diffusion language models. DUS divides sequence positions into non-adjacent dilated groups and unmasks them in parallel to minimize an upper bound on joint entropy gain at each step. Experiments on math (GSM8K, MATH500), code (HumanEval, MBPP), and reasoning benchmarks (BBH, MMLU-Pro) show that DUS outperforms confidence-based planners under the same compute budget.

### Strengths
1. The method is entirely training-free, requiring no model modification or additional supervision, which makes it simple and practical for accelerating MDLM inference.

2. The authors conduct experiments on a broad and representative set of benchmarks covering mathematical reasoning, code generation, and general knowledge tasks, demonstrating the method’s general applicability.

### Weaknesses
1. **Poor citation formatting.**   
The paper consistently uses `\citet{}` where `\citep{}` should be applied.

2. **Confused mathematical reasoning.**  
   (1) In Lines 160–161, the authors assume that $X$ is a sequence of random variables forming a stationary, ergodic, first-order Markov chain with fast-mixing properties, but provide no explanation of the validity or relevance of this assumption.  
   (2) The paper adopts **minimizing conditional entropy** as the optimization objective but does not justify why this objective is reasonable. Typically, minimizing the KL divergence between data and model distributions, or between the distributions induced by the original slow sampler and the new fast sampler, is more common. The rationale for using conditional entropy minimization needs further clarification.  
   (3) The authors derive that the conditional entropy of the confidence-based sampling method satisfies $\le \sum_{j=1}^k H(X_{i_j}|S_t)$, while that of the proposed DUS method approximately satisfies the same inequality. Even if minimizing conditional entropy is the goal, it is unclear why this theoretical result implies that the proposed method performs better.  
(4) In addition, the definition of $H(X|S_0)$ in Eq. (2) is never introduced. Since the distribution $p_{\theta}(X|S_0)$ is fixed once the model is trained, it is unclear why $H(X|S_0)$ can be treated as an optimizable quantity.

3. **Outdated baseline.**  
   Widely used confidence- or entropy-threshold-based methods can already achieve nearly lossless 2–3× acceleration [1, 2]. However, the authors choose only the earliest confidence-based acceleration strategy, which suffers severe performance degradation even at a 2.7× speedup.

[1] Wu et al. Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding. arXiv 2025.05.

[2] Ben-Hamu et al. Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking. arXiv 2025.05.

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
1

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
The paper introduces Dilated Unmasking Scheduler (DUS) to speed up masked diffusion LMs by revealing non-adjacent tokens in parallel using a coarse-to-fine, dilated pattern (groups roughly double each round). Under a first-order Markov assumption, this fixed schedule aims to minimize joint entropy per step; it’s inference-only (no retraining) and contrasts with learned planners like LLaDA or DDPD. Empirically, DUS completes decoding in O(log N) steps, recovers most quality lost by naive parallelism, and beats confidence-based planners on GSM8K, HumanEval, and MBPP at matched speedups—approaching sequential/AR quality while much faster. Net: a model-agnostic, principled scheduling rule that narrows the parallel–sequential gap.

### Strengths
The proposed dilated unmasking schedule is grounded in an information-theoretic rationale rather than ad-hoc heuristics. By maximizing the distance between concurrently unmasked tokens, DUS minimizes their mutual information and achieves the joint entropy reduction bound at each step under a first-order Markov model This is a more rigorous approach than greedy confidence-based strategies that ignore token interdependencies and it's model-free. DUS is a drop-in decoding method requiring no changes to model architecture or training It can be applied to any masked diffusion LM (e.g. the authors test it on different 7B–8B models) without fine-tuning. 

DUS yields logarithmic-time decoding in the sequence length, dramatically cutting the number of denoiser without much degradation of perofrmance. with the same average parallelism, it consistently scores higher than the standard self-confidence unmasking heuristic and nearly recovers the single-token diffusion performance (denoted as SOTA at 1× speed) even when unmasking multiple tokens per step

### Weaknesses
The “dilated” schedule, while motivated by theory, ultimately rests on a strong independence assumption that may not hold universally. The analysis assumes a fast-mixing first-order Markov chain for token dependencies. For the selected benchmarks, there's evidence that there no such complex dependencies. Real language sequences can have long-range dependencies (e.g. rhyme schemes in poetry or variable usage in code), perhaps evaluating the model on natural language tasks would give different results. Seems necessary to expand the scope of datasets, and report metrics like perplexity or diversity on large-scale text corpora

The lack of learned planner is less expressive than alternative, also efficient approaches with abilty to learn task-specific planners. Similarly, there's no mechanism for changing a token that has been unmasked. recent approaches to discrete diffusion can address this issue, and can fix a mistaken token in later iterations

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
