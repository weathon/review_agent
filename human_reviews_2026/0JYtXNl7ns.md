# Building Reliable Long-Form Generation via Step-Wise Hallucination Rejection Sampling

- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Large language models (LLMs) have achieved remarkable progress in open-ended text generation, yet they remain prone to hallucinating incorrect or unsupported content, which undermines their reliability. This issue is exacerbated in long-form generation due to hallucination snowballing, a phenomenon where early errors propagate and compound into subsequent outputs. To address this challenge, we propose a novel inference-time scaling framework, named Step-wise HAllucination Rejection Sampling (SHARS), that allocates additional computation during decoding to detect and reject hallucinated content as it is produced. By retaining only confident information and building subsequent generations upon it, the framework mitigates hallucination accumulation and enhances factual consistency. To instantiate this framework, we further introduce a new uncertainty-based hallucination detection method, named HalluSE, for long-form generation, improving upon the prior semantic entropy approach. The combined system enables models to self-correct hallucinations without requiring external resources such as web search or knowledge bases, while remaining compatible with them for future extensions. Empirical evaluations on standardized hallucination benchmarks demonstrate that our method substantially reduces hallucinations in long-form generation while preserving or even improving the informativeness of generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces an inference-time scaling framework (SHARS), aiming to allocate additional computational resources to detect and mitigate hallucinations during decoding.
As a main component of the framework, the uncertainty-based hallucination detection method HalluSE which aims to improve semantic entropy is introduced.
HalluSE is evaluated on long-form hallucination benchmarks, showing improved performance over some baseline methods.
The overall SHARS framework is evaluated by the FactScore benchmark, indicating less unsupported answers and high factual precision, at the cost of decreased response rate, showcasing scaling behavior with increased compute.

### Strengths
- The paper tackles an important problem and shows interesting scaling behavior, where more compute leads to higher factual correctness.
- The experiments use recent models (Llama 3.1 and QWEN 3) with both 8B and 32B parameters.
- Experiments tackle hallucinations in long-form generation, which is often omitted in prior work.
- HalluSE aims to improve the performance of semantic entropy in such long-form answer settings

### Weaknesses
- It would help to more clearly state that the overall framework depends on an auxiliary NLI model. Line 164 notes that Semantic Entropy (which HalluSE seeks to improve) requires mapping generated sequences to semantic clusters and refers to Farquhar et al. for details; in those works (and in the original work on semantic entropy - Kuhn et al., 2023), this mapping is performed with a BERT-style NLI model. However, the present paper does not discuss the specific NLI model used. This omission is important in light of the claims in lines 293ff that HalluSE "(1) is training-free and domain-agnostic, and (2) does not rely on external models, tools, or reference knowledge sources." Given the dependence on an NLI model, the "training-free", "domain-agnostic" and "no external models" characterizations are not justified if the NLI model's training data is domain-specific. Even when utilizing some a priori trained model as in Farquhar and Kuhn et al., I recommend explicitly describing the NLI component (architecture, checkpoint, and training data) and revising the scope of these claims. This would strengthen the paper's transparency and help readers assess generality. In the current form, I find the presentation not acceptable and misleading.
- The changes to the original semantic entropy algorithm appears minor, in essence it is about different prompting strategies to generate answers rather than any methodological advancement in how to calculate the uncertainty. However, while pretty much ad-hoc without any theoretical grounding, calculating long-form semantic entropy with HalluSE might be more useful than the originally proposed approach to scale Semantic Entropy to long form generation which was arguably suboptimal.
- The text in 5.1 does not clearly mention if only HalluSE is evaluated in this experiments or if the full SHARS framework is used as well to generate the answers.
- Baselines for the experiments reported in Table 1 are very limited, the literature on uncertainty estimation in LLMs is vast and there are many methods such as SAR (Duan at al.) or INSIDE (Chen et al.) that showed very strong results recently and could easily be incorporated.
- Overall, the proposed framework explains a high-level workflow that orchestrates multiple calls to LLMs. While it may be useful for particular practical settings, it relies heavily on the correct working of subcalls for fact decomposition, question generation and answer sampling. For me, the proposed SHARS framework does not pass the bar on profound methodological advancement I would expect from an ICLR paper. It might be better positioned for a venue focused more on applied or empirical studies, such as EMNLP, where the practical insights and orchestration design could be particularly appreciated.

---

Lorenz Kuhn, Yarin Gal, Sebastian Farquhar (2023) Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation, ICLR

Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, Kaidi Xu (2024) Shifting Attention to Relevance: Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models, ACL

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, Jieping Ye (2024) INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection, ICLR

### Questions
- What is the accuracy @ 1.0 for Table 1? Are the same answers by the QWEN model evaluated for different methods?
- The results in 5.2 are lacking a comparison for computational costs, e.g. simply by avg. walltime for answer generation or by token counts. The authors are upfront that their method incurs additional computational costs, but how much exactly? Similarly, it would be interesting to see such a comparison for the two considered sampling strategies (rewriting and Following). The headline results in Figure 1b suggests they are extreme, like up to 50 times compared to the standard strategy. It would be interesting to have more insight into the distribution of runtimes over a given dataset.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The submission introduces SHARS/HalluSE - Step-wise HAllucination Rejection Sampling, a method that is intended to address some classes of the hallucinations by resampling the outputs that have high semantic entropy.
The method uses a multistage pipeline that breaks down the model outputs into atomic facts, formulates them as subquestions, samples and evaluates semantic entropy (SE) on each of them individually. 
This improves the hallucination detection on test data compared to regular SE.

### Strengths
The authors consider the problem of applying semantic entropy to longer generations.

### Weaknesses
1. A large volume of state of the art uncertainty estimation literature is ignored, both in writing and in evaluation ( e.g. [1,2,3]).
2. The method consists mostly of prompt engineering. 
3. Limited evaluation, the results are presented for two models and two datasets, which is way below the standard of the field even if slack is given for compute availability ([4]). 
4. Theoretical justification is absent. The authors spam algorithms, yet fail to derive any identities for what they are doing (even though they are mentioning Rejection Sampling, which could be quite interesting, even the separation of the output into facts could be treated in an interesting fashion).

### References
1. Aichberger, L., Schweighofer, K. & Hochreiter, S. Rethinking Uncertainty Estimation in Natural Language Generation. Preprint at https://doi.org/10.48550/arXiv.2412.15176 (2024).
2. Duan, J. et al. Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models. Preprint at https://doi.org/10.48550/arXiv.2307.01379 (2023).
3. Manakul, P., Liusie, A. & Gales, M. J. F. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. Preprint at https://doi.org/10.48550/arXiv.2303.08896 (2023).
4. Ielanskyi, M., Schweighofer, K., Aichberger, L. & Hochreiter, S. Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation. Preprint at https://doi.org/10.48550/arXiv.2510.02279 (2025).

### Questions
1. Could this method generalize to problems that do not have a very specific 'fact' structure? For example generation of code or tool calls?
2. What happens if the model is still confident about hallucinated fact (i.e. low semantic entropy) after the "Hallucination Identification" step.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the "hallucination snowballing" effect in long-form generation, where early factual errors propagate and degrade overall reliability. The authors propose a novel inference-time framework, Step-wise HAllucination Rejection Sampling (SHARS), which operates incrementally at the sentence level. Instead of post-hoc verification, SHARS assesses each new sentence for factuality as it is generated. Hallucinated sentences are either discarded or rewritten, ensuring that subsequent generation is conditioned only on verified content. To enable this, the authors introduce HalluSE, an improved uncertainty-based hallucination detector that refines prior semantic entropy methods. A key feature is that the system is self-contained, not requiring external knowledge sources, though it remains compatible with them. Extensive experiments on benchmarks like FactScore and LongFact demonstrate that SHARS significantly reduces hallucinations and improves factual precision, often while increasing the total amount of supported factual information.

### Strengths
1. The method is supported by compelling empirical evidence across multiple benchmarks (FactScore, LongFact) and models (Llama3, Qwen3). The results consistently show that SHARS significantly improves factual precision and reduces the number of unsupported claims.
2. The paper is well-written and clearly structured, making the complex methodology easy to follow and understand.

### Weaknesses
1. My primary concern is the substantial increase in computational cost and latency at inference time. Each sentence requires a multi-step process of decomposition, question generation, sampling, and potential rewriting, making it significantly slower than naive decoding. This overhead may be prohibitive for real-time or resource-constrained applications.
2. The entire pipeline is highly dependent on the instruction-following capabilities of the base LLM. As the authors note, the framework's effectiveness diminishes with smaller or less capable models that struggle to perform the complex auxiliary tasks required.
3. The framework introduces several new hyperparameters (e.g., number of probe questions Q, sampled answers A, entropy threshold θ) that require careful tuning. The paper shows that optimal settings vary across models and objectives, suggesting that adapting the method to new use cases could be a complex and costly process.
Overall, the proposed method trades significant inference efficiency for improved hallucination mitigation. Given the complexity of the pipeline, I have reservations about its practical feasibility in real-world scenarios.

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Step-wise Hallucination Rejection Sampling (SHARS), an inference-time framework aimed at improving the factual reliability of large language models (LLMs) in long-form generation. The key idea is to allocate additional compute during decoding by detecting and rejecting hallucinated sentences as they are produced, preventing “hallucination snowballing.”

Empirical results on FactualBio, FactScore, and LongFact benchmarks (using Llama3.1-8B-Instruct and Qwen3-32B) demonstrate that SHARS with HalluSE reduces hallucination rates by 20–26% and improves factual precision, with a consistent positive scaling trend with increased inference-time computation.

### Strengths
1. Introduces a new inference-time paradigm—step-wise hallucination rejection—for long-form generation.

2. Well-specified algorithms and transparent design choices.

3.  Figures, tables, and prompts (Appendix A–B) are clear and reproducible.

### Weaknesses
1. Only two models (Llama3.1-8B, Qwen3-32B) and few baselines are tested. Lack of comparison with recent inference-time or training-time hallucination mitigation approaches (e.g., DoLa [ICLR 2024], Integrative Decoding [Cheng et al., ICLR 2025], Mask-DPO [ICLR 2025]) weakens claims of superiority.

2. The framework increases inference-time cost substantially, but quantitative trade-offs (runtime vs. factual gain) are not reported. This omission limits the paper’s practical relevance for deployment.

### Questions
1. Can the authors provide quantitative runtime scaling (e.g., FLOPs or wall-clock ratios) relative to naïve decoding? This would clarify practical feasibility.

2. Why were approaches like DoLa, FactAlign, or Mask-DPO not included for comparison? Even small-scale or qualitative comparisons would strengthen the argument.

### Soundness
2

### Presentation
2

### Contribution
2
