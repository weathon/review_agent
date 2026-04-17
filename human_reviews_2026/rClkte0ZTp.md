# Efficient Test-Time Scaling for Small Vision-Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Small Vision-Language Models (VLMs) provide a computationally efficient alternative to larger models, at the cost of weaker generalization abilities and downstream task performance. These shortcomings could be addressed by test-time scaling techniques, but existing methods are typically computationally demanding, contradicting the resource-efficient design goals of small models. To address these limitations, we propose two novel and efficient test-time scaling strategies that leverage the model-internal features rather than external supervision: (i) Test-Time Augmentation (TTAug), which generates multiple augmented inputs and aggregates outputs at the token level without parameter updates, and (ii) Test-Time Adaptation (TTAdapt), which adapts model parameters during inference using consensus-based pseudolabels from TTAug. Through extensive experiments across nine benchmarks, we demonstrate consistent performance improvements while maintaining computational efficiency suitable for resource-constrained environments. The generality of our approach is demonstrated both within models at different scales and across different VLMs without additional tuning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes two efficient test-time scaling strategies, Test-Time Augmentation and Test-Time Adaptation, for improving the generalization performance of existing methods. Test-Time Augmentation aggregates the outputs of multiple augmented inputs at the token level without parameter updates. Test-Time Adaptation further improves the performance of Test-Time Adaptation by using it’s outputs as pseudolabels and updating model parameters. The effectiveness of the two methods is verified by extensive experiments.

### Strengths
* This paper proposes a new approach for aggregating multiple outputs at the token level.

* The experimental results are sufficient.

### Weaknesses
* The explanations on some of experimental results seems insufficient. For instances, in Table 1-5, the value of performance evaluation is zero on some datasets. It is better to explain such results. 

* There is a lack of comparison on the computational cost across different methods. Test-Time Adaptation must update model parameters and thus requires more computational resources than Test-Time Adaptation and similar methods. The authors can provide some theoretical analyses or empirical results.

### Questions
* In Table2, if we exclude the dataset AI2D, it seems that answer-level aggregation with Self-Consistency is not worse than token-level aggregation with simple averaging. Besides, for the same dataset, answer-level aggregation with Sample-and-Rank is also better than token-level aggregation with simple averaging. We can observe the same results on the GQA datasets and answer-level aggregation with Self-Synthesizer. Therefore, it seems that token-level aggregation does not consistently outperforms answer-level aggregation. Could the authors explain those experimental results?

* How can the proposed methods implement early-stopping for low-quality generations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the performance limitations of small Vision-Language Models (VLMs) under domain shift by proposing two efficient test-time scaling strategies: Test-Time Augmentation (TTAug) and Test-Time Adaptation (TTAdapt). TTAug generates multiple semantically equivalent inputs through augmentations and aggregates predictions at the token level without parameter updates. TTAdapt extends this by fine-tuning model parameters during inference using pseudolabels derived from TTAug consensus. The methods are evaluated across nine diverse benchmarks, demonstrating consistent improvements while maintaining computational efficiency suitable for resource-constrained environments. The approach generalizes across model architectures and scales without additional tuning.

### Strengths
+ The paper introduces a novel framework for test-time scaling in small VLMs, different from existing methods that rely on external models or answer-level aggregation. The token-level aggregation strategy is novel, addressing limitations of global confidence measures by leveraging fine-grained signals during generation. 
+ The work is technically sound, with extensive experiments covering nine benchmarks and multiple model families. Ablation studies evaluate different design choices. Theoretical analysis in the appendix justifies some key decisions. 
+ The paper is well-structured, with good motivations, method descriptions, and visual demonstrations.
+ The work addresses the gap in deploying small VLMs under resource constraints. The proposed method can achieve practical improvements without heavy computational costs, making them potentially accessible for real-world applications.

### Weaknesses
- Limited comparison to baselines:​​ While the paper compares against methods like self-consistency and self-selector, it does not include more test-time adaptation techniques for VLMs. The comparison with these approaches would strengthen the claim of superiority.
- Computational Efficiency Claims:​​ The analysis in Section 4.3 and Appendix E reports memory and runtime overhead but lacks a breakdown of latency across different hardware. More explanation on this concerns are necessary.
- The benchmarks focus on question-answering and captioning, but those tasks requiring complex reasoning are not tested. The reviewer cannot confirm if this method will bring benefits for these tasks.

### Questions
- The results in Figure 4 show that optimal aggregation layers vary by task. Could the authors explain how these layer choices were determined for each benchmark? Is there a principled criterion behind the selection, or was it based on empirical validation?
- The paper claims the methods are suitable for resource-constrained environments but only provides results for an A100 GPU. Could the authors clarify if any tests were conducted on consumer-grade hardware?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes two efficient test-time scaling methods for small vision-language models (VLMs): Test-Time Augmentation (TTAug), which aggregates token-level predictions from semantically preserved input augmentations without updating model parameters, and Test-Time Adaptation (TTAdapt), which further refines model parameters during inference using pseudolabels derived from TTAug consensus. Evaluated across nine benchmarks, both methods consistently improve performance while maintaining computational efficiency suitable for resource-constrained environments, outperforming existing answer-level test-time scaling approaches. The study also demonstrates that input perturbations with greedy decoding and token-level aggregation are more effective than temperature sampling and answer-level aggregation, respectively.

### Strengths
- Proposes token-level aggregation, a novel and empirically superior alternative to answer-level methods
- Demonstrates consistent gains across nine diverse benchmarks, including challenging open-ended tasks like captioning and VQA, where prior methods fail due to reliance on extractable final answers.
- Provides a comprehensive ablation study on augmentation strategies, aggregation layers, and modality-specific contributions, revealing task-dependent optima (e.g., early layer aggregation for visual reasoning).

### Weaknesses
- The paper's claims of efficiency are undermined by significant computational overhead: TTAug increases inference time by 3.3× and GPU memory usage by 1.9× at 16 augmentations, which contradicts the goal of resource-constrained deployment.
- TTAdapt requires iterative fine-tuning during inference and parameter resets per sample, introducing non-trivial latency and complexity that are not adequately justified by marginal gains on most benchmarks.
- The experimental evaluation lacks ablation on real-world hardware constraints, such as latency-sensitive edge devices or energy budgets, making the practical viability of the methods questionable.
- Text augmentation dominates performance gains, yet the simplest classical perturbations (e.g., keyboard typos) often match or outperform more sophisticated self-paraphrasing, casting doubt on the necessity of complex augmentation design.
- The model-agnostic claims are weakened by hyperparameters (e.g., 16 augmentations, classical high-strength augmentations) being optimized solely on SmolVLM2-2.2B and not validated for transfer across all model families.
- The generative image augmentation failure is attributed to text exclusion, but the paper ignores alternative approaches like synthetic text insertion or text-aware augmentations that could preserve OCR-relevant content.
- Qualitative results show that TTAug frequently fixes errors by “stitching” correct tokens from noisy augmented inputs, suggesting the model’s base competence is fragile and the method acts more as a heuristic ensemble than a principled improvement.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

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
The article studies how to perform test-time scaling for small vision-language models. In particular, it investigates different design choices in terms of test-time augmentation (TTAug) and test-time adaptation (TTAdapt), analyzing how performance changes in 9 benchmarks w.r.t. diversity induced methods, aggregation levels, number of augmentations, text/image augmentation methods, and adaptation strategies. Each tested design choice comes with recipes and findings that, when put together, lead to consistent improvement across various VLMs.

### Strengths
1. This is an extensive study on how a large variety of design choices impacts the performance of test-time scaling. The study is well conducted, exploring a large variety of alternatives with regard to both test-time augmentation and adaptation. The results can serve as a blueprint for developers of test-time scaling techniques and a reference for practitioners and researchers in this field.

2. The choices have been primarily analyzed for SmolVLM2-2.2B, a small-scale model, but they also partly generalize to larger ones of different families, as shown with the experiments of Fig. 3.

3. The appendix is remarkable, providing extensive details on the implementation as well as theoretical analyses. This level of detail is important not only to ground the design choices (with the theoretical part) but also to allow for easy reproducibility.

### Weaknesses
1. The gaps across alternative strategies are sometimes small and might be hard to draw general conclusions from. For instance, 5 methods in Tab. 4 have only a 0.4 gap in performance, with the caption claiming that classical augmentations (L-M-H)  are the best, while the text-only strategy (0) performs comparably or even better. I am aware that experiments are costly, but reporting error bars or analyses on the significance of the gap could help strengthen the main takeaways.

2. Related to the general nature of the results, after each analysis, the best approach is selected for subsequent experiments/as a reference. Given the small gaps, optimal choices for one architecture might not be transferable to other ones. This behaviour can be seen from Fig. 3, as i) only the SmolVLM2 family shows consistent improvements (e.g., compared to Ovis2 and InternVL2), but only for SmolVLM2 2B, those are large. Moreover, even within that family, test-time augmentation and test-time scaling may show different behaviours (e.g., SmolVLM2 0.3B has TTAug helping more than TTAdapt, while SmolVLM2 0.5B shows the opposite trend). Performing the same analyses but considering more architectures/families would strengthen the applicability of the core takeaways.

3. The choice of benchmarks from VLMEvalKit (Duan et al. 2024) needs some clarification. There are 20 benchmarks in the VLMEvalKit (Duan et al. 2024), but 9 are considered: why is this subset considered the most representative? Note also that the AI2D benchmark on diagram understanding polarizes some of the results: e.g., self-consistency in Tab. 1 outperforms or is comparable to self-selection alternatives everywhere but for AI2D, where there is a huge gap (>60% points), and the same happens in Tab. 2 (1 vs 2 and 4). As the choice of benchmarks may have a high impact on the results and takeaways, the choice of benchmarks may need some clarification, as well as the use of the average to assess the best-performing method.

4. (relatively minor) From the tables, various alternatives are presented, but an overall comparison with the state-of-the-art is unclear (yet performed when testing various alternatives, e.g., Tab. 1 (Chen et al. 2024a; Pruner et al. 2025, Wang et al. 2023b). Given that one of the aims of the article is to outperform existing test-time scaling approaches (e.g., lines 81-85), it would be helpful to report a conclusive summary table where the final strategy is explicitly compared with existing works. 

**Minor points:**

5. While the article focuses on small models, to my knowledge, the tested strategies are not tailored to small models in particular. While I understand that test-time scaling is more helpful in the context of lower-performing VLMs, showing how the final findings generalize to large models could give further insights into the main outcomes. I deem this as a minor point, as the analysis is still helpful and several models have already been tested.

6. Section 3.1 provides a clear formalization of TTAug. However, 3.2 describes TTAdapt without any formalization. While excessive notations or unnecessary formulas may hinder the clarity, providing a general definition of TTAdapt with a mathematical formalization would be helpful to avoid potential confusion in the reader and make the manuscript (3.1 and 3.2) more consistent.

### Questions
Related (point-wise) to the weaknesses above:

1. Given the small gaps, how general might the result hold for other architectures?
2. Are there insights on why TTAdapt and TTAug could show inconsistent behaviours? 
3. Could you elaborate on the choice of benchmarks and how to select the best strategy (given the variance of results across benchmarks)?
4. How does the final model compare with existing alternatives?

### Soundness
3

### Presentation
3

### Contribution
3
