# When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations—superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches designed to address hallucinations caused by spurious correlations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the severe impact of spurious correlations on hallucination detection methods for large language models (LLMs). The authors construct a controllable synthetic experimental environment by introducing spurious correlations between surnames and attributes, and systematically evaluate the performance of various hallucination detection methods (e.g., confidence-based, internal-state probing, IDK fine-tuning) under different correlation strengths. The experiments show that as spurious correlations increase, the performance of existing detection methods drops significantly, making hallucination detection more difficult; even model scaling and IDK fine-tuning fail to effectively mitigate this issue. Theoretical analysis further reveals that in kernel regression models, spurious correlations cause the model to be overconfident in “rule regions”, making hallucinations harder to detect.

### Strengths
1. The problem is important: The paper focuses on spurious correlations, an overlooked source of hallucinations, and exposes the fundamental limitations of existing detection methods, demonstrating strong theoretical and practical significance.

2. Support for reproducibility: Detailed experimental settings, data construction, model configurations, and prompt templates are provided, facilitating reproduction and validation in future research.

### Weaknesses
1. Limited experimental scale and model coverage: Although multiple models are included, the experiments mainly focus on small- to medium-scale models (e.g., 1.7B parameters), with insufficient systematic evaluation on larger models (e.g., tens to hundreds of billions of parameters). It is recommended to include results for larger-scale models.

2. Narrow definition of spurious correlations: Currently, spurious correlations are introduced only via surname-attribute mappings, whereas in reality, spurious correlations can take various forms (e.g., context, word frequency, co-occurrence). It is suggested to expand experiments to include more types of spurious correlations.

3. Lack of exploration of effective mitigation strategies: The paper primarily reveals the problem but provides limited discussion on how to mitigate it. Future work could propose and evaluate methods for hallucination mitigation that target spurious correlations.

4. Readability of some figures could be improved: Some figures (e.g., heatmaps, linear probing results) have unclear labels. Optimizing legends and annotations would enhance visualization clarity.

5. Theoretical section is somewhat obscure: The derivations involving kernel regression and NTK may be difficult for readers without a theoretical background. It is recommended to add more intuitive explanations or illustrative diagrams in the main text.

### Questions
1. Limited experimental scale and model coverage: Although multiple models are included, the experiments mainly focus on small- to medium-scale models (e.g., 1.7B parameters), with insufficient systematic evaluation on larger models (e.g., tens to hundreds of billions of parameters). It is recommended to include results for larger-scale models.

2. Narrow definition of spurious correlations: Currently, spurious correlations are introduced only via surname-attribute mappings, whereas in reality, spurious correlations can take various forms (e.g., context, word frequency, co-occurrence). It is suggested to expand experiments to include more types of spurious correlations.

3. Lack of exploration of effective mitigation strategies: The paper primarily reveals the problem but provides limited discussion on how to mitigate it. Future work could propose and evaluate methods for hallucination mitigation that target spurious correlations.

4. Readability of some figures could be improved: Some figures (e.g., heatmaps, linear probing results) have unclear labels. Optimizing legends and annotations would enhance visualization clarity.

5. Theoretical section is somewhat obscure: The derivations involving kernel regression and NTK may be difficult for readers without a theoretical background. It is recommended to add more intuitive explanations or illustrative diagrams in the main text.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper identifies an important failure mode of hallucination detection in large language models (LLMs), attributing it to spurious correlations in training data. It provides a rigorous theoretical analysis and carefully designed synthetic and real-world experiments to support the claim that spurious correlation leads to high confident hallucinations  that are hard to detect under current detection schemes.

### Strengths
This paper provides rigorous theoretical derivation supporting the their hypothesis. They propose well-controlled synthetic experiments and pair with evolution on real LLMs. The paper demonstrates consistent empirical trend showing the degradation of detection under strong spurious correlations.The paper identifies an important failure mode of hallucination detection in large language models (LLMs), attributing it to spurious correlations in training data. It provides a rigorous theoretical analysis and carefully designed synthetic and real-world experiments to support this claim.

### Weaknesses
My main concerns is regarding the novelty and practical applicability of the paper. The core problem — that spurious correlations can lead models to make incorrect predictions — is well established in prior robustness and fairness literature. While this paper contextualizes the issue within LLM hallucination detection and LLM confidence, the theoretical analysis is based on a kernel regression formulation that is too simplified to capture the dynamics of modern LLMs. Moreover, the theoretical bound derived does not appear to provide actionable insights for model design or detection improvement, limiting its practical relevance. 

Overall, I find the work interesting and rigorous, but I am uncertain whether the contribution level is sufficient for acceptance as a theory paper.

[1] Yang, Yu, Eric Gan, Gintare Karolina Dziugaite, and Baharan Mirzasoleiman. "Identifying spurious biases early in training through the lens of simplicity bias." In International conference on artificial intelligence and statistics, pp. 2953-2961. PMLR, 2024.

[2] Ye, Wenqian, Guangtao Zheng, Xu Cao, Yunsheng Ma, and Aidong Zhang. "Spurious correlations in machine learning: A survey." arXiv preprint arXiv:2402.12715 (2024).

### Questions
see weakness

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
3

### Summary
The paper studies the cause of a specific kind of hallucinations, when spurious correlations in the data cause LLMs to confidently generate hallucinated content. Through controlled synthetic data usage for pretraining and fine-tuning, the paper shows how spurious correlations present in the data can make the LLM learn 'shortcut' connections, resulting in hallucinations that are not detected by most SOTA detection techniques, and not mitigated by most SOTA mitigation techniques.

### Strengths
1. The paper focuses on a targeted research question, and does a good job of providing an in-depth discussion.
2. The theoretical discussion and the controlled experiments with synthetic data across a wide range of models both strongly support the hypothesis proposed in the paper.
3. The paper is well written, and I enjoyed reading it. The Related Work is mostly well done, the Theoretical discussion is broadly easy to follow (although it can be made more accessible), and the experiment details are clear based on the main paper, the appendix, and the supplementary code (I strongly recommend adding a README to the code, though).

### Weaknesses
1. My only complaint, the 'real world validation' is not as strong, and thus leaves open the question of how to actually recognize spurious correlations in the wild. It is not clear to me how the Jaccard similarity of entities between the question and the generated answer is a measure of spurious correlations. While I do believe this is a good paper, even without the real world validation, clarification of these experiments would be appreciated. In their current form, I'm not convinced these results support the larger story.

### Questions
1. Why is the Jaccard similarity of entities between the question and the generated answer a measure of spurious correlations?

A Comment on Related Works
------
There has been a lot of work recently on trying to map out the 'knowledge' of an LLM [1, 2], or 'consistency' of hallucination evaluations [3]. While not exactly the same, these related works can also provide an interesting discussion for the paper. In my opinion, incorrect 'knowledge' in LLMs (or 'consistent' hallucinations) is probably an effect of spurious correlations detected in this paper.

References -

[1] Yin, Xunjian, et al. "Benchmarking Knowledge Boundary for Large Language Models: A Different Perspective on Model Evaluation." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

[2] Gekhman, Zorik, et al. "Inside-out: Hidden factual knowledge in llms." arXiv preprint arXiv:2503.15299 (2025).

[3] Ganesh, Prakhar, et al. "Rethinking hallucinations: Correctness, consistency, and prompt multiplicity." ICLR 2025 Workshop on Building Trust in Language Models and Applications. 2025.

### Soundness
3

### Presentation
4

### Contribution
3
