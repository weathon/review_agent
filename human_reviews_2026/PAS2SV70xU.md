# Echoes of the Visual Past: Test-Time Prompt Tuning with Multi-Scale Visual Memory

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Test-time prompt tuning (TPT) aims to adapt pre-trained vision-language models (VLMs) to various downstream tasks by learning textual prompts using unlabeled data at test time. However, existing TPT methods exhibit a performance gap compared to a line of prompt-engineering-based methods that leverage hand-crafted or LLM-generated prompts for VLM adaptation. We attribute this gap to a core limitation of previous TPT approaches: they learn prompts from only limited class-specific visual knowledge derived from a single test image. As a result, the learned prompts underperform compared to hand-crafted and LLM-generated prompts enriched with diverse, class-specific knowledge. To address this limitation, we propose $\textbf{T}$est-time $\textbf{P}$rompt $\textbf{T}$uning with $\textbf{M}$ulti-scale visual $\textbf{M}$emory ($\text{M}^2\text{TPT}$). Specifically, the memory is constructed to store past seen class-relevant image patches as multi-scale visual descriptions for each class. For each test image, we use it to query the memory and learn the textual prompt using both the test image and the retrieved class-relevant visual memory. Additionally, we introduce holistic visual memory to better handle holistic visual recognition tasks that require global image-level context, and an irrelevance suppression strategy to mitigate the impact of noisy memory entries at test time. We evaluate our method on 15 commonly used benchmark datasets and show that it outperforms existing TPT methods. Furthermore, our framework can incorporate human-designed prompts and achieves state-of-the-art performance compared to recent VLM adaptation methods that use hand-crafted or LLM-generated prompts.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work extends test-time prompt tuning (TPT) for vision-language models by introducing a multi-scale visual memory mechanism that stores class-relevant patch features from past test samples and uses them to guide prompt adaptation. Experiments on 15 datasets show consistent gains over previous TPT and competitive performance with hand-crafted/LLM prompts.

### Strengths
- It pinpoints a clear weakness in existing TPT methods, relying on a single image, and offers a conceptually reasonable solution via memory augmentation.
- Extensive experiments on 15 benchmarks, both in-distribution and OOD, with clear ablation.

### Weaknesses
- With regard to multi-scale visual memory, what does the term "multi-scale" refer to, and how is it initialized?
- The idea of maintaining a memory of past features (e.g., HisTPT, DynaPrompt) is not new. The main difference here lies in multi-scale patch granularity and explicit cross-promotion between memory and prompt, which is incremental rather than conceptually ground-breaking. It's better to compare with HisTPT and DYnaPrompt if possible.
- Since the memory is class-relevant, it costs 18.96G for ImageNet, so the scalability and memory efficiency are limited.
- Since only two of the compared methods were published in 2024 or 2025, are there any other recent related works?
-  Visualization of retrieved patches and how they influence prompt tokens would strengthen the paper’s interpretability.

### Questions
See the comments in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a TPT method that differs from prior approaches relying solely on a single test image and its augmented views for prompt optimization.  By introducing multi-scale visual memory, holistic memory, and an irrelevance suppression mechanism, the method aims to achieve more effective TPT. Extensive experiments across 15 datasets demonstrate competitive performance.

### Strengths
- The evaluation is comprehensive, including two experimental settings across 15 diverse datasets.
- The overall writing is clear and easy to follow.

### Weaknesses
- The method suffer from practical inefficiency. Test-time prompt tuning introduces substantial inference latency and computational overhead due to full back-propagation and multi-step forward inference. On top of this burden, the proposed approach further maintains a memory queue, which can significantly increase the computational cost, especially in tasks with a large number of classes. I am seriously concerned about the deployability of the method in real-world scenarios.
- The novelty is limited and some strongly relevant works are missing. The idea of introducing a memory mechanism into TPT is not new, as HisTPT [1] has already explored similar concepts. BoostAdaptor [2] incorporates augmented views of test images as multi-scale information in memory, which is closely related to the “multi-scale memory” proposed here. Moreover, recent approaches such as DPE [3] and GS-Bias [4] have shown more efficient test-time learning via prototype or bias updates. In addition, compared with the latest training-free method MCP [5], the proposed approach does not show clear performance advantages.

[1] Historical Test-time Prompt Tuning for Vision Foundation Models. NIPS2024

[2] BoostAdapter: Improving Vision-Language Test-Time Adaptation via Regional Bootstrapping. NIPS2024

[3] Dual Prototype Evolving for Test-Time Generalization of Vision-Language Models. NIPS2024

[4] GS-Bias: Global-Spatial Bias Learner for Single-Image Test-Time Adaptation of Vision-Language Models. ICML 2025

[5] Multi-Cache enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models. ICCV 2025

### Questions
- The final results in Eq (8) are not solely obtained through prompt tuning, making it difficult to determine whether the method’s effectiveness primarily stems from prompt optimization.
- The tuned prompts may potentially have negative effects.  For example, TPT has been observed to decrease performance on the Pets dataset.  Such negative effects could also compromise the quality of memory samples.
- The paper does not report how many steps of prompt tuning were used, leaving unclear the computational cost and convergence behavior of the proposed method.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces M²TPT, a test-time prompt tuning method that enhances vision-language models by incorporating a multi-scale visual memory of past class-relevant image patches, allowing prompts to be learned from richer, accumulated visual context rather than just a single test image. By jointly optimizing prompts and updating memory in a mutual promotion loop—supplemented by a holistic memory for global context and an irrelevance suppression mechanism to filter noise—it outperforms existing test-time methods and even rivals performance of hand-crafted or LLM-generated prompt approaches, without requiring prior knowledge of test datasets.

### Strengths
- Introduces a novel test-time prompt tuning framework that bridges the performance gap between TPT and hand-crafted/LLM-generated prompts by incorporating multi-scale visual memory of past class-relevant patches.  
- Achieves SOTA results on 15 benchmark datasets, outperforming prior TPT methods and prompt-engineering approaches even without human-designed prompts.  
- Maintains computational efficiency with minimal overhead compared to existing TPT methods, making it practical for deployment without requiring backpropagation through the full VLM.  
- Rigorous evaluation across in-distribution and out-of-distribution settings, with reproducible implementation and clear ablation studies.

### Weaknesses
- The method relies on pseudo-labels for memory update and retrieval, making it vulnerable to early prediction errors that can accumulate and degrade performance over time.  
- Memory requires storage of visual features across the test stream, which may not be feasible in memory-constrained or real-time deployment scenarios.  
- The approach assumes a fixed, known set of classes beforehand, limiting applicability to dynamic or open-category settings where new classes emerge online.  
- The irrelevance suppression mechanism introduces additional complexity and hyperparameters (γ, α, β) with limited analysis of their robustness across diverse tasks or domains.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
