# Can visual input Be Compressed? A visual input token compression benchmark for large multimodal models

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 4

## Abstract
Large multimodal models (LMMs) often suffer from severe inference inefficiency due to the large number of visual tokens introduced by image encoders. While recent token compression methods, such as pruning and merging, have shown promise in reducing redundancy, their evaluation remains fragmented and inconsistent. In this work, we present \textbf{UniPruneBench}, a unified and extensible benchmark for visual token pruning in multimodal LLMs. UniPruneBench provides standardized protocols across six ability dimensions and ten datasets, covering ten representative compression algorithms and three families of LMMs (LLaVA-v1.5, Intern-VL3, and Qwen2.5-VL). Beyond task accuracy, it incorporates system-level metrics such as runtime and prefilling latency to provide a holistic view. Our experiments uncover several key findings: (1) random pruning is a surprisingly strong baseline, (2) no single method consistently outperforms others across scenarios, (3) pruning sensitivity varies significantly across tasks, with OCR being most vulnerable, and (4) pruning ratio is the dominant factor governing performance degradation. We believe UniPruneBench will serve as a reliable foundation for future research on efficient multimodal modeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a unified benchmark UniPruneBench for evaluating different vision token reduction methods in MLLMs. The paper includes common MLLM benchmarks of different categories and evaluates the performance of several token reduction methods on three MLLM structures under different reduction ratios. The paper provides observations and discussions about the included token reduction methods based on the experimental results.

### Strengths
The benchmark covers multiple models, datasets, and pruning strategies in a unified framework, addressing the inconsistency of prior fragmented evaluations. The inclusion of system-level metrics such as runtime and prefill latency is also commendable for assessing efficiency beyond reduction ratio and accuracy.

### Weaknesses
1. The proposed UniPruneBench primarily aggregates existing datasets and applies a unified pruning rule (fixing pruning at layer K=2). While this standardization may ease comparison, it does not clearly justify why future token-compression methods should adopt this particular benchmark. The benchmark appears procedural rather than conceptual, lacking a strong motivation or theoretical insight into why or how it enables better research on token reduction, which I will detail below.

2. The benchmark selection seems somewhat arbitrary. A benchmark for token reduction should be designed from the perspective of information loss sensitivity, not merely from the perspective of general multimodal ability coverage.
For instance, finer-grained or high-resolution visual tasks—where local details matter—would better expose the strengths and weaknesses of compression methods. The paper could have categorized benchmarks by the proportion of critical visual information or image size, providing a more principled testbed for evaluating pruning methods.

3. The benchmark results have limited reusability and practical comparability. First, different pruning methods may have distinct optimal reduction configurations (e.g., varying the pruning layer K or pruning ratio), so enforcing a fixed setting like K = 2 may not reflect each method’s true capability. Second, achieving fair comparison in real scenarios requires aligning not only the token reduction ratio but also the prefilling and generation latency, which are strongly hardware- and implementation-dependent. Consequently, researchers adopting this benchmark would still need to re-run all baseline methods under their own hardware and runtime environment to obtain meaningful and fair results, undermining the intended convenience and standardization of UniPruneBench.

4. The experimental findings provide limited conceptual insights that could guide the design of future pruning strategies. A more meaningful contribution would be to abstract common mechanisms across pruning methods (e.g., similarity-based, attention-based, hybrid) and perform controlled experiments to derive actionable insights about token redundancy and semantic preservation.

5. The study exclusively considers training-free pruning. However, many practical compression pipelines involve fine-tuning or joint training with pruning to recover lost performance. Including this perspective would have made the benchmark more representative and useful for real-world model optimization.

6. The paper should also include the down-sampling baseline besides the random dropping.

### Questions
1. Since different pruning methods may have distinct optimal pruning configurations (e.g., varying pruning layers or ratios), how do the authors justify fixing K = 2 across all methods? Would results change significantly if each method used its best-performing configuration?
2. Some pruning or token reduction methods may not be compatible with commonly used attention acceleration mechanisms (e.g., FlashAttention). How should such incompatibilities be handled to ensure fairness and consistency when comparing runtime efficiency across methods?

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
4

### Summary
This paper introduces UniPruneBench, a unified visual token pruning evaluation benchmark designed to systematically assess visual token compression methods in large multimodal models (LMMs). The benchmark targets the crucial inefficiency caused by the large number of visual tokens in multimodal systems, which hinders inference speed and scalability. UniPruneBench offers standardized evaluation protocols across 6 ability dimensions and 10 datasets, including 10 representative pruning algorithms. The study reveals several key findings, providing a more comprehensive perspective on the trade-off between efficiency and accuracy in pruning tasks for LMMs.

### Strengths
1. **Clear Research Motivation**: This paper addresses the challenge of visual symbol redundancy in multimodal reasoning by designing a benchmarking platform. This platform enables fair and reproducible comparisons across different models, datasets, and compression algorithms, tackling a major bottleneck in real-world multimodal deployments.

2. **Comprehensive Experimentation**: UniPruneBench evaluates 10 pruning algorithms across diverse datasets. The paper systematically categorizes methods into pure ViT, pure LLM, and hybrid models, providing detailed quantitative results that reveal universal trends in pruning behavior across different architectures.

3. **Standardized Evaluation**: The benchmark offers a unified assessment protocol, standardized metrics, and will publicly release its implementation. Unlike prior studies focused solely on task accuracy, UniPruneBench also examines runtime details such as pre-filling latency and pruning overhead.

4. **Critical Findings**: Through analysis of experimental results, this paper presents critical insights for multimodal pruning tasks. For instance, random pruning serves as a strong baseline, indicating room for improvement in current method designs. Furthermore, compression effectiveness is jointly influenced by task, model, and pruning ratio, providing clear directions for future optimization efforts.

### Weaknesses
1. **Scope of Evaluation Metrics**: Current metrics primarily focus on accuracy and runtime, yielding intuitive conclusions such as higher pruning ratios leading to more pronounced model performance degradation. However, this evaluation lacks semantic alignment measures and fails to capture qualitative changes beyond numerical performance declines. Introducing higher-level metrics like semantic fidelity or visual localization consistency would enhance the benchmark's assessment capabilities.

2. **Cross-Dataset Metric Consistency**: UniPruneBench uniformly employs accuracy as the performance metric across tasks. Yet datasets like MME, MMBench, and POPE adopt distinct evaluation paradigms. Uniform scale normalization may obscure task-specific challenges, preventing comprehensive assessment.

3. **In-depth Analysis**: While the paper demonstrates through extensive experiments that different pruning methods exhibit significant performance variations across models, tasks, and pruning ratios, it lacks an analysis of the underlying mechanisms driving these differences. For instance, it remains unexplained why SparseVLM outperforms DivPrune under light pruning on LLaVA-v1.5, yet DivPrune becomes more robust under heavy pruning. Authors could appropriately supplement the analysis by exploring the relationship between method characteristics and model architecture/task requirements. For instance, examining attention distributions, visual fusion layer design, and task-dependent features could provide more insightful references for selecting and designing compression methods.

4. **Evaluation Visualization**: Aggregated charts such as heatmaps or comparison diagrams could be provided to enhance result readability. Additionally, reporting statistical confidence intervals or standard deviations, incorporating reliability metrics, would strengthen the final experimental conclusions.

5. **Limited Focus on Visual Modality Pruning**: The paper primarily focuses on pruning methods that operate within the visual modality, such as those based on self-attention or feature maps within the ViT. However, a notable gap is the lack of attention to an emerging and important area: using text prompts to guide the pruning of visual tokens dynamically. In multimodal tasks, the relevance of a visual token is often context-dependent and influenced by the accompanying textual query. Recent studies (e.g., **AdaptInfer**, **HoloV**, **Recoverable Compression**) have shown that integrating text semantics into the pruning process, through cross-modal attention or other specialized techniques, leads to more accurate and context-aware pruning. These approaches are particularly effective in maintaining task-relevant information even when high compression ratios are applied. We suggest that the authors expand the related work section to include a review of this text-guided pruning approach. Incorporating these methods into future versions of UniPruneBench would provide a more complete and insightful perspective on the field.

   **References**:
   1. Zhang, Weichen, et al. "AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference with Dynamical Text Guidance." *arXiv preprint arXiv:2508.06084* (2025).
   2. Zou, Xin, et al. "Don't Just Chase Highlighted Tokens in MLLMs: Revisiting Visual Holistic Context Retention." *arXiv preprint arXiv:2510.02912* (2025).
   3. Chen, Yi, et al. "Recoverable compression: A multimodal vision token recovery mechanism guided by text information." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 39. No. 2. 2025.

### Questions
Please refer to the weakness.

### Soundness
3

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
4

### Summary
This paper introduces UniPruneBench, a unified and extensible benchmark for evaluating visual token pruning methods in large multimodal models (LMMs). Unlike prior fragmented studies, UniPruneBench standardizes evaluation across 6 capability dimensions (e.g., OCR, reasoning, hallucination) and 10 datasets, covering 10 pruning algorithms and 3 major model baselines (LLaVA-v1.5, Intern-VL3, and Qwen2.5-VL). It combines accuracy and system-level metrics such as runtime and prefilling latency, offering a holistic view of both performance and efficiency. Through extensive experiments, the authors find that random pruning is a surprisingly strong baseline, no single method is consistently superior, task sensitivity varies widely (with OCR being most affected), and pruning ratio chiefly governs the accuracy–efficiency trade-off. UniPruneBench thus establishes the first comprehensive and reproducible framework for assessing visual token compression in multimodal LLMs, providing key insights and a foundation for designing more efficient and scalable multimodal systems.

### Strengths
1. The paper is well-motivated. The field of LMM efficiency is advancing rapidly, but as the authors correctly point out, evaluation is fragmented, making it difficult to compare new methods. This work provides a standardized, comprehensive, and well-designed benchmark that will be highly valuable for future research and establishing common baselines.

2. The experimental setup is thorough and rigorous. Evaluating 10 pruning methods (categorized into ViT-only, LLM-only, and hybrid) across 3 major model baselines and 10 datasets represents a significant and valuable empirical undertaking.

3. The paper delivers several important findings. The most critical is that random pruning serves as a "surprisingly strong baseline" , outperforming several "designed" methods. This is a crucial, humbling finding for the field.

### Weaknesses
1. While the focus on pruning is well-motivated and thorough, the paper does not explore other compression strategies, such as token merging(only took 1 method?) or quantization, which could also play a key role in reducing the visual token burden in LMMs.

2. Most tables present single numbers without interval ranges or seed variability. Given that “random pruning” is a key message, showing variance across different runs is crucial to ensure that the superiority of random baselines is not merely an artifact of sampling. Moreover, other comparison methods should also report the performance range across multiple trials.

3. (minor) The paper excels at what is happening but is light on the why. For instance, why is random pruning so effective? Is it because visual token redundancy is so high that any selection is good enough? Why exactly do ViT-only methods outperform LLM-only methods on some architectures? It would be great that adding a section to do a shallow analysis.

### Questions
1. Given that task sensitivity varies significantly, can the authors explore task-specific pruning strategies? For example, would pruning techniques be more effective if optimized for specific task domains?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents UniPruneBench, the first unified benchmark for evaluating visual token pruning methods in large multimodal models, covering 10 algorithms across 3 LMM families (LLaVA-v1.5, InternVL3, Qwen2.5-VL) and 10 datasets spanning 6 capability dimensions. The benchmark reveals counter-intuitive findings: random pruning surprisingly outperforms many sophisticated methods, no single approach dominates universally, and task sensitivity varies dramatically with OCR being most vulnerable while instruction-following remains robust. While the work addresses a critical need for standardized evaluation in this fragmented field, it suffers from missing recent state-of-the-art methods (DivPrune, G-Prune, SparseVLM), lacks theoretical analysis explaining why findings occur, and provides limited practical guidance for method selection.

### Strengths
First standardized evaluation framework covering diverse methods (ViT-only, LLM-only, hybrid), multiple LMM families with different architectures, 10 datasets across 6 capability dimensions, and multi-metric evaluation (accuracy, runtime, prefilling latency). This addresses critical fragmentation in the field where prior work used inconsistent evaluation protocols.

The finding that random pruning achieves competitive performance challenges fundamental assumptions about importance-based token selection. Task-specific sensitivity analysis (OCR vulnerable, instruction-following robust) provides actionable deployment insights. Cross-model consistency strengthens confidence that findings reflect fundamental principles rather than architecture-specific artifacts.

### Weaknesses
1. DivPrune (CVPR 2025), G-Prune (AAAI 2025), SparseVLM (ICML 2025 - achieves 54% FLOPs reduction with 97% accuracy retention, outperforming FastV by 34.4% on video). The "random pruning competitiveness" finding may not hold against these sophisticated baselines, severely limiting the benchmark's completeness and relevance for understanding current state-of-the-art.

2. Purely empirical without investigating why findings occur. No explanation for random pruning effectiveness (token redundancy patterns? attention analysis? information flow?), no mechanistic understanding of task sensitivity differences (why OCR vulnerable vs. instruction-following robust?), and no investigation of model scale robustness factors. Without theoretical grounding, the benchmark cannot guide future method development beyond trial-and-error.

3. (1) No multi-turn conversation evaluation despite being essential for practical deployment, (2) No video understanding tasks where efficiency matters most.

### Questions
refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
