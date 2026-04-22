# SiTu: A Simple Training-Free Thinking-with-Image Approach via Uncertainty Guidance

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Large Multimodal Models (LMMs) have shown great promise in complex reasoning by incorporating images as intermediate steps, a paradigm known as "thinking with images". However, most current ``thinking with images" techniques are training-based, incurring significant computational costs, limiting model versatility, and risking catastrophic forgetting. To bridge this gap, we propose SiTu, a simple, training-free framework for "thinking with images" that leverages an LMM's inherent uncertainty to achieve test-time scaling for multimodal reasoning. The core of our approach is the discovery of a stable, entropy-based uncertainty estimation native to LMMs, which reliably guides the dynamic combination of diverse perception enhancement paths. We implement three simple perceptual actions, categorized as visual highlighting and irrelevant information suppression, and demonstrate a notable scaling phenomenon: as the number and diversity of these actions increase, the LMM's reasoning ability improves consistently.  Our extensive experiments on fine-grained visual understanding benchmarks like \textit{V}$^*$, HR-Bench 4K, HR-Bench 8K, and MME-realworld show that SiTu significantly outperforms existing training-free perception enhancement methods. Surprisingly, SiTu even surpasses the performance of current state-of-the-art training-based "thinking with images" methods, highlighting the immense potential of test-time scaling for multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SiTu, a training-free framework for “thinking with images” in large multimodal models, aimed at improving fine-grained visual reasoning without additional training or fine-tuning. SiTu leverages an entropy-based uncertainty metric, native to LMMs, to guide and select optimal perception enhancement strategies at test time. The framework incorporates several simple, model-intrinsic perception actions such as visual highlighting and suppression of irrelevant regions, dynamically combining their outputs via uncertainty-guided selection. Experiments conducted on high-resolution benchmarks demonstrate that SiTu surpasses both prior training-free perception enhancement pipelines and good “thinking-with-image” techniques. The paper also includes extensive ablation studies and interpretable visualizations illustrating the efficacy and scaling effects of SiTu’s multi-action approach.

### Strengths
1. The paper targets a relevant and under-explored direction—training-free, test-time improvement of multimodal reasoning—addressing computational and generalization bottlenecks arising from specialized fine-tuning. This is well motivated both by computational cost and risk of catastrophic forgetting inherent in training-based methods
2. Through multiple, plug-in perception operations (Draw Boxes, Grounding Crop, Grid Crop), SiTu demonstrates effective “scaling up” as more actions are incorporated, as explicitly visualized in Figure 3 (Page 6). The observed monotonic accuracy gains with increased action diversity are well-supported and argued.

### Weaknesses
3. Although the paper proposes a token-entropy–based “uncertainty” metric (Eq. (2)–(3)) to select the final answer path, the theoretical soundness of this metric is not well justified. For instance, why should the path with the lowest entropy necessarily correspond to the optimal perception? Does low entropy always imply a correct answer, or could it instead reflect an overconfident but wrong prediction? The authors assume that the pipeline of “multi-perception strategies → multiple candidate answers → select the one with the lowest entropy” works effectively, yet they do not analyze potential failure modes in depth—such as cases where all strategies yield incorrect but low-entropy predictions, or when different perception strategies share correlated biases, making the selection ineffective.
4. The action space (Draw Boxes, Grounding Crop, Grid Crop) is manually defined and relatively narrow. Section 5 acknowledges this as a limitation, but experiments make no effort to quantify how action diversity or selection procedure may scale in larger or less curated action spaces. There is also little discussion on the challenges or computational costs that might arise as the number of candidate actions is increased substantially (Page 9).
5. The uncertainty metric is defined in terms of average per-token entropy (Equation, Page 3), and is empirically validated to correlate with answer quality. However, the link between minimum entropy and answer correctness—especially in open-ended, generative contexts with semantic ambiguity—remains untheorized and might fail in practice for certain answer distributions (i.e., over-confident wrong answers). There is no formal analysis on its reliability in the presence of ambiguous or unlikely answer spaces, nor a theoretical discussion of its potential limitations compared to other aggregation metrics.
6. While Table 1 reports strong improvements over both training-based and training-free methods, the comparison may be confounded by choice of backbone models or implementations. For instance, it is unclear if all competing methods are re-executed or adapted to the same underlying LMM (Qwen2.5-VL-7B) as SiTu, or if some results refer to published numbers from different base models, possibly inflating the comparative gains. The paper would be strengthened by a more transparent comparison pipeline (Experiments, Page 5).
7. The ablation study in Table 3 (Page 7) demonstrates that single perception actions can degrade performance in certain sub-tasks (notably FCP), but the paper does not provide a systematic failure or error analysis explaining why this occurs, or under what conditions the aggregation truly outperforms naive or single-action settings. The analysis lacks a fine-grained breakdown of when and why aggregation via uncertainty guidance can fail or be unreliable.
8. The performance gains appear strongest where the base LMM struggles—namely, in extremely fine-grained or high-resolution domains (see Table 1 and Figure 3). It is not clear how much of the observed benefit translates to standard-resolution, lower-difficulty benchmarks, where perception enhancements might add little or introduce noise. No low-resolution are included for contrast.
9. The experiments omit full details regarding the hyperparameters, sampling strategies, and input preprocessing for various perception actions (e.g., exact prompts for bounding box queries, image partitioning specifics for Grid Crop, etc.; see Section 2.4 and Appendix). Although the method is training-free, there is a risk of hidden engineering or tuned heuristics affecting reproducibility.
10. Despite a strong general literature positioning, the paper omits several directly relevant recent works at the intersection of multimodal reasoning, implicit prompt engineering, fairness/accountability in perception, and scalability laws in contrastive language-image learning. 
11. Figure 3 (Page 6) provides compelling evidence of empirical scaling as actions are incorporated, but the paper does not offer a theoretical or even intuitive explanation for why such monotonic improvement should occur or be robust to additional action diversity. The risk of performance plateauing or even degrading with overly heterogeneous, noisy, or poor actions is not addressed.
12. As the framework scales up, the combinatorial action space grows rapidly, but there is no benchmarking or analysis on computational tractability, either in inference cost or in the practical number of actions before returns diminish.
13. Although the method is positioned as a general framework for multimodal reasoning, all empirical validation is strictly limited to fine-grained visual understanding on high-resolution datasets (V* Bench, HR-Bench, MME-RealWorld), as admitted in the paper’s own limitations (Conclusion, Page 9). There is no experimental or conceptual analysis supporting applicability to other multimodal reasoning domains (e.g., synthetic visual question answering, text-audio, or video).
14. More importantly, the motivation and novelty are relatively weak. Although the “training-free perception enhancement” direction is appealing, the proposed perception strategies (cropping, framing, grid cropping) are not inherently novel in visual tasks. While the authors combine uncertainty estimation with these strategies, the core operation still boils down to “selecting a crop/highlight and then performing reasoning.” The paper claims that “training-free reflective perception” represents a new paradigm, but it does not sufficiently explain why existing methods (e.g., agent-based perception workflows or zero-shot tuning strategies) cannot directly incorporate uncertainty measures, or why current designs are fundamentally limited. In other words, the motivation—specifically, why existing approaches cannot be applied—is not convincingly argued. In the field of multimodal reasoning, frameworks that follow the “perceive + crop + reason” pipeline are already common. The authors need to clarify their unique contributions and distinctions more explicitly, as this part of the paper currently appears rather weak.

### Questions
15. Can you provide empirical or conceptual justification for SiTu’s ability to scale and generalize to non-visual modalities or broader multimodal reasoning challenges (e.g., audio-text, video, etc.), or across standard-resolution tasks?
16. How does the averaging per-token entropy handle open-ended or highly ambiguous response spaces, especially if the LMM produces low-entropy (confident) yet incorrect answers? Have you observed any systematic failures of this metric versus, for instance, self-consistency or perplexity?
17. Is there experimental evidence or theoretical analysis for how SiTu’s performance scales as the cardinality and diversity of perception actions continue increasing? Is there a point where more actions harm rather than help due to action noise?
18. Can you quantify the inference cost or latency of SiTu, especially as the number of perception actions grows, in comparison to both naive LMM inference and training-based/multi-stage pipelines?
19. Will you publish the code, full prompt templates, and configuration settings for the various perception actions to support reproducibility? Are there hidden heuristics (thresholds, post-processing rules) that need to be known by future users?
20. Is SiTu’s observed scaling effect specific to high-resolution or fine-grained tasks? Do you have results or hypotheses on how performance changes in standard VQA, lower-res, or “mainstream” visual-linguistic benchmarks?
21. In the few cases where single perception actions degrade performance (Table 3), what are the common causes, and does the uncertainty-guided mechanism always recover these errors?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SiTu, a training-free framework for enhancing LMMs through uncertainty-guided perception enhancement. The key innovation is using mean entropy across generated tokens to select optimal perception paths from a set of simple visual manipulation operations (Draw Boxes, Grounding Crop, Grid Crop). The method achieves strong performance on fine-grained visual understanding benchmarks without requiring training, surpassing current training-based approaches on these specific tasks.

### Strengths
1. The paper addresses real limitations of training-based method with a simple but effective training-free alternative.

2. The idea of using a native uncertainty metric as a signal for selecting among perception/preprocessing strategies is appealing.

3. The method achieves higher improvement over baseline on V* Bench, HR-Bench 4K, outperforming existing training-based methods on fine-grained visual understanding.

4. Its compute practicality, a single A40 48g can run the whole stuff.

### Weaknesses
1. Insufficient theoretical justification: No theoretical analysis of why mean entropy is optimal or when it might fail; empirical comparison with other uncertainty metrics is limited.

2. Very limited scope and generalization: Only validated on fine-grained visual understanding tasks; MME-RealWorld results (Table 2) show marginal improvements on OCR (+0.5%) and complex reasoning tasks, raising concerns about broader applicability.

3. Single model evaluation: Only tested on qwen2.5vl-7b; generalization to other LMMs (InternVL, etc.) not demonstrated.

4. Specific prompts, hyperparameters (grid size, crop strategies), and reproducibility details are insufficient.

### Questions
1. Provide justification or calibration for why mean token-entropy predicts correctness; include counterexamples and comparisons to margin, energy, variance, MC-dropout.

2. May be evaluate on OCR-heavy and complex reasoning suites with per-category results and failure analyses.

3. Use alternative LMMs and report stability of gains and entropy–accuracy correlation.

4. Provide detailed prompts/seeds/hyperparameters.

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
This paper proposes a training-free thinking-with-image method to improve the perception and reasoning ability of VLM. For the input high-resolution image, their method first implements three simple perceptual actions, to gain different visual clues, then generates different answers. Finally they use entropy-based uncertainty score to select final answer. As the number and diversity of perceptual actions increase, the LMM's reasoning ability improves consistently. They outperform current "thinking with images" methods on fine-grained visual understanding benchmarks.

### Strengths
1. This is a training-free method, so it is easy to follow and use.
2. Experiments demonstrate the effectiveness of their method. The ablation is comprehensive. Especially, the first ablation about actions shows the importance of number and diversity of perceptual actions. Because it increases the number of sample reasoning paths.

### Weaknesses
1. The method novelty is limited. The contribution is to propose some perception actions to enhance images to generate different paths, then select answer according to entropy-based uncertainty.
2. The uncertainty-guided selection is not a creative way, which is similar to the sequence confidence.

### Questions
See weakness.

### Soundness
3

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
This paper introduces SiTu, a training-free framework for “thinking with images” that enhances Large Multimodal Models (LMMs) through an entropy-based uncertainty metric guiding visual perception at test time. The method defines three perception actions to highlight or suppress visual regions, dynamically selected by the model’s mean token entropy. Evaluations on V∗ Bench, HR-Bench 4K/8K, and MME-RealWorld demonstrate substantial accuracy gains over both training-based and training-free baselines.

### Strengths
1. The paper presents a simple yet effective uncertainty-guided test-time framework. It identifies a stable entropy-based metric that allows LMMs to adapt perceptual focus dynamically without retraining.
	
2. The results are strong and consistent. SiTu achieves significant improvements on challenging high-resolution benchmarks and surpasses existing training-free and training-based methods, validating the framework’s effectiveness.

3. The method is practical, interpretable, and reproducible. It requires only a single GPU, introduces no extra training cost, and provides a transparent perception process that can be extended to other multimodal settings.

### Weaknesses
1. The perception actions are limited in number and fully predefined, making the framework difficult to generalize to other domains or task types. In scenarios requiring multi-step reasoning or abstract understanding, the fixed action space restricts adaptability.

2. The entropy-based uncertainty idea has been extensively explored in large language models (LLMs), such as ARPO and other uncertainty-driven control or tool-generation methods. Therefore, it is difficult to clearly assess the conceptual novelty of SiTu and how substantially it differs from prior work.

3. The experimental scope lacks diversity. All results are based solely on Qwen2.5-VL-7B, without testing across larger models or different architectures. For a method claiming to be training-free, validation on only one mid-sized model is insufficient to demonstrate scalability or generality across parameter sizes and model families.

### Questions
1. The paper should explicitly clarify how the proposed uncertainty-guided framework differs from existing uncertainty-based approaches developed for LLMs.
2. The authors should include additional experiments on diverse training-free architectures and model scales to better demonstrate the generality and robustness of SiTu across different settings.

### Soundness
2

### Presentation
3

### Contribution
2
