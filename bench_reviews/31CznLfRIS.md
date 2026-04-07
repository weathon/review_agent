## Summary
VideoJudge introduces a bootstrapping framework to train specialized, small MLLM judges (3B/7B) for evaluating video understanding model outputs. The method uses an iterative generator-evaluator pipeline to synthesize a large-scale training dataset with aligned quality ratings, avoiding costly human annotation. The trained judges match or outperform much larger (e.g., 32B/72B) general-purpose MLLMs on several meta-evaluation benchmarks and can generate instance-specific evaluation rubrics.

## Strengths
- **Novel and scalable bootstrapping framework:** The paper presents a clearly described, iterative generator-evaluator loop (Algorithm 1) that creates over 100k rating-aligned training examples without human annotation. This addresses a critical data scarcity problem for video evaluation.
- **Strong empirical performance with small models:** VideoJudge-7B consistently matches or surpasses significantly larger models (Qwen2.5-VL-32B/72B) on multiple benchmarks (e.g., VideoJudgeLLaVA, VideoJudgeVCG, LongVideoBench). The rubric-trained VideoJudgeR-3B demonstrates that specialized fine-tuning can close the performance gap to models 10x its size.
- **Comprehensive analysis and released resources:** The paper includes valuable ablations (frames, decoding temperature) and a detailed error analysis. The release of code, models, bootstrapped datasets, and meta-evaluation benchmarks promotes reproducibility and provides essential resources for the community.

## Weaknesses
- **Substantial "closed-loop" evaluation concern:** The training data and two primary pointwise meta-evaluation benchmarks (VideoJudgeLLaVA, VideoJudgeVCG) are constructed using the same bootstrapping pipeline. While results on independent benchmarks (VATEX, VideoAutoArena, LongVideoBench) are positive, the strongest performances are on the bootstrapped benchmarks, leaving uncertainty about true generalization to human judgment.
- **Severe calibration issues and overestimation bias:** The error analysis (§6.2) reveals a consistent and critical flaw: judge models are poorly calibrated and exhibit a strong overestimation bias. For example, 46.6% of rating-3 responses are incorrectly scored as 5, and 81.3% of rating-4 responses are inflated to 5. This undermines the reliability of the judges for precise evaluation.
- **Missing methodological details affecting reproducibility:** Key parameters of the bootstrapping algorithm are not specified in the main text, such as the acceptance threshold α (used in Algorithm 1) and the exact identity of the generator (`G`) and evaluator (`E`) models during data synthesis. While some information is in the appendix, these omissions hinder precise replication.

## Nice-to-Haves
- **Ablation on bootstrapping components:** Studying the contribution of the iterative feedback loop versus a single generation pass would better characterize the framework's necessity and robustness.
- **Deeper diagnostic analysis of overestimation bias:** Investigating the root cause (e.g., data imbalance, loss function, rubric design) would move beyond observation to inform fixes.
- **Explicit cost-benefit analysis:** Quantifying the computational cost (GPU hours/API cost) of the bootstrapping phase versus the benefit of small, efficient judges would provide a more complete picture of the method's practicality.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strengths:** "The paper is well-written" (generic).
- **Weaknesses:** Demands for comparison to proprietary models (GPT-4V, Gemini) – the paper's scope is open-source models and the comparison is fair within that context.
- **Weaknesses:** Criticism that the generator/evaluator identity is unspecified – the models are identified in Section 4.1 and Appendix A.2.
- **Weaknesses:** Requests for societal impact analysis (bias amplification, environmental cost) – while valid considerations, they are not standard core requirements for a technical methodology paper at ICLR.
- **Weaknesses:** Claim that human evaluation of rubrics is unfair because baseline models were not fine-tuned – the comparison demonstrates the value of the full training recipe, which is a valid contribution.

## Novel Insights
The paper demonstrates that a carefully bootstrapped, self-consistent pipeline can generate high-quality supervision for a complex multimodal task, enabling small models to specialize and rival the evaluation capability of much larger general-purpose models. A key finding is that providing visual input is crucial for video evaluation (MLLMs outperform text-only LLMs), and extended chain-of-thought reasoning does not compensate for its absence. Furthermore, the framework shows that models can be trained to generate instance-specific evaluation rubrics, a step towards interpretable and context-grounded automated assessment.

## Suggestions
- To address the closed-loop concern, prioritize and expand analysis on all available *independent*, human-annotated benchmarks (e.g., seek additional pointwise datasets beyond VATEX) and present those results as primary evidence of generalization.
- Actively tackle the calibration weakness in revision. Propose and test a concrete mitigation, such as incorporating a balanced loss, temperature scaling, or augmenting training data with synthetically generated "hard negatives" near the top of the rating scale.
- In the main methodology section, explicitly state the acceptance threshold α and the specific models used as the generator (`G`) and evaluator (`E`) during bootstrapping to ensure full reproducibility.