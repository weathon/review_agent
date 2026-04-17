# Cosmos-Eval: Towards Explainable Evaluation of Physics and Semantics in Text-to-Video Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recent text-to-video (T2V) models have achieved impressive visual fidelity, yet they remain prone to failures in two critical dimensions: adhering to prompt semantics and respecting physical commonsense. Existing benchmarks, including VideoPhy and VideoPhy-2, formalize these axes but provide only scalar scores, leaving model errors unexplained and hindering reliable evaluation. To address this, we present Cosmos-Eval, an explainable evaluation framework that jointly assesses semantic adherence and physical consistency. Cosmos-Eval produces fine-grained 5-point scores with natural-language rationales, leveraging the physically grounded ontology of Cosmos-Reason1 and an LLM-based rationale refinement pipeline. This enables precise identification of semantic mismatches and violations of physical laws, such as floating objects or momentum inconsistencies. Experiments on VideoPhy-2 show that Cosmos-Eval matches state-of-the-art auto-evaluators in score alignment (Pearson 0.46 vs. 0.43 for semantics; Q-Kappa 0.33 vs. 0.33 for physics) while also delivering state-of-the-art rationale quality (e.g., best BERTScore F1 and BLEU-4 on both SA and PC). Beyond this benchmark, our framework generalizes to other evaluation suites, establishing a unified paradigm for explainable physics-and-semantics reasoning in T2V evaluation and enabling safer, more reliable model development.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Cosmos-Eval, an explainable evaluation framework for text-to-video (T2V) generation that assesses both semantic adherence (SA) and physical commonsense (PC). Unlike existing evaluators that only output scalar scores, Cosmos-Eval provides 5-point scores alongside natural language rationales explaining the evaluation. The framework consists of multiple stages: Stage 0 uses frozen VideoPhy-2 scorers for initial scoring, Stage 1 generates reference rationales through ensemble/selection, Stage 2 employs a judge-verified controller with iterative refinement strategies, and Stage 3 distills the pipeline into a lightweight model via two-run SFT. Experiments on VideoPhy-2 show that Cosmos-Eval matches state-of-the-art score alignment (SA Pearson 0.46 vs 0.43; PC Q-Kappa 0.33 vs 0.33) while achieving superior rationale quality.

### Strengths
- Important and timely problem. The paper tries to address a real gap in T2V evaluation - existing benchmarks like VideoPhy-2 only provide numeric scores without explaining why a video fails. Having explainable evaluations is clearly valuable for model development and debugging.
- Well-designed multi-stage pipeline. The framework is thoughtfully structured with clear motivations for each component. 
- Strong empirical validation. The paper includes comprehensive experiments with proper baselines (Qwen-2.5-VL, VideoLLaMA3, InternVL variants). The results show Cosmos-Eval achieves competitive score alignment while providing substantially better rationales. The Stage-1 ablation (Table 4) with hit-rate analysis provides convincing evidence that the reference rationales improve usability.
- Reproducibility and generalizability. The paper provides extensive implementation details (Table 6, Appendix H with full prompts), making reproduction feasible.

### Weaknesses
- Limited exploration of model scale. The paper primarily uses 7B-72B parameter models, but doesn't investigate whether much larger frontier models like Qwen3-VL-235B or GPT-4V could perform direct explainable evaluation without the complex multi-stage pipeline. Given the rapid improvement in VLM capabilities, it's unclear whether this elaborate framework is necessary or if a well-prompted large model could achieve similar results more efficiently. Some comparison or discussion would strengthen the positioning.
- Missing benchmark of T2V generation models. Since the paper presents an evaluation framework, it would be valuable to actually evaluate and rank current state-of-the-art T2V models (e.g., Sora2, Veo-3.1, Kling, Pika, Lumiere, Stable Video Diffusion) to provide a reference benchmark. This would demonstrate the framework's practical utility and give the community concrete comparisons. Currently, the experiments only validate the evaluator itself against human judgments, not its application to comparing different generators.
- Limited dataset diversity. All experiments are conducted only on VideoPhy/VideoPhy-2, which consists of synthetic model-generated videos. The generalization to real-world videos or other evaluation suites like T2VPhysBench or PhyCoBench is mentioned as future work, but not validated. This raises questions about whether the approach will transfer to different video distributions and evaluation scenarios.
- Computational cost not analyzed. The multi-stage pipeline with ensemble models, multiple sampling, iterative controller strategies, and judge verification seems computationally expensive, but the paper doesn't report wall-clock time, cost estimates, or throughput comparisons against baselines. For a proposed evaluation framework, understanding the practical overhead is important for adoption.

### Questions
See Weaknesses above.

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
3

### Summary
The paper proposes Cosmos-Eval, an explainable evaluation framework for T2V models that jointly assesses Semantic Adherence (SA) and Physical Commonsense (PC). Unlike prior score-only evaluators, Cosmos-Eval outputs 5-point scores plus concise, natural-language rationales grounded in a physics ontology derived from Cosmos-Reason1. On the VideoPhy-2 testset, the method attains score alignment comparable to the official automatic evaluator while improving rationale quality according to text metrics and a judge-verified pipeline. 
The system is built via a four-stage process: Stage-0 uses frozen VideoPhy-2 scorers to obtain 5-point labels; Stage-1 generates score-aligned reference rationales; Stage-2 runs a reference-seeded, judge-verified controller to produce evidence-grounded chains-of-thought; Stage-3 performs two-run SFT. The authors report state-of-the-art rationale quality and competitive agreement with human labels, and present ablations isolating the contribution of each stage.

### Strengths
1.	Clear problem framing and practical value
The paper addresses a recognized gap: score-only evaluators provide limited diagnostics. Cosmos-Eval augments SA/PC scores with concise, physics-grounded rationales that make failures actionable, directly improving auditability and trust. 
2.	Well-structured, modular methodology
The Stage-0→1→2→3 pipeline is logically decomposed: label priors (Stage-0), reference rationale (Stage-1), judge-verified controller with explicit strategies (backtracking, exploration, verification, correction; Stage-2), and scores-first, reasons-conditioned SFT (Stage-3). 
3.	Competitive core agreement with added explainability
On VideoPhy-2, Cosmos-Eval reaches top or near-top results on several core metrics while producing rationales (e.g., SA Pearson 0.4643; PC Acc 0.3912 / Q-κ 0.3301). This reduces the traditional accuracy–interpretability trade-off. Cosmos-Eval’s rationales outperform general VLMs by wide margins on BERTScore/ROUGE/BLEU.

### Weaknesses
1.	Training supervision may inherit biases from the frozen scorer
Stage-0 relies on VideoPhy-2-AutoEval to obtain 5-point labels that then condition the entire pipeline (and are textualized for SFT). While evaluation is against human labels on VideoPhy-2, this setup risks teacher-student bias and may cap the achievable human alignment if the teacher diverges from humans in systematic ways. A more direct calibration to partial human labels (or other supervision) would strengthen claims. 
2.	Reliance on LLM/VLM judges introduces circularity risks
The judge-verified controller (Stage-2) and ablations use a VLM judge with task-specific rubrics; re-mapping rationales to scores uses another strong LLM. Although this is standard practice, it raises judge-choice sensitivity and circularity concerns. The discussion notes judge bias but provides limited cross-judge robustness or multi-judge agreement analyses. 
3.	Statistics for the main tables lack uncertainty estimates
Core results (Table 1) are presented as point estimates without confidence intervals or significance tests across seeds/subsets. Given modest absolute correlations (e.g., SA Pearson 0.46) and close deltas vs. the baseline evaluator, intervals would clarify whether improvements are statistically meaningful. 
4.	Lack of case study
Cosmos-Eval claims that it outputs 5-point scores plus concise, natural-language rationales, but specific cot text data, or more about generating cases and error analysis, are not mentioned much in the main text.

### Questions
See weaknesses

### Soundness
3

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
3

### Summary
The authors present Cosmos-Eval, an interpretable benchmark built on VIDEOPhy-2 that aims to address its limited explanatory feedback.
The proposed framework appears to maintain close alignment with VIDEOPhy-2-AutoEval scores while producing explanations that better correspond to human-written gold rationales than those from several existing VLM baselines, suggesting improved reliability and interpretability.

### Strengths
1. Enhances VIDEOPhy-2 by introducing explicit reasoning on both semantic alignment and physical consistency, improving interpretability across two key evaluation dimensions.
2. The four-stage generation–verification–distillation pipeline is well structured, functioning as data augmentation for rationales.
3. The reported ablation and comparative experiments provide reasonable evidence that each module contributes to both score consistency and explanation quality.

### Weaknesses
1. All rationales are generated by large closed-source VLMs, which may introduce systematic biases and limit interpretive diversity.
2. While those closed-source VLMs serve as generators and judges, the paper does not include a direct comparison of their explanation quality against Cosmos-Eval, leaving the extent of improvement somewhat uncertain.
3. The evaluation mainly relies on surface-level textual metrics (e.g., BLEU, BERTScore), which may not fully capture reasoning accuracy or factual soundness.
4. I still have concerns about the level of novelty and the amount of work in this paper. The contribution seems to be mostly incremental, as the proposed method is built on top of existing benchmarks such as VIDEOPHY-2, with only limited innovation. The core approach essentially relies on prompting VLM to generate explanations, which does not appear particularly novel. If the authors can reasonably and thoroughly address this concern, I would be open to increasing my score.

### Questions
1. Could the authors elaborate on how the “consensus extraction” step using the aggregator LLM combines outputs from multiple VLMs? In particular, how are disagreements handled to ensure consistency in the selected reference rationale?
2. As several closed-source VLMs (e.g., GPT-4V, Claude, Gemini) are involved as generators and judges, could the authors provide direct or comparative examples showing how Cosmos-Eval’s explanations differ from those produced by these models?
3. Beyond BLEU and BERTScore, did the authors conduct or consider any human evaluation or factual-consistency analysis to assess the correctness of the generated rationales?

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
4

### Summary
Summary:   
This paper addresses a critical limitation of existing text-to-video (T2V) model evaluators: they only output scalar scores for Semantic Adherence (SA, prompt-video alignment) and Physical Commonsense (PC, adherence to physical laws) without explaining the reasoning behind scores. To solve this, the authors proposeCosmos-Eval, an explainable evaluation framework that combines 5-point SA/PC scores with physics-grounded natural-language rationales.  

Contributions:  
（1）Explainable SA/PC Evaluation Paradigm: Cosmos-Eval is the first framework to pair fine-grained 5-point SA/PC scores with actionable, physics-grounded rationales, resolving the interpretability gap of score-only evaluators and enabling targeted diagnosis of model failures.    
（2）Score-Aligned Rationale Generation: By integrating Cosmos-Reason1’s physical ontology and a judge-verified CoT controller, it ensures rationales are tightly linked to scores—avoiding the accuracy-interpretability trade-off seen in generic VLM-based evaluators.  
（3）Generalizable, Distillable Pipeline: Its four-stage workflow is dataset- and scorer-agnostic, making it adaptable to other T2V evaluation suites beyond VIDEOPHY-2.  
（4）State-of-the-Art Rationale Quality: Cosmos-Eval outperforms generic VLMs in rationale similarity to human-like references, with best-in-class BERTScore F1 and BLEU-4 scores.

### Strengths
Strengths:   
（1）Originality  
It proposes Cosmos-Eval, the first T2V evaluation framework pairing 5-point SA/PC scores with physics-grounded rationales to solve the interpretability gap of score-only tools. Its four-stage pipeline and two-run fine-tuning innovatively balance score alignment, rationale quality, and deployment efficiency.  
（2）Quality  
Its pipeline follows a rigorous, reproducible process, with each stage validated to ensure reliability. Experiments cover 6 T2V models and 9000+ videos, with metrics matching top auto-evaluators in score alignment and outperforming VLMs in rationale quality.  
（3）Clarity  
It follows a clear "gap→solution→validation" structure, explaining technical details in plain language. Consistent terminology and visual aids help readers easily understand the framework’s design and results.  
（4）Significance  
As an open, standardized tool, it enables T2V model failure diagnosis and guides real-world applications. It shifts T2V evaluation to prioritize interpretability, aligns with scientific integrity, and reduces educational misinformation risks.

### Weaknesses
Weaknesses:    
（1）The comparison with the methods of predecessors is not sufficient.    
A comparison between the dataset and some previous related datasets, such as VideoREPA, WISA, NewtonGen, etc.?    
（2）LLM Self-Reinforcement Risk  
It heavily relies on LLMs for rationale generation and evaluation, risking bias toward LLM-style explanations over human needs. The paper does not address this closed-loop issue or propose human-in-the-loop calibration.    
（3）Limited Video Complexity and Duration Coverage  
Validated mainly on VIDEOPHY-2, it lacks testing on long-duration videos or complex physical scenarios, failing to assess escalated inconsistencies in extended content.  
（4）Dependence on Predecessor Tools and Narrow Data  
It relies on frozen VIDEOPHY-2-AUTOEVAL for initial scores, inheriting its limitations. Training and evaluation are restricted to one benchmark, with no adaptability tests for diverse suites or non-English prompts.  
（5）Insufficient Computational Efficiency Analysis  
While claiming a lightweight distilled model, it provides no detailed compute cost breakdown for pipeline stages. No inference speed or resource comparison with score-only evaluators is given.  
（6）Noisy Input Robustness Gap  
It does not test performance on noisy T2V outputs (e.g., motion blur, occlusion), leaving reliability for real-world imperfect videos unconfirmed.

### Questions
Questions:    
（1）Questions About LLM Self-Reinforcement Bias  
Your framework relies heavily on LLMs for both generating reference rationales and evaluating final quality, which may create a self-reinforcement loop favoring LLM-style outputs over human-aligned explanations. Have you conducted comparative studies to measure how much this bias deviates from genuine human interpretability needs? Additionally, do you have plans to integrate human-in-the-loop calibration to mitigate such closed-loop dependency in future iterations?  
（2）Generalization to Complex Video Scenarios  
Cosmos-Eval is primarily validated on the VIDEOPHY-2 benchmark, which lacks long-duration videos and complex physical interactions. Have you tested its performance on extended video content where physical inconsistencies typically escalate? If not, what technical challenges do you anticipate in adapting your trajectory extraction and rationale generation modules for such scenarios?  
（3）Suggestions for Reducing Dependence on Predecessor Tools  
Your framework inherits initial scores from frozen VIDEOPHY-2-AUTOEVAL without independent validation, which may propagate its predecessor’s limitations. We suggest adding a parallel validation step: compare scores from VIDEOPHY-2-AUTOEVAL with human-labeled ground truth for a subset of samples to quantify inherited biases. This would enhance trust in your pipeline’s foundational score accuracy.  
If the author can effectively solve my doubts, I will consider improving my score.

### Soundness
2

### Presentation
3

### Contribution
3
