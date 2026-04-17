# On alignment of unified multimodal large language models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Unified Multi-Modal Large Language Models (U-MLLMs) have demonstrated strong capabilities in text-to-image (T2I) generation, but most post-training methods still rely on sparse, image-level rewards and place limited emphasis on safety. In this work, we take an exploratory view of \emph{dense} reward signals for U-MLLMs: token-level feedback derived from existing reward and evaluation models. Rather than proposing a new RL algorithm, We study how dense rewards can be extracted, how they behave, and how they can be integrated into the standard Group Relative Policy Optimization (GRPO) framework. Concretely, we investigate four questions: (1) how to obtain dense token-level rewards from scalar reward models such as HPSv2; (2) what the empirical behavior and distribution of dense rewards over image tokens look like; (3) how to incorporate dense rewards into GRPO via token-weighted advantages while preserving group-wise sample rankings; and (4) how different interpretability methods compare as providers of dense reward, including trade-offs in localization, computational cost, and downstream performance. On WISE and GenAI-Bench, dense-reward variants of a Janus-Pro-7B U-MLLM achieve competitive image quality (e.g., WISE: 0.50) with slightly smoother training dynamics compared to a sparse-reward T2I-R1 baseline. As a preliminary case study, we also instantiate a safety-focused variant that combines safety reward and observe a 59.4\% reduction in unsafe content on the MMDT benchmark relative to the base model. Overall, our results suggest that dense reward is a promising but nuanced design axis for U-MLLM post-training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies post-training alignment of unified multimodal LLMs for text-to-image (T2I). It injects dense visual attributions (RAHF heatmaps, SHAP/LIME) into GRPO by converting attribution scores into token-level weights during policy optimization. For safety, it adds a composite negative reward combining Toxic-BERT and an NSFW detector. Across WISE, GenAI-Bench, MMDT, and T2I-Safety, the method maintains or improves image quality while reducing unsafe generations on MMDT.

### Strengths
1.Clear, well-specified token-weighted GRPO with explicit equations and design choices.

2.Safety reward is simple, transparent, and easy to implement.

3.Substantial safety gains reported on MMDT.

4.Broad benchmark coverage (WISE, GenAI-Bench, MMDT, T2I-Safety) with ablations.

### Weaknesses
1.Limited novelty: the improvement over T2I-R1/GRPO appears small: (1) reuse of GRPO; (2)token weights derived from standard attributions (RAHF/SHAP/LIME); (3) a straightforward safety penalty using off-the-shelf classifiers.

2.Missing experimental details: batch size, training steps, and prompt curation specifics (e.g., for T2I-CompBench).

3.Underspecified safety hyperparameters: weights $w_{toxic}$ and $w_{nsfw}$ are not clear. 

4.Scope mismatch: the paper frames U-MLLM alignment broadly but does not evaluate I2T alignment.

### Questions
1.Beyond reweighting, how does token-weighted GRPO change optimization dynamics relative to scalar-reward GRPO？

### Soundness
2

### Presentation
3

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
This paper presents a framework for addressing safety alignment in unified multimodal large language models (U-MLLMs), which are designed to process both text and image modalities within a single architecture. The authors propose a reinforcement learning approach named Dense-GRPO, an extension of Group Relative Policy Optimization (GRPO) that incorporates token-level dense reward weighting. This dense reward is derived from visual attribution methods—such as SHAP, LIME, and RAHF—and is intended to deliver fine-grained feedback for safety-aware training.
Comprehensive experiments conducted across multiple benchmarks, including MMDT, WISE, GenAI-Bench, and T2I-Safety, demonstrate a substantial improvement in safety metrics—with up to approximately 59% reduction in unsafe generations—while only minimally compromising image quality. The paper argues that fine-grained reward modeling enables the joint optimization of safety and visual quality in multimodal alignment.

### Strengths
Focusing on the vital issue of safety alignment for multimodal large language models, this paper compellingly bridges a gap in a field that has largely centered on text-only models. The authors' approach to jointly optimizing safety and quality within a single RL framework is well-conceived. Their innovation of dense, token-level feedback effectively addresses the challenge of sparse rewards in multimodal contexts. Targeting a key obstacle to the real-world deployment of unified models, this research represents a valuable contribution to both theoretical and applied AI safety.

### Weaknesses
While the paper is well-motivated, its technical formulation and experimental validation remain limited.

 First, the claimed dense reward is only used as token-level weighting rather than integrated into the advantage computation. This represents a conceptual misuse of “dense rewards” and does not address sparse-return issues in reinforcement learning.

Second, the “Dense-GRPO” method introduces only marginal modifications to GRPO, lacking substantial algorithmic innovation.

Third, since the approach essentially reweights unsafe samples, a weighted Supervised Fine-Tuning (SFT) baseline should have been included to evaluate whether reinforcement learning is truly necessary.

Moreover, the study does not examine how the weighting term influences policy stability, convergence behavior, or reward variance.
On the experimental side, the emphasis is placed heavily on safety metrics, while standard image-quality assessments such as Geneval are overlooked. Additionally, the individual contributions of different reward components (e.g., those based on SHAP, LIME, and RAHF) are not adequately disentangled.

Overall, while the work offers promising empirical findings, it lacks the methodological depth and theoretical grounding needed to fully substantiate its claims.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a token-weighted GRPO framework for aligning Unified Multimodal Large Language Models (U-MLLMs) in text-to-image (T2I) generation. The method introduces dense, spatially localized rewards (from RAHF and SHAP/LIME attributions) and safety-specific penalties (from toxic-CoT and NSFW detectors). By assigning token-level weights during GRPO optimization, the model seeks to improve both image quality and safety alignment. Experiments on several benchmarks (WISE, GenAI-Bench, MMDT, and T2I-Safety) show improved safety and comparable visual quality relative to baselines.

### Strengths
1. This paper addresses a meaningful and timely problem of balancing visual quality and safety alignment in multimodal LLMs through a clear and conceptually simple framework.
2. The token-weighted GRPO design is straightforward and can be easily integrated into existing RLHF-style pipelines, offering a practical engineering solution.
3. The paper reports results across multiple benchmarks, showing that the approach generalizes to both quality and safety objectives.

### Weaknesses
1. The approach is more of an engineering extension of existing GRPO/DPO frameworks rather than a fundamentally new algorithm.
2. Safety gains might stem from reusing the same toxic/NSFW evaluators in both training and testing, and key training details (e.g., λ, β schedules, G×K, random seeds) are missing.
3. The framework's applicability to diffusion or flow-based models is claimed but not validated.
4. The experiments are limited to Janus-Pro-7B.
5. The model improves safety but may over-suppress valid or creative outputs; this balance is not analyzed.

### Questions
1. Can authors please clarify the theoretical motivation for token-level weighting—e.g., how localized rewards stabilize optimization or mitigate sparse-signal variance?
2. Is it possible to include independent cross-evaluator tests and provide training hyperparameters to ensure reproducibility?
3. Can authors please provide a brief validation or discussion of how token weighting would transfer to diffusion backbones or other U-MLLM architectures?

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
2

### Summary
This paper addresses safety and quality alignment in Unified Multimodal Large Language Models (U-MLLMs)—models capable of both image-to-text (I2T) and text-to-image (T2I) generation. The authors argue that while recent U-MLLMs achieve strong generative performance, their safety alignment remains under-explored, and existing reinforcement learning (RL)-based alignment methods rely on sparse scalar rewards.To address these issues, the paper proposes a token-level dense reward framework integrated into Group Relative Policy Optimization (GRPO). Experiments on benchmarks such as WISE and MMDT show competitive image quality (WISE score: 0.50) and a 59.4% reduction in unsafe content compared to the baseline.

### Strengths
1. The implementation of usin fine-grained reward within GRPO is technically sound. 
2. The ensemble of reward models (Table 1)—spanning aesthetic, compositional, grounding, and safety signals—demonstrates significant engineering rigor. The dual-path evaluation (safe vs. unsafe prompts) is thoughtful.
3. The paper is clearly written, with intuitive figures (e.g., Fig. 2–3) and a logical flow from problem formulation to method to evaluation. The distinction between quality-oriented and safety-oriented reward pathways is well articulated.

### Weaknesses
1. The central claim that “safety alignment has been under-explored” appears overstated. While U-MLLMs may be a recent architecture, T2I safety alignment has been actively studied [1,2]. These and other works suggest that safety in T2I is not unexplored, even if not yet fully adapted to autoregressive U-MLLMs. Alternatively, in my view, the safety problem in U-MLLMs is not fundamentally different from that in standalone LLMs, T2I models, or I2T models. Therefore, existing safety alignment frameworks developed for LLMs, T2I, or I2T models are largely applicable to this setting.
2. Loose Coupling Between Contributions: The paper presents two seemingly orthogonal contributions: (1) Introducing dense rewards for fine-grained quality optimization; (2) Adding safety-specific rewards to suppress harmful content. However, these are not meaningfully integrated. A more compelling story would be: “Existing RL alignment for U-MLLMs lacks fine-grained safety signals; we propose dense safety-aware rewards that jointly optimize quality and safety at the token level.” Instead, safety remains coarse-grained, undermining the paper’s emphasis on “dense” alignment.

> [1] Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding proposes prompt-level safety intervention via embedding projection.
> [2] AlignGuard: Scalable Safety Alignment for Text-to-Image Generation introduces scalable red-teaming and safety fine-tuning for diffusion models.

### Questions
No question

### Soundness
2

### Presentation
2

### Contribution
2
