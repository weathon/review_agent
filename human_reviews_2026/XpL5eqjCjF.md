# More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Reasoning has emerged as a pivotal capability in Large Language Models (LLMs). Through Reinforcement Learning (RL), typically Group Relative Policy Optimization (GRPO), these models are able to solve complex tasks such as mathematics and code generation. Building on these advances, recent research has sought to extend reasoning to Vision-Language Models (VLMs), yielding promising results across diverse visual tasks. Despite this progress, our study uncovers the dual nature of multimodal reasoning: while it substantially enhances logical inference and facilitates performance on challenging problems, longer reasoning length may gradually impair perceptual grounding, leading to recognition failures on otherwise basic visual questions. Through further analysis, we attribute this phenomenon to visual forgetting, wherein prolonged reasoning ength causes models to disregard visual input. To address this, we propose Vision-Anchored Policy Optimization (VAPO), a simple yet effective method that explicitly steers the reasoning process toward visually grounded trajectories. Our result model, VAPO-Thinker-7B, significantly strengthens the model's reliance on visual information and achieves new state-of-the-art results on various benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates the dual nature of multimodal reasoning in vision-language models, showing that while early reasoning often improves performance, prolonged reasoning can weaken perceptual grounding and reduce accuracy on visually simple questions. Using an “early decision” protocol that forces answers at intermediate points, the authors find that accuracy first increases and then declines as reasoning continues. Error analysis shows perception errors dominate and many could be avoided by stopping earlier. This phenomenon is attributed to visual forgetting, where attention to visual tokens decays over time. Two inference-time fixes—visual replay and focus prompts—temporarily restore visual attention and improve accuracy. To address the issue during training, the authors propose VAPO (Vision-Anchored Policy Optimization), which inserts visual claim anchors along the reasoning process and introduces a perception reward that encourages correct yes/no judgments of image-grounded claims. With late-emphasis weighting and an accuracy-gated reward, VAPO-Thinker-7B achieves state-of-the-art results on multiple benchmarks, particularly those requiring strong visual grounding, while maintaining efficiency. Ablations confirm the effectiveness of more anchors, later-anchor emphasis, stable training dynamics, and superiority over attention-only rewards and inference-time fixes.

### Strengths
- Original analysis of multimodal reasoning dynamics: the early-decision protocol provides a fine-grained view of how different reasoning segments contribute to final accuracy, revealing a non-monotonic curve where longer reasoning can hurt (Sec. 3.1, Fig. 2A) and that the harm is largely perceptual (Fig. 2B–C).
- Clear causal story with measurements and interventions: the visual-attention decay analysis (Fig. 3A) and simple test-time remedies (visual replay, focus prompt) that immediately spike attention and improve accuracy (Fig. 3B) support the visual forgetting hypothesis (Sec. 3.2).
- Methodological novelty and simplicity: VAPO’s claim-anchored perception reward operationalizes “visual grounding” during RL without architectural changes. The late-emphasis weighting targets the observed late-stage forgetting (Sec. 4.2, Eq. 3–4).
- Strong empirical results and breadth: consistent gains across 10 benchmarks with notable improvements on vision-intensive datasets (MMStar, HallusionBench) and competitive math performance (Tables 1–2, 11). Thorough ablations on anchor number K, weighting β, reward weight γ, KL coefficient, and baselines with test-time remedies (Fig. 7; Tables 3, 4, 7–9, 14).
- Efficiency: small training-time overhead over GRPO and larger accuracy gains than a stronger RL baseline (DAPO) (Table 10).
- Analysis depth: learning curves (Fig. 6), attention trends versus baseline (Fig. 5A), accuracy versus reasoning progress (Fig. 5B), and anchor-score diagnostics (Fig. 13) collectively strengthen the argument.

### Weaknesses
- Dependence on a closed-source claim generator (GPT-5) and limited auditing of claim correctness and vision-dependency. While Appendix A.6 provides prompts and Table 6 explores variants, there is no quantitative estimate of claim label noise or a human audit; method sensitivity to generator quality is acknowledged but not systematically evaluated.
- The “recoverable by early decision” metric aggregates across multiple potential cut points (Fig. 2 caption) and may overestimate recoverability. A stricter, single-cut metric or randomized stopping analysis would better calibrate the effect size.
- Potential overlap and contamination risks are not fully ruled out: ViRL39K composition (Sec. 5) merges several sources; explicit deduplication against evaluation sets is not reported. This is especially relevant for MMStar and MathVista where many public pipelines exist.
- The accuracy-gated perception reward can bias learning toward easier instances where final answers are already correct, potentially under-incentivizing perception on hard items. An ablation that relaxes the gate or uses shaped rewards conditioned on partial correctness would clarify trade-offs.
- Anchor placement strategy is mostly random with late weighting; more analysis of position selection (e.g., semantic segment boundaries versus uniform) would strengthen claims about targeting late-stage forgetting beyond the β ablation (Fig. 7B).
- Comparison to closely related RL formulations emphasizing perception (e.g., perception-aware policy optimization [Wang et al., 2025c]) could be deepened to delineate conceptual/implementation differences and when claim-anchoring is preferable.
- The attention metric is an indirect proxy; while useful, further triangulation (e.g., causal masking of image tokens at anchor points, feature attribution) would reduce the risk of over-interpreting attention as reliance.
- Some related works about RL+Reasoning in VLM are missed. 
[1] Wang, Jiaqi, et al. "Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models." arXiv preprint arXiv:2505.16854 (2025).

### Questions
- Claim quality and robustness:
  - What fraction of GPT-5 claims were mislabeled upon manual audit? Can you report inter-annotator agreement or a small-scale human validation of claim correctness and vision-dependency (Sec. A.6; Table 6)?
  - How sensitive is VAPO to weaker generators? Please provide results with an open-source vision-capable LLM to generate claims and quantify the drop from GPT-5.
  - Do you filter claims for redundancy or triviality (e.g., color naming in synthetic diagrams)? Any diversity constraints?
- Reward design:
  - How often does the accuracy gate mask perception reward during training? A histogram versus steps would reveal whether hard items are under-optimized.
  - Could curriculum gating (e.g., gradually reducing the accuracy condition) further improve performance on harder instances?
  - Have you tried per-claim calibrated rewards (e.g., confidence-weighted yes/no) or multi-class verification to reduce sparsity?
- Anchor placement:
  - Beyond late-emphasis β, did you compare placement by semantic boundaries (sentence/step ends) versus uniform random? Any gains on MMStar/HallusionBench?
  - How does K interact with sequence length at test time? Do longer-thought samples benefit disproportionately from more anchors?
- Evaluation rigor:
  - Please provide explicit data deduplication checks between ViRL39K and each benchmark, including image hash and near-duplicate detection.
  - The “recoverable” metric (Fig. 2B–C) aggregates over multiple early stops. Can you report accuracy when stopping at a single, pre-specified percentile for all samples (e.g., 40% of tokens), to avoid optimistic selection?
  - Reasoning length distribution: does VAPO change average reasoning length versus baselines? If so, how does that correlate with attention and accuracy?
- Comparisons and scope:
  - Clarify distinctions from perception-aware policy optimization (Wang et al., 2025c): is the novelty primarily the claim-anchored verification and late-weighting? Could their reward be used as an auxiliary alongside anchors?
  - Any results on multi-image or video reasoning using multi-frame claims (future Sec. B.10 suggests feasibility)? A small-scale pilot would be informative.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates a "dual nature" of reasoning in Vision-Language Models (VLMs), claiming that while reasoning enhances logical inference, it can impair perceptual grounding. The authors attribute this phenomenon to "visual forgetting," where prolonged reasoning causes the model to disregard visual input. They propose VISION-ANCHORED POLICY OPTIMIZATION (VAPO), an RL-based method with a "perception reward" to force the model to maintain visual grounding

### Strengths
1. The "early decision" mode of analysis (Fig. 2A) is a clever and effective method for demonstrating this phenomenon of the degradation of perceptual accuracy as the generation process continues. 

2. VAPO showcase an effective solution of rewarding the model for maintaining perceptual grounding,

### Weaknesses
The primary weakness of this paper lies in its central conceptual claim. The argument that "reasoning harms perception" is a significant overstatement that is neither logically sound nor intuitively convincing.

1. **Flawed Causal Claim (Logical):** The evidence presented demonstrates a **correlation between reasoning length and perceptual decline, but it fails to establish causation.** The "prolonged reasoning" is perfectly confounded with the length of the generated text context. The paper provides no evidence to disprove the more parsimonious hypothesis: any sufficiently long context (it could be textual reasoning or it could be many image tokens) would cause the model's attention to "drift" from the visual input, leading to perceptual errors. The paper mistakes correlation for causation.

2. **Flawed Premise (Intuitive):** The claim is also nonsensical from a cognitive perspective. In human intelligence, perception and reasoning are synergistic, not conflicting. Perception provides the "ground truth" (premises) upon which reasoning operates. Reasoning, in turn, can direct perceptual focus (e.g., "look for the red car"). Claiming an inherent conflict in AI models based on this confounded evidence is an extraordinary claim that would require extraordinary proof, which the paper does not provide. It is far more likely that the model is simply failing at a known, mundane problem (long-context attention drift) than it has developed a special cognitive conflict different from how human visual system and brain work. 

The main experiment results do not convincingly show the significance of perceptual grounding, without comparing with state-of-the-art RL approaches on VLM that do not explicitly facilitate perceptual grounding, such as MM-Eureka and VL-Rethinker. 

Since the central argument is flawed and misleading, my true rating for this paper is 3. From my perspective, the current version is not ready for publication.

### Questions
1. On the Confounding Variable: Did the authors run any control experiments to disentangle the act of reasoning from the length of the textual context? For example, what happens to perceptual accuracy if the model is fed a long context of another question-solution or a few other images ?
2. On the Experimental Design: The "visual replay" intervention (Fig. 3) simply re-inserts the image. This intervention successfully alleviates the performance drop. Does this not strongly suggest that the problem is a standard attention failure (i.e., the model "forgot" the image in its context) rather than a more complex cognitive "harm" caused by the reasoning process itself?
3. On the Method's Motivation: Given the lack of evidence for a true "reasoning vs. perception" conflict, isn't VAPO more accurately described as a powerful regularization technique to combat visual context drifting? Why frame it as solving a cognitive duality when the evidence only points to a context length/attention failure?
4. On the Error Analysis: In Fig. 2, the paper claims reasoning induces perception errors. Could the causality be reversed? That is, could the model make an early logical error (e.g., path 1 in Fig 1, deciding 7.4+7.2+8.1=22.7 instead of 20.5) and then hallucinate visual "evidence" to justify its flawed reasoning path, rather than the reasoning itself degrading its ability to see?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates a key contradiction in Vision-Language Models (VLMs), termed the “dual nature” problem. While Chain-of-Thought (CoT) reasoning improves logical inference, excessively long reasoning chains can impair visual grounding, leading to errors on simple visual tasks. The authors attribute this to “visual forgetting,” where attention to the image fades as reasoning progresses.
To validate this, the paper introduces two inference-time interventions:
1. Visual Replay – reintroducing the image mid-reasoning.
2. Focus Prompt – instructing the model via text to re-attend to the image.
 Both methods mitigate degradation, supporting the visual forgetting hypothesis.
To address the issue fundamentally, the authors propose VISION-ANCHORED POLICY OPTIMIZATION (VAPO), an RL-based algorithm that inserts visual anchors (short, image-related statements) into reasoning paths. The model judges their correctness and receives a perception reward, encouraging sustained visual grounding.

### Strengths
1. Important and Novel Problem: Identifies a counterintuitive phenomenon—“more reasoning can reduce accuracy.” This challenges a major assumption in VLM research.
2. Strong Empirical Validation: The “visual forgetting” hypothesis is convincingly supported through attention visualizations and well-designed interventions (Visual Replay, Focus Prompt).
3. Elegant Method Design: VAPO elegantly integrates perception evaluation into RL training through visual anchors and perception rewards, directly addressing the root cause.
4. Comprehensive Evaluation:
  - Covers 10 benchmarks across reasoning and VQA tasks.
  - Compares against both closed- and open-source SOTA models.
  - Provides detailed analyses and ablations confirming the method’s robustness.

### Weaknesses
1. Reproducibility: Reliance on a closed-source, super-capable large model (GPT-5) may create difficulties for other researchers trying to reproduce or extend this work.     
2. Cost and Efficiency: Generating a large number of visual claims for the training data introduces considerable computational overhead.    
3.  Quality Bottleneck: The upper bound of VAPO's effectiveness may be limited by the quality of the claim generation model. Low-quality claims could negatively impact training.
4. Added Complexity: Introduces additional steps, hyperparameters (K, β, γ), and tuning difficulty compared to standard GRPO.
5. Efficiency Discussion Limited: Although runtime overhead is small (Table 10), total pipeline cost—including data generation—remains higher than baseline.

### Questions
1. Sensitivity to Claim Quality: How does performance change if visual claims are generated by weaker models (e.g., LLaVA-1.6, Qwen-VL-Max)? Are there filtering mechanisms?
2. Universality of Visual Forgetting: This study experiments with models based on Qwen2.5-VL. Do you believe the "visual forgetting" phenomenon is prevalent across all types of VLM architectures? 
3. Generalization of VAPO: Does the perception training generalize to fine-grained or spatial reasoning tasks beyond truth verification?
4. Alternative Anchors:  Besides using GPT-5 to generate explicit text claims, have you considered other forms of "anchors"? For example, could you explore self-supervised or reconstruction-based anchors to reduce dependence on external models?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies a dual nature of reasoning in VLMs: while extended reasoning enhances logical capability, it simultaneously weakens visual perception. The authors provide empirical evidence across multiple reasoning benchmarks showing that early-stage reasoning improves accuracy, but performance declines in later stages, with most errors arising from visual perception failures. To address this, the paper first introduces an early-decision mode, where the model is prompted to terminate reasoning mid-trajectory, which mitigates visual forgetting and yields performance gains. Visual Replay and Focus Prompt are also designed to increase attention to visual tokens. Finally, the paper proposes VAPO, a simple yet effective policy-gradient algorithm that inserts visual anchors along the reasoning trajectory to encourage consistent visual perception. Experimental results demonstrate that the proposed VAPO-Thinker-7B improves the model's reliance on visual input and achieves average gains of 2-4% over strong baselines on established benchmarks.

### Strengths
- The paper is clearly written and easy to follow.

- The experimental setup is sound and convincingly supports the core claim that prolonged reasoning leads models to neglect visual cues and commit perception-related errors. The quantitative error analysis is comprehensive, showing that perception errors dominate.

- The training framework is effective and well-motivated. I like the design of visual anchors and the perception reward.

### Weaknesses
- Since both the insertion of visual anchors and the sampling of reasoning prefixes are random, some anchors may appear in contexts that are unrelated to the visual content described by the corresponding claim, or lack sufficient logical cues for the model to judge the correctness.

- The generation of visual claims relies on GPT-5, which may introduce noise or bias in their quality. This has been pointed out in their limitations.

### Questions
- I am interested in the error distribution of the trained VAPO-Thinker. What is the current proportion of perception errors, and do recoverable errors still exist?

- Did the authors verify the reliability of GPT-5–based annotation and claim generation? How was the quality of these generated annotations/claims evaluated?

### Soundness
3

### Presentation
4

### Contribution
3
