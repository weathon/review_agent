# Temporal Preference Optimization of Large Multimodal Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Despite recent advancements in video large multimodal models (video-LMMs), accurate temporal grounding remains a key challenge. In this work, we introduce Temporal Preference Optimization (TPO)—a post-training framework that unlocks superior temporal reasoning in video-LMMs without requiring human annotations. TPO enables preference modeling by manipulating video inputs to generate contrastive responses, ensuring that preferred responses are more temporally grounded than dis-preferred ones. Through preference learning, TPO enhances the model’s capability for more comprehensive video understanding with better temporal reasoning. Extensive experiments on LongVideoBench, MLVU, and Video-MME demonstrate that TPO significantly improves temporal grounding across multiple video-LMMs.   Notably, LLaVA-Video-TPO achieves state-of-the-art performance among 7B models on Video-MME, establishing TPO as a scalable and effective solution for advancing temporal understanding in video analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Temporal Preference Optimization (TPO), a post-training framework designed to enhance the temporal grounding of video-LMMs without human annotations. TPO works by generating contrastive responses from manipulated video inputs, using preference learning to teach the model to favor answers that are more temporally accurate. Experiments demonstrate that TPO improves temporal reasoning on multiple benchmarks, establishing it as a scalable method.

### Strengths
The intuition of the paper is straightforward: use DPO, but for temporal understanding. With a simple yet effective pipeline, they generate both preferred and dis-preferred responses using full and incomplete clips. The results show significant performance improvements on various temporal benchmarks and outperforms other methods such as pure SFT or classic DPO.

### Weaknesses
1. Though the authors introduced two types of data for training, namely generating with 1) irrelevant and 2) incomplete information, there is no explanation as to how this was chosen as the criteria for data curation. Is it because you examined some models' failure cases and found out that the failure modes, or is there some other reason?
2. Do we really need to optimize with both a) and b) data? Even as the authors have claimed, a) simulates *extreme* scenarios. For me, I understand the intuition of b) since models do not do well on retrieving *all* the correct frame(s) for temporal tasks; on the other hand, a) seems too extreme and may not be necessary since there is literally nothing to reason about, and the models may not gain as useful signals as b). Did the authors do any experiments to see if they only train on a) and only train on b), would the performance gains be different?
3. It would be great if the authors can evaluate on new temporal benchmarks like TempCompass [1], TemporalBench [2], and Vinoground [3].
4. It would also be good if the authors can demonstrate that using TPO with the data curated does not hurt general benchmark performances.

[1] Liu et al., 2024, TempCompass: Do Video LLMs Really Understand Videos?

[2] Cai et al., 2024, TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models

[3] Zhang et al., 2024, Vinoground: Scrutinizing LMMs over Dense Temporal Reasoning with Short Videos

### Questions
See Weaknesses.

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
3

### Summary
This paper proposes Temporal Preference Optimization (TPO), a lightweight post-training framework that improves video-LMMs’ temporal grounding and reasoning without human temporal annotations by generating contrastive preference pairs from the same query answered on original (relevant) versus corrupted or incomplete frames, filtering noisy pairs with a small LLM, and then optimizing with Direct Preference Optimization plus a minor auxiliary SFT loss so the model prefers responses aligned with when evidence occurs, not just what appears; across LongVideoBench, MLVU, and Video-MME, TPO consistently outperforms baselines, with LLaVA-Video-TPO achieving state-of-the-art among 7B models, and the recipe is practical (e.g., ~4 hours on 8×A100 with fixed 32 sampled frames). A scalable annotation-free temporal supervision pipeline via input manipulation and LLM post-filtering; a preference-learning objective instantiating DPO (with small SFT) tailored to temporal grounding; and strong empirical gains across three long-video benchmarks and multiple bases

### Strengths
- Simple, efficient training recipe. The method trains in roughly four hours on 8×A100 (80 GB) with fixed 32 sampled frames shared across data generation and training, indicating practical scalability.
- Targeted objective that preserves general ability. DPO on temporal preference pairs is positioned to enhance temporal reasoning while retaining pretrained knowledge.

### Weaknesses
- Limited benefit on short-video settings. The authors note performance is only comparable to SFT baselines on the Video-MME-short subset, suggesting gains concentrate on longer temporal contexts.
- Fixed-frame sampling could bottleneck long-horizon reasoning. The design uses a constant 32 frames for both generation and training; the implications for very long or high-motion videos are not extensively studied.
- Dependence on an external LLM for curation. The data pipeline requires GPT-4o-mini for question curation and post-filtering, introducing cost/availability/bias considerations that are not thoroughly quantified.

### Questions
## Temporal grounding definition & evidence.
Could you specify which aspects of “temporal grounding” are under-defined here and what concrete operational tests (e.g., frame-level localization, counterfactual masking) you would need to see to accept that the method truly improves when-aware reasoning rather than generic accuracy?

## LLM post-filter robustness & bias.
Which failure modes of the LLM-based filtering (e.g., label leakage, stylistic bias, prompt sensitivity) most concern you, and what ablations or audits (alternative teachers, temperature sweeps, bias probes) would convincingly address them?

## Evaluation reliability & generalization.
Where do you believe the current results are insufficiently reliable (e.g., short-video subsets, high-motion segments), and which additional analyses—stratified metrics with uncertainty (bootstrap CIs), new baselines, or zero-shot datasets—would meaningfully change your assessment?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Temporal Preference Optimization (TPO), a post-training framework to enhance temporal reasoning in Video-LMMs without requiring manual annotations. TPO generates contrastive supervision by manipulating video inputs to create preferred and dis-preferred responses, which are refined through a lightweight LLM-based post-filtering step. The approach is evaluated on LongVideoBench, MLVU, and Video-MME, showing modest improvements over baseline models.

### Strengths
- Proposes a scalable, annotation-free approach to improve temporal reasoning in Video-LMMs.
- Efficiently generates contrastive data via input manipulation, avoiding costly manual annotations.
- Addresses the critical challenge of temporal grounding in Video-LMMs.

### Weaknesses
- The paper lacks detailed evaluations directly targeting temporal reasoning (*e.g.*, adversarial temporal testing or failure case analysis).
- The experiments focus on general video understanding rather than explicitly validating improvements in temporal reasoning.
- Limited discussion with related work (*e.g.*, Hound-DPO, VistaDPO) reduces clarity on TPO’s unique contributions.
- Over-reliance on synthetic contrastive data without evaluation of its quality or generalizability.
- The impact of the LLM-based post-filtering step is not analyzed in detail.
- Missing studies to isolate the contribution of individual components (*e.g.*, post-filtering, preferred vs. dis-preferred responses).

### Questions
Please see Weaknesses for details.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Temporal Preference Optimization (TPO), a post-training framework that enhances temporal reasoning and grounding capabilities in large video-language models (video-LMMs) without requiring manual annotations. TPO generates contrastive preference pairs by comparing model responses to original versus temporally corrupted video clips, then refines models through Direct Preference Optimization (DPO). Experiments on LongVideoBench, MLVU, and Video-MME demonstrate consistent performance gains across multiple backbones.

### Strengths
1. **Clarity of methods.** The contrastive setup between relevant and manipulated frames, combined with LLM-based post-filtering, is simple yet effective.
2. **Strong results.** TPO demonstrates consistent improvements across diverse benchmarks and models (LongVA-TPO and LLaVA-Video-TPO), outperforming baselines.

### Weaknesses
1. **Limited technical contribution.** The idea is simply generating positive/negative captions pairs and optimizing with DPO, which seems to be limited in their novelty or technical contribution.
2. **Comparison with recent RL-based approaches.** Direct comparisons with recent reinforcement or segmentation-based optimization methods (e.g., Time-R1, Grounded-VideoLLM) are missing.

### Questions
Typo in FIgure 3-(a); MLUV → MLVU

### Soundness
3

### Presentation
3

### Contribution
2
