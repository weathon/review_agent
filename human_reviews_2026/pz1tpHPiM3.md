# From Scores to Preferences: Redefining MOS Benchmarking for Speech Quality Reward Modeling

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Assessing the perceptual quality of synthetic speech is crucial for guiding the development and refinement of speech generation models. However, it has traditionally relied on human subjective ratings such as the Mean Opinion Score (MOS), which depend on manual annotations and often suffer from inconsistent rating standards and poor reproducibility. To address these limitations, we introduce MOS-RMBench, a unified benchmark that reformulates diverse MOS datasets into a preference-comparison setting, enabling rigorous evaluation across different datasets. Building on MOS-RMBench, we systematically construct and evaluate three paradigms for reward modeling: scalar reward models, semi-scalar reward models, and generative reward models (GRMs). Our experiments reveal three key findings: (1) scalar models achieve the strongest overall performance, consistently exceeding 74% accuracy; (2) most models perform considerably worse on synthetic speech than on human speech; and (3) all models struggle on pairs with very small MOS differences. To improve performance on these challenging pairs, we propose a MOS-aware GRM that incorporates an MOS-difference-based reward function, enabling the model to adaptively scale rewards according to the difficulty of each sample pair. Experimental results show that the MOS-aware GRM significantly improves fine-grained quality discrimination and narrows the gap with scalar models on the most challenging cases. We hope this work will establish both a benchmark and a methodological framework to foster more rigorous and scalable research in automatic speech quality assessment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a benchmark called MOS-RMBench, which consists of five in-domain datasets and one out-of-domain dataset. The MOS scores are converted into paired preferences. The paper compares three training strategies on the benchmark and proposes a MOS-aware GRM to enhance the model’s ability to capture fine-grained perceptual quality differences.

### Strengths
This paper focuses on an important task: automatically assessing speech quality. This is a crucial research direction for enabling reproducible, consistent, and efficient evaluation of synthesised speech.

### Weaknesses
Regarding the benchmark:

1. The authors state that MOS scores “suffer from inconsistent rating standards and poor reproducibility,” but the benchmark construction still relies on MOS scores to derive preference pairs. This raises a conceptual inconsistency.
    
2. Preference pairs are only formed within each dataset. While I understand that cross-dataset pairing poses difficulties, allowing comparisons across datasets would better leverage the larger data volume and diversity.
    
3. It seems that no margin filtering was applied when converting MOS differences into binary preferences. Given the noisy nature of MOS annotations, for pairs with very small ΔMOS, it is questionable whether such differences reflect a meaningful perceptual preference even for human listeners. Moreover, the current formulation treats all pairs in the same way—whether the ΔMOS is 0.2 or 2.0—forcing both into a binary positive/negative decision and ignoring the different levels of confidence implied by the margin size. Especially in the situation where, as shown in Figure 2c, a large proportion of pairs lie in the small-margin region (e.g., ΔMOS < 0.5), meaning that ambiguous cases are not the minority but a dominant part of the benchmark. Introducing a threshold to treat small-gap pairs as similar quality might lead to a more meaningful and stable evaluation protocol.
    

Regarding evaluated models:

4. It seems that only a single backbone model (Qwen2-Audio) was used, which may not be sufficient to support the general conclusions drawn in Section 5.1. It would be more convincing to evaluate both LLM-based and non-LLM-based backbones to verify whether the observed trends and patterns generalise across architectures. In addition, important training details are missing, such as model size and whether the backbone was fully fine-tuned or adapted via PEFT.

5. The terminology “reward model” in Section 4.2 is somewhat confusing. Both classic scalar reward models and semi-scalar reward models appear to refer to supervised fine-tuning strategies applied to the Qwen2-Audio backbone, whereas generative reward models involve RL training with an accuracy-based reward model—effectively introducing a reward model inside another reward model setup. Clarifying this terminology hierarchy would improve conceptual clarity.

6. For GRM, it is stated that “each pair is compared across the same four dimensions and provided with overall quality scores” while the RL reward is binary. It is unclear whether there is any explicit supervision or guidance on how the model should assess the four dimensions or produce the overall score during RL training. Design RL training based on the text description rather than a rule-based reward function might help.

Regarding experiments and results:

7. In Table 2, what is “classic + xx loss” and “cloud + xx loss” means. Explanation seems missing.

8. The authors state that “all models are particularly prone to errors on sample pairs with small MOS differences,” which seems expected given the issue noted earlier: very small ΔMOS values may not correspond to meaningful perceptual preferences and are likely to be dominated by label noise. The observation that “fine-grained discrimination of speech quality remains a critical bottleneck” also appears to follow directly from the decision to polarize all preference pairs regardless of the actual MOS difference.

9. Regarding the MOS-aware reward function, it is not clearly stated where the MOS gap used in the reward function comes from. If the gap is predicted by the model itself, it is unclear how this prediction is trained and, if it were accurate, whether additional RL optimisation would still be necessary. If the gap is instead provided as metadata derived from ground-truth MOS, it raises concerns about practical applicability, as this information would not be available at inference time in real-world usage scenarios.

### Questions
See weakness section.

Line 212 "spanning six languages: English, Chinese, Taiwanese Mandarin, Japanese, French, and German." -- not sure whether Chinese and Taiwanese Mandarin should be treated as separate languages or different accents.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitations of traditional Mean Opinion Score (MOS) evaluations—such as inconsistent standards and poor reproducibility—by introducing MOS-RMBench, a unified benchmark that transforms heterogeneous MOS datasets into a preference-comparison framework. The authors systematically evaluate three reward modeling paradigms (scalar, semi-scalar, generative) and identify key challenges: scalar models achieve the highest overall accuracy (~80%), most models underperform on synthetic speech, and all struggle with fine-grained discrimination for pairs with small MOS differences. To mitigate this, a MOS-aware Generative Reward Model (GRM) is proposed, which incorporates MOS-difference-based rewards to adaptively scale learning signals. Experimental results show consistent performance improvements, especially on challenging pairs, establishing both a benchmark and methodological foundation for speech quality assessment.

### Strengths
Novel Unified Benchmark: MOS-RMBench fills a critical gap by integrating six diverse MOS datasets into a consistent preference-based framework, enabling rigorous cross-dataset and cross-domain evaluation—an advancement over fragmented existing resources.
Comprehensive Paradigm Comparison: The systematic evaluation of scalar, semi-scalar, and generative reward models, along with LLM-as-a-judge and MOS prediction baselines, provides valuable insights into their relative strengths and weaknesses for speech quality assessment.
Targeted Innovation with MOS-aware GRM: The proposed MOS-difference-based reward function directly addresses the core challenge of fine-grained quality discrimination, delivering measurable improvements on high-difficulty sample pairs and enhancing model interpretability.
Rigorous Methodology: The paper demonstrates strong reproducibility through detailed dataset statistics, training configurations, prompt templates, and error analysis, adhering to high academic standards for empirical research.

### Weaknesses
Limited Language and Quality Dimensions: MOS-RMBench covers only six languages and focuses solely on perceptual quality, neglecting important dimensions like prosody, emotion, and style—key aspects of real-world speech generation evaluation.
Potential Bias from LLM Annotations: The reliance on Gemini-2.5-Pro for generating natural-language critiques and preference labels introduces potential annotation biases, which are not fully discussed or validated against human judgments.
Narrow Analysis of Domain Gap: While the paper notes poorer performance on synthetic speech, it lacks in-depth exploration of the root causes (e.g., acoustic differences, data distribution shifts) or strategies to bridge this gap beyond the proposed GRM.

### Questions
How does the annotation quality of Gemini-2.5-Pro compare to human annotators for preference labeling and critique generation? Are there any cases where LLM annotations diverge significantly from human judgments, and how might this impact model training?
Given the focus on fine-grained discrimination, have the authors tested the MOS-aware GRM on pairs with extremely small MOS differences (e.g., ∆MOS < 0.2) to further validate its effectiveness in the most challenging scenarios?

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
3

### Summary
The work is well-motivated: MOS is costly, inconsistent across datasets, and difficult to unify; preference-based framing is more comparable. The benchmark integrates six datasets spanning multiple languages, vocoder types, and tasks (TTS, SE, VC, SVC, singing, noisy/reverberant speech). It introduces MOS-RMBench, a benchmark that reformulates multiple MOS datasets into pairwise preference comparisons to enable consistent training and evaluation of speech-quality reward models. Based on this unified preference view, the authors benchmark three reward-modeling paradigms — scalar, semi-scalar, and generative reward models (GRMs) — implemented on Qwen2-Audio. Major findings include: (i) scalar RMs perform best overall, reaching ~80% accuracy; (ii) models perform substantially worse on synthetic speech; and (iii) all models struggle when MOS differences are small. To address this, the authors propose a MOS-aware GRM that incorporates a MOS-difference-dependent reward shaping term, improving discrimination on difficult pairs.

### Strengths
* A very interesting study tackling a difficult problem, with a commendable effort to collate and repurpose existing MOS crowdsourced annotation (which are generally low-resource and heterogeneous) into a more structured preference-based benchmark.
* Systematic evaluation of three RM paradigms; the proposed formulation of classic, semi-scalar, and generative reward modeling is well-structured and novel as a comparative framework.

### Weaknesses
* **(No New Human Preference Labels)** All labels originate from MOS and are reinterpreted into preferences. Without new preference annotation, MOS-derived ordering is assumed valid at instance level, even though MOS reliability is often debated at utterance granularity. This risks embedding existing scoring noise and dataset-specific biases directly into the benchmark.
* **(Pair Construction Rules May Create Artificial Comparisons)** The benchmark groups samples not only by shared content but also by shared speaker or system. Grouping by content aligns with CMOS/MUSHRA practices, but grouping by speaker or system is less natural for perceptual quality comparison, since two samples may differ semantically and still be forced into a preference pair. It is unclear that such comparisons reflect meaningful perceptual judgments. Moreover, MOS is a score reported after averaging across multiple utterances and still widely criticized; here, using utterance-level MOS to infer pairwise preferences introduces additional unreliability.
* **(No External Validation of Benchmark Quality)** There is no evidence of human validation of the resulting preference pairs or benchmark sanity checking. Without external human verification or comparison against established tests (e.g., MUSHRA/CCR), it is difficult to assess whether these preference labels are perceptually valid. One indicator would be correlation of benchmark-derived rankings with other evaluation protocols.
* **(No Study of Preference Reversals or Ranking Fidelity)** Accuracy is reported, but not whether predicted preferences preserve global ranking or avoid cycles

### Questions
1. How are “unreliable annotations” identified and filtered? The paper states that samples are filtered but does not describe the criteria, annotation confidence, or thresholds used.

2. Pair construction: grouping by shared content makes sense, but grouping by shared speaker/system seems less principled. What is the perceptual motivation? Why is this considered a meaningful comparison when MOS was not collected in a paired setting?

3. Was any human review or quality check conducted on a subset of constructed preference pairs to evaluate benchmark validity?

4. Is there evidence that model rankings on MOS-RMBench correlate with rankings from other benchmarks?

### Soundness
2

### Presentation
3

### Contribution
2
