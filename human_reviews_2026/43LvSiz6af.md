# TTS-Hub: Leveraging Modular LoRAs and  Arithmetic Composition for Controllable Text-to-Speech

- Avg Score: 4.40
- Decision: Reject
- Scores: 6, 2, 4, 4, 6

## Abstract
Controllable text-to-speech (TTS) aims to generate speech from text while allowing control over prosodic and speaker-related attributes such as pitch, age, and accent. Existing controllable TTS methods primarily rely on natural language prompts to guide the synthesis process or utilize reference audio cloning to achieve control. However, prompt-based approaches often struggle with the cross-modal semantic gap between textual descriptions and intended speech attributes, leading to imprecise and coarse-grained control. Conversely, cloning methods depend heavily on reference audio samples and struggle to generalize beyond the characteristics seen in those samples, resulting in limited flexibility. To overcome these challenges, this paper proposes TTS-Hub, a novel controllable TTS framework that employs modular Low-Rank Adaptation (LoRA) components and their arithmetic-based composition to achieve fine-grained and flexible controllable TTS. Specifically, we construct a comprehensive Data Hub, which covers 6 high-level attribute categories and 32 fine-grained speech attributes. Leveraging this attribute-specific data, we fine-tune two mainstream TTS frameworks to obtain a corresponding LoRA Hub, where each modular LoRA is specialized for a specific speech attribute. At inference time, TTS-Hub selects the required LoRA modules and combines them through simple arithmetic composition to produce a fused LoRA that simultaneously encodes multiple attribute representations, enabling flexible and extensible multi-attribute control without retraining the backbone. Extensive experiments show that individual LoRAs provide precise single‑attribute control, while arithmetic composition yields flexible and interpretable multi‑attribute speech and consistently outperforms prompt‑based baselines. Code and data are available in supplementary materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Building on existing TTS backbones, the paper develops an attribute-specific Data Hub and fine-tunes LoRA modules for each characteristic, resulting in a LoRA Hub where every adapter is dedicated to controlling a specific speech attribute. By combining selected adapters through basic arithmetic operations, the system offers intuitive multi-attribute control in TTS. The approach is direct and uncomplicated. Experimental results indicate that individual adapters effectively manage their assigned attributes, and certain arithmetic combinations allow users to intuitively blend, reduce, or amplify these attributes, to some extent.

### Strengths
The method provides interpretable and versatile control over various predefined speech characteristics—such as accent, age, emotion, and gender—and can be easily extended by adding new adapters to enhance expressive potential. 

The framework is simple and readily applicable to different TTS backbones.

### Weaknesses
Relying on predefined attributes may limit the breadth of semantic representation. Additionally, interactions between certain attributes might be non-linear or even conflicting, which means straightforward arithmetic composition may not always produce satisfactory outcomes.

It still requires a sizeable, attribute-annotated dataset encompassing all specified attributes to train the LoRA modules.

### Questions
Training data and attribute annotations are essential for controllable TTS. The baseline, such as Parler-TTS, used different data and fewer attributes, reducing expressiveness and potentially affecting comparisons between text prompts and LoRA-hub. The quality of synthesis is also dependent on backbone TTS. More detailed explanations of this issue are recommended. 

When utilizing the GPT-4o-Audio API for speech evaluation, is there ablation study to measure its effectiveness comparing with human annotation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces TTS-Hub, a novel framework for controllable text-to-speech synthesis. Current prompt-based systems often lack precision, while audio-cloning systems are not flexible. The proposed solution is a modular system built on Low-Rank Adaptation (LoRA). The authors first create a large dataset with clear attribute labels. They then use this data to train a collection of specialized LoRA modules, where each module acts as an expert for a single speech attribute, such as a specific accent or emotion. During synthesis, these expert modules can be combined through simple arithmetic operations like addition, subtraction, and scaling. This allows for fine-grained control over multiple speech attributes without needing to retrain the large backbone model. The authors validate their method on two different TTS backbones and show that it provides superior attribute control compared to a prompt-based baseline.

### Strengths
1. Novel Architecture: This work is the first to use multiple LoRA modules as distinct experts to achieve fine-grained, multi-attribute control over TTS. This modular and compositional approach is a significant and original contribution.
2. Comprehensive Experiments: The paper thoroughly evaluates the proposed framework. It systematically tests the effects of adding, subtracting, and scaling the LoRA modules. The authors also employ a diverse set of metrics to robustly measure the effectiveness of attribute control.
3. High Reproducibility: The framework is built entirely on open-source datasets and models. This commitment to using publicly available resources makes the paper's strong results highly reproducible for the research community.

### Weaknesses
1. Limited Trustworthiness of AMOS Scores: In Section 5.1, the paper notes that the Attribute Mean Opinion Score (AMOS) was evaluated using only 16 samples per setting. While the difficulty of conducting large-scale human studies is understandable, such a small sample size raises concerns about the statistical significance and reliability of the human evaluation results.
2. Unverified LLM-based Metrics: The paper relies on GPT-4o-Audio-Preview for its Win-Attr and Win-Qual metrics. However, the authors do not provide any evidence to validate the reliability of this LLM as an evaluator for this specific audio task. The credibility of these metrics would be much stronger if the paper included results showing a high correlation between the LLM's judgments and human judgments on a validation set.
3. Insufficient Baselines: The paper only compares its method against a prompt-based version of its own backbone model. It does not include comparisons with other state-of-the-art controllable TTS models mentioned in the Related Works section. This makes it difficult to judge the true performance of TTS-Hub in the context of the current state of the art.
4. Minor Writing and Citation Issues: There are minor errors in the manuscript that detract from its polish. For example, a citation key in Section 5.2 appears as plain text (hu2022lora) instead of a proper reference, indicating a potential issue in the citation process.

### Comment Not Affecting Score
1. Table 3, which shows the trade-off between two weighted attributes, could be more intuitive if presented as a line graph instead of a table.

### Questions
1. Could you please clarify the mechanism for LoRA selection from the LoRA Hub? Figure 2 suggests a process where a user's request is parsed into attributes. Is this selection based on simple keyword matching from a user's structured input, or is there a more advanced mechanism for interpreting free-form text requests?
2. In Section 5.3, the attributes $a_1$ and $a_2$ are used as generic placeholders for the subtraction experiments. While the appendix provides specific examples, for clarity in the main text, could you specify which attribute pair was used to generate the results shown in Table 5?

### Soundness
2

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
4

### Summary
This paper introduces TTS-Hub, a novel framework for controllable text-to-speech (TTS) synthesis. The key idea is to leverage modular Low-Rank Adaptation (LoRA) adapters, each fine-tuned for a specific speech attribute (e.g., accent, age, emotion), and to enable flexible, fine-grained control by composing these adapters arithmetically at inference time. The authors construct a comprehensive data hub covering 32 fine-grained speech attributes and demonstrate the approach on two backbone TTS models (Parler-TTS and VoiceLDM). Extensive experiments show that TTS-Hub outperforms prompt-based baselines in both single- and multi-attribute control, as measured by human and automatic metrics.

### Strengths
1. The paper is the first to systematically explore modular LoRA composition for controllable TTS, moving beyond prompt-based or reference-based control.

2. The authors provide thorough experiments,  and is validated on two different TTS architectures, demonstrating robustness and generality.

### Weaknesses
1. While the method generally outperforms baselines, there are cases in the demo where the improvement is marginal or the quality drops. It's better to do some qualitative analysis or audio examples highlighting these failure modes and their possible causes.

2. The paper does not propose or analyze advanced fusion strategies to mitigate this, nor does it deeply investigate the causes of interference.

### Questions
1. Have you explored more advanced composition strategies (e.g., gating, attention, or learned fusion) to address the observed degradation when fusing many LoRA modules?

2. Can you provide more insight or analysis (e.g., ablations, visualizations) into why attribute interference occurs and how it might be mitigated?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TTS-Hub, a controllable TTS framework that fine-tunes attribute-specialist LoRA adapters (e.g., for accent, age, emotion, gender, energy, pitch) on curated, attribute-labeled subsets drawn from four public corpora. At inference, multiple adapters are algebraically composed (addition, subtraction, scalar weighting) and injected into a frozen backbone to realize multi-attribute control. The paper evaluates with human Attribute MOS (AMOS), LLM-judge win rates (Win-Attr/Win-Qual using GPT-4o-Audio-Preview), and objective metrics (MCD, F0-RMSE, SSIM), reporting consistent gains over prompt-only control and reasonable behavior under composition

### Strengths
- The modular attribute-specialist LoRA design is simple, extensible, and model-agnostic: train once per attribute, compose at inference without touching the backbone. The paper demonstrates this on Parler-TTS (main) and VoiceLDM (appendix), showing architectural robustness.

- Single-attribute control clearly improves over prompt-only baselines in AMOS and Win-Attr across many attributes; the tables quantify consistent, meaningful margins (e.g., ">60 age" gains). This supports the claim that LoRAs capture sharper acoustic factors than natural-language prompts alone.

### Weaknesses
- While the proposed method is promising, novelty seems to be incremental. Composing LoRAs arithmetically is well-trodden in NLP/vision (e.g., LoRAHub for cross-task generalization; Mixture-of-LoRA-Experts), and LoRA for TTS has been used for emotion/style/personalization (e.g., StyleSpeech/EELE/LorP-TTS). I would suggest the authors compare with dynamic composition (Huang et al., 2024a; Wu et al., 2024) and TTS-specific LoRA works (Lou et al., 2024; Qi et al., 2024; Bondaruk & Kubiak, 2025).

- Evaluation validity and reliance on LLM-as-judge: Win-Attr/Win-Qual depend on GPT-4o-Audio-Preview, which may encode hidden priors and can be sensitive to prompting and audio loudness/length. Stronger human listening (e.g., MUSHRA/CMOS with inter-rater reliability) and task-relevant automatic metrics (ASR WER, F0/energy controllability accuracy via independent estimators) would strengthen claims.

- Limited baselines/backbones and scaling limits: Comparisons are primarily to prompt-only control on Parler-TTS; many contemporary TTS systems expose structured control tokens/encoders (e.g., style tokens, prosody/vector controls) that would be stronger baselines.

### Questions
- What is the runtime/memory overhead of injecting multiple adapters, and how does latency scale with the number of composed LoRAs on Parler-TTS vs. VoiceLDM?

- Can you report fairness diagnostics (e.g., accent/gender/age bias) and failure cases for rare attribute combinations or cross-lingual accents?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of controllable text-to-speech (TTS) and introduces a modular framework called TTS-Hub. The method builds on LoRA (Low-Rank Adaptations), where each adapter focuses on a specific speech attribute (e.g., accent, emotion, gender), and multiple attributes can be combined at inference time through simple arithmetic operations on LoRA weights.
A reconstructed multi-dataset Data Hub (covering 32 detailed attributes) is used to train the adapters. Experiments on two backbone models (Parler-TTS and VoiceLDM) demonstrate superior single- and multi-attribute control compared to prompt-only baselines, along with intuitive arithmetic effects such as subtraction and scaling.

### Strengths
* LoRA-based adapters have been used for TTS before, and arithmetic-based composition has appeared in other domains, but this paper is novel in explicitly exploring arithmetic composition of those adapters for controllable TTS.

* Keeping the backbone frozen while composing LoRAs arithmetically is lightweight, modular, and easily extensible. This is clearly the main highlight of this paper.

* The arithmetic weighting, subtraction, and scaling experiments make the controls intuitive and demonstrate that the model can effectively isolate targeted attributes.

* The paper provides public datasets, detailed descriptions, and code/configs in the supplementary materials, which improves reproducibility. The Data Hub itself is also a valuable contribution to the community.

### Weaknesses
* Some intuitive questions remain unanswered. It would be useful to analyze interference between conflicting attributes (e.g., “high pitch” and “low pitch” adapters) to understand the limits of arithmetic composition. It would also help to show how subtracting high pitch versus adding low pitch differs in practice, if at all. Similar logic applies to age attribute as well.

* While the evaluation is generally strong, it lacks several ablations. For example, it would be helpful to know how the choice of layers for LoRA insertion affects performance, or whether deeper layers influence attribute control more strongly.

* As with prompt-based or reference-based methods, this approach depends on the available data and adapters. If a desired attribute is not represented in the dataset or a corresponding LoRA is unavailable, it becomes impossible to synthesize that target attribute.

* The paper argues that existing approaches relying on natural-language prompts or reference audio cloning have limitations, but the experiments only compare against prompt-based control. If no such baseline for reference cloning is available, it might be better to slightly adjust the initial claim in the paper.

* Some reported metrics (e.g., Win-Attr, AMOS) appear subjective and prompt-dependent. It would strengthen the evaluation to report the true variance or ground-truth checks for these metrics.

Minor issues (did not affect score)

* Some citations (e.g., hu2022lora) did not compile correctly.
* Line 078: “flexibel” → “flexible.”
* Figure 1: “Unselcted” → “Unselected.”

### Questions
What are the most common examples where arithmetic composition fails? For instance, accent and pitch may be difficult to isolate — keeping one without affecting the other could introduce artifacts.


Can you show results for attribute combinations not seen together during training i.e., zero-shot composition?

### Soundness
3

### Presentation
3

### Contribution
3
