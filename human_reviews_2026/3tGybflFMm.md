# Intentional-Gesture: Deliever your Thoughts by Gestures for Speech

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
When humans speak, gestures help convey communicative intentions, such as adding emphasis or describing concepts. However, current co-speech gesture generation methods rely solely on superficial linguistic cues (\textit{e.g.} speech audio or text transcripts), neglecting to understand and leverage the communicative intention that underpins human gestures. This results in outputs that are rhythmically synchronized with speech but are semantically shallow. To address this gap, we introduce \textbf{Intentional-Gesture}, a novel framework that casts gesture generation as an intention-reasoning task grounded in high-level communicative functions. First, we curate the \textbf{InG} dataset by augmenting BEAT-2 with gesture-intention annotations (\textit{i.e.}, text sentences summarizing intentions), which are automatically annotated using large vision-language models. Next, we introduce the \textbf{Intentional Gesture Motion Tokenizer} to leverage these intention annotations. It injects high-level communicative functions (\textit{e.g.}, intentions) into tokenized motion representations to enable intention-aware gesture synthesis that are both temporally aligned and semantically meaningful, achieving new state-of-the-art performance on the BEAT-2 benchmark. Our framework offers a modular foundation for expressive gesture generation in digital humans and embodied AI.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces **Intentional-Gesture**, a framework that treats gesture generation as an intention reasoning task. A key contribution is the creation of the **InG (Intention-Grounded) dataset**, which augments BEAT-2 with gesture-intention annotations inferred using large vision-language models. This results in a structured dataset aligning gestures with high-level communicative functions. Additionally, the paper proposes the **Intentional Gesture Tokenizer**, which injects intention semantics into discrete motion representations, enabling gesture generation that is both temporally synchronized and semantically expressive. The method achieves state-of-the-art performance on BEAT-2, providing both data and methodology for intention-driven gesture generation.

### Strengths
1. **Strong dataset contribution**: The InG dataset links gestures with semantic intentions, offering the first large-scale intention-annotated gesture dataset.
2. **Integration of intention semantics into motion**: The Intentional Gesture Tokenizer embeds high-level intention information into discrete motion tokens, enhancing semantic expressiveness and interpretability.
3. **Modular and reproducible data processing**: The pipeline from BEAT-2 augmentation to intention annotation and motion tokenization is clear and extensible, providing a replicable framework.
4. **Comprehensive experimental validation**: Both quantitative and qualitative results demonstrate the effectiveness of the intention annotations in improving gesture generation quality.

### Weaknesses
1. **Lack of complete example demonstration**

   The paper describes inputs that include audio, text transcriptions, and intention information, but the current presentation does not clearly show the correspondence between these components or provide concrete examples. It is recommended to include intention information in Figure 6 or provide complete examples in the appendix to help readers intuitively understand the relationship between inputs and generated gestures.

2. **Insufficient intention analysis and visualization**

   The paper lacks visualization and analysis of intention information. After adding intention visualization, it is suggested to incorporate intention-related content in the analysis of experimental results. Additionally, comparing generation results with and without intention inputs would further support and strengthen the contribution of intention control. Demonstrating the intention control results mentioned in the paper would provide a more direct validation of the claims.

3. **Writing and structural improvements**

   The content of the paper is solid overall, but the writing is somewhat rough. It is recommended to move some lengthy content to the appendix and strengthen the description of the dataset and intention-related analysis in the main text to improve clarity and structure. At the same time, the appendix could be better organized to present the information more clearly and comprehensively.

4. **Improvement of figure clarity**

   Some figures have small and slightly blurred text. It is recommended to enlarge the font size and improve the figure resolution to enhance readability.

5. **Incomplete description of the data annotation process**

   The description of the data annotation pipeline is currently fragmented, lacking a complete overview. It is suggested to include a full process diagram in Figure 2 or Section 3, so that readers can intuitively understand all the steps and their logical relationships.

6. **Missing templates for the rule-based parser**

   The appendix does not provide an intuitive description of how the rule-based parser converts motions into captions. It is recommended to include relevant templates or prompt examples to help readers better understand the parser’s output logic and generation process.

### Questions
See the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Intentional Gesture, a new framework for co-speech full-body gesture generation that explicitly models the communicative intentions underlying human speech. Rather than relying solely on linguistic or rhythmic cues, the work reframes gesture synthesis as an intention reasoning problem, seeking to bridge high-level semantic functions (e.g., emphasis, deixis, affirmation) with low-level motion patterns.

To enable this, the authors construct InG (Intention-Grounded) Dataset, which extends BEAT-2 and Audio2PhotoReal with structured annotations of communicative functions and inferred speaker intentions. These annotations are generated through a vision-language-model (VLM)-based pipeline that integrates motion analysis, functional derivation, gesture–function mapping, and intention summarization, followed by human-in-the-loop validation to ensure annotation quality.

Building on this dataset, the framework introduces two key modules:

H-AuMoCLIP, a CLIP-style multimodal encoder that jointly aligns audio, transcript, and intention embeddings via a hierarchical contrastive learning scheme. This model fuses acoustic and linguistic information with intention semantics to build an intention-aware representation space.

Intentional Gesture Tokenizer, a global multi-codebook quantization module that embeds intention semantics directly into motion representations, supervised by features from H-AuMoCLIP. This design allows discrete gesture tokens to carry both structural and semantic meaning.

Experiments on BEAT-2 demonstrate that the proposed method achieves state-of-the-art performance in realism, diversity, and synchronization, and generalizes effectively to unseen speakers. Ablation studies confirm that intentional supervision and structured annotation significantly improve both alignment and generation quality. A user study further verifies that the generated gestures are more natural, interpretable, and intention-consistent compared with prior methods such as GestureLSM and EMAGE.

### Strengths
- The paper introduces a conceptually clear and distinctive formulation of gesture generation as an intention-grounded reasoning problem, moving beyond traditional rhythm- or style-based conditioning.

- The InG dataset and the VLM-based annotation pipeline are thoughtfully designed and well justified. The multi-stage annotation process—spanning motion analysis, communicative function derivation, and intention inference—demonstrates a scalable way to obtain semantically grounded gesture data, supported by quantitative validation and human-in-the-loop filtering.

- Experimental results are comprehensive. The method shows consistent gains over recent state-of-the-art approaches (e.g., GestureLSM, EMAGE) in both objective metrics and user studies, while also preserving real-time inference efficiency.

- The paper is well-structured and readable, with clear figures, detailed methodology descriptions, and thorough supplementary material.

### Weaknesses
- Although the authors provide extensive quantitative results, the evaluation scope is still limited to standard gesture-generation metrics (e.g., FGD, beat constancy, diversity). These capture motion quality but not communicative effectiveness. Additional user studies or perception-based tasks assessing how well the generated gestures convey intended meanings would strengthen the claim of intention awareness.
- The core architectural elements (contrastive CLIP-style encoders, multi-codebook vector quantization) are evolutions of existing techniques. The novelty of the work primarily lies in the systematic integration of these elements under an intention-grounded framework rather than in fundamentally new modeling paradigms.

### Questions
1. In the InG dataset, each word is associated with a single keyframe for motion grounding. For longer or highly expressive utterances, this might undersample the gesture dynamics. Have the authors explored multi-frame sampling or adaptive temporal windows that scale with speech duration? How might such adjustments affect the fidelity or semantic consistency of the inferred intentions?
2. Since the motivation centers on communicative intention, it would be informative to see human perception studies or downstream interaction tasks (e.g., how well viewers interpret or respond to the generated gestures). Have the authors considered incorporating such tests to substantiate the communicative effectiveness of intention-aware gestures?

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
This paper introduces Intentional-Gesture, a framework that models communicative intentions for co-speech gesture generation. By augmenting BEAT-2 with automatically generated intention annotations and proposing an intention-aware motion tokenizer, the method produces gestures that are both semantically meaningful and well-aligned with speech, achieving state-of-the-art results on BEAT-2.

### Strengths
- This paper presents a comprehensive system that covers the entire pipeline, from dataset construction to method design.  
- It also identifies and addresses the issue of semantic shallowness in gesture generation.

### Weaknesses
- In fact, the semantic signals conveyed by gestures are often subtle and ambiguous in most situations. This paper lacks an analysis of how well VLM handles this aspect.  
- It is also difficult to discern the enhanced semantic signals in the gesture examples presented in the paper.

### Questions
n/a

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
The paper aims to address the lack of semantic expressiveness in existing co-speech gesture generation systems. Current methods mainly focus on synchronizing gestures with prosodic patterns in speech (from audio or text), while ignoring the deeper communicative intentions that actually drive human gestures. The authors propose a new framework, Intentional-Gesture, which includes:

* Re-defining gesture generation as intention reasoning, and constructing an Intention-Grounded (InG) dataset. A VLM is used to automatically annotate the BEAT-2 dataset by linking speech, gestures, and communicative intentions (such as “emphasis” or “giving examples”).

* Introducing H-AuMoCLIP, a CLIP-style multimodal model for learning a joint representation of audio, text, intention, and motion. Based on this, an Intentional Gesture Tokenizer is built with vector quantization to discretize motion, while using intention-aware features for semantic supervision.

* Integrating the intention encoder and tokenizer into GestureLSM, achieving state-of-the-art results on BEAT-2.

### Strengths
- Presents a complete and scalable pipeline to automatically infer communicative intentions from speech, producing a well-structured dataset.  
- Effectively integrates intention information both as conditional input and as semantic supervision loss.  
- Clear writing and well-organized experiments.

### Weaknesses
- The annotation pipeline relies on GPT-4o-mini, which is inherently noisy and biased. The model is trained to reproduce VLM-generated semantic labels rather than discovering genuine intention–motion relations from real data, which limits its theoretical depth and generalization. Real communicative intentions are far more ambiguous, culture-dependent, and context-sensitive than the fixed categories defined here.

- It is unclear how the “human-written baseline” in Figure 4 was produced. The authors do not explain the annotators’ background or guidelines. A preference for VLM outputs might simply reflect more uniform formatting and fluent wording, rather than better semantic accuracy. Given that even humans struggle to annotate gestures reliably, the conclusion that “VLM outperforms humans” is questionable. 

- Lack of failure-mode analysis and limited baselines. The paper does not show how the model behaves under sarcasm, ambiguity, or complex emotions. Evaluation is limited to EMAGE, CAMN, and GestureLSM, while many relevant SOTA models (e.g., on the GENEA leaderboard) are missing from both objective and subjective comparisons. Meanwhile, the reported MOS scores (1–2 for EMAGE and CAMN) seem implausibly low, calling their evaluation reliability into question. The claimed semantic improvements are not supported by a metric that truly measures semantic correctness, and gains on FGD/BC/etc do not demonstrate intention alignment.

- Training–testing mismatch. During training the system sees both gestures and transcripts, but at test time the intention is inferred only from transcript text. Figure 4 already shows a large quality gap between “text-only” and “text+motion” annotation. In addition, splitting speech into 1–2 second windows is problematic, since gestures and speech often do not align at such a rigid granularity. The appendix also acknowledges long-sequence drift, yet Table 13 reports longer sequences performing better, which raises further doubts about the underlying method and evaluation.

- Limited interpretability in demos. The demo videos do not clearly indicate which gestures correspond to which intentions. Adding textual and intention annotations in the demo timeline would make the “intention controllability” more evident.

- Unclear distinction between intention and text semantics. The paper claims to move beyond text-based semantic alignment, but the results do not clearly demonstrate the qualitative difference between text-derived semantics and intention-driven control. This makes the framework appear more like an extension of text-based CLIP gesture models (e.g., GestureDiffuCLIP) rather than a fundamentally new paradigm. Better illustrating the difference between directly using text embeddings and LLM-summarized intention features could further clarify the contribution and help readers understand the distinct roles of each.

### Questions
- Since communicative intentions are subjective and often allow multiple valid interpretations, how does the system handle segments with more than one reasonable intention? To what extent is the final motion influenced by the wording or verbosity of the VLM-generated text labels?

- For Figure 4, what is the exact protocol for the “human-written baseline”? Who were the annotators—domain experts or crowd workers? Did they receive prompts comparable in structure to the VLM prompt? In addition, can the authors show that different intentions (e.g., “emphasis” vs. “questioning”) lead to clearly different gestures for the same speech segment? The current videos show only modest visual improvement—why is that?

- Where do the test-time transcripts come from? If ASR is used, how robust is the system to transcription errors? How much do such errors degrade intention inference and gesture naturalness, especially for rhythm-driven or listener-feedback gestures, which are common in real interaction?

- Many tables only report mean values without variance, confidence intervals, or statistical tests.

### Soundness
2

### Presentation
3

### Contribution
3
