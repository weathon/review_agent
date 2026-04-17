# SAKE: Towards Editing Auditory Attribute Knowledge of Large Audio-Language Models

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Knowledge editing offers an efficient way to update model knowledge without full retraining, but prior work has concentrated almost exclusively on textual or visual modalities. We introduce SAKE, the first benchmark specifically designed for editing auditory attribute knowledge in Large Audio-Language Models (LALMs). Unlike factual updates, SAKE targets several abstract auditory attributes, capturing knowledge types that go beyond conventional textual and visual domains. We benchmark eight editing methods on two LALMs along four dimensions: reliability, generality, audio/text locality, and portability. Results highlight challenges such as preserving intra-attribute knowledge unrelated to the edit, generalizing edits to multimodal reasoning, and maintaining edits under sequential updates. SAKE provides a principled framework to study how knowledge editing extends to the auditory modalities, opening new directions for maintaining and adapting LALMs in more diverse real-world scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SAKE, the first benchmark specifically designed to evaluate knowledge editing in Large Audio-Language Models (LALMs). The authors posit that editing auditory attribute knowledge (e.g., speaker emotion, animal sounds) presents unique challenges not found in text or vision, as these attributes are abstract, high-level perceptual concepts rather than discrete facts.

### Strengths
* **Pioneering Problem Definition:** This is the first work to formally define and systematically benchmark the problem of editing auditory attribute knowledge in LALMs, moving beyond traditional text/vision fact editing.
* **Rigorous Benchmark Framework:** The SAKE benchmark is methodologically strong. Its design, particularly the granular "Audio Locality Type 2" (intra-attribute) and "Portability" (reasoning propagation) metrics, astutely captures the unique, abstract challenges of the auditory modality.
* **Significant and Actionable Findings:** The paper delivers clear, impactful results. The discovery of "intra-attribute entanglement" (e.g., editing "dog" breaks "cat") and the documented failure of IKE methods for LALMs are crucial findings that highlight specific, unsolved challenges for the field.
* **High-Quality Presentation:** The paper is exceptionally clear. Figure 1, in particular, serves as an excellent visual abstract that effectively communicates the benchmark's complex design and evaluation dimensions.

### Weaknesses
* **Mismatch in Attribute Scope:** The paper's motivation hinges on editing "abstract and continuous" auditory concepts, yet the benchmark's attributes (e.g., animal sound, language) are predominantly evaluated as discrete classification labels. This under-delivers on the core premise, as the challenges of editing truly continuous attributes (like pitch or prosody) remain unexplored.
* **Superficial Analysis of IKE Failure:** The paper reports the stark failure of In-Context Editing (IKE) but attributes it to a generic "limited in-context learning ability." This analysis is shallow; it fails to investigate the specific failure mechanism, such as whether the LALM struggles to process in-context audio or fails to apply textual instructions to its auditory processing.
* **Limited Model Diversity:** Key findings, such as "intra-attribute entanglement," are derived from only two LALMs. While acknowledged as a limitation, this narrow scope makes it difficult to ascertain if these significant challenges are fundamental to LALMs or are artifacts of the specific architectures tested.

### Questions
1.  **IKE Failure Mechanism:** Given the failure of In-Context Editing, is this due to the LALM's inability to process in-context *audio examples*, or a more general failure to apply *textual instructions* to its auditory processing? A test using only text-based instructions could isolate the precise point of failure.
2.  **Intra-Attribute Entanglement Mechanism:** What is the hypothesized cause for the severe "Audio Locality Type 2" failure (e.g., editing "dog" breaks "cat")? Is it (a) **acoustic entanglement**, where representations are similar in the audio encoder, or (b) **semantic entanglement**, where the LLM backbone co-locates these concepts?
3.  **Scope of Portability Failure:** The portability test (animal sound $\rightarrow$ diet) fails. How wide is this reasoning disconnect? For the "frog" $\rightarrow$ "dog" edit, do other related concepts like "habitat" (pond $\rightarrow$ house) or "classification" (amphibian $\rightarrow$ mammal) also fail to update, or is the failure isolated to the tested attribute?

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
This paper introduces SAKE, the first benchmark designed to evaluate knowledge editing in Large Audio-Language Models. SAKE targets auditory attribute knowledge, such as speaker gender, emotion, spoken language, and animal sounds.
The benchmark evaluates seven editing methods across four dimensions: Reliability, Generality, Locality, Portability
Experiments were conducted on two strong LALMs: DeSTA2.5-Audio and Qwen2-Audio.
The results show that while existing editing methods can successfully change specific auditory knowledge, they struggle to generalize, maintain unrelated knowledge, and support multiple sequential edits.
The paper concludes that new methods are needed to handle abstract, perceptual auditory knowledge more robustly

### Strengths
- First benchmark focused on auditory attribute knowledge editing, extending a well-studied concept from text and vision into the audio domain.
- Significant for maintaining and updating multimodal model knowledge efficiently.
- Results are well-analyzed, identifying causes of poor generality and locality in existing methods.

### Weaknesses
- Only 2 LALMs are evaluated. 
- Only 4 attributes are covered. Can other auditory attributes like environmental sound types, etc. be considered for a stronger benchmark?

### Questions
- Are the paraphrased text human-checked for semantic consistency?
- Can the authors show evaluation on recent LALMs like Audio Flamingo 2, etc.?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SAKE, the first comprehensive evaluation benchmark specifically designed for auditory attribute knowledge editing in Large Audio-Language Models. 
The SAKE benchmark evaluates editing capabilities across four core auditory attributes (speaker gender, emotion, language, and animal sounds) along four key dimensions: reliability, generality, locality, and portability. The authors conducted experiments on two leading LALMs (DeSTA2.5-Audio and Qwen2-Audio), evaluating seven common editing methods, including fine-tuning, Knowledge Editor, MEND, and In-Context Knowledge Editing in both single and sequential editing settings.

### Strengths
1. The topic is very interesting and well-motivated.
2.  The SAKE benchmark is designed with four critical dimensions: reliability, generality, locality, and portability. The benchmark has a potential impact on the following studies.

### Weaknesses
1. The evaluated scope of knowledge editing methods is limited. While seven editing methods were evaluated, some SOTA editing methods can be considered, such as WISE, AlphaEdit, UltraEdit.
2. The paper primarily focuses on evaluating the effects of editing (what works/doesn't work, and where). However, there's a relative lack of deeper mechanistic explanations for why certain methods are effective or ineffective in the auditory modality. For instance, which parameters or layers are most crucial for auditory attribute knowledge editing? What changes occur in the internal representations of the model? How is knowledge of different auditory attributes encoded and interlinked within the model? Incorporating interpretability analyses (e.g., feature attribution, probing tasks) to delve into the internal workings of knowledge editing on LALMs would provide more profound insights.
3. The considered auditory attributes are limited to 4. This may be due to the speech modality itself, but a more solid analysis is needed.
4. The edited performance can be related to the audio generation performance of the original model (without any editing). More baselines can be useful for discussion.
5. The editing task is limited to audio understanding, while audio generation can be a  much more significant scenario.

### Questions
Please see Weakness

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
The paper introduces SAKE, a benchmark for editing auditory attribute knowledge in Large Audio-Language Models (LALMs). It targets four attributes, including speaker gender, speaker emotion, spoken language, and animal sounds, and evaluates seven editing methods on two LALMs.

### Strengths
1. It is the first systematic benchmark for auditory attribute editing.
2. It tests two competitive LALMs, multiple attributes, single vs. sequential editing, and a comparative suite of seven methods.

### Weaknesses
1. Though the contribution is positioned primarily as a benchmark, the sample of edits provided is quite narrow and under specified. For example, when editing an attribute (e.g., changing “sad” → “angry”), the paper does not clearly define whether the edit is restricted to a specific sound instance or intended to generalize across all instances of the “sad” attribute. Without this scope clarity, it is difficult to interpret whether the model is simply mapping the one training instance or truly generalizing the attribute change.

2. In the locality evaluation, the benchmark requires that unrelated knowledge (i.e., non-edited items) remains unaffected. However, some of the design choices undermine this. For instance, in Figure 4 the locality sample has the answer “sad,” which exactly matches the original (pre-edited) attribute. This raises a concern: if the “locality” example uses the same attribute value as the edited item, then a correct response could simply reflect propagation of the edit rather than demonstrating true preservation of unrelated knowledge. The benchmark therefore may not reliably distinguish between genuine locality preservation and unintentional overlap with the edited attribute.

### Questions
Please refer to the Weakness.

### Soundness
2

### Presentation
3

### Contribution
2
