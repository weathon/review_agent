# Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
As Retrieval-Augmented Generation (RAG) evolves into service-oriented platforms (Rag-as-a-Service) with shared knowledge bases, protecting the copyright of contributed data becomes essential. Existing watermarking methods in RAG focus solely on textual knowledge, leaving image knowledge unprotected. In this work, we propose \textit{AQUA}, the first watermark framework for image knowledge protection in Multimodal RAG systems. \textit{AQUA} embeds semantic signals into synthetic images using two complementary methods: acronym-based triggers and spatial relationship cues. These techniques ensure watermark signals survive indirect watermark propagation from image retriever to textual generator, being efficient, effective and imperceptible. Experiments across diverse models and datasets show that \textit{AQUA} enables robust, stealthy, and reliable copyright tracing, filling a key gap in multimodal RAG protection.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes **AQUA**, the first watermarking framework dedicated to safeguarding **image knowledge copyright** in **Multimodal Retrieval-Augmented Generation (RAG)** systems, filling a critical gap left by existing text-only methods. AQUA addresses the challenge of *indirect watermark propagation* (image input to textual output) and *unapparent distribution shifts* through two complementary methods: **$AQUA_{acronym}$**, which embeds uncommon acronyms into images, and **$AQUA_{spatial}$**, which uses synthetic images with unusual spatial relationships, both leading to textual verification signals in the RAG output. Experiments across various Multimodal RAG models and datasets demonstrate that AQUA is highly effective, harmless, stealthy, and robust against common attacks, enabling reliable copyright tracing with high efficiency and statistical significance.

### Strengths
1. This problem and the proposed method is novel.
2. This paper is well-structured and easy to follow.
3. The evaluation consider multiple attack methods.

### Weaknesses
1. The space between each paragraph seems small.
2. It seems no adapative attacks are considered.
3. There are only a few baselines to compare.

### Questions
1. According to Table 4, it seems $AQUA_{acronym}$ consistently outperforms $AQUA_{spatial}$. Are there specific scenarios where only $AQUA_{spatial}$ is applicable or can those two varients be combined?

2. Are image watermarking methods totally inapplicable in this problem?

3. Apart from copyright detection, can AQUA be extended for the attribution of copyright as well [1]?

[1] Watermark-based Attribution of AI-Generated Content.

4. What value of Rank can be considered as good, as it ranges from 1 to 10 in table 2?

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
This paper introduces AQUA, a framework to address copyright protection for image knowledge in multimodal Retrieval-Augmented Generation (RAG) services. The authors identify novel challenges, such as indirect watermark propagation (embedding a watermark in an image that is detected in generated text) and the need for an explicitly retrievable watermark that doesn't cause an obvious data distribution shift. It introduces two variants: AQUA_acronym: Embeds rare acronyms and their full names into synthetic images, leveraging a VLM's OCR capability for verification. AQUA_spatial: Generates images with unusual spatial relationships for models with limited OCR, leveraging spatial reasoning for verification. A comprehensive evaluation demonstrates that AQUA is effective with high detection rates, harmless, stealthy, and robust against image attacks.

### Strengths
- This is the first work to formally tackle image copyright protection in multimodal RAG. The problem formulation, particularly identifying "indirect watermark propagation" as a core challenge, is a novel and significant contribution. The two proposed methods are creative and well-designed solutions.
- This work fills a critical, unaddressed gap in AI data governance as RAG services increasingly rely on proprietary multimodal data. AQUA provides a practical solution and sets a strong baseline for an important new research area.
- The paper is exceptionally clear. Figures 1, 2, and 3 provide excellent visualizations of the RaaS problem, the core challenges, and the AQUA methodology. Key concepts, like the "Trigger" and "Instruction" components of a probe query, are precisely defined and aid understanding.

### Weaknesses
- The paper's threat model, which only considers one defender and one adversary, overlooks the multi-tenant nature of RaaS platforms. It's unclear how AQUA would prevent "collisions" where multiple providers independently create the same watermark (e.g., the same acronym or spatial concept), which could lead to false accusations of misuse.
- The methods' reliance on VLM capabilities (OCR, spatial reasoning) is also a potential fragility. A future, more advanced VLM might identify AQUA_spatial images as "unnatural" and refuse to answer. Conversely, an adversary could fine-tune their model to specifically ignore text overlays or unusual object pairings, defeating the watermark. The robustness tests focus on image transformations, not model-level adaptations.
- For AQUA_spatial, the semantic trigger must be as rare as the image content. While the paper shows 0% retrieval for 10000 benign queries, it's unclear if this query set was stress-tested with queries semantically similar to the triggers. A benign user could accidentally issue a query that matches the trigger, retrieving the watermark.

### Questions
I got several questions for this paper:
- How does AQUA prevent watermark collisions in a RaaS platform with hundreds of data providers? Does this framework require a centralized "watermark registry" managed by the platform?
- Have you considered failure cases where a VLM's safety or "common sense" guardrails cause it to identify AQUA_spatial images as "unnatural" and refuse the probe query? How robust is AQUA against an adversary who fine-tunes their model to ignore these specific watermark types?
- How do you guarantee the semantic uniqueness of the AQUA_spatial trigger? Was the benign query set in Section 5.4 specifically tested for queries that are semantically similar, though not identical, to your triggers?

### Soundness
2

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
This paper introduces AQUA, a novel watermarking framework designed to safeguard image knowledge copyrights in Multimodal Retrieval-Augmented Generation (RAG) systems. With the rise of RAG-as-a-Service (RaaS) platforms, where data providers contribute knowledge to a shared pool used by external services, the need for protecting copyright has become critical. Existing watermarking methods have largely focused on text-based RAG systems, leaving image knowledge unprotected. AQUA addresses this gap by embedding semantic signals into images through two complementary watermarking methods: AQUAacronym (embedding uncommon acronyms and their full names) and AQUAspatial (using spatial relationships in the image). These techniques ensure that the watermarks survive indirect propagation from image retrievers to textual generators, making them efficient, effective, and imperceptible. Experiments demonstrate that AQUA is robust, stealthy, and effective in tracing copyright, even in the face of attacks like image transformations and regeneration.

### Strengths
1. AQUA introduces a groundbreaking watermarking method for Multimodal RAG systems, focusing on the protection of image knowledge, an area previously neglected in watermarking research. By using semantic-based signals (acronyms and spatial relationships), it provides a new approach to watermark embedding that spans both image and text modalities.

2. The watermarking techniques, particularly AQUAacronym and AQUAspatial, are shown to be robust against various image transformations and attacks, including rescaling, rotation, compression, and regeneration. They maintain their imperceptibility to end-users and cannot be detected by unauthorized filtering mechanisms.

3. The framework has been extensively tested across different RAG models and multimodal datasets (MMQA and WebQA). The paper provides a thorough evaluation of AQUA’s effectiveness, harmlessness, stealthiness, and robustness, with results indicating that AQUA outperforms baseline methods and maintains high retrieval success and generation success rates.

4. AQUA is adaptable for both black-box and white-box scenarios, meaning it can be used in various real-world RAG systems without requiring direct access to the internal model or dataset. Its design also ensures easy deployment and provides a solid baseline for future research in the protection of multimodal datasets in RaaS environments.

### Weaknesses
1. Focuses on 7B-scale VLMs (LLaVA-NeXT, InternVL3, etc.) without assessing performance on larger models (e.g., 32B+ VLMs) or lightweight models for edge deployments.

2. Does not assess how watermark detection performance degrades over time with retriever/generator updates, fine-tuning, or dataset drift.

3. While mentioning a reference distribution for practical verification, it provides only a single example without guiding how to adapt it to diverse dataset characteristics or RAG system configurations.

### Questions
Please refer to the weakness

### Soundness
2

### Presentation
2

### Contribution
3
