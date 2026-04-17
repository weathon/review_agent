# Benchmarking Open-ended Segmentation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Open-ended segmentation requires models capable of generating free-form descriptions of previously unseen concepts and regions. Despite advancements in model development, current evaluation protocols for open-ended segmentation tasks fail to capture the true semantic accuracy of the generated descriptions. We empirically demonstrate that embedding‐based similarity score mappings diverge significantly from human judgments. To address this issue, we introduce a novel mapping function that considers multiple lexical relationships between free‐form outputs and test‐vocabulary labels, yielding much closer alignment with human annotations. We integrate this mapping into a robust evaluation framework and re‐benchmark previous state‐of‐the‐art methods. Additionally, we present the first Multi-modal Large‐Language Model trained with a contrastive objective to jointly align visual regions and textual descriptions, achieving new state‐of‐the‐art results in open‐ended panoptic segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses flaws in open-ended segmentation (which needs models to generate free-form descriptions of unseen visual concepts, not select predefined labels) evaluation: it notes existing embedding-based metrics (e.g., Sentence-BERT) conflict with human judgments, and other metrics/studies are misaligned or impractical. It proposes a lexical relationship-based mapping (exact matches, synonyms, hyponyms, meronyms) integrated into the LAC framework, aligning better with human annotations. It also introduces OPAL, the first contrastive-trained MLLM for the task, achieving state-of-the-art results and better robustness.

### Strengths
It innovatively proposes a lexical relationship-aware mapping function integrated into the Lexical Alignment Curve framework and introduces OPAL, the first contrastive-trained MLLM for open-ended segmentation.

The LAC framework provides a reliable evaluation benchmark, OPAL sets a new SOTA.

It ensures rigor by comprehensively re-benchmarking SOTA models on ADE20K and Cityscapes via its proposed framework, and commits to releasing resources upon acceptance to guarantee reproducibility.

### Weaknesses
The paper’s lexical vocabulary lacks verification of its comprehensiveness (e.g., whether it covers low-frequency or niche concepts), which impacts the reliability of the evaluation results.

The paper has no ablation studies on critical lexical mapping designs (e.g., LLM selection basis, noun list filtering rules), undermining the reproducibility of the mapping.

The paper only validates the framework on ADE20K and Cityscapes, failing to include widely used datasets like COCO, limiting the generalization verification of its methods.

The paper does not report OPAL’s efficiency metrics such as inference latency and memory usage, making it hard to assess its practical deployment potential.

### Questions
Could you provide evidence of your lexical vocabulary’s coverage of rare/long-tail concepts?


could you add LLM-comparison ablation, release the noun list/LLM outputs, and quantify how noun list size/filtering affects mapping performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits the problem of evaluation of open-ended segmentation models.
The authors propose:
1. a lexical alignment–based evaluation protocol (LAC) that integrates hierarchical word relations—Exact, Synonym, Hyponym, Meronym—for mapping free-form text to target categories.
2. OPAL, a multi-modal large language model trained with a dual generative–contrastive objective to better align visual regions and textual outputs.

### Strengths
- The proposed lexical mapping and curve alignment method are novel. The paradigm provides a systematic and interpretable way to evaluate free-form segmentation outputs, filling an important methodological gap.
- The method is aligned with human verification (90% with human judgement).

### Weaknesses
- Dependence on external LLMs (e.g., GPT-4) for lexical relation extraction may reduce reproducibility and introduce hidden biases due to model updates.
- The metric maybe be hacked by the inclusion of hyponyms and meronyms.
- The 7B OPAL, as a tool in metric, is not convinent to deploy (specific enviroment). It may cause mismatched aligning and unfair comparison.

### Questions
- Is this method, maybe be used not only for open-ended segmentation? what about open-ended detection?
- What about contextual semantics or polysemy? (e.g., “apple” as fruit vs. company). How does the system handle cases where semantically distinct words share lexical proximity (e.g., “truck” → “car”)?
- How stable are the GPT-4–derived lexical relations across different LLMs, prompts, or randomness?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of evaluating open-ended segmentation, where models generate free-form textual descriptions instead of choosing from a fixed label set. The authors demonstrate that existing embedding-based evaluation methods (e.g., Sentence-BERT) poorly align with human judgment and propose a new Lexical Alignment Curve (LAC) metric that accounts for multiple lexical relationships—such as exact matches, synonyms, hyponyms, and meronyms—to better capture semantic correctness. They further introduce OPAL, the first multimodal large language model trained with a contrastive objective to jointly align visual regions and textual descriptions. Extensive experiments and human verification studies show that the proposed evaluation framework achieves stronger alignment with human perception and that OPAL sets new state-of-the-art results on open-ended segmentation benchmarks.

### Strengths
**Originality:** The paper tackles a highly relevant and underexplored problem—how to objectively evaluate open-ended segmentation models where outputs are free-form text rather than fixed labels. The proposed Lexical Alignment Curve (LAC) introduces a novel evaluation perspective that bridges semantic granularity levels (exact, synonym, hyponym, meronym). This is a creative and conceptually elegant formulation that advances beyond traditional embedding-based similarity scoring.

**Quality:** The methodology is systematic and carefully designed, combining empirical human studies, metric development, and benchmarking experiments.

**Clarity:** The paper is well-structured and logically presented—the motivation is clearly articulated, the limitations of existing methods are explained with concrete examples, and the proposed framework is illustrated with intuitive figures.

**Significance:** The work addresses a critical bottleneck in open-ended visual understanding: the lack of accurate, scalable evaluation metrics that align with human perception. The introduction of OPAL demonstrates practical impact by achieving measurable improvements on standard datasets, indicating both theoretical and empirical significance.

### Weaknesses
- The proposed lexical mapping relies heavily on automatic subject extraction to identify the key semantic concept in a sentence. However, this assumption breaks down for many common sentence structures where the grammatical subject is not the true visual referent (e.g., “A man is walking a dog,” “This is a picture of a dog,” or “On the grass, there is a dog”). Moreover, the method does not account for contextual modifiers such as adjectives or compound nouns, which can drastically alter meaning (e.g., “toy dog,” “hot dog,” “drawing of a dog”). As a result, the current approach may conflate semantically distinct entities or misidentify the referent.
- The lexical mapping depends on lexical resources (synonyms, hyponyms, meronyms) extracted using large language models and linguistic heuristics. However, this approach may **also** inherit biases and incompleteness from the LLM itself.
- While the proposed Lexical Alignment Curve framework effectively addresses the shortcomings of embedding-based mappings, its evaluation scope is relatively narrow. The analysis focuses primarily on Cityscapes and ADE20K, which, although standard, may not fully represent the diversity of open-ended segmentation scenarios (e.g., fine-grained or domain-specific datasets such as LVIS or COCO-Stuff).
- Although the paper is titled “Benchmarking Open-Ended Segmentation,” the experimental setup assumes that masks are provided as input, meaning that the model does not perform mask generation or spatial localization itself. Consequently, the task formulation aligns more closely with open-ended region or object recognition, rather than full open-ended segmentation.
- Several closely related works [1,2,3] are missing from the discussion. 

[1] Shin, Heeseong, Chaehyun Kim, Sunghwan Hong, Seokju Cho, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. "Towards open-vocabulary semantic segmentation without semantic labels." Advances in Neural Information Processing Systems 37 (2024): 9153-9177.

[2] Ülger, Osman, Maksymilian Kulicki, Yuki Asano, and Martin R. Oswald. "Auto-vocabulary semantic segmentation." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 24266-24275. 2025.

[3] Rewatbowornwong, Pitchaporn, Nattanat Chatthee, Ekapol Chuangsuwanich, and Supasorn Suwajanakorn. "Zero-guidance segmentation using zero segment labels." In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1162-1172. 2023.

### Questions
- How does the proposed lexical mapping handle cases where the grammatical subject does not correspond to the true visual referent (e.g., “A man is walking a dog”)?
- Does the method explicitly account for modifiers such as adjectives or compound nouns (e.g., “toy dog,” “hot dog,” “drawing of a dog”)?
- Since the lexical mapping depends on large language models for synonym, hyponym, and meronym extraction, how do the authors address potential biases or inconsistencies in these LLM-generated resources?
- Could the authors discuss how their framework might generalize to fine-grained or domain-specific datasets such as LVIS or COCO-Stuff?
- Given that the experiments assume pre-defined masks as input, does the proposed setup still align with the notion of “open-ended segmentation”?

### Soundness
3

### Presentation
3

### Contribution
3
