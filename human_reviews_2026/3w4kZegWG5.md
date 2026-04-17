# Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems

- Decision: Accept (Poster)
- Scores: 8, 2, 4, 8

## Abstract
Bongard Problems (BPs) provide a challenging testbed for abstract visual reasoning (AVR), requiring models to identify visual concepts from just a few examples and describe them in natural language. Early BP benchmarks featured synthetic black-and-white drawings, which might not fully capture the complexity of real-world scenes. Subsequent BP datasets employed real-world images, albeit the represented concepts are identifiable from high-level image features, reducing the task complexity. Differently, the recently released Bongard-RWR dataset aimed at representing abstract concepts formulated in the original BPs using fine-grained real-world images. Its manual construction, however, limited the dataset size to just $60$ instances, constraining evaluation robustness. In this work, we introduce Bongard-RWR+, a BP dataset composed of $5400$ instances that represent original BP abstract concepts using real-world-like images generated via a vision language model (VLM) pipeline. Building on Bongard-RWR, we employ Pixtral-12B to describe manually curated images and generate new descriptions aligned with the underlying concepts, use Flux.1-dev to synthesize images from these descriptions, and manually verify that the generated images faithfully reflect the intended concepts. We evaluate state-of-the-art VLMs across diverse BP formulations, including binary and multiclass classification, as well as textual answer generation. Our findings reveal that while VLMs can recognize coarse-grained visual concepts, they consistently struggle with discerning fine-grained concepts, highlighting limitations in their reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper is about Bongard Problems (a classical test of abstract visual reasoning).

The first contribution is a pipeline to generate such problems at scale, with synthetic photo-realistic images.

The second contribution is an evaluation of existing models on the resulting set of new problems.

### Strengths
The paper addresses a gap in the literature: there was no large dataset of such photo-realistic BPs.
BPs are a classical test so there is interest in keeping to use this paradigm to test AI models.
The literature review does a good job of reviewing the extensive related work in abstract visual reasoning.

The evaluation uses several (4) current large models. It includes several formulations of the task, and multiple interesting variations and ablations (model size, color vs. grayscale,

### Weaknesses
I don't see any clear weakness to this paper.

### Questions
N/A

---
Tips for the presentation:

- Typo on the 2nd line of the abstract
- Figure 5: may be interesting to mark the "random choice" ("chance") baseline for each K, e.g. as a small gray line?
- Inconsistent capitalization in paragraph headers: e.g. L309 "Concept selection" (no need to capitalize "selection", it's still a simple noun even if it's the name of an experiment), L342, L357, L465
- Figure 8: I don't think these are "ablation" experiments; ablation means *taking away* (some part of the model, usually). A correct name for these experiments could be, for example, a "sensitivity analysis".
- In the experiments, it may be worth reminding a few more times what some key symbols mean (K, P), for the readers who will scan through the paper without reading everything top to bottom. Another option may be to replace these "unguessable" symbols by something more explicit (mabye something like N_candidates? I'm not totally sure it's a good idea, maybe it's too cumbersome). Or maybe include/define these symbols in one of the figures that describe the dataset/setup?

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
5

### Summary
This paper introduces Bongard-RWR+, a dataset of 5,400 Bongard Problems that uses AI-generated real-world-like images to represent abstract visual concepts from classical Bongard Problems. The authors develop a semi-automated pipeline using vision-language models to generate images, then evaluate state-of-the-art VLMs on multiple task formulations including binary classification, multiclass concept selection, and free-form text generation. Their experiments reveal that while current VLMs can recognize coarse-grained visual concepts, they consistently struggle with fine-grained concept recognition, highlighting limitations in their abstract visual reasoning capabilities.

### Strengths
The paper addresses a relevant problem by scaling up the Bongard-RWR dataset from 60 to 5,400 instances through a semi-automated generation pipeline, which represents a reasonable engineering contribution to the abstract visual reasoning benchmark landscape. The experimental evaluation is comprehensive, covering multiple task formulations (binary classification, multiclass selection, text generation) and including useful ablation studies on factors like model size, color versus grayscale, and number of demonstrations. The paper is generally well-structured and clearly written, with detailed appendices documenting the generation process, prompts, and bias analysis. The findings consistently demonstrate that current VLMs struggle with fine-grained visual reasoning, which confirms existing concerns about their capabilities.

### Weaknesses
The paper's core limitation is that it provides dataset scaling rather than methodological innovation. The reliance on generated images raises validity concerns, especially given significant demographic bias (79.9% White figures) and the number of exclusion of original concepts due to generation failures, suggesting the approach cannot capture full abstract reasoning complexity. The experimental analysis is shallow—it confirms known VLM limitations without investigating why models fail, lacks detailed error analysis or failure mode characterization, and dismisses notable performance differences  between real and generated images rather than examining their implications. The supervised baselines achieve random performance but receive minimal analysis, missing opportunities to understand whether difficulties stem from data, architecture, or fundamental reasoning gaps. Additionally, the finding that explicit image captioning improves performance suggests models may be exploiting caption artifacts rather than demonstrating genuine visual reasoning.

### Questions
no

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
3

### Summary
The paper introduces Bongard-RWR+, a new large-scale benchmark for abstract visual reasoning, expanding on the Bongard Problems by generating 5,400 instances using a vision-language model pipeline. Despite promising results with coarse-grained concepts, state-of-the-art VLMs struggle with fine-grained reasoning tasks, highlighting significant gaps in their capabilities for multi-image abstract visual reasoning.

### Strengths
1. This paper introduces Bongard-RWR+, an innovative extension of the Bongard Problems, using a semi-automated pipeline with vision-language models (VLMs) to generate large-scale, fine-grained real-world images, addressing the limitations of manual datasets like Bongard-RWR.
2. The methodology is solid, ensuring image diversity and offering valuable insights through ablations on model size, color, and image diversity.
3. The paper is clear and well-structured, with effective visuals. Its significance lies in advancing AVR, providing a scalable benchmark that exposes gaps in current VLM capabilities and guides future research in multimodal reasoning.

### Weaknesses
1.The paper lacks a detailed exploration of how the semi-automated pipeline can be further refined for more reliable image generation. Manual filtering still plays a key role, and automated verification could improve scalability and reduce bias.
2.Although the experiments cover multiple tasks, the evaluation of fine-grained reasoning is limited. It would benefit from including more diverse models or comparing performance with human-level reasoning.
3.Lastly, a deeper analysis of errors, especially in fine-grained tasks, is needed. Understanding specific challenges faced by the models would help guide future improvements.

### Questions
1.	The paper claims that the main advantage of Bongard-RWR+ lies in its scalability. However, the image selection still relies on manual review, and according to Appendix E.2, as much as 30.2% of images were discarded. This significantly weakens the argument of "automation" in dataset construction. Why not further quantify the impact of manual intervention on the results? Is there any evidence of systematic bias in image selection due to reviewer subjectivity? 
2.	The paper only compares InternVL2.5, Qwen2-VL, LLaVA-Next, and MiniCPM-o. However, stronger closed-source models, like GPT-4V and Gemini, have been used in several AVR benchmark tests. Why were these stronger models not included in the comparison? 
3.	The authors intentionally maximize intra-side visual diversity (e.g., in Bongard-RWR+/LP) to aid concept identification. However, this strategy risks introducing distracting, spurious concepts. For instance, representing "Vertical" with a tree, a building, and a person may also activate unrelated concepts like "Nature" or "Architecture," potentially confusing the model. The claim that greater diversity makes concepts "easier to identify" is not systematically verified. The authors should analyze if an optimal level of diversity exists by examining whether accuracy on the Bongard-RWR+/LP variants plateaus or even decreases as P (and thus diversity) increases. 
4.	Although the paper claims that “models perform poorly with fine-grained concepts,” there is no comparison with human-level benchmarks. The absence of such a comparison makes it hard to interpret whether the models’ performance is truly “bad.” 
5.	As shown in Figure 16 and Appendix E, many discarded images were due to unclear structure, background confusion, or perspective errors. This suggests that the main issue might lie in the ambiguity or noise within the images themselves, rather than the models' poor reasoning capabilities. The authors should provide a quantitative analysis of “image quality vs. accuracy” to demonstrate that the problem lies in reasoning rather than perception.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces Bongard-RWR+, a large-scale benchmark (5,400 matrices) for abstract visual reasoning (AVR) that preserves the *fine-grained concepts* of classic Bongard Problems while using real-world-like images synthesized by a VLM-assisted pipeline. Starting from the concepts in Bongard-RWR, the authors (i) describe exemplar images via an I2T model, (ii) augment descriptions with a T2T model, (iii) render images with a T2I model, and (iv) perform human verification. The benchmark supports six task formulations—binary and paired image(s)/description(s)-to-side(s), multiclass concept selection, and free-form concept generation—and includes grayscale and varying-shots variants to analyze the role of color and number of demonstrations.

Across 4 strong open VLMs, results reveal a consistent pattern: performance is decent on coarse concepts (e.g., shape/size) but drops on fine-grained cues (e.g., contour, angle, rotation). Caption-then-reason pipelines help, and concept-selection accuracy scales with model size, but free-form concept generation remains weak. The authors further test functional equivalence between generated and real images and show broadly similar scaling trends, supporting the dataset’s validity.

### Strengths
- Clever use of I2T/T2T/T2I with human vetting yields large, diverse matrices that preserve fine-grained Bongard concepts rather than only coarse, object-level cues.  
- Binary/paired side assignments, concept selection, and free-form concept generation give a multifaceted picture of AVR.  
- Grayscale shows color is a distractor; more demonstrations help certain models; generated vs. real yields similar trends, supporting external validity.  
- Per-concept-group breakdowns (size/shape vs. contour/angle/rotation) isolate where VLMs fail, informing future model design.  
- A simple similarity classifier is competitive in image-based tasks, underscoring that current VLM prompting often fails to internalize the concept—a useful, humbling baseline.

### Weaknesses
- Caption/augmentation quality inherits biases and blind spots of the I2T/T2T models; although human verification mitigates this, a quantitative *masking* or *counterfactual* stress test of pipeline robustness would help.  
- BLEU/ROUGE/CIDEr/BERTScore only weakly capture conceptual correctness and fine-grained relations; a concept-aware rubric (or human evaluation on a subset) would better reflect success/failure in CG.  
- Most core experiments use a small pool of open VLMs; including stronger closed models (or stronger open ones as they appear) would calibrate the difficulty spectrum more completely.  
- While the release mitigates this, adding inter-annotator agreement for the acceptance filter directly in the main paper (not just appendix) would strengthen transparency.

### Questions
1. Have you considered automatic checkers that evaluate whether generated concepts *logically partition* the two sides (e.g., via structured parsers / learned validators), beyond surface n-gram overlap?  
2. What fraction of generated candidates fail human vetting per concept family (e.g., angle vs. shape)? Any systematic failure patterns in the T2I step?  
3. How sensitive are results to chain-of-thought or *explicit analogy* prompting in I1S/I2S (e.g., “explain the difference, then decide”) and to few-shot textual exemplars?  
4. Could the benchmark be extended to interactive *concept discovery* (iteratively request descriptions or crops) and do models improve with interaction?  
5. Beyond the bias audit, do you plan to ship per-matrix *concept tags* and *hardness annotations* so researchers can target specific failure modes?
6. The manuscript shows “ICLR 2025” in the PDF header but the submission targets ICLR 2026.

### Soundness
3

### Presentation
3

### Contribution
3
