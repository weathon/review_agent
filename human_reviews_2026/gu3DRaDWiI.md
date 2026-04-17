# ROVER: Benchmarking Reciprocal Cross-Modal Reasoning for Omnimodal Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Unified multimodal models (UMMs) have emerged as a powerful paradigm for seamlessly unifying text and image understanding and generation. However, prevailing evaluations treat these abilities in isolation, such that tasks with multimodal inputs and outputs are scored primarily through unimodal reasoning, i.e., textual benchmarks emphasize language-based reasoning, while visual benchmarks emphasize reasoning outcomes manifested in the pixels. We introduce ROVER to address this pressing need to test reciprocal cross-modal reasoning, the use of one modality to guide, verify, or refine outputs in the other, an ability central to the vision of unified multimodal intelligence.
ROVER is a human-annotated benchmark that explicitly targets reciprocal cross-modal reasoning, which contains 1,312 tasks grounded in 1,876 images, spanning two complementary settings. Verbally-augmented reasoning for visual generation evaluates whether models can use verbal prompts and reasoning chains to guide faithful image synthesis. Visually-augmented reasoning for verbal generation evaluates whether models can generate intermediate visualizations that strengthen their own reasoning processes for question answering.
Experiments on 17 unified models reveal two key findings:  (i) Cross-modal reasoning determines visual generation quality, with interleaved models significantly outperforming non-interleaved ones; notably, combining strong unimodal models fails to achieve comparable reasoning.  (ii) Models show dissociation between physical and symbolic reasoning: they succeed at interpreting perceptual concepts literally but fail to construct visual abstractions for symbolic tasks, where faulty reasoning harms performance. These results highlight reciprocal cross-modal reasoning as a critical frontier for enabling true omnimodal generation. Homepage: https://roverbench.github.io

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ROVER, a benchmark designed to evaluate the reciprocal cross-modal reasoning capabilities of Unified Multimodal Models (UMMs). The authors argue that existing benchmarks fail to assess how models use one modality to guide or verify outputs in another, instead testing text and image abilities in isolation. ROVER addresses this gap with over 1,200 tasks that require integrated reasoning across modalities, focusing on verbally-augmented reasoning for visual generation and visually-augmented reasoning for verbal generation. By testing 17 state-of-the-art UMMs, the study finds that cross-modal reasoning skills strongly correlate with visual generation performance. However, it also reveals that current models are severely limited in visually-augmented reasoning, showing particular weakness in logical tasks compared to perception and physical modeling.

### Strengths
The paper introduces the first benchmark that requires generating both visual and textual content for joint visual and textual reasoning, effectively unifying the two modalities. The authors conduct extensive experiments demonstrating that incorporating multimodal generation improves performance compared to text-only generation during evaluation. The finding that models with stronger image–text interleaving capabilities outperform image-editing models is also noteworthy. Overall, the paper provides clear evidence that text generation supports image generation, and image generation, in turn, enhances textual reasoning.

### Weaknesses
1. The paper's evaluation methodology relies heavily on the quality of generated images, particularly for Reasoning Visual (RV), which requires generating coherent images to facilitate correct reasoning. However, the use of a VLM as a judge is questionable. Figure 8 reveals a significantly low correlation (0.63) and a high MAE of nearly 1.0 between GPT's evaluations and human judgments. Assuming the four human evaluators provide a more reliable gold standard, this discrepancy undermines the validity of using VLMs to assess RV quality. This concern is amplified by prior work demonstrating that even state-of-the-art VLMs struggle with fundamental spatial and temporal reasoning.
2. In addition to the questionable reliability of the VLM-as-a-judge paradigm, the paper fails to address the financial costs associated with using the GPT for evaluation, a notable omission given the large volume of generated images involved.
3. While the authors present Figure 7b to show the correlation between different task types, this finding is largely unsurprising due to the semantic definitions of the tasks. Crucially, it remains unclear how this analysis provides actionable insights or how it might guide the development of future models.

### Questions
See weaknesses.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes ROVER, a benchmark with over 1,200 tasks and 2,048 images for reciprocal cross-modal reasoning. ROVER has two parts: ROVER-IG (language guiding image generation) and ROVER-TG (vision aiding text generation). They tested 17 UMMs with a VLM judge plus expert checks, finding cross-modal reasoning ties to visual generation, but models struggle with vision-aided logic tasks.

### Strengths
1. This paper is generally well-written and easy to follow, with clearly illustrated figures.
2. ROVER covers a wide range of both language-reasoning tasks and visual-reasoning tasks, and uses a comprehensive evaluation method (VLM + expert validation) to ensure reliability.
3. The authors evaluate 17 unified multimodal models and provide insightful findings.

### Weaknesses
1. The benchmark heavily depends on a "VLM-as-a-judge" for scoring complex reasoning qualities. The paper's own user study (Figure 8) shows that while correlation is good, there are noticeable discrepancies, especially for reasoning-related metrics. This introduces a potential bias, where the benchmark might favor models whose outputs align with the judging VLM's own reasoning patterns.
2. As listed in Table 3, language-only models often match or exceed the performance of unified models on reasoning tasks, questioning whether the current task design truly requires cross-modal reasoning for optimal results ("thinking with images").​

### Questions
1. How does the up-to-date Gemini-2.5-pro perform on this benchmark?

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
4

### Summary
This paper introduces ROVER, a new benchmark designed to evaluate reciprocal cross-modal reasoning in unified multimodal models (UMMs), i.e., the ability to use one modality (text or image) to guide reasoning and generation in the other. Existing evaluations tend to isolate modalities, emphasizing either textual or visual reasoning in isolation, which fails to capture the intended integration of modern UMMs.

ROVER fills this gap through over 1,200 human-annotated tasks grounded in 2,048 images, spanning two complementary settings: (1) verbally-augmented reasoning for visual generation, where structured verbal reasoning guides faithful image synthesis, and (2) visually-augmented reasoning for verbal generation, where models generate intermediate visualizations to support their reasoning.

The authors evaluate 17 state-of-the-art UMMs and find that cross-modal reasoning ability correlates with visual generation performance, especially for interleaved text–image tasks. However, most models remain weak in visually-augmented reasoning, particularly in logical reasoning scenarios. 

Overall, the work is a good first step towards analyzing cross model reasoning abilities of current UMMs and avenues of improvement, but still requires more work to solidify its usability and interpretability.

### Strengths
1. Importance of cross model reasoning in UMM and problem formulation in two complementary settings of verbally-augmented reasoning for visual generation (ROVER-IG) and visually-augmented reasoning for verbal generation (ROVER-TG) is interesting, useful and novel.
2. Careful dataset design into top level domains and subtasks for both ROVER-IG and ROVER-TG
3. Detailed metrics that aim to provide a holisitic understanding of the model performance in either settings.
4. Interesting analysis like coherence between reasoning substasks.

### Weaknesses
The paper gives a good shot to cover a novel perspective but falls short in these following areas:

1. Stretch / Over claims: 
a) "Pg 5 section 4.1 (last para) the authors claim that gaps in reasoning process and alignment is the fundamental driver of diminished visual generation performance" but as seen for table 2, if you look at natural science or logic for instance for both closed and open source model, similar RP and align scores show great variability in RV scores. 
b) "Pg 7 section 4.2 Models demonstrate superior interleaved reasoning performance on physical world and visual perception tasks compared to logical reasoning challenges" is not supported in table 3, model perform similarly for the best for visual perception only and they have similar low performance in logic and physical world domains.

2. Clarifications
a). It is difficult to infer anything from the % reported in the paper, none of them mention if its absolute, or relative and relative with respect to what ?
b) visual generation performance on pg 5 last paragraph is vague. from reading context, i can map it to RV but would urge the authors to make explicit connections between numbers and metrics, especially when they define them
c) Section 4.3 Cross-modal Reasoning matters for UMMs: Could not follow through this analysis, CLIP-1 and edit world are introduced out of the blue without prior context. Fig on pg 9 top right has the corresponding details but is not reference in text and the figure itself is unclear, with some bars having +ve/-ve value and being of different lengths + no caption. Could not make sense of this at all

3. Judge reliability evaluation
a) Human correlation of RV one of the important metrics for IG is low
b) Only IG metrics undergo reliability evaluation what about TG metrics which are also llm judges ?
c) The models used for judge calibration are either closed source models or the strongest open-source model. This could potentially add bias in score calibration, having a weaker open source model being part of calibration can ensure that the entire spectrum of scores is callibrated.

4. Missing details
a) Fig 4, it would be nice to see the reasoning generated by the models in addition to input text and generated image, to better analyze reasoning alignment.
b) Table 2 does not clearly indicate which models are interleaved vs single turn, image editing only vs UMM but uses these terminologies in the analysis section when table 2 is referenced. Appendix does provide some insight to them but still models like Show-o2, Blip3o-8b, Janus-Pro-7b, etc have not been classified, making it difficult to relate to the claims in the paper

5. Overall the paper writing needs to improve, better references to figures and tables, improved captions

### Questions
I have added my questions as part of weakness itself, would urge the authors to respond to them. In its current state the work does not merit publication, however if the authors make the necessary clarifications and substantiate their claims, the benchmark would be more useful and i can consider bumping my score.

### Soundness
1

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ROVER, the first human-annotated benchmark explicitly designed to evaluate reciprocal cross-modal reasoning in Unified Multimodal Models (UMMs). It addresses the fundamental limitation of existing benchmarks, which treat understanding and generation abilities in isolation, failing to assess how one modality can guide, verify, or refine outputs in the other. 17 SOTA UMMs have been evaluated on the visual generation and text generation settings.

### Strengths
- The paper is well structured and presented overall, with a helpful project page.
- It addresses an important gap in UMMs by benchmarking and evaluating reciprocal cross-modal reasoning.

### Weaknesses
- Table 1 should include comparisons across more aspects. Additional explanations are needed in both the text and the table caption: benchmark dataset scale, whether it is for VG/TG/both, and clarifications on the multi-dimensional and hybrid evaluations and the types.
- This work's emphasis on intermediate reasoning as a core signal for multimodal reasoning distinguishes it from existing benchmarks. However, the data curation process for these progressive reasoning steps is under-specified, especially for the TG setup. The paper should clarify exactly what intermediate data is curated for various sub-tasks, dataset statistics, and how it is used for evaluation.
- The current evaluation depends only on GPT-based judgment. Introducing objective, automatically computed metrics would improve the reliability of the fine-grained reasoning evaluation. 
- Would it also be beneficial to include the text reasoning chain for the TG task? Clarification is needed on how the progressive visual reasoning steps are validated as active reasoning components rather than decorative elements, as claimed in line 246.

### Questions
Please see the weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
