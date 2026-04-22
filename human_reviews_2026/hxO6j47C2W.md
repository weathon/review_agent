# MMGenBench: Fully Automatically Evaluating LMMs from the Text-to-Image Generation Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large Multimodal Models (LMMs) demonstrate impressive capabilities. However, current benchmarks predominantly focus on image comprehension in specific domains, and these benchmarks are labor-intensive to construct. Moreover, their answers tend to be brief, making it difficult to assess the ability of LMMs to generate detailed descriptions of images. To address these limitations, we propose the MMGenBench-Pipeline, a straightforward and fully automated evaluation pipeline. This involves generating textual descriptions from input images, using these descriptions to create auxiliary images via text-to-image generative models, and then comparing the original and generated images. Furthermore, to ensure the effectiveness of MMGenBench-Pipeline, we design MMGenBench-Test, evaluating LMMs across 13 distinct image patterns, and MMGenBench-Domain, focusing on generative image performance. A thorough evaluation involving over 50 popular LMMs demonstrates the effectiveness and reliability of both the pipeline and benchmark. Our observations indicate that numerous LMMs excelling in existing benchmarks fail to adequately complete the basic tasks related to image understanding and description. This finding highlights the substantial potential for performance improvement in current LMMs and suggests avenues for future model optimization. Concurrently, MMGenBench-Pipeline can efficiently assess the performance of LMMs across diverse domains using only image inputs. All code and data will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces MMGenBench, a fully automated pipeline for evaluating LMMs’ image understanding and detailed description. Given an input image, an LMM generates a fine-grained caption; a text-to-image (T2I) model then synthesizes an auxiliary image from that text; finally, a single image encoder (Unicom) computes feature-level similarity (SIM, FID) between the original and synthesized images. Two datasets are provided: MMGenBench-Test (13 image patterns, 1,284 images) and MMGenBench-Domain (~10k images). Results are reported for 50+ LMMs; the best SIM is < 0.6, and a human-alignment study reports 88.27% agreement on 1,850 pairs.

### Strengths
1.Clear, modular pipeline with explicit metric definitions (SIM/FID).

2.Broad coverage (50+ LMMs) with series-wise and pattern-wise analyses.

3.Practical cross-domain evaluation requiring only images.

### Weaknesses
1.Metric robustness not established.  Features come from a single encoder (Unicom), and main results default to one T2I model (FLUX.1-dev); no ablations on encoder/T2I/seed variation or rank correlations are reported.

2.Length/style bias. Shorter descriptions tend to receive lower scores.

3.Stochasticity not controlled. Evidence: Randomness is acknowledged (Eq. 2), but no multi-run statistics or confidence intervals are provided.

4.Human alignment under-analyzed. Only a single overall figure (88.27%) is given;

### Questions
1.What is the per-model seed variance of SIM?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MMGenBench, a fully automated evaluation framework for large multimodal models (LMMs) from a text-to-image generation perspective. The core idea is to assess the image understanding and descriptive capabilities of LMMs through a closed-loop pipeline: (1) generating textual descriptions from input images via the tested LMMs; (2) using state-of-the-art text-to-image diffusion models to regenerate images from those descriptions; and (3) quantitatively evaluating the similarity between the original and regenerated images using SIM-Score and FID-Score. To support this pipeline, the authors build two benchmarks: MMGenBench-Test, covering 13 well-defined image patterns (e.g., Natural, Artistic, Motion, Contextual, Symbol, etc.), and MMGenBench-Domain, focusing on the “generated image” domain. Over 50 representative LMMs are evaluated.

### Strengths
1. The paper introduces a fully automated, cross-domain benchmarking framework that reduces manual labeling cost. It combines LMM-based image-to-text and diffusion-based text-to-image processes into a self-consistent evaluation loop.

2. More than 50 LMMs, both open- and closed-source, are evaluated using unified metrics.

### Weaknesses
1. My main concern about this paper lies in the closed-loop evaluation design, which involves three interconnected stages. The first stage is the LMM’s understanding of an image (which is exactly what the paper aims to evaluate); the second stage converts the generated textual description into an image using a text-to-image model; and the third stage measures the similarity between the regenerated image and the original one to assess the LMM’s image comprehension ability.

Theoretically, this approach is valid only if two key assumptions hold: (a) the text-to-image model must be sufficiently powerful to accurately render every object described in the text, and (b) the similarity metric must be capable of reliably quantifying semantic correspondence between images. In practice, however, both assumptions are problematic. Current text-to-image models are still imperfect, which raises doubts about whether the regenerated image can faithfully reflect the textual content. Likewise, existing similarity metrics struggle to capture semantic-level consistency accurately.

2. Although this method is theoretically feasible and can be fully automated, it lacks interpretability. The system outputs only a single numerical score, without providing insight into which aspects of the LMM’s understanding are inaccurate or incomplete. This makes it difficult to diagnose specific weaknesses in the evaluated model.

### Questions
1. In Lines 106–107, the authors state that “numerous LMMs excelling in existing benchmarks fail to address the basic tasks of image understanding and description.” Could the authors clarify what specific tasks are referred to here? Please elaborate on what constitutes these “basic tasks” and how they are defined or measured in the proposed framework.

2. What is the difference between MMGenBench-Test and MMGenBench-Domain? Why does MMGenBench-Test require specific annotations and human verification, whereas MMGenBench-Domain does not?

3. In Table 1, GPT-4o, which generally outperforms open-source models on most existing benchmarks, does not achieve the best results here. This raises questions about the reliability and rationality of the proposed evaluation method. The authors should discuss this discrepancy in depth—why does GPT-4o underperform, and why do larger models in the same series not consistently outperform smaller ones? Is this phenomenon caused by limitations or biases in the proposed evaluation pipeline itself, or does it genuinely reflect the intrinsic capability differences between models? This discussion is crucial for demonstrating whether the proposed benchmark is both valid and capable of challenging or redefining existing evaluation systems.

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
3

### Summary
This paper introduces MMGenBench, a novel evaluation framework designed to assess Large Multimodal Models (LMMs) from the perspective of text-to-image generation. The central contribution is the MMGenBench-Pipeline, which proposes a "fully automated" method: an LMM generates a textual description from an input image, this description is then fed to a text-to-image (T2I) model to create an auxiliary image, and finally, the similarity between the original and auxiliary images is quantitatively measured using an image representation model. To support this pipeline, the authors develop two benchmarks: MMGenBench-Test, featuring 1284 images categorized into 13 distinct patterns (e.g., Surreal, Color, Motion), and MMGenBench-Domain, focused on generative images. The paper conducts an extensive evaluation of over 50 popular LMMs, revealing their limitations in generating detailed descriptions and adhering to instructions, even for models that excel in existing benchmarks.

### Strengths
1. The core idea of using text-to-image models to "reconstruct" an image from an LMM's description is a creative and insightful way to quantify the fidelity and detail of LMM understanding. This "information compression and restoration" perspective is a fresh contribution to LMM evaluation. The paper effectively highlights critical shortcomings of current LMMs, such as poor instruction following, inability to generate sufficiently detailed descriptions, and instances of "safety" overfitting. These findings are highly significant for guiding future LMM research and development.
2. The paper conducts a thorough evaluation across more than 50 diverse LMMs, including both open-source and proprietary models. This broad assessment provides a valuable overview of the current landscape of LMM capabilities and weaknesses.
3. The introduction of MMGenBench-Test, with its 13 distinct image patterns, offers a more granular and structured approach to evaluating LMMs across various visual characteristics. MMGenBench-Domain further extends this to generative images. The inclusion of qualitative examples provides compelling visual evidence for the observed LMM failures, making the paper's points tangible and understandable.

### Weaknesses
1. The most significant weakness is the repeated and misleading assertion of a "fully automated" benchmark. The construction of MMGenBench-Test explicitly involves "human check" and "manually filtered" steps. This fundamental contradiction undermines the paper's credibility and its claims of scalability and objectivity in benchmark creation.
2. The entire pipeline's accuracy and fairness are inherently tied to the performance and potential biases of the underlying GPT-4o, text-to-image models (FLUX, SD3.5, etc.), and the Unicom image representation model. The paper lacks a rigorous analysis or quantification of how these dependencies affect the LMM evaluation results. Without this, it's difficult to ascertain whether observed LMM failures are due to the LMM itself or limitations propagated from the auxiliary models.
3. The LMMs are specifically prompted to generate an "image caption-prompt" for a T2I model, with a word count constraint (20-60 words). This framing might inadvertently bias LMM outputs towards a specific style or level of detail amenable to T2I models, rather than truly assessing their general capability for any detailed image description. The word count limit, in particular, can hinder genuinely comprehensive descriptions for complex images.
4. While the paper effectively identifies common LMM problems (instruction following, lack of detail, overfitting), the analysis largely remains descriptive. There's minimal deeper investigation into the underlying causes (e.g., specific training data deficiencies, architectural limitations, prompt engineering sensitivity) or potential avenues for mitigation. This limits the actionable insights for model developers seeking to improve LMMs.
5. While the overall MMGenBench-Test dataset has 1284 images, some of the 13 patterns have relatively few examples (e.g., "Orientation" with 114 images, "Motion" with 160). This could lead to less robust or generalizable evaluation results for those specific categories.

### Questions
1. Could the authors explicitly clarify and rectify the "fully automated" claim, particularly concerning the benchmark construction? Please detail the extent of human involvement (e.g., person-hours, number of annotators) in the "human check" and "manually filtered" stages of MMGenBench-Test creation. How does this manual effort impact the scalability and objectivity claims?
2. Given the heavy reliance on external models (GPT-4o, T2I models, Unicom), what specific experiments or analyses were conducted (or are planned) to understand and quantify the impact of their potential biases, failure modes, or performance ceilings on the LMM evaluation results? For example, if a T2I model struggles with counting objects, how does this affect the evaluation of an LMM that accurately described the count?
3. The LMM prompt asks for an "image caption-prompt" for a T2I model, with a 20-60 word limit. How might this specific instruction and length constraint influence the LMMs' output compared to a more open-ended request for a general "detailed image description"? Have you explored variations in this prompt, and if so, what were the effects?
4. Beyond SIM-Score and FID-Score, have the authors considered or explored other metrics that could capture more fine-grained aspects of descriptive accuracy, such as object presence, attribute correctness, or relational understanding, perhaps by integrating object detection or semantic segmentation models on the generated images?
5. For the "Model Overfitting" weakness, particularly the "safety" example, can the authors provide more quantitative evidence or analysis across the benchmark to show how widespread this issue is among LMMs and its overall impact on their scores?
6. You state that "MMGenBench-Domain includes 10,000 images, thereby improving the accuracy of its FID-Score measurement. Therefore, we propose using SIM-Score as the primary metric." Could you elaborate on why FID-Score is considered less reliable for MMGenBench-Test (due to fewer images) and why SIM-Score is deemed a more suitable primary metric across both benchmarks, despite FID being a common generative metric?
7. Please provide more details on the "human metric based on votes from 9 human experts" mentioned for pipeline effectiveness. What was the exact task given to humans, how were "comparable" cases defined, and what was the inter-annotator agreement?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This benchmark paper presents MMGenBench, an automated pipeline to evaluate the caption capability of LMMs. The key idea is to reconstruct an auxiliary image via powerful text-to-image models, followed by representation-level comparison between the original and generated images. On top of the pipeline, the authors build two benchmarks: MMGenBench-Test, which covers 13 carefully summarized image patterns derived from JourneyDB, and MMGenBench-Domain, which targets the “generated images” domain. Evaluating 50+ popular LMMs shows: (i) current models are far from perfect (best SIM < 0.6); (ii) models strong on existing VQA/caption/ocr leaderboards can still fail at detailed description; and (iii) the proposed pipeline aligns with human judgment in 88.27% of sampled cases.

### Strengths
- The motivation is clear and the idea is interesting. The paper targets a real gap: most current LMM benchmarks emphasize short answers and specific domains, while real applications need long, faithful, instruction-following image descriptions. Meanwhile, the proposed automated evaluation pipeline is scalable.

- The Large-scale empirical study demonstrates interesting insights. Evaluating 50+ LMMs and reporting pattern-wise weaknesses (context, orientation, count, motion) gives the community actionable signals.

- Human alignment check. The 88% agreement suggests the metric is not completely drifting away from human judgment.

### Weaknesses
- An important issue is the metric entanglement with the T2I model. The final score is a function of (LMM description quality) × (T2I controllability) × (image encoder). Even though four T2I models are tried, the paper does not quantify how much ranking changes if the T2I model is weaker/safer/biased. A sensitivity analysis is needed.

- Using only Unicom as the image representation back-end makes the whole pipeline hinge on one model’s inductive biases. Showing results with different encoders such as CLIP and DINOv2 would make the claim of “fully automatic and reliable” more convincing.

- Both MMGenBench-Test and Domain come from JourneyDB-like, style-rich, often synthetic images. It is unclear whether the same pipeline will hold for photos, documents, Med-VQA, or low-res, cluttered, user-uploaded images. A small real-photo subset would strengthen the story.

- Since the pipeline favors descriptions that lead to well-conditioned T2I prompts, a model that always outputs long, enumerated, style-heavy prompts may score higher than a model that is actually more faithful but concise. The paper partially discusses instruction-following failure (Fig. 9) but does not experiment with length-controlled outputs.

### Questions
How is the consistency of the human annotators? Does the inter-annotator agreements correlate with certain patterns (e.g., Contextual, Orientation, Count), where T2I models naturally struggle?

### Soundness
3

### Presentation
3

### Contribution
3
