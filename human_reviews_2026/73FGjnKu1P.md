# Seeing What’s Wrong: A Trajectory-Guided Approach to Caption Error Detection

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Error detection is critical for enhancing multimodal dataset reliability and downstream model performance. Existing error filters, while increasingly powerful, typically rely on a single similarity score per image–caption pair. This is limiting: captions with subtle errors (e.g., mislabeled objects, incorrect colors, or negations) can still score highly, while correct but imprecisely worded captions may score poorly. To address this, we introduce the notion of a caption trajectory: an ordered sequence of captions produced by iteratively editing a caption to maximize an image-text relevance score. This trajectory carries rich signals for error detection. Correct captions typically stabilize after minor edits, while erroneous captions undergo substantial improvements. Building on these insights, we introduce TRACED, a cost-efficient and model-agnostic framework that leverages trajectory statistics for more accurate caption error detection. Beyond detection, TRACED also serves as an interpretable tool for identifying the origins of errors. We further demonstrate that, in the case of error correction, this interpretable token-level error information can be provided to VLMs to enhance the alignment score of the generated captions. On MS COCO and Flickr30k, TRACED achieves up to 2.8% improvement in accuracy for error detection across three noise types. Our code is available at https://github.com/mazumder-lab/TRACED.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TRACED, a new method for caption error detection. By looking at caption trajectories – iterative edit traces starting from the original caption – TRACED looks for atomic edits that greatly improve an image-text alignment score, finding that these frequently correspond to fine-grained misalignments between the original caption and image. Results show that TRACED is effective at identifying and correcting caption errors, potentially complementing existing methods.

### Strengths
The idea of using caption trajectories to identify fine-grained misalignments is clever and appears novel.

The framework is model-agnostic, making it flexible and potentially applicable to a wide range of settings and improvements in vision-language models.

The results show a modest but significant improvement in error detection using TRACED relative to baseline methods.

Overall the paper is clearly written.

### Weaknesses
The evaluation seems partly circular, as the results in Sec 4.4 show that TRACED, which is designed based on single-token edits, improves error detection for synthetic edits which may be largely single-token based (L367–374). In addition TRACED relies on an automatic alignment score using models which may be biased towards the types of errors generated synthetically. This seems partly in line with Table 4 where improvements for random caption substitution are mostly very small.

There are some evident limitations of the types of trajectories used which are not clearly discussed. Conceptually, since VLMs like CLIP and BLIP struggle with compositionality, it is not clear that they will be robust to fine-grained compositional misalignments (e.g. “a jacket” vs. “no jacket”). [1] Since trajectories only use single token edits, it is not clear how or if TRACED could handle misalignments due to multi-token words for phrases. 

Optimization-based edits lead to non-fluent captions, as seen in Fig 6–8 (“vehicles on near …”, “thecoat”, “vehiclesl” etc.), making it unclear whether the trajectory signal is fully meaningful. Flagged words include function words (L931, e.g. “is”). Limitations and/or robustness to these aspects should be explicitly discussed.

The improvements in error detection seem modest (Tab 1), raising questions about whether it is worth the added computational overhead. While the paper claims the method is “efficient” in several places, some of the runtimes reported in Tab 3 are rather high (although I acknowledge that this is a one-time cost).

[1] Yuksekgonul et al. When and why vision-language models behave like bags-of-words, and what to do about it? ICLR 2023

### Questions
How are flagged words (as in Fig 4) determined? Is it a fixed threshold applied to changes in s(...) or c(...)?

Other works explore caption filtering, recaptioning, or error correction showing improvement on downstream VLM performance. Have you tested this? How does your method compare, or is it complementary to these approaches?

Is the interpretability application (Sec 3.3) purely impressionistic, or have you tested that TRACED flagged errors correlate statistically with human judgements of the source of misalignments?

It seems like Elimination is usually more effective than the more complex GCD-based methods while being much more efficient. What is the motivation for testing GCD? Could it be improved to be competitive or does it have other conceptual or practical benefits?

In Tab 3, Fast GCD is approximately x240 slower than elimination, vs. L1608 where it is about x9 slower. Is this correct?

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
4

### Summary
The paper proposes **TRACED**, a model-agnostic framework for detecting erroneous image–caption pairs by analyzing a **trajectory** of edits applied to the caption that monotonically seek to improve an image–text relevance score $s$. Instead of trusting a single similarity score, TRACED constructs a sequence of candidate captions via token deletions and/or replacements, tracks how (s) evolves, and also measures semantic drift relative to the original caption. A classifier is then trained on trajectory features to predict whether the original pair is correct. The authors evaluate TRACED on MS-COCO, Flickr30k, and MM-IMDb, showing consistent AUC gains over CLIP, BLIP, and LEMoN baselines; they also introduce a “fine-grained” noise process (minimal GPT-4o-mini edits) to stress subtle errors. Beyond detection, the trajectory highlights token sources of error; these token-level hints can guide a VLM to produce higher-scoring corrected captions.

### Strengths
1. Moving from a single score to **trajectory statistics** is intuitive and broadly compatible with popular alignment functions (CLIP cosine, BLIP ITC/ITM, LEMoN). The paper’s Algorithm-1-style procedure and the use of both alignment evolution and semantic similarity are well-motivated. 
2. Trajectories naturally localize problematic tokens (e.g., removing “no” resolves a negation error). This is a useful diagnostic for dataset cleaning pipelines and for prompting downstream caption correction.
3. Supplying token-level “flagged words” to a VLM improves BLIP alignment of corrected captions (especially with larger models + CoT prompting), supporting the claim that trajectories provide actionable signals, not just detection.

### Weaknesses
1. TRACED *optimizes and evaluates* using alignment metrics like BLIP ITM and CLIP similarity. While convenient, this risks **circularity**: if trajectories explicitly maximize (s), improved AUC and “correction quality” may partly reflect over-optimization of the very metric used for labels/evaluation rather than true semantic correctness.
2. A large part of the evaluation hinges on **synthetic** perturbations, including the proposed fine-grained GPT-4o-mini edits. While this nicely induces subtle errors, it is still simulated. It remains unclear how well TRACED generalizes to **real, organically noisy** web-scale datasets.
3. We’re told using both alignment and semantic-similarity signals beats using either alone, and that Elimination often wins; however, a fuller **ablation** would help isolate what matters most: trajectory length (T), candidate count (N), semantic similarity choice, and the classifier architecture.
4. While the paper argues scalability, **single-pass** baselines are very cheap; TRACED adds multiple forward passes per pair. The reported 6.5h/1M pairs (BLIP-ITM + Elimination on 4×L40) is reasonable, but not trivial for very large corpora; clearer guidance on **cost/quality trade-offs** (when Elimination suffices vs. when Fast GCD pays off) would aid adoption.

### Questions
N/A

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
This paper aims to detect and correct errors in vision-text data pairs, which is an interesting issue. To tackle this issue, a trajectory-guided method, named TRACED, is proposed, which edits the caption iteratively. This is a plug-and-play framework for different scoring multi-modality models. Experiments show that such a detection and correction manner is helpful and useful.

### Strengths
1.	The motivation of this paper is interesting.
2.	The proposed TRACED model is effective in detecting caption errors and finding the most influential tokens in a sentence.
3.	The experimental results demonstrate the effectiveness of this model.

### Weaknesses
1.	From my perspective, it’s true that current vision-text training data may contain mistakes, which affects the model's learning. But the authors don’t provide the number and the proportion of mislabelled instances of a specific dataset, like MS-COCO, Flickr30K. Hence, I have doubts about whether the pre-trained datasets have this issue. 
2.	Table 1 is not easy to read and follow. 
3.	In Appendix A.4, the authors propose to construct a new benchmark, but I cannot see more details about this benchmark. 
4.	This paper is able to detect errors in captions. Hence, I think it would be better to represent how the correction procedure improves the multi-modal models’ understanding performance, such as the image-text retrieval task, the captioning task, etc. For now, Figure 12 still evaluates the data itself with different models after correction.

### Questions
1.	I cannot fully understand Table 1. Take the Flickr30K dataset as an example. What does the AUC here mean? Then what about the improvement? What’s the baseline result to compute the improvement? 
2.	For each image-text data pair, how many noised captions does the pipeline generate? 


------
This motivation is interesting to me. I will be happy to raise my score if my concerns can be solved.

### Soundness
3

### Presentation
2

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
This paper proposes TRACED (Trajectory Creation for Error Detection), a framework for detecting errors in image-caption datasets. Unlike existing methods that rely on a single similarity score, TRACED iteratively edits captions to create a "trajectory" and analyzes how alignment scores and semantic similarity evolve. The key insight is that correct captions stabilize after minor edits, while erroneous captions show substantial improvement potential.

**Main contributions:**
1. A novel trajectory-based approach using token deletion (Elimination) and replacement (GCD, Fast GCD)
2. A model-agnostic framework applicable to existing methods (CLIP, LEMoN, BLIP)
3. Introduction of fine-grained noise generated by GPT-4o-mini, more realistic than random/noun swaps
4. Up to 2.8% AUC improvement on MS COCO, Flickr30k, and MM-IMDb
5. Token-level error localization for interpretability and caption correction, achieving up to 14.5% alignment score improvement

### Strengths
1. **Novel and intuitive approach**: The trajectory-based idea is creative and well-motivated. The insight that correct captions require minimal edits while erroneous ones show substantial improvement is clear and compelling. The framework is model-agnostic and consistently improves multiple baselines (CLIP, LEMoN, BLIP).

2. **Practical value with interpretability**: Beyond detection, TRACED provides token-level error localization and enables caption correction. The 14.5% alignment score improvement on InternVL3-14B demonstrates practical utility.

3. **More realistic evaluation**: The fine-grained noise benchmark better reflects real annotation errors than wholesale caption swaps. The finding that TRACED's largest gains occur under fine-grained noise (where baselines struggle most) validates the approach's practical relevance.

### Weaknesses
1. **Limited demonstration of generality (Critical)**: The same "no jacket" example appears repeatedly (Figures 1, 3, and text), raising cherry-picking concerns. The paper lacks diverse examples showing different error types (object misidentification, color errors, numerical errors, relationship errors, distributed errors) and failure cases. This is essential for convincing readers of the method's generality, especially given concerns about ambiguous images, subjective captions, and varying caption lengths.

2. **Circular reasoning and evaluation concerns**: Caption correction is evaluated using BLIP alignment scores while being optimized with BLIP-based TRACED. Independent metrics (METEOR, CIDEr, SPICE) or human evaluation are needed to verify whether "correction" actually fixes factual errors rather than simply adjusting to BLIP's preferences. Additionally, critical results are relegated to the appendix: noise-type breakdown (Table 4), trajectory importance (A.12), and maximization vs. minimization (A.13) should be in the main text to support key claims.

3. **Insufficient validation of fine-grained noise**: While conceptually appealing, the fine-grained noise lacks validation against real annotation errors. The use of GPT-4o-mini raises reproducibility concerns, and the omission of MM-IMDb due to cost limits generalizability claims. The 50% noise rate is also unrealistic, performance at lower rates (10%, 20%) is needed. Furthermore, key questions remain unaddressed: Why is maximization better than minimization? What about token addition (not just deletion/replacement)? What is the sensitivity to hyperparameters N and T?

### Questions
1. Can you provide diverse examples showing different error types (object, color, number, relationship, distributed errors) and failure cases beyond the repeated "no jacket" example?

2. How did you validate that GPT-4o-mini generated noise reflects real annotation errors? Can you compare against datasets with actual annotation errors (e.g., Conceptual Captions)?

3. Can you evaluate caption correction using independent metrics (CIDEr, SPICE) or human evaluation, not just BLIP scores? Does "correction" fix factual errors or just adjust to BLIP preferences?

4. Why are critical ablation results (Table 4 on noise types, A.12 on trajectory importance, A.13 on maximization vs. minimization) in the appendix rather than the main text? Can you provide specific numbers showing fine-grained noise improvements in the main paper?

5. How does the method perform at realistic noise rates (10%, 20%) rather than 50%? What about very long/short captions, ambiguous images, or subjective descriptions?

6. Why not consider token addition, only deletion/replacement? What is the sensitivity to N and T? Why does maximization outperform minimization (A.13 only says "seem to be obtained")?

### Soundness
3

### Presentation
2

### Contribution
3
