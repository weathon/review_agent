---
job_id: a050b715-c92c-446a-b845-e506320bda49
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: yEpE0QPpf8.pdf
paper: Grounding-IQA: Grounding Multimodal Language Model for Image Quality Assessment
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is on multimodal large language models, grounding, and image quality assessment, squarely within ICLR’s scope of representation learning, multimodal learning, and datasets/benchmarks.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present and in English. The work proposes a clear paradigm (Grounding-IQA), an automatically constructed dataset (GIQA-160K), and a benchmark (GIQA-Bench) with quantitative and qualitative results. No fatal methodological or theoretical errors are apparent from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any text in the paper that attempts to manipulate automated reviewing or contains hidden prompts; formatting and content are standard for a research paper.

---

# Expected Review Outcome:

## Summary

The paper proposes **Grounding-IQA**, a new image quality assessment paradigm that integrates multimodal referring and grounding into IQA, instantiated via two subtasks: **GIQA-DES** (quality descriptions with explicit bounding boxes for key regions) and **GIQA-VQA** (quality-focused VQA with spatial references in the question or answer).  

To support this paradigm, the authors build **GIQA-160K**, a 160K-sample instruction-tuning dataset generated through an automated pipeline leveraging existing IQA datasets, LLMs, and GroundingDINO, and design **GIQA-Bench**, a 100-image benchmark with 250 expert-annotated test samples and metrics covering description quality, VQA accuracy, and grounding precision. Experiments across several MLLMs show that fine-tuning on GIQA-160K substantially improves performance on GIQA-Bench compared to general, grounding-only, and IQA-only baselines.

## Strengths

1. **Clear and timely problem formulation.**  
   The paper identifies a real gap between existing MLLM-based IQA methods (which provide global or loosely localized descriptions) and the need for fine-grained, spatially grounded quality assessment. The definition of the two subtasks, **GIQA-DES** and **GIQA-VQA** (Sec. 3.1, Fig. 2 and Fig. 5), is conceptually clean and ties well to existing grounded captioning and referring-expression tasks but with a low-level quality focus.

2. **Well-thought-out data construction pipeline.**  
   The automated annotation pipeline in Sec. 3.2, visually summarized in **Figure 3**, is relatively sophisticated:  
   - Stage-1 uses Llama3 to extract triplets \(\{\mathcal{T}_r, \mathcal{T}_q, \mathcal{T}_e\}\), where \(\mathcal{T}_e\) is used to filter non-critical objects.  
   - Stage-2 uses GroundingDINO conditioned on descriptive phrases rather than bare object names, addressing multi-instance ambiguity, as illustrated by **Figure 4**.  
   - Stage-3 introduces **IQA-Filter** using Q-Instruct and **Box-Merge** (Algorithm 1) to favor regions whose quality matches the described attribute and reduce redundant boxes.  
   - Stage-4 encodes coordinates as textual tokens with grid discretization (Eqs. (1)–(2)).  
   This is a reasonably coherent pipeline, and the paper backs some of these design choices with ablations.

3. **Non-trivial ablations and analysis about annotation design.**  
   The ablation in **Table 2a** shows that applying box refinement (IQA-Filter + Box-Merge) yields better grounding performance (mIoU 0.5851 vs 0.5624, Tag-Recall 0.5497 vs 0.5045) and improved language metrics (BLEU@4 from 20.97 to 23.67, LLM-Score from 61.00 to 61.75). Similarly, **Table 2b** indicates that the **discrete coordinate representation** slightly improves Tag-Recall and language scores compared to normalized coordinates. **Figure 6** further supports that refinement shifts the GIQA-160K box-area distribution closer to GIQA-Bench, which is exactly the sort of sanity check one would want for an automatically generated dataset.

4. **Strong empirical evidence that GIQA-160K improves multiple models.**  
   **Table 4** (“Data Compatibility”) shows that fine-tuning different base MLLMs on GIQA-160K consistently boosts both GIQA-DES (Tag-Recall and LLM-Score) and GIQA-VQA (Tag-Recall and Acc(Total)). For example, mPLUG-Owl2-7B’s LLM-Score on GIQA-DES rises from 48.25 to 63.0 and total VQA accuracy from 0.5633 to 0.7417. This suggests the dataset is not over-specialized to one architecture.

5. **Comprehensive comparison across model families.**  
   **Table 5** benchmarks four groups: general models, grounding-focused models, IQA-focused models, and “Ours” (general models fine-tuned on GIQA-160K). This table is central to the paper’s claim: the “Ours” models largely dominate on the combined axes of description quality, VQA accuracy, and grounding (mIoU, Tag-Recall). For instance, Grounding-IQA (mPLUG-Owl2-7B) attains Tag-Recall 0.5474 (GIQA-DES) and 0.7372 (GIQA-VQA) with high VQA accuracy (Acc(Y)=0.8444, Acc(W)=0.5875), while IQA-only models have high description quality but no grounding metrics, and grounding models have strong mIoU/Tag-Recall but much weaker quality description scores (e.g., Ferret-7B LLM-Score 43.75). The radar-style visualization in **Figure 1** nicely summarizes these trade-offs and visually supports the quantitative claims.

6. **Qualitative examples concretely demonstrate the claimed behavior.**  
   **Figure 7** shows side-by-side outputs on GIQA-Bench for both GIQA-DES and GIQA-VQA. In the left example, standard models give vague or partially wrong explanations, while Grounding-IQA produces a detailed description with boxes explicitly marking regions of blur or artifacts. The right example shows that the Grounding-IQA model answers spatially precise quality questions more reliably. These visualizations make it much easier to grasp what is actually gained over Q-Instruct or generic grounding models.

7. **Multi-task training design is empirically validated.**  
   **Table 3** shows that training on GIQA-DES alone primarily helps description and grounding, GIQA-VQA alone primarily helps question-answering, and joint training (GIQA-160K) yields the best of both worlds (e.g., GIQA-VQA Tag-Recall 0.7372 vs 0.4872/0.5577). This supports the claim that the two subtasks are complementary and that the dataset is sensibly constructed.

## Weaknesses

1. **Benchmark scale and diversity are limited, raising concerns about overfitting and external validity.**  
   GIQA-Bench contains **only 100 images** with 250 test samples (Sec. 3.4, **Table 1**), and all quantitative claims hinge on this very small benchmark. There is no evidence that the observed performance improvements transfer to other held-out datasets or “in-the-wild” user tasks. The authors mention further experiments (e.g., “traditional score-based IQA tasks” and user study) only in the supplementary material, but the main paper does not provide numbers. For a work whose primary contribution is a new *paradigm* and benchmark, relying on such a small and somewhat opaque evaluation set weakens the significance of the results. At minimum, some results on an independent IQA dataset (e.g., Q-Pathway held-out subset, DQ-495K subset, or Q-Bench metrics) should be summarized in the main paper to demonstrate that models trained on GIQA-160K genuinely generalize beyond the authors’ curated benchmark.

2. **Heavy reliance on LLM-based evaluation introduces circularity and potential bias.**  
   Several main metrics are LLM-derived:  
   - Description quality uses **LLM-Score** (Llama3) in addition to BLEU@4 (Sec. 3.4).  
   - VQA accuracy for open-ended questions uses another LLM scoring 0–4, normalized (Acc(W)).  
   This is particularly problematic because Llama3 is also used in *data construction* (Sec. 3.2) and *evaluation*. So the same or similar model family both generates and judges outputs, which risks overestimating performance, especially for models trained on those generated patterns. The paper does not explore how robust results are to the choice of judge (e.g., different LLMs), does not report inter-rater agreement with human annotators, and does not show any human evaluation on text quality in the main paper. This undermines the strength of claims about “more accurate” or “fine-grained” assessments, which might just align better with the judge’s style.

3. **Automatic annotation pipeline quality is not rigorously validated beyond self-consistency.**  
   While **Table 2** and **Figure 6** show that certain pipeline components improve internal metrics, the paper does not provide *human* verification of annotation quality for GIQA-160K, e.g., what fraction of generated bounding boxes and descriptions are judged correct by humans. Since the dataset is fully automatic and uses GroundingDINO + Llama3 + Q-Instruct, systematic biases or failure modes are very plausible. For instance:  
   - Stage-3 uses Q-Instruct to answer “Is the image quality \<T_q\>?”. This procedure presumes Q-Instruct is already reliable at judging local quality, yet earlier sections argue current methods are limited for fine-grained assessment. No error analysis or sanity checks are given (e.g., examples where IQA-Filter fails).  
   - Algorithm 1’s Box-Merge has a somewhat ad-hoc area threshold \(T_a=0.256\) and IoU threshold \(T_o=0.95\), but there is no justification or sensitivity study.  
   Overall, the reader is asked to take on faith that GIQA-160K is “high-quality,” but the evidence presented is limited to improved training performance on the authors’ own benchmark rather than real data quality audits.

4. **Mathematical formulation of coordinate discretization is under-specified and potentially inconsistent.**  
   In Stage-4 (Sec. 3.2), the authors define discrete indices via Eq. (1):  
   \[
   \mathrm{idx}_l = y_1 \cdot m \cdot n + x_1 \cdot n, \quad \mathrm{idx}_r = y_2 \cdot m \cdot n + x_2 \cdot n.
   \]  
   and remap them back via Eq. (2):  
   \[
   x_1' = (\mathrm{idx}_l \% n + 0.5)/n, \quad y_1' = (\mathrm{idx}_l / n + 0.5)/m,
   \]  
   with analogous expressions for \(x_2', y_2'\). This formulation is confusing and appears inconsistent:  
   - If the grid is \(n \times m\), a natural linearization index would be \(y \cdot n + x\) or \(y \cdot m + x\); multiplying by both \(m\) and \(n\) as in \(y_1 \cdot m \cdot n\) does not match the later decomposition using modulo \(n\) and division by \(n\), which assumes an index range of \([0, nm-1]\), not \([0, n^2 m - 1]\).  
   - The text states that grids are “numbering from 0 to \(nm-1\),” which contradicts the formula.  
   This is not just a cosmetic issue: incorrect coordinate encoding/decoding can systematically distort boxes and affect mIoU, yet there is no clarification or derivation. At rebuttal, the authors should either correct Eq. (1) and Eq. (2) rigorously, or clearly explain their indexing scheme and its effect on spatial precision.

5. **Positioning with respect to closely related work is incomplete, especially around mix-grained IQA.**  
   The paper cites Q-Ground (Chen et al., 2024b) as an IQA grounding work, but does not discuss in any depth how Grounding-IQA compares conceptually or empirically (e.g., Q-Ground already performs degradation-region grounding; how is GIQA-DES/GIQA-VQA meaningfully beyond that?). Moreover, important recent work such as **Dog-IQA (2024)**, which explicitly tackles mix-grained IQA with MLLMs and standards guidance, is not mentioned at all (see “Potentially Missing Related Work”). This makes it harder to assess the true originality of the paradigm: is “grounding + IQA” genuinely novel, or largely an extension/combination of Q-Ground, Dog-IQA, and grounded captioning models? The paper could do a much better job articulating conceptual differences and trade-offs beyond “we also support referring” and “we generalize existing IQA datasets.”

6. **Limited exploration of failure modes and qualitative error analysis.**  
   The few qualitative examples in **Figure 7** all showcase successes of the proposed method, but there is no systematic investigation of where Grounding-IQA fails: e.g., cluttered scenes with many small objects, severe distortions where object detectors fail, or cases where multiple quality factors interact (noise + blur + compression). Since the entire pipeline is layered (LLM + object detector + IQA model), compounding errors are expected. Without error analysis, it is difficult to judge robustness and to understand when users should trust or distrust the model’s localized quality claims.

7. **Some experimental design choices and baselines remain under-specified.**  
   While **Table 5** is extensive, several aspects are not clearly documented in the main paper:  
   - For grounding baselines like Ferret or GroundingGPT, how exactly are they prompted for quality descriptions or IQA-VQA? If they are not fine-tuned for quality-related language, their underperformance may partly reflect prompting rather than inherent limitations.  
   - General and IQA models have N/A entries for mIoU/Tag-Recall, yet in principle grounded outputs might be coaxed from them with bounding-box tokens or coordinate expressions. This asymmetry could overstate the advantage of the proposed approach.  
   - Training is only 2 epochs with batch size 64 (Sec. 4.1), but there is no indication of dataset split (train/val) or early stopping criteria; all hyperparameters for fine-tuned baselines are glossed over.  
   Reproducibility is therefore weaker than it could be, and readers cannot easily tell whether some baselines are unfairly disadvantaged.

8. **GIQA-Bench annotation and metric definitions lack detail in the main text.**  
   Sec. 3.4 states that each sample is annotated “in multiple rounds by at least three experts,” but provides no specifics: annotation protocol, disagreement resolution, inter-rater agreement, or how “key objects” are selected. Similarly, **Tag-Recall** is only defined briefly as requiring IoU and object name similarity > 0.5. There is no explicit formula for object-name similarity (e.g., embedding cosine similarity, exact match, edit distance), which complicates interpretation of Tag-Recall values in **Tables 2, 3, 4, and 5**. This under-specification of evaluation metrics and annotation procedures is a non-trivial weakness for a benchmark paper.

9. **GIQA-160K granularity and coverage are not deeply characterized.**  
   **Figure 5** gives two nice example instances, but there is little statistical breakdown beyond **Table 1**. There is no analysis of how many distinct distortion types, object categories, or box sizes per image are included, nor of the distribution of quality labels (e.g., positive vs negative vs neutral). For a dataset whose core value is “fine-grained quality perception,” more insight into what fine-grained phenomena are actually covered would be useful. This also makes it hard to compare to other IQA datasets (ManIQA, Dog-IQA, etc.) in terms of richness.

## Potentially Missing Related Work

1. **K. Liu, Z. Zhang, W. Li, “Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessment,” 2024.**  
   - **Why directly related:** Dog-IQA explicitly targets mix-grained IQA using MLLMs, aiming for fine-grained quality assessment while addressing OOD generalization and training cost. This substantially overlaps with the problem space of Grounding-IQA, which also claims to enhance fine-grained quality perception via MLLMs.  
   - **How/where to add:** It should be discussed in **Sec. 2.1 (Image Quality Assessment)** alongside Q-Instruct and DepictQA, with a clear articulation of differences: e.g., Dog-IQA focuses on zero-shot mix-grained scores/feedback, while Grounding-IQA centers on explicit spatial grounding and instruction-tuning. If possible, a conceptual or empirical comparison (e.g., whether Dog-IQA could be adapted to GIQA-VQA-style tasks) would significantly strengthen positioning.

2. **More explicit comparison to Q-Ground (Chen et al., 2024b).**  
   While Q-Ground is cited, the discussion is very brief and does not fully clarify how the proposed GIQA-DES/GIQA-VQA paradigm generalizes beyond “degradation region grounding” offered by Q-Ground. A more detailed comparison in **Sec. 2.2 or Sec. 3.1**, and ideally an empirical row for Q-Ground in **Table 5** (or at least a qualitative comparison), would be appropriate given how close the topics are.

If additional works on grounded or mix-grained IQA exist (e.g., standards-guided or region-aware IQA), they should similarly be integrated into Sec. 2.1–2.2 to more sharply position the contributions.

## Questions

1. **Coordinate discretization correctness.**  
   Could the authors clarify and correct the coordinate discretization scheme in Eqs. (1)–(2)? Explicitly:  
   - What are the exact discrete ranges of \(x_1, y_1, x_2, y_2\) before linearization?  
   - How do you derive \(\mathrm{idx}_l\) and \(\mathrm{idx}_r\) from these, and how does the modulo/division in Eq. (2) invert this mapping?  
   A precise derivation and perhaps a 1D/2D toy example would help confirm that the discretization is mathematically sound.

2. **Human validation of GIQA-160K annotations.**  
   Have you conducted any human evaluation of the automatically generated GIQA-160K annotations (both bounding boxes and descriptions)? For example, what percentage of boxes are judged by human annotators as correctly locating the described region, and what fraction of object-quality associations (\(\mathcal{T}_q\)) are accurate? Presenting even a small-scale audit would significantly strengthen the dataset’s credibility.

3. **Robustness to different LLM judges.**  
   Since Llama3 is used both in data creation and evaluation, how sensitive are your conclusions to the choice of evaluation LLM? Have you tried using a different judge (e.g., GPT-4o or another open-source model) for LLM-Score and Acc(W)? If not, could you comment on potential biases and how you might mitigate them?

4. **Comparison to Q-Ground and Dog-IQA.**  
   Could you explicitly compare Grounding-IQA to Q-Ground and Dog-IQA, both conceptually and empirically? For example, can Q-Ground’s model be adapted to your GIQA-Bench to provide mIoU/Tag-Recall on quality-related regions, and can Dog-IQA outputs be used to answer GIQA-VQA-style questions? Clarifying this would help establish how much additional capability GIQA-160K and GIQA-Bench truly provide.

5. **Details on GIQA-Bench annotation protocol and Tag-Recall.**  
   Please elaborate on:  
   - How experts selected “key objects” and bounding boxes when multiple regions contribute to quality.  
   - How inter-annotator agreement was measured and resolved.  
   - The exact definition of “object name similarity > 0.5” in Tag-Recall (token-level, embedding-based, etc.).  
   These details are important to interpret the mIoU and Tag-Recall numbers in **Table 5**.

6. **Baseline prompting and training details.**  
   For grounding baselines in **Table 5** (Ferret, GroundingGPT, etc.), what prompts and settings were used to make them answer quality-related questions and output boxes? Were they fine-tuned on any quality-related data, or are they used off-the-shelf? More transparency here is needed to assess whether the comparison is fair.

7. **Generalization beyond GIQA-Bench.**  
   Could you provide, at least in the rebuttal, some quantitative results of Grounding-IQA models on an *independent* IQA benchmark (e.g., Q-Bench metrics, traditional IQA datasets) to support the claim that your models are useful beyond GIQA-Bench?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methods are technically reasonable overall, and experiments are consistent with claims, but evaluation is limited to a small benchmark and some mathematical and procedural details (e.g., coordinate indexing, annotation quality, LLM-based judging) are under-specified.

## Presentation Rating

3: good.  
The paper is generally clear and well-structured, with helpful figures (especially Figures 2, 3, 4, 6, and 7) and tables (2–5). However, some crucial details about metric definitions, coordinate formulas, and annotation protocols are missing or ambiguous.

## Contribution Rating

3: good.  
The work offers a useful dataset and benchmark to push MLLM-based IQA toward spatial grounding, with a plausible new task formulation and empirical evidence across multiple base models. Novelty is moderate, as it sits between existing grounded captioning and IQA-LLM works, and stronger positioning vs. very recent related efforts is needed.

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper presents a timely and practically valuable combination of grounding and IQA, with a non-trivial dataset construction pipeline and consistent empirical gains across several MLLMs. At the same time, the small size and somewhat opaque evaluation protocol of GIQA-Bench, the lack of rigorous human validation of GIQA-160K, and under-specified mathematical and methodological details prevent a higher recommendation. With clarifications and additional validation (especially on annotation quality and generalization), this work could become a solid reference for grounded IQA.

## Reviewer Confidence

4: confident.  
I am familiar with MLLMs, grounded vision-language tasks, and IQA, and have carefully checked the technical and experimental sections, though some practical annotation details remain opaque without the supplementary material.