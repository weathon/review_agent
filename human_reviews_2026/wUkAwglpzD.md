# Fine-grained Contrastive Learning for ECG-Report Alignment with Waveform Enhancement

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Electrocardiograms (ECGs) are essential for diagnosing cardiovascular diseases. However, existing ECG-Report contrastive learning methods focus on whole-ECG and report alignment, missing the link between local ECG features and individual report tags.
In this paper, we propose FG-CLEP (Fine-Grained Contrastive Language ECG Pre-training), which achieves fine-grained alignment between specific ECG segments and each tag in the report via tag-specific ECG representations. Furthermore, we found that nearly 55\% of ECG reports in the MIMIC-ECG training dataset lack detailed waveform features, which hinders fine-grained alignment. To address this, we introduce a coarse-to-fine training process that leverages large language models (LLMs) to recover these missing waveform features and validate the LLM outputs using a coarse model. Additionally, fine-grained alignment at the tag level, rather than at the report level, exacerbates the false negative problem, as different reports may share common tags. To mitigate this, we introduce a semantic similarity matrix to guide the model in identifying and correcting false negatives. 
Experiments on six datasets demonstrate that FG-CLEP significantly improves fine-grained alignment, outperforming state-of-the-art methods in both zero-shot prediction and linear probing. Meanwhile, the fine-grained reports we generate also enhance the performance of other methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents FG-CLEP, a fine-grained ECG–report alignment model that links ECG patches with report tags.

### Strengths
- Introduces a cross-attention patch-tag alignment and LLM-based fine-grained training pipeline for ECG-text pretraining.

### Weaknesses
- This work has limited technical novelty. The work depends on the LLM usage instead of core modeling. Also, the LLM-generated GF reports are clinically unverifiable and risk introducing noise or bias and especially even label leakage when doing zero-shot experiments. 

- Lack comparisions with recent ECG-text modeling works.

### Questions
- How to ensure ECG patches with specific patch length align to any of clinical report tags?  

- Please compare the work with recent advances on ECG-Text pretraining works [1,2]. In many evaluation setting, the performance are poorer than them, whether the work uses “FG-” or not.  

- Furthermore, [2] also point out the close problem with false negative samples (report level). Please provide additional experiments to compare N3S in [2] and the FNM in this work. 

- Please add linear probing experiment with METS model and compare with the work. 

- In ablation study, w/o Fine-Grained Alignment performance is very close (<1%) with the proposed model (default) while the w/o Fine-Grained Report performance is clearer. Does this show that the work highly rely on additional LLM usage and used trick like ensemble? 

- Furthermore, looking at linear probing results (table 2), where only ECG encoder is used, the gap between with “FG-“ and without “FG-” is minor. Does this mean the “FG-” just slightly boost the ECG encoder (which I suppose, is the core component in real-world deployment). 

- Figure 4 is not mentioned in any text.  

- It seems to be unable to access to the code provided.  

[1] Wang, Fuying, Jiacheng Xu, and Lequan Yu. "From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining." Forty-second International Conference on Machine Learning.  

[2] Hung, Manh Pham, Aaqib Saeed, and Dong Ma. "Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners." Forty-second International Conference on Machine Learning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents FG-CLEP, an innovative framework that advances ECG–text alignment by introducing fine-grained contrastive learning between ECG patches and report tags.

### Strengths
The paper presents FG-CLEP, an innovative framework that advances ECG–text alignment by introducing fine-grained contrastive learning between ECG patches and report tags. Its coarse-to-fine training pipeline, which integrates large language models (LLMs) for recovering missing waveform features, demonstrates strong methodological creativity and practical relevance. Extensive experiments across six datasets show consistent performance gains in both zero-shot and linear probing tasks, validating the model’s robustness and generalizability. Visualizations (activation maps and retrieval results) provide qualitative support for fine-grained alignment.

### Weaknesses
The dependence on LLMs for generating waveform features may introduce bias or inconsistency, even with CLEP-based validation.  
The evaluation lacks human expert assessment of fine-grained alignment quality beyond AUC metrics, limiting interpretability claims. 
The training efficiency and computational cost of multi-stage fine-tuning and LLM querying are not clearly quantified, raising concerns about scalability in clinical deployment. 
The model’s reliance on tag-level alignment assumes structured report formats, which may not generalize across healthcare systems or languages. 
While the paper claims improvements in zero-shot settings, statistical significance testing is not reported. 
There are several writing issues, i.e., the legend of Fig.2 is incomplete (line 230); ref information is incomplete (line 286).

### Questions
How does FG-CLEP perform on free-text clinical reports rather than tag-based structured ones?

What are the computational and time costs of the coarse-to-fine training pipeline when scaling to larger datasets?

Have you evaluated the clinical interpretability of tag-specific ECG activations with cardiologists?

How sensitive is FG-CLEP to LLM hallucinations or errors during waveform feature generation, especially when the validation threshold varies?

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
4

### Summary
The paper introduces a coarse-to-fine training process that leverages large language models (LLMs) to recover these missing waveform features and validate the LLM outputs using a coarse model. Besides, authors introduce a semantic similarity matrix to guide the model in identifying and correcting false negatives. Experiments demonstrate the superior performance of the proposed approach.

### Strengths
1. The problem of ECG–language pretraining is important and has clear clinical relevance.

2. The proposed framework is well-motivated and technically sound; the LLM-enriched waveform features are novel.

3. Experiments across multiple tasks show competitive or superior performance.

### Weaknesses
1. In the “Fine-Grained Contrastive Learning Objective,” it appears that a single tag is randomly sampled per ECG. If so, this could ignore information from other relevant segments and reduce feature richness. Please clarify the sampling procedure and consider reporting results with multi-tag aggregation or coverage-controlled sampling.

2. While the tag sampling strategy plausibly aligns better with zero-shot text prompts and could help zero-shot performance, the improvements on linear probing are small. This suggests the proposed fine-grained objective may not meaningfully strengthen the learned ECG representation. A targeted ablation contrasting zero-shot and linear-probe gains would help.

3. The LLM-enriched waveform reports are an interesting idea. However, Table 3 indicates that mitigating false negatives has a larger impact than the fine-grained training itself. Please isolate these effects: show results applying the false-negative technique to the original reports (without LLM enrichment) to quantify each component’s contribution.

4. Figure 5 is difficult to interpret and adds limited insight. An ECG retrieval analysis (e.g., text→ECG or ECG→text or ECG→ECG retrieval with qualitative examples and recall@k) would be more informative for assessing alignment quality.

### Questions
1. How are ECG patches defined (temporal windows, lead-wise splits, or both)? What does $N_{lead}$ denote precisely? If patches are fixed windows, how do you ensure complete morphology is captured (e.g., P–QRS–T cycles) rather than fragmented?

2. How are labels for “false negative” tags identified or constructed in training? Please detail the detection heuristic, thresholds, and any human verification.

3. Are the results in Table 3 from linear probing or zero-shot classification, and on which dataset(s)? Briefly state the evaluation protocol to make the comparison interpretable.

4. Please highlight the best results (e.g., boldface) to improve readability.

### Soundness
3

### Presentation
3

### Contribution
2
