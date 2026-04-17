# Closing the Modality Gap for Mixed Modality Search

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Mixed modality search, retrieving information across a heterogeneous corpus composed of images, texts, and multimodal documents, is an important yet underexplored real-world application. In this work, we investigate how contrastive vision-language models, such as CLIP, perform on the mixed modality search task. Our analysis reveals a critical limitation: these models exhibit a pronounced modality gap in the embedding space, where image and text embeddings form distinct clusters, leading to intra-modal ranking bias and inter-modal fusion failure. To address this issue, we propose GR-CLIP, a lightweight post-hoc calibration method that removes the modality gap in CLIP’s embedding space. Evaluated on MixBench, the first benchmark specifically designed for mixed modality search, GR-CLIP improves NDCG@10 by up to 26% over CLIP, surpasses recent vision-language generative embedding models by 4%, while using 75x less compute.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the problem of mixed modality search, where queries must retrieve relevant content from a corpus containing text-only, image-only, and multimodal documents. It identifies a key limitation in contrastive vision-language models like CLIP: a modality gap in the embedding space that causes ranking bias and poor fusion of modalities.
To address this, this paper proposes GR-CLIP, a lightweight post-hoc calibration method that removes the modality gap by subtracting modality-specific mean embeddings. 
This approach requires no retraining and introduces negligible computational overhead.
Experiments on the newly introduced MixBench benchmark show that GR-CLIP improves retrieval performance by up to 26% over CLIP and outperforms generative embedding models like VLM2Vec while using significantly less compute. 
The proposed method generalizes across CLIP variants, datasets, and modalities, demonstrating its robustness and practical utility.

### Strengths
## Originality 
This paper introduces a new problem formulation—mixed modality search—which reflects realistic retrieval scenarios involving heterogeneous corpora. 
Contrastive vision-language models are well-studied, but this work uniquely highlights the modality gap in mixed modality search settings.
GR-CLIP is a simple yet creative post-hoc calibration method that builds on prior theoretical insights but applies them in a new and practical way.

## Quality 
The proposed methodology is sound and well-motivated. The current experiments are designed to isolate and evaluate the effects of the modality gap across different settings. The use of MixBench, a custom benchmark tailored to the task, strengthens the empirical foundation. The results are consistent and show clear improvements over strong baselines, including generative models.

## Clarity
This paper is clearly written and well-structured for readers. The problem is defined precisely, and the figures effectively illustrate key concepts such as the modality gap and its impact on retrieval. The explanation of GR-CLIP is concise and accessible, and the appendix Foprovides thorough implementation details.

## Significance
The current work addresses a practical and underexplored challenge in multimodal retrieval. 
By demonstrating that a lightweight calibration can substantially improve performance, this paper demonstrates practical utility through improved retrieval metrics on MixBench.
Its findings could influence future designs of retrieval systems and multimodal embedding models.

### Weaknesses
## Insufficient definitions
While the formulation of mixed modality search is novel and practically relevant, the current work does not sufficiently compare its definition or task setup with prior work in multimodal retrieval or modality gap analysis. 
For example, methods like I0T (Embedding Standardization) and CMD (Central Moment Discrepancy) offer model-agnostic approaches to modality gap reduction, and their absence from the comparison limits the assessment of GR-CLIP’s novelty and generality.

## Experiments setup
These experiments focus exclusively on CLIP-style models. Although authors claim generalizability across CLIP variants, they do not evaluate GR-CLIP on non-contrastive or generative models like BLIP, SmolVLM, or LLM2CLIP. This restricts the scope of the findings and leaves open the question of whether the modality gap and the proposed correction method apply more broadly.

## Task settings
The datasets used in MixBench are well-chosen, but the document structure is simplified to single image-text pairs. Real-world multimodal documents often contain multiple images, longer texts, or interleaved modalities. Evaluating GR-CLIP in such settings would strengthen the claim that it addresses realistic retrieval challenges.

## Practical significance
While this paper shows strong improvements in retrieval metrics, it does not explore downstream effects such as user satisfaction, relevance diversity, or robustness to noisy queries. These aspects are important for assessing the practical significance of the method in real-world search systems.

## Modality gap
The theoretical framing of the modality gap as a constant vector is compelling, but the current work could benefit from deeper analysis of when and why this approximation holds, especially across different domains and modalities.
Addressing these points would enhance the paper’s impact and help validate GR-CLIP as a broadly applicable solution to modality gap issues in multimodal retrieval.

### Questions
Q1. Could you clarify whether the modality gap correction method (mean subtraction) generalizes to non-CLIP models, such as BLIP, SmolVLM, or LLM2CLIP? If not, what structural assumptions of CLIP make GR-CLIP effective, and how might those differ in other architectures?

Q2. The current paper compares GR-CLIP primarily against CLIP variants and VLM2Vec. Could you explain why other modality gap reduction methods like I0T or CMD were not included in the comparison? Would you consider adding these to strengthen the evaluation?

Q3. MixBench is a valuable benchmark, but the document structure is limited to single image-text pairs. Do you plan to extend the benchmark to include more complex multimodal documents (e.g., multi-image, multi-paragraph, interleaved formats)? How might GR-CLIP behave in such settings?

Q4. The modality gap is modeled as a constant vector. Could you provide more theoretical or empirical justification for this assumption? For example, does this approximation hold across different domains, modalities, or embedding distributions?

Q5. This paper shows strong improvements in retrieval metrics, but does not evaluate downstream effects such as user satisfaction, diversity of results, or robustness to noisy queries. Would you consider adding such evaluations to better assess practical significance?

Q6. GR-CLIP is described as lightweight and post-hoc. Could you elaborate on its integration cost in real-world systems? For example, how does it affect latency, memory usage, or compatibility with existing retrieval pipelines?

Q7. In the ablation studies, GR-CLIP is shown to work with fine-tuned CLIP models. Could you clarify whether the modality gap persists or changes after fine-tuning, and whether GR-CLIP needs to be re-applied or re-estimated in such cases?

Q8. The paper mentions generalization to audio and video modalities. Could you provide more details or examples of how GR-CLIP performs in these settings, and whether modality-specific characteristics (e.g., temporal structure) affect its effectiveness?

### Soundness
3

### Presentation
3

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
he paper presents GR-CLIP, a post-hoc calibration method designed to close the modality gap in CLIP-based models for mixed modality search. The authors identify a critical limitation in existing contrastive vision-language models—CLIP, where the embeddings for images and texts form distinct clusters, causing inter-modal fusion failure and ranking bias. To overcome this issue, they propose a lightweight approach that involves subtracting modality-specific means from embeddings, thereby aligning image and text embeddings more effectively.

### Strengths
1. The paper tackles a well-defined and important problem—mixed modality search. The identification of the modality gap and the proposed post-hoc calibration method are interesting contributions to the field of multimodal retrieval.
2. The authors perform extensive evaluations on several datasets and compare GR-CLIP with both CLIP and state-of-the-art generative methods, such as VLM2Vec, which adds credibility to their claims.

### Weaknesses
* The authors address the issue of image and text alignment in CLIP and attempt to overcome the gap between embeddings of different modalities. However, this is a classic problem, and the authors fail to compare their method with some important approaches. For instance, UniVL-DR [1] extends images with captions and jointly encodes both the image and caption to address the embedding gap. The authors should compare their method with these approaches.
* While CLIP's extracted embeddings indeed exhibit modality bias, a well-known solution to this problem is to separately handle image retrieval and text retrieval, and then re-rank the recalled images and texts together. This framework has been shown to work effectively in multimodal retrieval, but the authors do not compare or discuss this method.
* The authors only use text queries in their experiments. However, in real-world retrieval scenarios, text queries, image queries, and text-image mixed queries are very common. The authors should include experiments that involve these additional query types.
* The authors only report NDCG@10 as the evaluation metric. It is unclear why other commonly used retrieval evaluation metrics, such as MRR@1 or Recall@1, are not included. The authors should justify this choice and consider adding other relevant metrics.

[1] Liu Z, Xiong C, Lv Y, et al. Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval[C]//The Eleventh International Conference on Learning Representations.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies mixed modality search, where queries and corpus items can be composed of different modalities. It addresses the known problem of the modality gap, which causes systematic retrieval bias, and proposes GR-CLIP, a simple post-hoc calibration method that subtracts modality-specific means before computing similarity. Extensive experiments on the newly introduced MixBench benchmark demonstrate consistent and substantial improvements across models and modalities.

### Strengths
1 - The paper addresses the known modality gap issue, clearly positions the contribution within mixed-modality search and gives a clear course of action to solve it.  

2 - The proposed method, GR-CLIP, is both simple and effective, as it can be easily applied to various models and doesn’t require extensive resources.

3 - Introduction of a new benchmark for mixed-modality search, outlining the limitations of current models and the effectiveness of the proposed method.

4 - Comprehensive empirical study with respect to datasets, models and with a wide range of ablation analyses.

### Weaknesses
1 - The method presented in the paper lacks novelty, as the modality gap has already been studied in previous work.

2 - Lacking baselines to compare against, it is challenging to assess the relevance of GR-CLIP relative to other calibration methods.

3 - The theoretical explanations are limited; the paper would benefit from an analysis of where the gap originates and on which dataset it might be absent.

### Questions
1 - Are there cases where mean-centring hurts, since images and text can contain fundamentally different pieces of information?
2 - How much does the calibration depend on the number of samples used and their variety in the dataset?
3 - How does this method compare to other calibration methods, like whitening, projections or alignment?
4 - Can you detail the comparison of the compute used by the method vs. VLM2Vec? Be sure to detail the impact of calibration on compute use.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses mixed modality search—retrieving from heterogeneous corpora containing images, texts, and multimodal documents—and proposes GR-CLIP, a post-hoc mean-shift calibration method to close the modality gap in CLIP embeddings. The work introduces MixBench, the first benchmark for this task, and demonstrates substantial empirical improvements (up to 26 percentage points NDCG@10) over vanilla CLIP while achieving competitive performance against VLM2Vec with significantly lower computational cost.

### Strengths
The paper makes valuable contributions through 

1. formulating an important underexplored problem (characterization of ranking bias and fusion failure in retrieval context)

2. creating a well-designed benchmark (introducing MixBench)

3. demonstrating a simple, effective, and reproducible method (application to mixed modality retrieval

4. conducting systematic evaluation across models, datasets, and modalities (systematic evaluation demonstrating NDCG@10 improvements)

### Weaknesses
1. The core method is not novel. It directly applies mean-shift calibration proposed by “Diagnosing and Rectifying Vision Models” (Zhang et al., ICLR 2023) and formalized in “Connect, Collapse, Corrupt: Learning Cross-Modal Tasks with Uni-Modal Data” (Zhang et al., ICLR 2024). The prior work explicitly states: "We propose a simple technique to close the modality gap... During training, instead of feeding x to the model h, we feed it with x − E_x[x]." and the paper inadequately acknowledges this. The paper frames GR-CLIP as "we introduce" and "we propose" (Section 2.3) without clearly acknowledging that Zhang et al. (ICLR 2023) already proposed similar method. While the prior work is cited, the presentation may misrepresent the degree of novelty. I believe it would be better to to state: "We apply the mean-shift calibration method proposed by Zhang et al. (2023) to mixed modality retrieval" and position the paper as an application/benchmark paper rather than a methods paper.

2. Only NDCG@10 reported in main paper. Why not use other metrics too?

3. The equal distribution of image-only, text-only, and multimodal documents does not reflect real-world distributions, where text documents vastly outnumber images. Real search engines might have 70:20:10 or more skewed ratios. This artificial balance may inflate performance differences that would diminish in realistic scenarios. No justification is provided for this specific choice, and no evaluation at alternative ratios is conducted.


4. What will be other alternative calibration approaches? How about recent gap-closing methods like AlignCLIP (ICLR 2025)? Without such comparison, how can we know if mean-shift is the optimal approach for closing the modality gap?

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
