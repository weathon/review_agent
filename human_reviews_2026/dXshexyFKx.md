# How Do Medical MLLMs Fail?  A Study on Visual Grounding in Medical Images

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Generalist multimodal large language models (MLLMs) have achieved impressive performance across a wide range of vision-language tasks. However, their performance on medical tasks—particularly in zero-shot settings where generalization is critical—remains suboptimal. A key research gap is the limited understanding of why medical MLLMs underperform in medical image interpretation.
**In this work**, we present a pioneering systematic investigation into the visual grounding capabilities of state-of-the-art medical MLLMs. To disentangle *visual grounding* from *semantic grounding*, we design VGMED, a novel evaluation dataset developed with expert clinical guidance, explicitly assessing the visual grounding capability of medical MLLMs. 
We introduce new quantitative metrics and conduct detailed qualitative analyses. Our study across **eight** state-of-the-art (SOTA) medical MLLMs validates that they often fail to ground their predictions in clinically relevant image regions. We note that this finding is specific to medical image analysis; in contrast, prior work has shown that MLLMs are capable of grounding their predictions in the correct image regions when applied to natural scene images.
Motivated by these findings, we propose VGRefine, a simple yet effective inference-time method that refines attention distribution to improve visual grounding in medical settings. Our approach achieves SOTA performance across  6 diverse Med-VQA benchmarks (over 110K VQA samples from 8 imaging modalities) 
without requiring additional training or external expert models.  Overall, our work, for the first time, systematically validates inadequate visual grounding as one of the key contributing factors for medical MLLMs' under-performance.
Additional experiments are included in the Supp. Project Page: https://guimeng-leo-liu.github.io/Medical-MLLMs-Fail/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the failure modes of medical MLLMs, focusing on their visual grounding ability in Med-VQA. The authors introduce a diagnostic framework that evaluates attention alignment between model-generated attention maps and annotated regions of interest using three metrics—Attention Ratio (AR), Kullback–Leibler (KL) divergence, and Jensen–Shannon (JS) divergence—on the proposed VGMED benchmark. They further present VGRefine, a two-step training-free method that refines attention through (1) filtering low-confidence visual regions to create a high-confidence binary mask and (2) applying this mask to suppress irrelevant visual attention. Experiments on multiple Med-VQA benchmarks show improved grounding alignment and accuracy, with qualitative visualizations and expert evaluations supporting the method’s interpretability and effectiveness.

### Strengths
* The paper identifies a clear and meaningful problem—the weak visual grounding ability of medical MLLMs—and addresses it through a structured, quantitative framework. It turns a qualitative interpretability challenge into measurable alignment metrics and integrates them within a unified evaluation setting.

* The proposed inference-time method, VGRefine, is concise, computationally efficient, and easy to deploy. Despite its simplicity, it consistently enhances grounding quality and answer accuracy across diverse Med-VQA benchmarks, demonstrating both technical soundness and practical relevance.

* The experimental section is extensive, including evaluations on multiple datasets, ablation studies, and expert assessments. The combination of quantitative metrics and human evaluation adds credibility and depth to the analysis.

### Weaknesses
* The paper attributes model failure primarily to weak visual grounding but does not clearly disentangle it from semantic grounding or reasoning errors. It remains unclear which factor—semantic or visual—dominates the poor generalization of current medical MLLMs. Providing representative failure cases could clarify whether errors stem from misunderstanding the question or mislocalizing visual evidence.

* The choice to compute attention maps only from the last input text token lacks justification. Averaging or weighting attention across all input tokens might yield a more stable and comprehensive alignment estimation; offering an explanation or comparison for this design choice would strengthen the analysis.

* Attention Knockout is applied at a single layer, yet the broader influence on higher layers or cross-layer propagation of redundant visual information is not analyzed, which may limit the consistency of the refinement effect.

* Comparisons with general-domain vision-language models are limited. Including stronger or more recent baselines, such as Qwen2.5-VL or InternVL3, could better support the argument that grounding difficulty is domain-inherent rather than model-specific.

* While the study provides a thorough diagnostic view, it remains primarily descriptive. Further exploration into why grounding failures arise—such as biases in medical datasets or insufficient visual diversity—would deepen the explanatory value of the work.

### Questions
* Provide an explanation for using only the last input text token to compute attention maps. It would be helpful to test or discuss whether averaging or weighting attention across all input tokens offers a more reliable alignment estimation.

* Explain the rationale for applying Attention Knockout at a single layer (e.g., layer 16) and whether extending it to multiple or higher layers could further improve grounding consistency.

* Consider including newer or stronger vision-language models (such as Qwen2.5-VL or InternVL) in the comparison to confirm whether grounding difficulty is indeed domain-specific.

* Discuss possible training-related causes of grounding failure, such as dataset bias or limited visual diversity, and whether VGRefine could mitigate these underlying issues.

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
This work presents VGMED, a clinician-curated benchmark for evaluating visual grounding in medical MLLMs, and proposes VGRefine, an inference-time attention refinement strategy that significantly improves visual grounding and zero-shot Med-VQA performance. The authors systematically analyze failure modes in existing medical MLLMs and demonstrate that inadequate visual grounding is a major bottleneck. The study is well-motivated, carefully designed, and supported by strong empirical results across multiple tasks and modalities.

### Strengths
1. The paper isolates visual grounding as a distinct bottleneck in medical multimodal models, a perspective that is under-explored yet highly relevant for trustworthy deployment.
2. Layer- and head-specific analysis of cross-attention provides an interpretable pathway to understanding failure behavior, in line with mechanistic interpretability trends in multimodal research.
3. The inference-time attention refinement is conceptually clean, model-agnostic, and introduces no additional data or parameters. The performance gains across diverse modalities and benchmarks are substantial.

### Weaknesses
1. The reliance on attention as the primary grounding signal may be discussed more rigorously; a short reflection on alternative grounding indicators (e.g., gradient-based saliency, causal perturbation) would improve completeness.

2. The benchmark curation process is strong, but clearer quantitative annotation statistics (e.g. agreement checks, diversity distribution) would further support dataset rigor.

3. Method hyperparameters (selected heads/layers) are fixed; small discussion of adaptive or data-driven selection would enhance generality claims.

### Questions
See Weaknesses.

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
The paper investigates why medical multimodal large language models (MLLMs) underperform in medical image interpretation tasks, particularly in zero-shot settings where generalization is critical.
Through an empirical analysis of eight state-of-the-art medical MLLMs, the authors demonstrate that these models often fail to ground their predictions in clinically relevant image regions—a deficiency that appears specific to medical imaging. In contrast, prior studies show that general-domain MLLMs exhibit appropriate visual grounding when applied to natural image understanding tasks.
Motivated by this finding, the paper introduces VGRefine, a simple yet effective inference-time method that optimizes the model’s attention distribution to improve visual grounding and reasoning in medical contexts. The proposed method achieves state-of-the-art results on six Med-VQA benchmarks covering eight imaging modalities and more than 110,000 samples.

### Strengths
1. The study is interesting. By examining whether MLLMs can reuse their reasoning attention for localization, the authors observe that models’ attention during question answering is not fully concentrated on relevant image regions, and finally propose VGRefine to improve model performance.

2. The paper is the first to systematically verify that insufficient visual grounding is one of the key factors leading to poor performance of medical MLLMs. The proposed method achieves state-of-the-art results on six different Med-VQA benchmarks covering eight imaging modalities and more than 110,000 VQA samples.

### Weaknesses
1. The paper could include more mathematical analysis to explain the causes behind the observed phenomenon.

2. The paper claims that this phenomenon has not been observed in other domains. Do “other domains” refer only to general visual question answering on natural scenes? Or does the paper merely show results on some general scenes? If so, can this truly prove that the same phenomenon does not appear in any other specialized domains?

3. The models used in the experiments—such as Qwen-VL, LLaVA-Med, and LLaVA-1.5—are relatively outdated. On the latest visual models, whether domain-specific or general-purpose, can the observed phenomenon and the effectiveness of the proposed method still hold? Could the authors also provide experiments showing how medical models perform in general VQA scenarios?

### Questions
1. What do the authors believe are the reasons why this phenomenon occurs in medical scenarios?

2. In Figure 1 of the main text, the paper compares several domain-specific medical models on medical VQA tasks with LLaVA-1.5 on general VQA tasks. What if this comparison were reversed? Is the problem caused by these medical models themselves? If the proposed method were directly applied to stronger and more recent general MLLMs, such as Qwen2.5-VL or Qwen3-VL, would the same conclusions be obtained?

If the authors can address the above questions, the reviewer promises to raise the score.

### Soundness
3

### Presentation
3

### Contribution
3
