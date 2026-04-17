# Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 0

## Abstract
Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in aligning visual inputs with natural language outputs. Yet, the extent to which generated tokens depend on visual modalities remains poorly understood, limiting interpretability and reliability. In this work, we present EAGLE, a lightweight black-box framework for explaining autoregressive token generation in MLLMs. EAGLE attributes any selected tokens to compact perceptual regions while quantifying the relative influence of language priors and perceptual evidence. The framework introduces an objective function that unifies sufficiency (insight score) and indispensability (necessity score), optimized via greedy search over sparsified image regions for faithful and efficient attribution. Beyond spatial attribution, EAGLE performs modality-aware analysis that disentangles what tokens rely on, providing fine-grained interpretability of model decisions. Extensive experiments across open-source MLLMs show that EAGLE consistently outperforms existing methods in faithfulness, localization, and hallucination diagnosis, while requiring substantially less GPU memory. These results highlight its effectiveness and practicality for advancing the interpretability of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents EAGLE, a black-box attribution framework designed to interpret the token generation process of multimodal large language models (MLLMs). The method decomposes an image into superpixel subregions and employs a greedy search over ordered subsets to identify regions that are both sufficient and necessary for generating target tokens. The unified objective function combines an insight score (for sufficiency) and a necessity score (for indispensability). Additionally, the framework introduces a modality analysis that quantifies whether each token is primarily driven by visual evidence or language priors. Extensive experiments across multiple MLLMs demonstrate consistent improvements over gradient- and activation-based baselines on faithfulness, localization, and hallucination benchmarks.

### Strengths
The method is intuitive and straightforward, relying on interpretable probability changes on subregions rather than gradients or activations. Experiments are comprehensive and well-designed, covering multiple models and benchmarks. Results show consistent improvements.

### Weaknesses
However, I am not familiar with the XAI literature, so I am not fully confident about the method’s novelty relative to prior work. The proposed approach also depends on the quality of the image partitioning algorithm, which could affect stability and interpretability. Moreover, the greedy search requires a full forward pass through the MLLM at each iteration, potentially leading to very high computational cost compared to activation- or gradient-based baselines.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the EAGLE framework for explaining autoregressive token generation in multimodal large language models (MLLMs). EAGLE attributes each selected token to its relevant perceptual regions and quantifies the relative influences of language priors and perceptual evidence. The framework unifies two metrics—insight score and necessity score—optimized via greedy search, achieving faithful and efficient attribution. Experimental results show that EAGLE outperforms existing methods like LLaVA-CAM, IGOS++, and TAM in terms of faithfulness, localization, and hallucination diagnosis. Moreover, EAGLE mitigates hallucinations by removing minimal interfering regions, offering a promising tool for improving interpretability and reducing GPU memory usage.

### Strengths
1. EAGLE provides a complete attribution model by combining both insight (sufficiency) and necessity scores, ensuring that it captures both the minimal set of perceptual regions sufficient for token generation and those that are indispensable. This dual approach allows for a thorough and comprehensive explanation of which image regions and language priors drive the model’s decision-making process, offering a more complete understanding of the model’s internal workings.

2. EAGLE excels at visualizing attribution results, providing clear and intuitive saliency maps and influence scores. By visualizing the perceptual regions that influence token generation, it enables users to easily understand where the model focuses its attention. This enhances the interpretability of multimodal models, making complex decisions more transparent and accessible for researchers and developers.

### Weaknesses
1. As I see it, EAGLE appears to be a framework designed solely for token attribution analysis, lacking direct practical applications. Furthermore, the authors haven't leveraged this framework to derive any significant insights. The work reads more like a "tool paper"—simply introducing a new method without demonstrating its impact—which arguably doesn't meet the contribution bar for a top-tier conference.

2. On the topic of hallucination mitigation, my understanding is that, particularly for benchmarks like POPE, many hallucinations do not stem from language priors. Instead, they are often caused by a failure in fine-grained perception—an inability to accurately perceive specific objects. If that's the case, in such scenarios, wouldn't the saliency region (the area the model focuses on) and the hallucinated region (the area causing the hallucination) essentially become one and the same?

3. I also find the proposed method for hallucination correction to be counterintuitive. The paper introduces the Average Minimal Correction Region (AMCR) metric, defined as "the average proportion of regions that must be removed to correct hallucinations." This raises a fundamental question: why is the correction mechanism predicated on removing visual information? Rather than fixing the model's interpretation, this approach seems to simply block its perception of certain regions, which strikes me as a case of the cure being worse than the disease.

### Questions
1. Is the paper's core contribution limited to the tool itself? It appears to offer an attribution analysis framework but stops short of presenting any novel insights or applications derived from it. As a point of contrast, I examined a paper you cite. [1] That work not only introduces an analytical method but, crucially, leverages it to produce tangible insights. This combination of "tool plus discovery" is what I would consider a substantial contribution, a standard this paper does not seem to meet.

2. Could the author provide further explanation on the Evaluation Metrics proposed in Section 4.1? I believe the broader research community may not be familiar with the motivation or significance of such tasks. Are these metrics purely intended to measure the accuracy of the attribution?

[1] Zhang, Xiaofeng, et al. "From redundancy to relevance: Enhancing explainability in multimodal large language models." arXiv e-prints (2024): arXiv-2406.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
EAGLE is a black‑box attribution framework for MLLMs that (1) finds an ordered, compact set of image regions that best explains autoregressive token probabilities via a combined sufficiency and necessity objective, and (2) quantifies language‑prior vs. visual‑evidence reliance through an insertion trajectory. It shows strong gains on COCO/MMVP/RePOPE across several open‑source MLLMs and includes sufficient ablations/qualitative analyses.

### Strengths
1. Clear problem formulation and motivation for autoregressive token‑level attribution and modality disentanglement. 

2. Black‑box, model‑agnostic method that avoids gradients/activations; supports multi‑token explanations and interactive selection. 

3. Consistent empirical gains in faithfulness/localization and meaningful hallucination‑correction metrics across multiple models/tasks. 

4. Simple, implementable objective and transparent ablations for necessity/insight and granularity.

### Weaknesses
1. Moderate novelty. Although the method is well-motivated and practically useful, the core idea—perturbation-based subset selection with sufficiency and necessity terms—builds on long-standing concepts in black-box interpretability (e.g., RISE [1], meaningful perturbations [2], and submodular selection). The contribution lies more in adaptation to autoregressive MLLMs than in a fundamentally new formulation.

2. Weak submodularity claim is qualitative. The paper states that the objective is “weakly submodular” but does not provide an empirical estimate or theoretical bound of the submodularity ratio gamma. Without quantification, the theoretical justification for using a greedy approximation remains largely heuristic.

3. No runtime analysis. Although the paper reports reduced GPU memory, it omits actual wall-clock times or number of forward passes. Given the quadratic complexity O(|V|²) and typical superpixel counts (e.g., 64 regions ≈ 2,000 model calls), this omission makes it difficult to assess practical scalability. Runtime and sampling efficiency are crucial for determining whether EAGLE can be used interactively or at dataset scale.



[1] RISE: Randomized Input Sampling for Explanation of Black-box Models
[2] Interpretable Explanations of Black Boxes by Meaningful Perturbation

### Questions
1.  What are average wall‑clock times / number of forward passes per sample, especially for large |V|, like 64, across your three models? How do these compare to gradient-based methods like IGOS++ or TAM?

2. How are “object tokens” automatically identified and aligned with COCO categories in the Pointing-Game evaluation? How are compound nouns or tokenized subwords handled?

3. How sensitive are results to the choice of segmentation algorithm (SLICO vs. Felzenszwalb or SEEDS) and random seed initialization?

4. Can EAGLE be adapted to commercial or API-restricted MLLMs where token probabilities are not exposed—e.g., by sampling multiple continuations and estimating probabilities empirically?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper introduces EAGLE, a black-box attribution framework that explains how LVLMs generate text conditioned on different image patches. EAGLE attributes generated tokens to compact perceptual regions while quantifying their reliance on visual evidence. It unifies two interpretability principles, sufficiency and necessity, into a single objective optimized through a greedy search over sparsified image regions. Experiments on models such as LLaVA-1.5, Qwen2.5-VL, and InternVL3.5 across tasks like captioning, VQA, and hallucination diagnosis show that EAGLE achieves higher faithfulness, localization, and hallucination correction than prior gradient- or activation-based methods, while requiring far less GPU memory.

### Strengths
+ **Model-agnostic and training-free.** EAGLE operates as a black-box method applicable to diverse LVLMs (e.g., LLaVA, Qwen2.5-VL, InternVL) without requiring model access or retraining, which enhances its practicality, and can potentially achieve high impact in the field.
+ **Quantitative modality analysis.** The framework visualizes where the model attends but also quantifies what it relies on (language priors vs. perceptual evidence), which can be helpful in diagnosing the models' behavior of hallucination.
+ **Empirical thoroughness.** The experiments cover multiple models, datasets, and metrics (faithfulness, localization, hallucination correction), demonstrating consistent performance gains and strong generalization across tasks.
+ **Good presentation.** The paper presents the arguments and evidence in a well-organized way that makes it accessible to the audience.

### Weaknesses
+ **Limited novelty (my biggest concern).** While EAGLE combines sufficiency and necessity into a unified objective, the underlying ideas (greedy subset selection and region-based attribution) are adaptations of existing interpretability frameworks rather than fundamentally new algorithmic advances. Although I'm not from the community of interpretability, within 1min of search, I was able to find the following accepted work VPS [1] that has the exactly the same technique as in this paper: 
  + This paper has the same organization and content as the referenced paper. Especially in the methodology session, EAGLE's two scores (Insight Score, Necessity Score, and the way the are combined into one singular objective function) are exact copy of VPS. 
  + The experimental settings, tables, visualization, etc. are the exact copy of VPS. 
  + Hence it seems to me this paper simply changed the application domain from the pure Vision problem to LVLMs. I would not consider this paper having enough contributions.
+ **Lack of Computational Efficiency Analysis.** It is understandable that EAGLE has a clear edge over the gradient-based and activation-based methods in terms of memory usage. However, EAGLE has to have multiple runs (for search) and it should have a longer running time? it's a bit unclear to me why the authors failed to report the actual running time of the method and compare it to the other baselines. 
+ **On Optimality of the Greedy Search.** How do you make sure the greedy search is satisfactory enough? 

[1] Chen, Ruoyu, et al. "Interpreting object-level foundation models via visual precision search." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
+ How is the region masked/deleted during the analysis? Do we have more studies on this implementation? 
+ To me it's more straightforward to cut the image into square sub-regions before tokenization instead of doing SLICO. This way it will also make the consequent attribution straightforward (random patching embedding or directly deleting the patch). Why is SLICO necessary in this analysis? Do we have any sort of support? 
+ For the rest, see above (Weaknesses).

### Soundness
1

### Presentation
3

### Contribution
1
