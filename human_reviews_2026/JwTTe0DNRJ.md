# Background Prompt for Few-Shot Out-of-Distribution Detection

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Existing foreground-background (FG-BG) decomposition methods for the few-shot out-of-distribution (FS-OOD) detection often suffer from low robustness due to over-reliance on the local class similarity and a fixed background patch extraction strategy. To address these challenges, we propose  a new FG-BG decomposition framework, namely Mambo, for FS-OOD detection. Specifically, we propose to first learn a background prompt to obtain the local background similarity containing both the background and image semantic information, and then refine the local background similarity using the local class similarity. As a result, we use both the refined local background similarity and the local class similarity to conduct background extraction, reducing the dependence of the local class similarity in previous methods. Furthermore, we propose the patch self-calibrated tuning to consider the sample diversity to flexibly select numbers of background patches for different samples, and thus exploring the issue of fixed background extraction strategies in previous methods. Extensive experiments on real-world datasets demonstrate that our proposed Mambo achieves the best performance, compared to SOTA methods in terms of OOD detection and near OOD detection setting. The source code will be released a https://anonymous.4open.science/r/Mambo-CBC1.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Mambo, a framework for Few-Shot Out-of-Distribution (FS-OOD) detection. It addresses the limitations of existing FG-BG decomposition methods, such as over-reliance on local class similarity and fixed background patch extraction strategies. Mambo incorporates three key components:

1. A learnable prompt to extract background semantic information independently of class features.
2. A mechanism to refine background similarity using class similarity adaptively.
3. A dynamic strategy to extract a variable number of background patches tailored to each sample.

The authors validate Mambo on standard FS-OOD datasets and report that it outperforms state-of-the-art (SOTA) methods in terms of FPR95 and AUROC, while maintaining computational efficiency.

### Strengths
1. The paper effectively highlights key limitations in existing FG-BG decomposition methods, such as fixed patch selection strategies and over-reliance on local class similarity.
2. The empirical evaluation leverages multiple datasets (e.g., ImageNet-1K, iNaturalist, SUN, Places, Texture) and settings (e.g., OOD detection and near-OOD detection).
3. The ablation studies demonstrate the individual contributions of each proposed component.
4. Despite its improvements, Mambo remains computationally efficient, introducing minimal additional overhead compared to baseline methods like SCT and LoCoOp.
5. The paper provides detailed experimental settings and promises to release the source code, ensuring reproducibility.

### Weaknesses
1. This component is essentially a learnable prompt applied to the CLIP text encoder, which is a natural extension of existing prompt learning approaches (e.g., CoOp and Local-Prompt). The authors do not provide sufficient evidence that a learnable "background prompt" is superior to simpler alternatives, such as using predefined textual templates or leveraging existing negative prompts. For example, in Table 3, the ablation study shows that the baseline (without refinement or patch tuning) already achieves a relatively strong performance, with only modest improvements (e.g., FPR95 improves by ~2.2% on average) when adding the background prompt. This suggests that the background prompt alone does not bring transformative gains.

2. The refinement strategy combines local class similarity with background similarity in a weighted manner. While this is an effective mechanism, it lacks novelty, as it builds on existing weighting techniques commonly used in FS-OOD tasks (e.g., SCT's dynamic balancing). Moreover, The refinement weight $\delta_i$ in Eq (11) is heuristic and not well-justified. 

3. The dynamic patch extraction strategy is conceptually similar to SCT's self-calibrated tuning. The primary difference is the introduction of a threshold parameter ($\alpha$) to adjust the number of extracted patches. While this adds flexibility, it is not a fundamentally new idea.

4. SCT also uses a self-calibrated tuning mechanism to dynamically balance tasks. The patch self-calibrated tuning in Mambo appears to be a direct extension of this idea, with the addition of a heuristic threshold (Eq (12)). The authors do not sufficiently highlight how their approach is distinct or superior.

5. Local-Prompt leverages negative prompts and local outlier knowledge for FS-OOD detection. The background prompt in Mambo serves a similar purpose but focuses on extracting background features instead of negatives. The novelty here is limited, as both approaches aim to augment prompt learning with auxiliary prompts.

6. Several components, such as the refinement weight ($\delta_i$) and the threshold parameter ($\alpha$), are heuristic and not rigorously justified. This raises concerns about the robustness of the method across different datasets and tasks. For example, the hyperparameter sensitivity analysis (Fig. 9) shows that small changes in $\alpha$ can lead to noticeable performance drops, indicating that the method may require careful tuning.

Overall, the paper tackles important challenges in FS-OOD detection and shows good empirical results, but the contributions feel incremental, and the added complexity isn't fully justified. A broader evaluation and clearer differentiation from prior work would better highlight its significance.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a foregroudn-background decomposition framework for few-shot out-of-distribution  detection. The aim is to address the challenge of low robustness due to over-reliance on the local class similarity and a fixed background patch extraction.
 .

### Strengths
- The paper is well organized and written.
- The proposed method named Mambo is detailed and reproducible.
- The contribution addresses  a key challenge in the field.
- Experiments are well conducted showing the superiority of Mambo compared  to three previous methods.

### Weaknesses
None

### Questions
Please remind the citation [] after each acronym of the compared methods in Table 2.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the problem of erroneous OOD feature extraction in existing few-shot OOD detection methods and argues that over-reliance on the local class similarity and a fixed background patch extraction strategy are the two main reasons. It proposes Mambo, a new foreground-background decomposition framework which consists of a learnable background prompt refined by local class probabilities and patch self-calibrated tuning to adaptively extract background information. Experiments on ImageNet OOD benchmarks and OpenOOD benchmarks are conducted to verify its effectiveness.

### Strengths
1. The problem of erroneous OOD feature extraction in existing few-shot OOD detection methods is interesting.
2. Experiments on large-scale ImageNet OOD benchmarks and OpenOOD benchmarks are conducted to evaluate the proposed method.

### Weaknesses
1. **The presentation of the proposed method is incomplete and unclear**. First, the proposed background prompt is learnable but there is no mathematical formulation and explanation of the loss function to optimize the learnable prompt. Second, the meaning of $s_j^{class}$ in Eq.(11) is not clarified and there is no in-depth discussion about the design and implication of the refinement weight factor $\Delta_i$. It’s also recommended that the authors provide a clear algorithmic summary of the pipeline of **Mambo**.
2. **The OOD detection performance of the proposed method is not compelling**. For instance, on the ImageNet-1k OOD benchmark, **Mambo** only outperforms the best baseline, Local-Prompt, by less than 1 in terms of FPR95 and falls behind Local-Prompt in terms of AUROC under the 4-shot setting and falls behind Local-Prompt in terms of these two metrics under the 16-shot setting.
3. To **validate the motivation of this paper**, the reviewer suggests that the authors provide **sufficient illustrations or experimental results** to demonstrate the problem of over-reliance on the local class similarity and fixed background patch extraction strategies.
4. To **further verify the effectiveness** of the proposed method, the reviewer suggests that the authors add experiments on **different CLIP backbone structures**.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on FS-OOD detection with VLMs. The authors argue that existing FG-BG methods (e.g., LoCoOp, SCT) suffer from: (1) heavy dependence on potentially inaccurate local class similarityfor background identification, and (2) inflexible background extraction strategies (e.g., fixed top-K) that ignore sample diversity. To tackle these, Mambo proposes three core components. The method is evaluated on standard OOD and near-OOD tasks, outperforming existing zero-shot and few-shot baselines. Ablation studies and visualizations validate the design choices.

### Strengths
1. The paper accurately identifies the core weaknesses of FG-BG methods, providing a clear rationale for the proposed solutions.
2. The background prompt is a creative idea to model background semantics explicitly. The integration of refinement and self-calibration is logically structured and directly addresses the stated limitations.
3. Extensive experiments cover diverse datasets (e.g., ImageNet-1K, near-OOD benchmarks), few-shot settings, and ablation studies. Results consistently show Mambo’s superiority. Additional analyses (e.g., computational cost, OOD scoring strategies) add depth.

### Weaknesses
1. According to Table 1, compared to the baseline, the performance advantage of the method proposed in this paper is not significant. The method adds complexity (e.g., an extra prompt and hyperparameters).
2. Although Figure 5 visualizes the background prompt’s attention, a deeper and more analysis of the learned background concepts would strengthen interpretability. 
3. According to [1], the similarity between the embeddings of background regions and the corresponding real label’s text embedding is greater than the similarity between the embeddings of foreground regions and the corresponding real label’s text embedding.  This proposed method does not take this into account. 
4. The issue of inflexible background extraction strategies is mentioned in CLIP-OS [2].  

[1] Li, Yi, et al. "A closer look at the explainability of Contrastive language-image pre-training." Pattern Recognition 162 (2025): 111409.
[2] Sun, Hao, et al. "Clip-driven outliers synthesis for few-shot OOD detection." arXiv preprint arXiv:2404.00323 (2024).

### Questions
When there are multiple foreground objects or the foreground is very small in an image, do Local Background Similarity and Local Similarity Refinement still remain effective?

### Soundness
2

### Presentation
2

### Contribution
2
