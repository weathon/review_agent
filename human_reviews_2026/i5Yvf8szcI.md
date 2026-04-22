# Improving the Effective Coverage Space for Source-Free Domain Generalization via Visual-Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
With the widespread application of deep learning in computer vision, deep models often experience a significant drop in performance when facing unseen data, which negatively impacts their practical deployment. In this work, a dynamic feature construction and fusion method (DFCF) based on vision-language models is proposed for the task of source-free domain generalization. This method introduces the concept of Effective Coverage Space (ECS) and utilizes vision-language models to dynamically generate diverse feature representations and construct a virtual dataset, which transforms the source-free domain generalization into a supervised learning task. In the absence of source domain images, the effective coverage of the feature space is extended by improving the diversity of styles and features, thereby enhancing the model's adaptability to the unseen domain. Experimental results demonstrate that this method significantly improves performance of source-free domain generalization tasks across multiple datasets, effectively enhancing the generalization capability of the model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies source-free domain generalization (SDG) through the Effective Coverage Space (ECS) framework, with the motivation that a larger ECS implies higher generalizability. Specifically, a dynamic feature construction and fusion method (DFCF) is proposed to enhance the generalization of VLM-based models (e.g., CLIP) without requiring any specific source or target data available during training. DFCF first generates diverse styles based on random Gaussian noise and their interpolation, then incorporates a contrastive-based approach to fuse multi-template features by obtaining weights for each template. This approach enables source-free generalization to be solved through supervised training in one stage, in contrast to prior two-stage solutions.

The paper conducts comprehensive experimental results on existing benchmarks on SDG and prior works. Experimental results show that DFCF outperforms baselines overall while maintaining better efficiency.

### Strengths
- The motivation behind ECS is interesting, but requires experimental or theoretical support in the paper (see weaknesses)
- Better efficiency while higher performance compared to baselines; specifically, prior works (e.g., PromptStyle) are 2-stage and the propsoed method is one stage. Experimental results show higher performance across all benchmarks while requiring lower memory and time requirements compared to PromptStyler.
- The motivation is clear, and the text (mostly) is easy to understand; however, the paper presentation and content need to be improved (see weaknesses).

### Weaknesses
**Papers require experimental or theoretical support for  ECS motivation.**he main motivation of the paper is that the larger size of ECS is correlated with higher generalization; e.g., L075 "The concept of ECS is introduced, and it is demonstrated **both theoretically and experimentally** that the model’s generalization ability is closely linked to the size of its ECS.....the size of ECS is crucial for enhancing the model’s adaptability in unseen domains....".  However, no theoretical proof or experimental evaluation is provided to show the correctness of the motivation. The only provided result is the t-SNE in Fig. 6, which shows that combining multiple templates results in a higher M (a metric that was defined as part of the motivation) and a toy illustration in Fig.2.  Either a theoretical proof or comprehensive experiments are required to show that a higher M is correlated with higher generalization.

**Lack of connection between the method and motivation**. While the motivation of the paper is that higher ECS correlates with better generalization, it’s unclear how the actual proposed method is connected to ECS? Only the t-SNE in Fig. 6 shows that combining three templates shows higher ECS. However, this doesn’t really connect to the generation of Gaussian noises/styles and the contrastive weighting approach. Hence, the paper can benifit from theoretical or experimental support. For instance, one simple experiment could show that the style generation module, prompt ensemble, and fusion module via contrastive weighting yield higher M. This must be accompanied by W1 (i.e., higher M correlates with higher generalization) to effectively support the motivation and claims in the paper.


**Lack of required ablation and implementation details makes the paper’s contributions hard to understand and not reproducible.** Additional experiments/analysis are needed to effectively understand each component. (1) In L213 “... from which n embeddings are randomly selected.” What is the value of n in the experiments? How do different values of n impact the performance? (2) How is the initial pool of n styles generated in Algorithm 1? Is it all random or hybrid (3)? When and how is it decided between random and hybrid approaches for style generation?


** Method is partial explained; many imlementation detaisl are missing**. Fig.1 includes a supervised training phase and "classifier backone" . Also in the paper (e.g., L080 "The source-free domain generalization problem is transformed into a supervised learning problem") It's stated that the problem is formulated as supervised training. However, in the Method section (Sec. 3), no explanation/detail is given on how the supervised training is performed, what the loss function is, or what the architecture of the classifier backbone is. I assume that it’s similar to the baseline (i.e., PromptStyler), which also transforms the problem into a supervised learning problem. However, these details are important to be discussed in the paper so that readers can understand the method and reproduce it. 

**Mismatch between main text and Fig. 1**. The paper discusses how contrastive weighting is used to fuse the templates. However, Fig. 1 seems to add/concatenate each template feature and fused embedding T_f. However, such an explanation is missing.


**Ablation doesn’t support the claim of the paper.** Table 2 shows that adding contrastive fusion actually hurts performance, especially on DomainNet. This doesn’t support the claim in L250–252: “"However, if features from different templates are considered equally important, the model’s representational capacity could be limited", as the multi-prompt (M) performance is higher than feature fusion (F). Also, I assume M (i.e., Multiple Prompt) is a simple addition/average of prompt templates, as I can’t find any explanation of how it’s implemented.

**Missing implementation details hurt understanding the paper and reproducibility". In addition to the architecture of the classifier, the value of "n", and supervised training, other training/inference details must be provided. Also, is the number of templates 3 in the paper? How different number of templates impact the results? Why only three templates? 

**paper presentation needs to be improved** As mentioned before, there is a mismatch between the figure and the text. It’s not clear what the input of supervised training is. Also, in page 9, it seems like the explanation for Fig. 6 is in the middle of the explanation for Table 2. Please ensure the text appears coherent. I also suggest moving the method Fig to the method section rather than the introduction

**Limited contribution** The method is very similar to PromptStyler, just combined in one stage (i.e., generating diverse style prompts) and training a supervised model on textual data, which transfers to the image modality. given that the contribution fusion module is not clear (performance drops on average in Table 2), and more experimental or theoretical support is provided, and there is little or no connection between the proposed modules and the ECS motivation, I find the contribution of the paper in the current format limited and vague.

### Questions
- Suggestion: Drop the phrase "virtual training datasets," as it’s not referred to or mentioned in the main text other than the conclusion.
- What’s the connection between the method and ECS motivation? See weaknesses for more details.
- How to ensure sampling random noise to simulate different styles? Couldn’t this negatively impact/interfere with the content (e.g., class token such as dog)?
- What is the value of "n", and how do different values of "n" impact the performance?

- In contrastive training, are different z_i (e.g., z_i, z_j) different templates for the same class? How about style? It’s confusing, as Fig. 3 provides both examples of different templates and different styles. Further brief clarification is helpful.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper first introduces the concept of effective coverage space to state the limitations of source domain generalization. Then, it constructs a virtual dataset with different text and fusion vectors to mitigate the absence of source data. A dynamic feature construction and fusion module is further proposed with visual-language model under different style information. Extensive experiments validate that the method outperforms previous methods on multiple benchmarks.

### Strengths
1. The definition of ECS is plausible to clarify the source-free DG issue.
2. The experimental results on multiple benchmarks are supportive.

### Weaknesses
1. The technical contribution is not sufficient. Firstly, the method uses a perturbation technique to expand the style space within a pool of basic styles, which is similar to many existing literatures in DG field, such as,
[1] Kang, Juwon, et al. "Style neophile: Constantly seeking novel styles for domain generalization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
[2] Li, Xiaotong, et al. "Uncertainty modeling for out-of-distribution generalization." arXiv preprint arXiv:2202.03958 (2022).
The proposed simple perturbation of gaussian noise is not sufficient for style exploration, and please state how your proposed techniques differ from these ones.
2. The number of text templates is chosen as 3. Does that mean using 3 templates achieves the optimal performance? Or what if more templates are used for fusion? A more detailed ablation study is recommended.
3. The ablation study in Table 2 is not supportive. It seems that the feature fusion technique cannot improve the overall performance (74.0 vs 73.7). Moreover, it can be noticed that using only multiple templates achieves 74.0, which is even better than the combination of all the three proposed method, which strongly shows the incremental improvement of the two other techniques. This raises deep concern that the improvement of your method originates from the design of multiple templates.
4. As the novelty of the feature fusion is to design the dynamic weight across different views, it’ll be better to present a visualization to show the weight distribution on the training objective. A more detailed analysis is recommended to state how this dynamic weight improves the original performance.
5. The presentation of this paper is rather ambiguous. Firstly, the dynamic feature depicted in Figure 1 is not described in the main text. The virtual training datasets is also confusing. From the main text, the method computes contrastive loss between only three different embeddings in an end-to-end manner. Where T_f originates from?

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

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
The paper presents three key contributions: (1) development of an effective coverage space to quantify the limits of domain generalization, (2) generation of synthetic data via text and fusion vectors to address the absence of source data, and (3) design of a style-aware dynamic fusion module for vision-language models. Extensive evaluations confirm its state-of-the-art performance.

### Strengths
1. This paper proposes a style-aware dynamic fusion module for visual-language models. 

2. Experimental results achieve its state-of-the-art performance.

### Weaknesses
1. The novelty of this method is relatively low. Using multiple embeddings with perturbation for different styles has been explored in prior studies.


2. According to Table 2, using multiple templates alone achieves 74.0, which even outperforms the combination of all three proposed components. This raises questions that the performance gain of the proposed approach may primarily stem from the multi-template design, while the other techniques contribute little to the overall improvement.


3. As the baseline of this paper is PromptStyler, there is a significant gap between the reported results and the re-implementation results. Furthermore, the proposed method performs worse than the original reported results of PromptStyler and shows only minimal improvement compared with the re-implemented baseline results. Please carefully verify the correctness of the method.
 

4. The paper adopts three text templates, but it remains unclear whether this choice is empirically optimal or arbitrarily selected. A more detailed ablation study on the number of templates should be conducted to investigate whether using more templates could further improve performance and to justify the choice of three templates.

### Questions
1. What is the underlying reason: using multiple templates alone achieves a higher result that even outperforms the combination of all three proposed components?


2. Why is there a significant gap between the reported results and the re-implemented results?

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
4

### Summary
This paper proposes DFCF (Dynamic Feature Construction and Fusion), a one-stage framework leveraging vision-language models (VLMs) (specifically CLIP) to perform source-free domain generalization (SFDG). The key conceptual innovation is the Effective Coverage Space (ECS), a theoretical proxy for quantifying how well the model’s learned feature space covers unseen domain variations. The work is technically well-organized, but its novelty, evaluation, and theoretical clarity can be improved.

### Strengths
1. The idea of linking generalization capability to an "effective coverage space" is conceptually intuitive and offers a geometric interpretation that complements recent prompt-based DG methods.
2. The one-stage training pipeline improves efficiency while slightly improving accuracy.
3. The dynamic style generation and contrastive feature fusion are elegant extensions to CLIP's latent space.
4. Evaluation covers four DG benchmarks (PACS, VLCS, OfficeHome, DomainNet) and three backbones (ResNet-50, ViT-B/16, ViT-L/14), which is very comprehensive

### Weaknesses
1. Improvements over PromptStyler are modest ( +0.5 ~ 1% on average). Given the large number of comparisons, statistical significance is unclear.
2. ECS formulation lacks strong theoretical grounding. The motivation of ECS is not clear.
3. Some over-claiming ("substantially improves performance", and "superior performance") despite small gains and improvements.

### Questions
1. Equation (3) defines an optimization objective involving set unions/intersections, but it is not explicitly optimized; it is later replaced by the proxy metric M. The proxy (Eq. 4) is reminiscent of standard contrastive metrics and may not justify a new theoretical construct. A formal derivation or correlation study between M and actual generalization is missing. Clarify whether ECS optimization is implicit via contrastive learning or explicit via M maximization.
2. Figure 1 is not explained in detail. 
3. Why 3 is used for the number of Templats? This is not described.

### Soundness
3

### Presentation
3

### Contribution
3
