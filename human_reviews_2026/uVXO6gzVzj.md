# Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Recent years have witnessed remarkable progress in multimodal learning within computational pathology. Existing models primarily rely on vision and language modalities; however, language alone lacks molecular specificity and offers limited pathological supervision, leading to representational bottlenecks. In this paper, we propose STAMP, a Spatial Transcriptomics-Augmented Multimodal Pathology representation learning framework that integrates spatially-resolved gene expression profiles to enable molecule-guided joint embedding of pathology images and transcriptomic data. Our study shows that self-supervised, gene-guided training provides a robust and task-agnostic signal for learning pathology image representations. Incorporating spatial context and multi-scale information further enhances model performance and generalizability. To support this, we constructed SpaVis-6M, the largest Visium-based spatial transcriptomics dataset to date, and trained a spatially-aware gene encoder on this resource. Leveraging hierarchical multi-scale contrastive alignment and cross-scale patch localization mechanisms, STAMP effectively aligns spatial transcriptomics with pathology images, capturing spatial structure and molecular variation. We validate STAMP across six datasets and four downstream tasks, where it consistently achieves strong performance. These results highlight the value and necessity of integrating spatially resolved molecular supervision for advancing multimodal learning in computational pathology. The code is included in the supplementary materials. The pretrained weights and SpaVis-6M are available at: https://github.com/Hanminghao/STAMP.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces STAMP, a framework that enhances computational pathology by fusing pathology images with spatially-resolved gene expression profiles, addressing the lack of molecular specificity in current models. To train this, the authors built SpaVis-6M, the largest Visium-based spatial transcriptomics dataset to date. STAMP uses a spatially-aware, multi-scale alignment strategy and achieved strong performance across six datasets, demonstrating the value of integrating spatial molecular data.

### Strengths
1.	Dataset contribution: The authors constructed SpaVis-6M, the largest Visium-based spatial transcriptomics dataset to date.

2.	The paper proposes a novel, spatially-aware pretraining framework (STAMP) that effectively fuses pathology images with gene expression by using hierarchical multi-scale contrastive alignment and cross-scale localization mechanisms.

3.	The proposed model’s effectiveness is validated by its strong performance across six different datasets and four downstream tasks.

### Weaknesses
1.	The figure quality could be improved. For example, there are two subfigures a) and b) in Fig. 2, yet the caption of Figure 2 did not mention them. Moreover, in Fig. 3, the font size of the scale bar on the y-axis is too small.

2.	More metrics could be included, especially for the WSI classification tasks. Metrics such as ACC, sensitivity, and specificity should be calculated to provide a more comprehensive assessment of the proposed STAMP.

3.	The authors should carefully proofread for typos. For example, in the first paragraph of Section A.5.1, “GPFM” should be bold, as other model names are.

### Questions
1.	I see that the authors have included visualization for gene expression prediction tasks. Could the authors also include heatmap visualization for WSI classification tasks?

2.	The author mentioned that “Considering performance and resource requirements, UNI (ViT-L/16) (Chen et al., 2024b) was selected.”. UNI is a uni-modal foundation model. And in recent years we have seen many vision-language pathology foundation models, such as MUSK [1] and TITAN [2], or multi-modal pathology foundation models such as mSTAR [3]. An intuitive idea would be that these models have a better aligned feature space for multi-modality, including image-text and image-text-gene. Could the authors elaborate more on why not use one of these models as the pathological vision encoder?

$\quad$ [1]  Xiang, Jinxi, et al. "A vision–language foundation model for precision oncology." Nature 638.8051 (2025): 769-778.

$\quad$ [2] Ding, Tong, et al. "Multimodal whole slide foundation model for pathology." arXiv preprint arXiv:2411.19666 (2024).

$\quad$ [3] Xu, Yingxue, et al. "A multimodal knowledge-enhanced whole-slide pathology foundation model." arXiv preprint arXiv:2407.15362 (2024).

Overall, the method is novel, and the experiments are solid.

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
3

### Summary
The paper proposes **STAMP**, a multimodal transformer-based model for spatial transcriptomics, trained via three stages: (a) _gene-only masked modeling_ of transcriptomic profiles, (b) _spatially-aware training_ leveraging local neighborhood structure, and (c) _hierarchical contrastive learning_ aligning a gene encoder with a histopathology foundation model. To train the model, the authors collected ST data from 1,982 slices from 35 organs, called SpaVis-6M which will be published together with the model code and weights. This dataset is said to be the largest public ST dataset so far and therefore helps the community to advance understanding and training DL models for ST data.

### Strengths
- The paper is well written and well structured.
- For both modalities, histopathology & (spatial) transcriptomics, a sufficient amount of foundation models were benchmarked to set the STAMP model in to context. When possible, STAMP was evaluated in both the uni-modal as well as multi-modal setting to better compare it to current uni-modal FMs
- The dataset addresses current short-comings of sparse and well structured, coherent ST data
- The training setup is sophisticated and the single loss term well grounded and explained
- The model as well as the benefit of the newly proposed loss compartments is evaluated. 
- The evaluation of the models is statistically grounded and a good variety of downstream tasks is assessed

### Weaknesses
Weaknesses/Questions:
- The Gene Encoder architecture is not explained in the main text. It is not directly clear how the Embedding(T_i) in Eq. 2) is being inferred or how it is masked. 
- The drawbacks of imputing missing genes in the SpaVis-6M dataset is not discussed. To which extend would this affect the training of the models? Does this introduce batch effects w.r.t. to the input of the Gene Encoder or to the target predictions? How would this impact the evaluation if a gene is predicted without it being measured?
- Figure 1.) is not sufficiently described. What is evaluated? What are the labels to evaluate the clustering metrics? This should ideally be covered in both the caption as well as in the main text
- Most of the evaluations in the main text focus on datasets which were (partially) used during training of STAMP but are o.o.d. for the remaining models. In an evaluation setting for which all models are o.o.d., STAMP does not show a general improvement over the foundation model UNI with which it was initialized


Overall, while the dataset contribution is strong and the training design is compelling, the paper lacks architectural clarity on the gene encoder and imputation process. Releasing the dataset has good value and I appreciate the systematic training formulation.

### Questions
Mentioned along with Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author have built a model dubbed STAMP, (Spatial Transcriptomics-Augmented Multimodal Pathology) which utilizes a large dataset (SpaVis- 6M) of spatial transcriptomic data for patch level encoder training.  Unlike most prior efforts utilizing ST data the authors have not dwelled on the so-called "cell-level" data but have instead treated adjacent small tiles as micro-enviromental units. To accomplish this they have developed a special Gene Tokenization construct using a BERT inspired method. To train the Vision-Genomic model the authors trained the model using contrastive loss (InfoNCE) incorporating a muti-scale approach. 

Performance was evaluted on the human dorsolateral prefrontal cortex (DLPFC) and Human Breast Cancer (HBC) datasets benchmarking against other encoder, including VLMs, vision only, Genomics only (unclear relevance, I recommend removing this), and vision-genomic encoders. In the linear probe the model had SOTA performance.  The model was also benchmarked against other algorithms in Gene expression prediction and mutation prediction using the model as an encoder for with ABMIL classifier with mostly superior performance.

### Strengths
I believe this is a novel training strategy for a patch level encoder.  

The technique for providing summary tokenization for tissue regions is very interesting and likely to be a path forward for more ST based vision models. 

While I was skeptical, the experiment showing that the model has comparable to superior performance as a feature extractor slide level biomarker prediction is fascinating.  The imaging technique and quality from ST is quite different from standard WSI. While the LUAD mutations predictions are highlighted in the manuscript, I appreciate that the authors have included in appendix the tasks where performance was not as good compared to other models. 

Ablations are informative and support the final conclusions.

### Weaknesses
ST is not widely available and different platforms likely have different artifacts.  This manuscript, while a good start, does not conclusively determine that performance gained from using data from this very spatially detailed assay outweighs performance that can be gained on much more widely available data types. 

The encoder training is borrowed heavily from other VLM and VGM so is not novel. Thus this is a smart re-implementation but not a novelty. 

In Figure 3, there is a large descrepancy between what I have personally assessed the performance of UNI, Gigapath, H-optimimus-0 for the EGFR and KRAS tasks on the TCGA-LUAD dataset.  I recommend checking your code, assuring that you are only assessing FFPE images, and ensuring that you are only assessing the biologically relevant events.  In my benchmarking the cross validation median AUC performance for UNI and Gigapath is 0.79 and for Hoptimus0 is 0.81.  It is possible that you have chosen tile sizes that are more favorable to your method. Please use some effort and cross checking this. 

Need y-axis label on Figure 3

I would suggest putting limitations in the manuscript not in the appendix.  It is among the most important portions of the manuscript so relegating it to the end of the appendix is not appropriate.

### Questions
What fraction of training data is on cancerous vs. non-concerous tissue? 

Does training solely on cancer result in similar performance or does non-cancer tissue facilitate better features?

I understand that ST datasets are scarce but what led you to choose non-cancer dataset DLPFC to lead the study.  Do you think of this model is a more cancer focused model or more general purpose.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces STAMP, a framework for pathology-ST pretraining. STAMP works by aligning image patches and spatially resolved transcriptomic profiles via variety of reconstruction and contrastive learning objectives. This work also introduces SpaVis‑6M, a new dataset using 10X Visium with 1982 slices and 5.75m spots. STAMP was benchmarked on a variety of tasks including linear probe (DLPFC,  HBC), gene expression prediction (PSC, HHK, and HER2+) and LUAD mutation prediction (as well as other tasks) and compared against many pathology image encoders and ST prediction models.

### Strengths
- This is a good resource and benchmark paper for researchers working in pathology-genomics pretraining. SpaVis‑6M is a good contribution, as well as the pretrained weights and code used to pretrain on this dataset.
- Experimental design is overall strong, with diverse breadth of tasks and comparisons. Table 6 and Table 7 are important ablations, which respectively show (1) loss objectives used in STAMP improve multimodal pretraining performance, (2) performance gain of pretraining on SpaVis-6M versus HEST-1k.
- Figures are illustrative (Figure 5-9), with lots of good details included in the Appendix.

### Weaknesses
- What is the statistical significance of STAMP improvement? In Table 2, the standard deviation of performance is enormous for the PSC, HHK, HER2+ tasks. For the HER2+ task, most models have an average MSE of ~0.9 with a standard deviation of ~0.45.
- Is there a reason why STAMP was not evaluated on the HEST benchmark?
- There are many works looking at multimodal alignment of pathology and ST. While STAMP was compared against BLEEP and mclSTExp, it is missing many other comparisons such as HisToGene, Hist2ST, UMPIRE, and OmiCLIP, with OmiCLIP being the most relevant comparison that this work should compare against.
- It is not clear how all the different variations of STAMP are implemented (STAMP-V, STAMP-G, STAMP-F, STAMP-Reg, STAMP-Con). Are these all different models, or the same model but using the outputs of different heads and intermediate layers as the learned representation? Having a section on implementation details on how all the STAMP variants are set-up would be helpful.

This work makes a lot of good contributions to the biomedical community. However, the contributions are more empirical and less relevant for the fundamental ML/AI community. The most important contributions I see in this work is the introduction of the SpaVis-6M dataset and experimental design. Method-wise, the added loss components are mostly incremental but the authors still show they are helpful. While I think this paper would be a better fit for a more biomedical venue, I still lean towards acceptance.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
