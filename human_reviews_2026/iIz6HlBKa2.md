# Explaining How Visual, Textual and Multimodal Encoders Share Concepts

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Sparse autoencoders (SAEs) have emerged as a powerful technique for extracting human-interpretable features from neural networks activations. Previous works compared different models based on SAE-derived features but those comparisons have been restricted to models within the same modality. We propose a novel indicator allowing quantitative comparison of models across SAE features, and use it to conduct a comparative study of visual, textual and multimodal encoders. We also propose to quantify the *Comparative Sharedness* of individual features between different classes of models. With these two new tools, we conduct several studies on 21 encoders of the three types, with two significantly different sizes, and considering generalist and domain specific datasets. The results allow to revisit previous studies at the light of encoders trained in a multimodal context and to quantify to which extent all these models share some representations or features. They also suggest that visual features that are specific to VLMs among vision encoders are shared with text encoders, highlighting the impact of text pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a comparison of visual, textual and multimodal encoders based on Sparse Auto Encoder (SAE) derived features. It proposes two tools - weighted Maximum Pairwise Pearson Correlation and Comparative Sharedness for performing this comparison. Using these tools, the model examines three datasets of text-image pairs (COCO, Laion-2B and Oxford-102 flowers) and reports some findings. The key findings are that 1) Encoders from different modalities (visual and text) show greater similarities when considering representations from only the final layer, 2) cross-modal similarities appear to be correlated to the image-text alignment quality in the dataset, and 3) visual features specific to visual language models such as CLIP/DFN are more similar to those from LLMs (e.g. BERT) than from other visual foundation models (e.g. Dino v2).

### Strengths
* Proposes two new tools for comparing visual and text language models based on sparse auto-encoder (SAE) derived features
* Presents a comparison of multiple vision, text and multimodal encoders based on SAE derived features.
* Obtains some findings about the similarity of models, data quality and representations.

### Weaknesses
* The findings in the paper (e.g. importance of the last layer, the typology of features specific to visual language models are more similar to text language models rather than other visual foundation models) could have been obtained due to idiosyncrasies in the specific datasets and need to be confirmed by repeating the analysis on more datasets. Without such a confirmation, it is hard to know whether these findings will generalize to novel datasets and models.
* The paper employs SAE but does not present an overview of the SAE approach. Without such a presentation, it is hard to understand the significance of specific modifications that were employed in the paper (e.g. in Section 2).

### Questions
* Section 2: It would be good to present a short overview of SAE in the appendix. Without such an overview, it is hard to comprehend sentences like L062: "The SAE is trained with mean square error loss using all patches or token of the input text"
* L125: images -> examples
* L374: "The obtained typology is very similar to the one established while considering VLM visual encoders, pushing the hypothesis that previous observations could be caused by their text pretraining." Could there be other factors that could have resulted in this finding?

### Soundness
2

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
2

### Summary
The authors propose a novel indicator to compare multiple encoders using their SAE features. This allows the quantification of the “shared” features common to different models. The key contributions are the scale of comparisons and performing multimodal comparisons. The authors also propose 2 metrics – weighted Maximum Pairwise  Pearson  Correlation and Comparative Sharedness which enable comparison across models.

### Strengths
1.	Extensive experiments have been conducted – with/outside modality comparisons in popular models, multiple datasets
2.	Qualitative analysis to identify underlying concepts has been done

### Weaknesses
My main concern is that all the numbers reported are in terms of the proposed metric wMPPC. While I understand intuitively why the use of weighting is important, I think there needs to be a comparison between wMPPC and older metrics, with examples that show the need for proposing wMPPC.

Detailed questions and clarifications are listed in the Questions section.

### Questions
1.	In the introduction, please elaborate why handling multiple modalities is challenging, and why this has not been done in the past.
2.	The claim in the abstract that previous papers have only looked at single modalities may need more context. Other papers like [1-4] do appear to have considered multi modality. Please clarify why these were not included.
3.	In equation (6), can we not just use the term $S_i^M \times (\rho_i^{M -> A} - \rho_i^{M-> B})$? This term would decrease if the similarity between M and B increases, and increase if the similarity between M and A increase. I’m guessing it’s to handle the positive and negative values of $\rho$ but it would be good to clarify this.
4.	Likewise, in eq (7), please provide the intuition for choosing this particular form, as opposed to taking the absolute value of the difference.
5.	I may have missed it, but please highlight the area where models of different sizes have been experimented with.
6.	Could you also highlight all the 21 models mentioned in the abstract in the appendix? 
7.	In the implementation details, please describe how the hyperparameter tuning was done.
8.	The conclusion about the last layers of the LLMs being most important semantically is drawn from Figure 1, which is based on the proposed metric. Is this conclusion also supported with MPPC metric as well? 
9.	I do not fully understand how the conclusion made about the last layers being important holds, given that in Fig1, towards the inner layers (~20), the wMPPC values are extremely low across the layers.
10.	My main concern is that all the numbers reported are in terms of the proposed metric wMPPC. While I understand intuitively why the use of weighting is important, I think there needs to be a comparison between wMPPC and older metrics, with examples that show the need for proposing wMPPC.


[1] Isabel Papadimitriou, Huangyuan Su, Thomas Fel, Sham Kakade, and Stephanie Gil. Interpreting the linear structure of vision-language model embedding spaces. arXiv preprint arXiv:2504.11695, 2025.

[2] Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. Sparse autoencoders learn monosemantic features in vision-language models. arXiv preprint arXiv:2504.02821, 2025.

[3] Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, and Yifei Wang. Multi-faceted multimodal monosemanticity. arXiv preprint arXiv:2502.14888, 2025.

[4] Vladimir Zaigrajew, Hubert Baniecki, and Przemyslaw Biecek. Interpreting CLIP with hierarchical sparse autoencoders. arXiv preprint arXiv:2502.20578, 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces two new metrics: weighted Maximum Pairwise Pearson Correlation (wMPPC), which is an extension of the previous MPPC score, and Comparative Sharedness, to compare interpretable features extracted via Sparse Autoencoders (SAEs) across visual, textual, and multimodal encoders. The authors conduct a large-scale analysis on 21 transformer-based encoders from different modalities and datasets, identifying shared and modality-specific concepts. Results highlight that shared cross-modal information mainly resides in the final layers and that text pretraining drives high-level visual concepts in VLMs.

### Strengths
* Analyzing similarities and differences across visual, textual, and multimodal encoders is valuable, as it can inform how future models are trained and aligned

* The study spans a large and diverse set of 21 transformer encoders, offering broad coverage across modalities, datasets, and scales.

* The paper includes a detailed limitation section, showing good awareness of scope boundaries and possible extensions.

### Weaknesses
* I found a bit surprising that CLIP image features are more correlated with DINOv2 image or than with SigLIP image (trained similarly to CLIP), Tab. 1. Same for SigLIP image being more correlated to CLIP and BERT text rather than SigLIP text encoder ! This make me question the proposed metrics.

* It’s not clear how the observed correlations translate to real-world impact (measured with quantitative metrics), for example, whether they relate to model performance, bias, or hallucination behavior.

* The study focus on contrastive multimodal encoders. How these findings holds for encoders trained with reconstruction objectives such as AIMv2 [1] 

* The analysis and findings (e.g. increasing correlation in last layers, multimodal concepts ...) closely resembles earlier concept-based interpretability [2] and modality alignment papers [3], but the connection to those frameworks is not mentioned or clarified. The paper should position itself to these related lines of research.

* For this kind of papers, more visual illustrations might help to understand better the contributions.

[1] "Multimodal autoregressive pre-training of large vision encoders", CVPR 2025.

[2] "A concept-based explainability framework for large multimodal models." NeurIPS 2024.

[3] "Implicit multimodal alignment: On the generalization of frozen llms to multimodal inputs." NeurIPS 2024.

### Questions
Please check weaknesses section.

### Soundness
3

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
2

### Summary
The paper presents a new metric "weighted Maximum Pairwise Pearson Correlation" or wMPPC which is similarity measure between two models computed through the weighted expectation of per feature correlation by sampling the activations. The weighting allows focus on correlations that are high between a set of features in two models. The authors find that computing wMPPC on different models uncovers differences in the quality of image-text alignment between datasets, e.g. Laion-2B is worse that Coco.  Another metric "Generalized Comparative Sharedness" is proposed that allows probing of a model over individual concepts/features to determine how unique it is to a group/class of models. The former is global metric of model similarity, the latter is more focused metric of similarity. The sharedness metrics shows how some textual concepts are well shared between text and VLMs, but not visual foundation models. These two new metrics show some promise in being useful diagnostics to help understand encoders.

### Strengths
Two new metrics are proposed that help to understand the similarities and differences between models. The authors show how these metrics can be used to uncover interesting details like the quality of the original corpora or "shared concepts" learned between models. The paper provides clear details and pointers to scripts on reproducing results and works with public data sets and models so should be highly reproducible.

### Weaknesses
The paper provides a comparative study of visual, textual and joint vision-text models. It would be super interesting to see what insights these measures could provide with the addition of audio to the assessed modalities.

### Questions
Have you looked at adding audio as a modality to analyze with your two new metrics?

### Soundness
3

### Presentation
3

### Contribution
3
