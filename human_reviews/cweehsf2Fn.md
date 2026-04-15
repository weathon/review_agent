# Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 5

## Abstract
Contrastive Language-Image Pre-training (CLIP) shows promise in medical image analysis but requires substantial data and computational resources. Due to these restrictions, existing CLIP applications in medical imaging focus mainly on modalities like chest X-rays that have abundant image-report data available, leaving many other important modalities under-explored. Here, we propose one of the first adaptations of the full CLIP model to mammography, which presents significant challenges due to labeled data scarcity, high-resolution images with small regions of interest, and data imbalance. We first develop a specialized supervision framework for mammography that leverages its multi-view nature. Furthermore, we design a symmetric local alignment module to better focus on detailed features in high-resolution images. Lastly, we incorporate a parameter-efficient fine-tuning approach for large language models pre-trained with medical knowledge to address data limitations. Our multi-view and multi-scale alignment (MaMA) method outperforms state-of-the-art baselines for three different tasks on two large real-world mammography datasets, EMBED and RSNA-Mammo, with only 52% model size compared with the largest baseline. The code is attached in the supplement file and will be released on GitHub upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The authors proposed a pre-training scheme for mammography images using cross-view (images) and cross-modality (image and text in detailed scale) contrastive learning. Two public datasets (EMBED and RSNA-Mammo) are adopted for the comparison study. Superior results of the proposed method are reported in comparison to previous VLMs for nature images and texts. The manuscript is overall easy to follow. However, it also suffers from critical flaws as listed below.

### Strengths
- The authors attempt to build a pre-training model for mammography using multi-view and multi-scale information in images. 
- The manuscript is easy to follow overall.

### Weaknesses
- The presented work is limited in technical innovation. The idea of aligning detailed textual descriptions with the image regions has been commonly investigated in training the model for medical image analysis tasks, e.g., in attributes/image alignment for chest X-rays. Additionally, enforcing the contrastive learning between images (multi-views) and between images and texts are also pretty standard proxy tasks. 

- The motivation for performing the structured report construction is unclear. If all the words except the key information are the same for all reports, what's the point of including them, which is meaningless to learn during the training? Instead, why not input the key information directly, e.g., as a list of keywords for each case?

- linear classification -> linear probing? The results of linear probing are missing in Tables 1 and 2? and missing zero-shot results in Table 3. The selectively reported results make the conclusion less convincing.

- The linear probing actually achieve the best results as shown in Table 3 and 4, which is suprisingly higher than the full fine-tuning. It will be helpful if the author could provide some insight on this.

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a multi-view and multi-scale alignment method for contrastive language-image pre-training in mammography. Based on the characteristic that a single patient has multiple images in mammography data, the framework performs alignment for both image-image and image-text for the same patient simultaneously. This method has been evaluated on the two public datasets EMBED and RSNA-Mammo, surpassing some classic vision-language methods, such as CLIP and SLIP.

### Strengths
This paper explores contrastive language-image pre-training in the field of mammography, highlighting the potential of vision-language pre-training in this area.

### Weaknesses
1. The multi-view and multi-scale alignment method proposed in the article offers only incremental contributions compared to existing contrastive language-image pre-training frameworks, providing limited insights for future developments in this field. 

2. The literature review on related work is lacking, and the experimental comparisons are insufficient. The comparison results mainly focus on categories based on ViT variants, specifically comparing with DeiT and DiNOv2-based ViT. Clearly, this paper is focused on the vision-language domain rather than the vision transformer architecture, and it should compare with the latest methods in vision-language. A shortcoming is that the paper only compares against methods like CLIP, ConVIRT, and MGCA, all of which are from before 2022. It is well-known that this field is evolving rapidly, and the paper fails to include more recent vision-language methods from the past one or two years.

3. The similarity score includes a lot of additional information beyond the diagnosis in the images and reports, such as patient information and image information. Although a meta-info random masking training strategy is employed, it cannot guarantee that the model will not be biased towards irrelevant information during the local alignment process.

4. The paper states, “incorporate parameter-efficient fine-tuning (PEFT) of a pre-trained large language model (LLM) as our text encoder.” However, to my knowledge, BioMedLM is a decoder-based transformer, and the article does not explain how it is utilized as a text encoder.

5. Mammography images are generally of high resolution, such as 4096x4096. Resizing the images directly to 518x518 will result in a significant loss of detailed information, which seems inconsistent with clinical requirements.

### Questions
1. The EMBED dataset is very large and provides a rich set of assessment categories. Can we use the model trained on EMBED to directly evaluate it on RSNA-Mammo? I wonder if it is feasible to assess only cancer vs. non-cancer. 

2. Could you explain how to use the BioMedLM, a decoder-based transformer, as a text encoder?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a contrastive language-image pre-training framework that utilizes the multi-view images in mammography. MaMA  performs contrastive learning between the two views of the same study, and alignment between each view to the corresponding metadata formulated as text, in the same time. Additioanlly, symmetric local alignment module is proposed to enhance model focus on small regions of interest. This approach is validated on large datasets, EMBED and RSNA-Mammo, and demonstrates promising results.

### Strengths
1. The proposed SLA module is interesting and shown to be effective.
2. MaMA is shown to be effective in utilizing the multi-view images in mamography.

### Weaknesses
Although the proposed method outperforms other baselines, none of the baseline is mammography-specific. There is lack of comparison with domain-specific models (uni-modal trained on images or multi-modal trained in CLIP-style). How does the model compare to a dinov2 model trained on images only in linear probe and finetuning, on both internal and external classification tasks?

### Questions
1. I'm confused about how did you do zero-shot for vision-only models (e.g. ViT initialized with DINOv2 model weights)? in Baselines, line 406 "We pre-train and fine-tune all these baselines with the same settings as our model.", and in line 409, "All the baseline methods use fully fine-tuned BioClinicalBERT (Alsentzer et al., 2019) as text encoder." - did you just use the vision encoder in these respective baselines as the vision encoder in your framework and train with you setting? E.g. use ViT, DINOv2 initialized ViT, vision encoder in the original CLIP, vision encoder in SLIP, etc. as the vision encoder, and then finetune?
2. Did you try to use encoder for tabular data to encode mammography metadata instead of phrasing them to text?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper provides a multimodality textual-visual contrastive learning framework in mammography, leveraging multi-view image in global level image-report matching and local level multi-scale patch-sentence matching. The proposed method is evaluated on two public datasets and get leading results in downstream tasks.

### Strengths
- The code and public datasets are provided for reproducibility. 
- The datasets covers large data size and diverse tasks. 
- The proposed framework is compared with many other leading methods. The results show leading downstream linear classification performance and zero-shot and full finetuning classification performance.

### Weaknesses
- Main weakness: there are very limited novelty in methodology design. 
	1. The multi-view matching module is basically common multimodality visual-textual representation/contrastive learning design. The pretraining tasks include global level image-to-image task due to mammography dataset provides multiple views of the same case, and global level image-to-text task where report is matched to each image view. These global matching tasks are widely used as pretraining auxiliary tasks in multimodality representation learning models. The only difference, multi-view matching, is not a novel method or task design, but simply because mammography provides multiple image views of each case. 
	2. Meta-info masking is the same as language masking task commonly used as a basic pretraining task in all kinds of NLP/LLM models, which has no novelty. 
	3. The symmetric local alignment (SLA) module is very similar to an attention-based unsupervised text-image matching method called *DAMSM* in paper Attn-Gan [1], which are also already used in other medical tasks such as chest X-ray report representation learning [2]. The authors use the similar local level patch-sentence matching module without citing these papers, and have very limited novelty in method design. 
	4. LLM has been used as text embedding encoder and replaced BERT since 2023, and using PEFT to efficiently finetune LLM is not a new technique anymore. 
- Self-supervised multimodality matching has some problems in medical domain, especially for the chest x-ray image-report and mammography images. Since images and reports of different cases can be very similar to each other in mamography, e.g. multiple cases with tumor showing up at similar image regions, in which case the contrastive loss would not work very well due to the consine similarity may get high similarity scores of multiple image pairs or image-report pairs in the same batch. I would like to know if the authors have solved this issue by any specific batch sampling strategy or have any discussion about this potential limitation. 
- In 3.1, the authors build template-based report using all kinds of meta information and clinical findings, which might have some redundent textual information not shown up in the corresponding images. However, since the paper is pretraining the model using local matching tasks between image patches and report sentenses, there could be some issues in SLA when matching a report sentense that has no semantic visual marker or clue in any image patch. I think using only the clinical findings as report may reduce such *empty matching* issue and let the pretraining converge better. 


[1] Xu, T., et al. "Attngan: Fine-grained text to image generation with attentional generative adversarial networks." CVPR. 2018. 
[2] Ji, Z., et al. "Improving joint learning of chest x-ray and radiology report by word region alignment." MICCAI-MLMI 2021.

### Questions
As above, The paper adapts multimodel visual-textual representation pretraining framework in mammography dataset and results in leading performance in multiple downstream classification datasets. However, the entire method design of the paper is mostly based on the existing methods and has very limited novelty, thus I do not think the paper would bring enough new insights to the multimodel visual-textual representation learning area or medical domain. In addition, some potential issues, such as sentence empty matching and similar cases matching, are not well solved or discussed in this image-report pretraining framework in mammography. Considering both the strengths and weaknesses of the paper, I believe the paper is slightly below the borderline and rate as "borderline reject".

### Soundness
2

### Presentation
3

### Contribution
2
