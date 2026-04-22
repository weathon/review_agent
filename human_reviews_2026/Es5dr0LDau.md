# ReText: Text Boosts Generalization in Image-Based Person Re-identification

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 0, 4, 4

## Abstract
Generalizable image-based person re-identification (Re-ID) aims to recognize individuals across cameras in unseen domains without retraining. While multiple existing approaches address the domain gap through complex architectures, recent findings indicate that better generalization can be achieved by stylistically diverse single-camera data. Although this data is easy to collect, it lacks complexity due to minimal cross-view variation. We propose ReText, a novel method trained on a mixture of multi-camera Re-ID data and single-camera data, where the latter is complemented by textual descriptions to enrich semantic cues. During training, ReText jointly optimizes three tasks: (1) Re-ID on multi-camera data, (2) image-text matching, and (3) image reconstruction guided by text on single-camera data. Experiments demonstrate that ReText achieves strong generalization and significantly outperforms state-of-the-art methods on cross-domain Re-ID benchmarks. To the best of our knowledge, this is the first work to explore multimodal joint learning on a mixture of multi-camera and single-camera data in image-based person Re-ID. Code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents ReText, a method for generalizable person re-identification (Re-ID) to recognize individuals across cameras in unseen domains without retraining. It combines multi-camera Re-ID data with single-camera data enhanced by textual descriptions. ReText trains on three tasks: Re-ID, image-text matching, and text-guided image reconstruction. Experiments show it outperforms limited existing methods on cross-domain Re-ID benchmarks.

### Strengths
1. ReText introduces a pioneering framework by integrating multi-camera Re-ID data with single-camera data enriched by textual descriptions. The joint optimization of three tasks, Re-ID, image-text matching, and text-guided image reconstruction, is a fresh perspective in the Re-ID domain, leveraging multimodal learning to enhance generalization.
2. By complementing single-camera data with textual descriptions, ReText addresses the limitation of minimal cross-view variation in such data, potentially enriching semantic cues and improving robustness across unseen domains.
3. The method demonstrates significant improvements over state-of-the-art (SOTA) methods on cross-domain Re-ID benchmarks, highlighting its potential to handle domain gaps effectively.
4. Public Code Availability: The commitment to releasing code publicly enhances reproducibility and encourages further research in this direction.

### Weaknesses
1. Missing Common Evaluation Settings: While ReText compares against some SOTA methods, it lacks results for standard settings, such as fully supervised performance after downstream fine-tuning on target domains. This omission makes it difficult to fully assess the method’s practical utility and robustness compared to established Re-ID approaches. 

2. Limited Novelty Perception: The proposed method bears similarities to existing frameworks like Masked Autoencoders (MAE) and CLIP, which may reduce its perceived innovation. The authors need to better articulate the unique aspects of their joint optimization strategy to differentiate it from these prior works.

3. Lack of Generated Data Comparisons: The study does not compare ReText’s approach with other generated data methods, such as MALS or LUPerson-NL only, limiting the ability to evaluate the effectiveness of its textual augmentation strategy relative to existing data augmentation techniques.

4. Insufficient Clarity on Experimental Settings: Key experimental details, such as specific training configurations or evaluation protocols, are relegated to the supplementary material. This placement risks misinterpretation of the method’s performance and reduces the main text’s clarity and impact.

### Questions
1. Downstream Fine-Tuning Results: Why were fully supervised results after downstream fine-tuning on target domains not included? Could the authors provide these results to better contextualize ReText’s performance against SOTA methods in standard Re-ID settings?

2. Novelty Differentiation: How does ReText’s joint optimization of Re-ID, image-text matching, and text-guided reconstruction differ fundamentally from combining MAE and CLIP-like frameworks? Can the authors clarify the specific innovations in their approach?

3. Comparison with Generated Data: Why were comparisons with prior generated data approaches (e.g., MALS, LUPerson-caption only) not included? How does ReText’s textual description strategy compare to these methods in terms of semantic enrichment and generalization?

4. Placement of Experimental Details: Why were critical experimental settings (protocol) placed in the supplementary material rather than the main text? Would moving these details to the main paper improve clarity and prevent potential misinterpretation of the results?

5. Task Interaction Mechanisms: How are the three tasks balanced during joint optimization? Are there specific weighting strategies or loss functions that ensure effective learning, and how do they impact the overall generalization performance?

6. The proposed Structure-preserving Loss is still a InfoNCE loss in the image modality;  the identity matching loss is a soft cross-entropy loss, which is quite common for smooth labels. 

7. Missing some general person retrieval methods like All in one framework for multimodal re-identification in the wild (CVPR), Adaptive Uncertainty-Based Learning for Text-Based Person Retrieval (AAAI)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper aims to address the problem of generalizable image-based person re-identification. It points out that existing methods have found incorporating single-camera data to be beneficial. However, single-camera data lacks cross-view variation, which is a natural limitation. Logically, the problem to solve is this very deficiency. The paper then states that “single-camera data is less complex due to limited cross-view variation,” implying that increasing complexity is the key. The question then becomes: how to enhance this complexity? The authors propose that “pairing single-camera data with descriptive captions unlocks rich semantic cues.”
Intuitively, this suggests that the paper should focus on pairing the data with high-quality captions to maximize semantic richness. But, interestingly, the paper quickly points out that such datasets already exist—specifically, SYNTH-PEDES. This raises the question: what exactly does this paper contribute beyond the existence of SYNTH-PEDES?
The authors summarize their contribution as simultaneously addressing two major challenges: first, the scarcity of multi-camera data, which they mitigate by leveraging stylistically diverse single-camera data; and second, the underutilization of natural language supervision. By effectively incorporating the pre-existing SYNTH-PEDES dataset, the paper’s core work lies in designing losses and modules to fully exploit this resource.
This approach proves successful, as evidenced by their experimental results showing that ReText achieves very strong performance.

### Strengths
[+] GOOD performance.

### Weaknesses
[-] The paper offers almost no novelty. The losses and modules employed are off-the-shelf, and the dataset used is already publicly available.
[-] The motivation is somewhat confused. As the summary states, the paper argues that “single-camera data is less complex due to limited cross-view variation.” But why does the lack of cross-view necessarily imply less complexity? Take LUPerson, for example—a dataset collected from numerous videos across diverse scenes. Surely, that is complex data.
I agree that pairing images with captions can help explore additional complexity. However, if captioning data is so beneficial, why is multi-camera data left without captions? It could just as well be captioned. Adding captions to multi-camera data would: (1) increase data distribution complexity and improve generalization, and (2) simplify the model architecture by treating single-camera and multi-camera data uniformly, merging them into one dataset and using a shared pipeline.
Ultimately, the paper’s reliance on single-camera data with captions stems simply from the fact that multi-camera datasets lack ready-made captions. Therefore, the paper does not contribute anything new regarding dataset construction or augmentation.
In summary, the so-called challenges the paper claims to tackle have already been addressed by prior works. This paper mainly attempts to integrate existing components in a coherent manner.
[-] Although the experimental results are strong, the conclusions drawn are neither novel nor deep. The claims that “Single-camera data is simple” and “Captions add complexity” have been observed elsewhere. Similarly, conclusions like “Stylistic diversity improves generalization” are well-established in earlier domain generalization research. This paper does not offer particularly insightful perspectives.
[-] The text captions in the SYNTH-PEDES dataset used in this paper are generated by a captioner trained on CUHK and ICFG datasets. Since the images in CUHK and ICFG originate from CUHK and MSMT—datasets that overlap with the target domain—there is a significant risk of data leakage in the experiments presented. This overlap raises concerns about the validity and generalizability of the reported results.

### Questions
See Weaknesses

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
The paper proposes ReText, a method that enhances cross-domain generalization in Re-ID by training on a mix of multi-camera Re-ID data and single-camera data with textual descriptions. It jointly optimizes Re-ID, image-text matching, and text-guided image reconstruction, and achieves favorable experimental results.

### Strengths
1. The figures and tables of this paper are relatively clear, and its writing is easy to understand.

2. It uses mixed training of multi-camera Re-ID data and text-annotated single-camera data, addressing both the scarcity of multi-camera data and the lack of cross-view variation in single-camera data.

3. ReText achieves better experimental results on multiple datasets compared with multiple existing methods.

### Weaknesses
1. Multi-task joint learning is not a novel concept in fields such as person re-identification or person retrieval. Many works employ methods based on generation/reconstruction and image-text matching for joint learning with ReID. The methods used in each task in this paper are mostly the introduction or adaptation of existing technologies; thus, the improvements in framework design are incremental.

2. The authors note in the implementation details that ReText is trained on a hybrid dataset constructed from multi-camera data and single-camera data. This is in conflict with the settings (Protocol 1) presented in Table 5. If ReText is trained using mixed data while other methods adhere to the settings of Protocol 1, such a comparison clearly lacks fairness—we cannot determine whether the performance gain stems from the model itself or the large-scale data.

3. ReMix is the most relevant work to this paper. However, existing experiments only show the comparison between ReText and ReMix under their respective settings, and lack an intuitive and comprehensive comparison. For example, a comparison between ReText and ReMix based on the same data and settings.

4. By combining Table 3 and Table 5, we note that ReText, without introducing other tasks and only using a pure ReID model, has an accuracy that far exceeds all comparative methods. This is quite confusing, because ReText does not make any additional improvements to the ReID model. It adopts the same loss function as ReMix, yet the accuracy of its pure ReID model is 20% higher than that of the complete ReMix. The authors do not analyze the source of these gains.

5. ReID is an application-oriented task. Does the authors' adoption of multi-task joint learning increase the difficulty of actual training, as well as raise computational cost and model complexity? This paper lacks relevant analysis.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
3

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
The paper focuses on image-based person ReID, and proposes ReText, which utilizes both multi-camera ReID and single-camera ReID datasets for generalization. Specifically, ReText includes three losses: ReID Loss on multi-camera data, image-text matching loss on single-camera data, and image reconstruction loss on single-camera data. To further improve the performance, this paper also refines the original CLIP image-text matching loss into Identity-aware Matching loss (by softening) and a Structure-preserving loss to preserve the semantic during image-text matching. Experiments are conducted by combining several multi-camrea datasets and one single-camera dataset (SYNTH-PEDES). Ablations show the effectiveness of the proposed modules.

### Strengths
The description of the proposed method is easy to follow.

### Weaknesses
1. Lack of comparison with some important SOTA methods. (1) Given that the proposed method is trained on a large-scale dataset (see Tab.1, more than 4.7M images), recent large-scale pretraining methods for ReID should be included during comparison. For example, PLIP[1], which only uses SYNTH-PEDES but shows a better performance on Market. (2) Besides, this paper is also related to multimodel multitask training, therefore, Instruct-ReID[2] should also be included, which also achieves great performance on Market.

2. Regarding the experimental section, the comparative descriptions are not clearly presented. It seems that the method is first pretrained on the large-scale dataset (Tab.1), then finetuned on different settings (Tab.5). But I can not confirm.

3. Overclaim on "first work to explore multimodal joint learning on a mixture of multi-camera and single-camera data in image-based personRe-ID".  "multimodal joint learning on a mixture of multi-camera and single-camera data" seems like a subset of multimodal multitask learning Re-ID (Two tasks: traditional image-based Re-ID, image-text Re-ID). There are some existing works like Instruct-ReID, Instruct-ReID++[3]. 

4. Small weakness. The improvement of the image reconstruction task is small, but it includes more computation costs. 

[1] PLIP: Language-Image Pre-training for Person Representation Learning
[2] Instruct-ReID: A Multi-purpose Person Re-identification Task with Instructions
[3] Instruct-reid++: Towards universal purpose instruction-guided person re-identification

### Questions
Please refer to weaknesses. My concerns are mainly about setting (weakness 3), performance (weakness 1) of this paper. If the paper is truly pre-training on  a large-scale dataset (see Tab.1, more than 4.7M images), then finetuned, the shown performance is limited. Additionally, the ablations demonstrate the effectiveness of the proposed modules, allowing the community to have a better choice.

### Soundness
3

### Presentation
2

### Contribution
2
