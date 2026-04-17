# Matting Anything 2:  Towards Video Matting for Anything

- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
Video matting is a crucial task for many applications, but existing methods face significant limitations. They are often domain-specific, focusing primarily on human portraits, and rely on the mask of first frame that is challenging to acquire for transparent or intricate objects like fire or smoke. To address these challenges, we introduce Matting Anything 2 (MAM2), a versatile and robust video matting model that handles diverse objects using flexible user prompts such as points, boxes, or masks. We first propose Promptable Dual-mode Decoder (PDD), an effective structure that simultaneously predicts a segmentation mask and a corresponding high-quality trimap, leveraging trimap-based guidance to improve generalization. To tackle prediction instability for transparent objects across video frames, we further propose a Memory-Separable Siamese (MSS) mechanism. MSS employs a recurrent approach that isolates trimap prediction from potentially interfering mask memory, significantly enhancing temporal consistency. To validate our method's performance on diverse objects, we introduce the Natural Object Video Matting dataset, a new benchmark with substantially greater diversity. Extensive experiments show that MAM2 possesses exceptional matting accuracy and generalization capabilities. We believe MAM2 demonstrates a significant leap forward in creating a video matting method for anything.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces Matting Anything 2 known as MAM2, a prompt driven video matting framework that moves beyond human portrait centric settings by supporting points, boxes, and masks, by stabilizing trimap prediction for transparent and complex objects, and by reducing reliance on a first frame mask. The method adds a Promptable Dual-mode Decoder that predicts a segmentation mask and a trimap in one pass, and a Memory Separable Siamese mechanism that stabilizes trimap decoding for transparent or complex objects across time. The authors also introduce a Natural Video Matting benchmark with diverse non human portrait categories. Results report strong accuracy on both natural objects and human videos, and competitive image matting as well.

### Strengths
1. Clear problem motivation regarding portrait bias and reliance on a first frame mask for selection. The paper explains why transparent targets such as smoke or fire make mask prompting difficult and why lighter prompts like box or points are preferable for such cases.

2. Architectural idea with practical value. PDD extends the SAM2 mask decoder to produce both a mask and a trimap in one pass, and it leverages the strong mask quality of SAM2 as a spatial prior. In practice this yields cleaner boundaries and a more stable unknown band, which improves the final alpha matte.

3. Transparent object failure analysis and remedy. The paper identifies temporal collapse where unknown regions drift to foreground for later frames. MSS addresses this by running a second PDD pass to decode the trimap from memory free features using the first pass mask as a pseudo prompt. Parameters are shared between the passes.

4. New test dataset (Natural Video Matting) for generalization.

### Weaknesses
1. The mathematical specification of the pipeline is insufficient.
The paper lacks a complete equation level description of the forward computations, especially for the interaction between PDD and MSS.

2. Figures do not explain the full system behavior. Figure 2 leaves ambiguity about how segmentation data and matting data are used across training iterations and stages. It is unclear how features from the MSS pathway connect to the PDD pathway, which parts are trainable at each stage, and whether user prompts for MSS and PDD are shared or distinct. The figure should be redrawn to show the end to end pipeline with data sources, feature flow, prompt flow, and trainable versus frozen modules.

3. The boundary between the SAM2 decoder and the authors’ contributions is unclear.
The text and figures do not make it evident what is inherited from the original SAM2 mask decoder and what is newly introduced in this work. The paper should provide a module level figure that clearly labels inherited blocks and newly added blocks, together with explicit annotation of the mask output token, the trimap output token, and all entry points for user prompts.

4. Why the method works is not analyzed.
The paper states that the proposed method fixes the failure cases but gives little analysis of the mechanism. In particular, the paper should explain why the Mask Augment Feature and the preserved Feature without Memory lead to stable trimap prediction, with diagnostic evidence or ablations.

5. Experimental fairness and data accounting are not sufficiently documented.
According to Appendix 1, models appear to be trained on different data sources. The manuscript should quantify the training data for each method in comparable units such as number of images, clips, and frames, and include experiments where all methods are trained on the same data to isolate the effect of the proposed design. Table 8 also indicates a parameter gap of nearly nine times between MAM2 and the strongest baseline. This capacity difference makes it difficult to attribute the gains in MAD and GRAD to the proposed architecture rather than to model size.

### Questions
1. Figure 5 appears to report results only on the Natural Video Matting dataset. Could you add a companion figure that compares MAM2 with baselines on additional public video matting test sets?

2. Table 2 shows MAM2 performance with box and point prompts. In Table 4 these entries appear in faint text, which suggests they may be out of scope or not directly comparable. Could you create a separate table summarizing results for models that use both prompts box and point under the same data and evaluation protocol, reporting MAD and GRAD, so the two prompt setting can be compared fairly with single prompt settings?

3. Could you revise Figure 4 to also display the matted image together with the trimap for the same frames and prompts so readers can directly see how trimap quality translates into the visual result?

4. There are some typographical errors. For example, Figure 7,8,9 caption contains “visualizaiotn” instead of “visualization”. Could you proofread the paper and update all figure texts and captions, ensuring correct spelling, consistent capitalization?

5. Section 3.4 is difficult to read and follow. Could you provide a clearer rewrite during rebuttal?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a generic video matting algorithm for different objects including person, animal, fire, water etc. It proposes a two-branch network to conduct the segmentation and tri-map prediction tasks, respectively. It also proposes a new dataset named Natural Video Matting covering different object categories.

The generic video matting task is challenging, and the authors take an initial step towards it.

### Strengths
-	This work deals with general objects for video matting, which is more advanced than existing work mainly dealing with humans.
-	According to table 2, the proposed method (matting anything2) outperforms existing methods on the proposed new dataset and a human matting dataset.

### Weaknesses
-	The model architecture is largely Segment Anything 2 (SAM2) with some additional components. The box and point prompt capability are directly from SAM2. The novelty of the proposed framework is limited.
-	The evaluation dataset is small for general object categories. It only contains 50 clips.
-	In supplementary section A, it shows the proposed method and existing method uses different datasets for training the model. The proposed method used more image matting data for training, which may pose an unfair comparison with existing methods. Can authors provide a variant that is trained with same data (like what Matanyone used) to give a fair comparison?

### Questions
see above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents Matting Anything 2, a versatile video matting model designed to overcome the domain-specificity (e.g., human-centric) and restrictive first-frame mask requirements of existing methods. The core technical contributions are twofold. First, a Promptable Dual-mode Decoder (PDD) that jointly predicts segmentation masks and high-quality trimaps, leveraging trimap-based guidance for generalization. Second, a Memory-Separable Siamese (MSS) mechanism that recurrently isolates trimap prediction from interfering mask memory, crucially improving temporal consistency for challenging transparent objects. To validate these contributions, the authors introduce the new, diverse Natural Video Matting (NVM) dataset. Experiments demonstrate that MAM2 significantly outperforms state-of-the-art methods on both diverse natural scenes and human portraits, accepting flexible prompts like points or boxes.

### Strengths
1. The paper demonstrates compelling quantitative and qualitative results, significantly outperforming previous state-of-the-art methods.
2. The paper well extends SAM2's promptable, generalist architecture to handle the distinct and more complex task of alpha matting. 
3. The paper is well-written, clearly organized, and easy to follow.

### Weaknesses
1. The name Natural Video Matting is confusing. In matting literature, "natural" typically implies real-world, non-composited videos. Since NVM is synthetic (composited from assets), this name is a misnomer and should be revised to avoid ambiguity.
2. All experiments are conducted exclusively on synthetic (composited) videos. This leaves a significant gap in evaluation, as performance on real-world videos that contain artifacts like complex lighting, sensor noise, and motion blur remains unproven. The matting "anything" claim is therefore not fully substantiated.
3. The paper lacks a dedicated limitations. There is no discussion of potential failure cases.

### Questions
The paper presents an extension of SAM 2 to the matting domain, and the quantitative and qualitative results shown are excellent. My primary question, however, concerns the validation scope. All experiments were conducted on synthetic (composited) videos. This raises a question about the method's true capability to video matting "anything". To fully substantiate the paper's strong claims, I recommend that the authors provide qualitative results (and quantitative results if possible) on real-world video clips, such as those sourced from YouTube, to demonstrate the model's robustness to non-synthetic artifacts.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of generalized video matting beyond human-centric domains. 
It introduces Matting Anything 2, a robust model capable of handling diverse objects, including transparent ones, with flexible user prompts such as points, boxes, or masks. 
The authors propose a Promptable Dual-mode Decoder (PDD) that jointly predicts segmentation masks and trimaps to enhance matting quality and generalization. 
To address temporal instability for transparent objects, a Memory-Separable Siamese (MSS) mechanism is designed.
Extensive experiments on diverse exiting benchmark and the newly proposed Natural Video Matting (NVM) dataset demonstrate that MAM2 achieves state-of-the-art accuracy and strong generalization to diverse, real-world scenes.

### Strengths
1. The paper is overall easy to follow and understand. It provides a clear motivation and explains key ideas effectively with well-designed supporting figures (e.g., Figure 3 and 4).

2. The ablation studies are comprehensive and convincingly demonstrate the effectiveness of each proposed component.

3. The authors conduct extensive evaluations across diverse tasks (image and video matting) and environments, which strongly support the generality and robustness of the proposed model.

### Weaknesses
1. Although the model shows superior performance over baselines across multiple tasks, much of the improvement may stem from leveraging the powerful foundation model SAM2. The proposed method benefits greatly from SAM2’s strong generalization and semantic understanding, whereas most baselines are not based on such advanced foundation models. Therefore, a more in-depth comparison with other SAM1/SAM2-based matting models is essential. For the image matting task, paper [A], which also utilizes SAM, would be a particularly relevant and strong comparison.

2. The Natural Video Matting (NVM) dataset is presented as one of the main contributions, but its description lacks sufficient detail. While brief statistics (in Table 1) and a few visual examples (in the supplementary material) are provided, the paper should offer a more detailed breakdown of the dataset composition—such as domain categories and their relative proportions—especially since it emphasizes dataset diversity as a key feature.

3. Since the primary application of video matting lies in video editing, it would strengthen the paper if the authors demonstrated editing results using the generated alpha mattes, rather than only presenting matte outputs.

[A] ZIM: Zero-Shot Image Matting for Anything, ICCV 2025

### Questions
Minor issues include a few typos (e.g., “iamge” in line 406) and missing citations (e.g., MEMatte in line 203 is mentioned without a proper reference). A careful proofread and citation check would improve the paper’s overall quality.

### Soundness
3

### Presentation
3

### Contribution
3
