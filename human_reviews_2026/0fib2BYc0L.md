# GPT4Scene: Understand 3D Scenes from Videos with Vision-Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
In recent years, 2D Vision-Language Models (VLMs) have made significant strides in image-text understanding tasks. However, their performance in 3D spatial comprehension, which is critical for embodied intelligence, remains limited. Recent advances have leveraged 3D point clouds and multi-view images as inputs, yielding promising results. However, we propose exploring a purely vision-based solution inspired by human perception, which merely relies on visual cues for 3D spatial understanding. This paper empirically investigates the limitations of VLMs in 3D spatial knowledge, revealing that their primary shortcoming lies in the lack of global-local correspondence between the scene and individual frames. To address this, we introduce GPT4Scene, a novel visual prompting paradigm in VLM training and inference that helps build the global-local relationship, significantly improving the 3D spatial understanding of indoor scenes. Specifically, GPT4Scene constructs a Bird's Eye View (BEV) image from the video and marks consistent object IDs across both frames and the BEV image. The model then inputs the concatenated BEV image and video frames with markers. In zero-shot evaluations, GPT4Scene improves performance over closed-source VLMs like GPT-4o. Additionally, we prepare a processed video dataset consisting of 165K text annotation to fine-tune open-source VLMs, achieving state-of-the-art performance on all 3D understanding tasks. Surprisingly, after training with the GPT4Scene paradigm, VLMs consistently improve during inference, even without object marker prompting and BEV image as explicit correspondence. It demonstrates that the proposed paradigm helps VLMs develop an intrinsic ability to understand 3D scenes, which paves the way for a seamless approach to extending VLMs for 3D scene understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a vision-only framework for indoor scene spatial understanding, GPT4Scene. The paper introduces two main innovations: (1) feeding BEV images into the VLM to provide global scene perception, and (2) assigning consistent object-level markers across the BEV view and multiple frames to establish correspondence between global and local observations. Experiments demonstrate that this approach enhances spatial understanding in large-scale VLMs (70B) under zero-shot settings, while smaller models (2B/7B) benefit more from fine-tuning with the proposed dataset.

### Strengths
1. The paper offers an insightful perspective: the key to 3D understanding lies not in explicit 3D geometry but in maintaining global-local consistency. Point clouds merely provide geometric constraints; a VLM can achieve “pseudo-3D understanding” by learning such consistency through BEV and object markers.
2. The method is validated across models of various scales, and a fine-tuning dataset is constructed for smaller models.
3. Comprehensive experiments and ablations are conducted on multiple benchmarks.
4. The paper is well written, with clear figures and extensive supplementary material, making it easy to follow.

### Weaknesses
1. The suitability of BEV as a global representation for indoor scenes is questionable. BEV works well in outdoor settings (e.g., autonomous driving) because height variation is limited, making BEV a near-complete 2D representation. However, in indoor embodied scenarios, the global viewpoint changes dynamically as the robot moves, and BEV may miss critical global information. The authors should provide further justification or relevant experiments.
2. Beyond benchmark performance, the paper would benefit from feature visualization or attention map analysis to offer theoretical insights and interpretability.
3. Please include a comparison of computational overhead across components to help assess the method’s practical feasibility.

### Questions
See Weaknesses.

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
This work proposes GPT4Scene, an approach to improve the performance of vision-language models (VLMs) by incorporating a 3D BEV mask and spatio-temporal object (STO) markers. The authors also introduce ScanAlign, a new dataset that includes these modalities and shows performance gains when used for fine-tuning. Experimental evaluations are conducted on 3D Question Answering, 3D Captioning, and 3D Visual Grounding tasks.

### Strengths
1. Extensive quantitative experiments are provided.
2. A new dataset, ScanAlign, is introduced, containing video frames, BEV images with STO markers, and text annotations.
3. The use of spatio-temporal object markers is a well-motivated and effective idea that has not been explored before and yields improvements, particularly in 3D grounding tasks.

### Weaknesses
1. For 3D instance segmentation, the authors rely on Mask3D, which is pre-trained on ScanNet scenes.
How well do the predicted masks generalize to other 3D environments (e.g., ARKitScenes)?
2. While the performance improvements are appreciated, the BEV images have been explored in prior work. Thus, the main methodological contribution lies in the introduction of STO markers, which somewhat limits the overall novelty of the approach.

### Questions
1. (See Weakness 1) Could the authors provide more details on mask quality and generalization across datasets?
2. Why does only Qwen2.5-VL-7B surpass the baselines in the question answering task? Relatedly, why is the improvement on SQA3D (situated understanding benchmark) marginal?
3. The proposed dataset contains data exclusively from ScanNet. Would fine-tuning on more diverse 3D environments further enhance model performance?

### Soundness
3

### Presentation
3

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
The paper proposes a framework called GPT4Scene to enhance the spatial understanding of VLMs. Specifically given a video, the framework reconstructs it and then projects it to a BEV image. Then, it uses an off the shelf 3D instance mask segmentation network and project its predictions to the 2d frames and the BEV map, thus marking them. This becomes the input to the VLM. The paper first shows that with this input, bigger VLMs improve performance while smaller VLMs do not benefit much or slightly decrease in performance. Then, the paper proposes to fine-tune the VLMs with such marked representation of videos, and evaluates the models’ performance on 3D VQA, captioning and grounding. The paper shows that their model achieves similar or better performance than prior SOTA methods.

### Strengths
- The paper tackles an interesting and important problem of enabling 3D understanding using pre-trained 2D VLMs
- The ablations are thorough and support the main design choices of the paper

### Weaknesses
The paper appears to have several wrong claims or statements, and formatting errors.

- L199-200: “our method… matching or surpassing the Chat-Scene Chat-scene” in Table-1. However, from Table-1, it seems that the version with GPT4Scene never surpasses ChatScene. 
- From the introduction, it appears that a new datasets is being created and released i.e. ScanAlign. However, it is essentially a combination of existing (and popular) 3D grounding and captioning datasets with the addition of STO markers and the BEV images (which is nice). It would be nice to be more clear and explicit about this in introduction and section-2.3.
- The description of experimental results and results themselves are misaligned again in section 3.2: There are phrases like “outperform prior SOTA like chat scene” but the table shows that the SOTA is ROSS3D and the proposed method only matches the performance of this method. Besides, since the paper do not describe any of the main baselines and details like base VLMs or training data used for the baseline, it is hard to figure out if the comparisons are fair. The section claims that the fine-tuned versions significantly improves the performance of untuned baselines VLMs, but the results are missing from the table. 
- In Table-5 task-specific model section, the baselines are significantly old. Current SOTA on ScanRefer is UniVLG (https://arxiv.org/abs/2503.10745) an the authors can check Table-1 of UniVLG for additional recent baselines. Additionally, could the authors say more on how the evaluation is conducted? This benchmarks tests for bounding box predictions — is the proposed method trained to regress bounding boxes?
- The related work section of the paper is moved completely to the appendix and even there it reads like a bunch of citations. I understand the page limitations, but related works is an important section to contextualize the proposed work with the existing literature and highlighting the similarities and differences

The paper is riddled with formatting errors. A few examples:
- L200 / 222: repeated chat scene and scannet words instead of proper citations
- The bold markings in Table-7 look wrong i.e. the bolded numbers are not the best numbers in the table

### Questions
I am not particularly excited about this paper, especially due to several errors in the manuscript -- however, they can probably be fixed in the revision. Answers to the discrepancies pointed out in weakness section might help increase my rating.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces GPT4Scene, a framework designed to enhance the 3D spatial understanding abilities of Vision-Language Models (VLMs) using only visual inputs (video). The approach constructs Bird’s Eye View (BEV) images from egocentric video and overlays spatial-temporal object markers (STO-markers) to align global scene layout with local observations, enabling robust 3D comprehension without reliance on explicit 3D point cloud data. The authors provide both zero-shot and fine-tuning strategies, introduce a large aligned dataset (ScanAlign), and empirically demonstrate performance gains across multiple 3D reasoning, captioning, and grounding benchmarks, including comprehensive ablations and qualitative analyses.

### Strengths
1.  I think this paper is well motivated and has a clear conceptual advance over prior VLMs for 3D. Unlike previous methods that rely heavily on point clouds, the paper proposes a vision-only approach for 3D spatial reasoning, closely mimicking human perceptual processes for scene understanding.
2. The empirical results is solid. Across diverse 3D benchmarks (Tables 3–7, and full results in tables 12–16), GPT4Scene models consistently set new state-of-the-art or strongly outperform both point-based and previous vision-language SOTA methods in 3D question answering, dense captioning, and visual grounding.
3. The ScanAlign dataset is a valuable resource, which represents a practical resource with 165K aligned video–BEV–text triplets, supporting reproducibility, further research, and downstream benchmarking.
4. The proposed method is scalable and architecture-agnostic. It does not require architectural changes to VLMs and shows improvements to both large, closed-source and smaller, open-source models.

### Weaknesses
1. Since the pipeline is dependent on 3D reconstruction and instance segmentation, I have a concern about its robustness. The method assumes reliable 3D scene reconstruction and high-quality mask annotations for generating BEV images and STO-markers. While Table 7 ablation demonstrates some robustness, the reliance on Mask3D and BundleFusion or similar systems (Figure 2) acts as a performance bottleneck. Real-world deployment may face significant degradation under varied lighting, occlusions, or sensor/calibration noise.
2. Also,  the paper lacks of exploration of failure cases or qualitative weaknesses. The paper presents strongly positive qualitative and quantitative results (Figures 7–10, Table 3–7), but is light on concrete instances where GPT4Scene underperforms (e.g., when scenes are highly cluttered, objects are partially observed, or markers are mismatched). This restricts understanding of limitations and generalization bounds.
3. While Table 6 ablation finds that fine-tuning with BEV/STO-markers enables "intrinsic" 3D scene understanding (i.e., persists even if markers and BEV are not presented at inference), the paper does not fully analyze why the VLMs develop this transfer capability, nor does it probe what types of spatial relationships/generalizations are learned vs. memorized. Is this effect robust to entirely novel scenes or objects unseen in the BEV/marker format at train time?
4. Despite advocating a vision-only solution, the proposed method requires explicit 3D information (scene geometry, camera intrinsics/extrinsics, and point cloud segmentation) for preprocessing. While the input to the VLM is images, the pipeline uses classic 3D processing to derive BEV and object correspondences. This undermines the claim that the system "relies solely on vision" in practice, and should be explicitly discussed.

### Questions
Refer to the questions above.

### Soundness
2

### Presentation
2

### Contribution
3
