# LEO-VL: Efficient Scene Representation for Scalable 3D Vision-Language Learning

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Developing vision-language models (VLMs) capable of understanding 3D scenes has been a longstanding goal in the 3D-VL community. Despite recent progress, 3D VLMs still fall short of their 2D counterparts in capability and robustness. A key bottleneck is that current scene representations struggle to balance performance and efficiency: competitive performance comes at the cost of heavy token overhead, which in turn hampers the scalability of 3D-VL learning. To address this, we propose the condensed feature grid (CFG), an efficient scene representation featuring significantly reduced token overhead and strong perception capability. Building on CFG, we introduce LEO-VL, a 3D VLM trained on 700k 3D-VL data spanning four real-world indoor domains and five tasks such as captioning and dialogue. To enhance the robustness of 3D VLM, we further propose SceneDPO for post-training, which involves contrasts across answers and scenes. LEO-VL achieves state-of-the-art performance on various 3D QA benchmarks, including SQA3D, MSQA, and Beacon3D. Our extensive experiments highlight the efficiency of our representation, the benefit of task and scene diversity, consistent scaling effects, and the advantages of SceneDPO compared to SFT and GRPO. We hope our findings advance the efficiency, scalability, and robustness of future 3D VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a 3D Vision-Language Model built upon a condensed feature grid for efficient scene representation. The authors further construct a large-scale 3D Vision-Language instruction-tuning dataset to validate the scalability of their method, along with a reinforcement learning  post-training approach. The proposed method achieves state-of-the-art performance across various 3D understanding benchmarks.

### Strengths
1. The paper introduces new datasets for 3D understanding with detailed ablation studies and demonstrates strong scalability.
2. The proposed method achieves leading performance across multiple 3D understanding benchmarks.
3. The idea of post-training a 3D LLM with reinforcement learning is promising.

### Weaknesses
1. **Computation overhead:** The idea of projecting image features from 2D foundation models is not novel—3D-LLM has already explored this approach. Compared with image-based 3D LLMs such as LLaVA-3D, the back-projection and voxelization steps introduce additional computation. Although the scene encoding reduces tokens from 3,096 (LLaVA-3D) to 750, stricter comparisons in FLOPs, computation time, and memory usage are needed to validate the claimed efficiency.
2. **Dataset advantage:** Given the significantly larger 3D annotation dataset used, it remains unclear whether the improvement stems mainly from the enhanced 3D encoding or the dataset scale. No controlled ablation with a base model (e.g., LEO) is provided to isolate the dataset’s contribution.
3. **RL method analysis:** The RL approach introduces two additional loss terms with extra hyperparameter tuning, yet no ablation studies are provided to demonstrate the necessity of each term. Is the model sensitive to the hyperparameters mentioned in Lines 456–457?

### Questions
1. **DPO pair construction:** How are the DPO pairs constructed—by random sampling from all responses or from a pretrained model?
2. **GRPO results:** Why does GRPO achieve worse in-domain performance compared to the base model while improving generalization?
3. **Memory efficiency:** Does processing and storing the feature grid require additional memory, and if so, how does it affect parallel training performance?

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
4

### Summary
The paper introduces LEO-VL, a 3D vision-language model that leverages a novel, efficient scene representation called condensed feature grid (CFG) to reduce token overhead without sacrificing perception performance. Trained on 700k 3D-VL samples across diverse domains and tasks, LEO-VL incorporates SceneDPO, a contrastive post-training method to further enhance robustness. The model achieves state-of-the-art results on several 3D question answering benchmarks, demonstrating improved efficiency, scalability, and robustness over prior approaches.

### Strengths
1. The overall writing of the paper is clear and easy to follow.

2. The proposed token reduction method achieves a good trade-off between performance and efficiency.

3. The authors conduct comprehensive experiments involving various post-training methods and effectively demonstrate the effectiveness of SceneDPO.

### Weaknesses
1. Recent relevant works, such as VG-LLM [1] and 3DRS [2], have not been cited or compared in the paper. Including a discussion or comparison with these approaches would strengthen the related work section and contextualize the contributions of this work.

2. The CFG method compresses height information, which may cause the model to lose fine-grained details of layered objects (e.g., a pillow on a bed or a pot on a table). This limitation could impact the model's ability to accurately represent and distinguish between objects in complex scenes. Furthermore, the proposed token compression method might be incompatible with 3D localization tasks, since it leads to a loss of detailed spatial information.

3. The GRPO algorithm used for comparison is not fully implemented, as it lacks both the code-start stage and the format reward component. Additionally, the paper does not clearly explain how rewards are designed for non-verifiable tasks, such as captioning. 

References:

[1] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors. NeurIPS 2025.

[2] MLLMs Need 3D-Aware Representation Supervision for Scene Understanding. NeurIPS 2025.

### Questions
1. Please properly include recent papers related to this work.

2. Please elaborate how the GRPO algorithm is designed in this paper.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses the performance vs. efficiency challenge in 3D Vision-Language Models (VLMs). Current methods often have high token costs, which hinders scalable training. This paper proposes LEO-VL, a VLM using a "Condensed Feature Grid" (CFG). The CFG is created by back-projecting 2D features from multi-view RGB-D inputs into 3D voxels, encoding height with RoPE, and then condensing all voxels in a vertical pillar into a single token.

This CFG representation reduces token count (to 750) while retaining spatial information, which allows for larger-scale training. The authors use a 700k sample dataset, noting that data quality is more beneficial than naively scaling with low-quality data. The paper also introduces SceneDPO, a post-training objective to improve model robustness by contrasting both answers and scenes. LEO-VL achieves strong results on 3D QA benchmarks (SQA3D, MSQA, Beacon3D) with high efficiency.

### Strengths
1. The Condensed Feature Grid is an elegant and effective contribution. It achieves a ~3x token reduction (33% compression rate) compared to the raw voxel grid, appearing to improve performance by preserving spatial information. 
2. The analysis in Section 4.3 demonstrating that scaling with 669k of simplistic, low-quality QA data degrades performance is an important contribution. This focus on "quality over scale" is a useful finding for the field.
3. The model achieves new SOTA performance on multiple challenging 3D QA benchmarks, validating the effectiveness of the overall approach.

### Weaknesses
1. While the model is trained on five tasks, the primary evaluation is almost exclusively on QA. The model's proficiency at other tasks (grounding, captioning) is not as rigorously benchmarked.
2. The vertical condensation (pooling) inherently loses information about the vertical distribution of features within a single (x, y) pillar. The paper argues RoPE encoding of height mitigates this, but there is no analysis of failure cases. For example, it's unclear if the model can distinguish a cup on a table from a lamp above the table if they fall in the same pillar.
3. The work is entirely focused on indoor scenes. The scalability and effectiveness of the CFG representation for more complex, large-scale outdoor environments remain unexplored.

### Questions
1. Regarding the vertical condensation in CFG: Could you provide a qualitative failure analysis? Are there specific types of spatial questions (e.g., "What is directly on top of object A?" vs. "What is above object A?") where the pooling of an entire vertical pillar into one token causes the model to fail? How well does the RoPE height embedding handle ambiguity for multiple, distinct objects stacked vertically?
2. The "quality over scale" finding is excellent. The paper uses "top-15 answer template occupancy" as a metric for low quality. Could you elaborate on the curation process for the final 700k dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes a 3D vision-language model (3D-VLM) called LEO-VL. To reduce the number of tokens representing a scene, it introduces a method named CFG, which back-projects 2D image features into a voxelized 3D space and pools voxel features along the height dimension. The authors also curate existing datasets and generate new samples to construct a 700K 3D-VL instruction-following dataset. Finally, they introduce SceneDPO as an effective post-training objective for 3D-VLMs. Experimental results demonstrate that LEO-VL achieves state-of-the-art performance on several existing benchmarks.

### Strengths
1. The motivation to reduce scene tokens is reasonable and addresses an important problem in the field.
2. LEO-VL achieves state-of-the-art results on several benchmarks.
3. The paper also introduces a new 3D-VL dataset, which adds value to the community.

### Weaknesses
1. My main concern lies in the fairness of the benchmark comparisons. Although LEO-VL achieves state-of-the-art results on several benchmarks, its VLM backbone and training data differ from those of the compared baselines. Therefore, it is difficult to verify whether the proposed CFG is truly more efficient and effective than previous methods.
2. Regarding the evaluation benchmarks, although it is common practice in the 3D-VL field to use image captioning metrics such as CIDEr and machine translation metrics such as BLEU-4 (B-4) for the ScanQA task, I do not think these are reliable indicators of model performance. These metrics can be easily influenced by the length of the ground-truth answers and synonym variations, which reduces their validity for QA tasks.
For example, while LEO-VL outperforms 3D-LLaVA by +9.2 CIDEr points (101.4 vs. 92.6), it actually performs –3.9 points worse on BLEU-4 (13.2 vs. 17.1). Since both ScanQA and SQA3D are essentially question-answering tasks, the evaluation should rely on accuracy—measuring how many questions are answered correctly. This can be easily computed by having an LLM verify whether the model’s output matches the ground truth. I would like to see results reported in terms of accuracy.
3. Figure 3 is a key experiment intended to demonstrate the effectiveness of the proposed CFG. However, presenting results only on SQA3D is not sufficiently convincing. I recommend including results on additional benchmarks, particularly those reporting accuracy-based metrics, to strengthen the evidence.
4. In Table 6, the performance improvement of SceneDPO over SFT on SQA3D is only 0.5 points, which raises doubts about the effectiveness of SceneDPO.
5. From Lines 227–229, the paper states that the 3D object grounding task is excluded simply because it has a different formulation. However, this explanation is not sufficiently justified, especially since the proposed LEO-VL framework should, in principle, be compatible with object grounding as well.

### Questions
/

### Soundness
3

### Presentation
3

### Contribution
3
