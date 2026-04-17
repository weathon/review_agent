# StretchySnake: Flexible SSM Training Unlocks Action Recognition Across Spatio-Temporal Scales

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
State space models (SSMs) have recently emerged as a competitive alternative to transformers in various linguistic and visual tasks. Their linear complexity and hidden-state recurrence make them particularly attractive for modeling long sequences, whereas attention becomes quadratically expensive. However, current training methods for video understanding are tailored towards transformers and fail to fully leverage the unique attributes of SSMs. For example, video models are often trained at a fixed resolution and video length to balance the quadratic scaling of attention cost against performance. Consequently, these models suffer from degraded performance when evaluated on videos with spatial and temporal resolutions unseen during training; a property we call spatio-temporal inflexibility. In the context of action recognition, this severely limits a model's ability to retain performance across both short- and long-form videos.
Therefore, we propose a flexible training method that leverages and improves the inherent adaptability of SSMs. Our method samples videos at varying temporal and spatial resolutions during training and dynamically interpolates model weights to accommodate any spatio-temporal scale. This instills our SSM, which we call {\sc StretchySnake}, with spatio-temporal flexibility and enables it to seamlessly handle videos ranging from short, fine-grained clips to long, complex activities. 
We introduce and compare five different variants of flexible training, and identify the most effective strategy for video SSMs. On $6$ action video benchmarks, {\sc StretchySnake} outperforms vanilla VideoMamba by up to 28\%, while simultaneously delivering 3x speedups and a 90\% reduction in GFLOPs in low-resolution settings. On short-action (UCF-101, HMDB-51) and long-action (COIN, Breakfast) benchmarks, StretchySnake outperforms transformer and SSM baselines alike, with strong adaptability to fine-grained actions (SSV2, Diving-48). Therefore, our method provides a simple drop-in training recipe that makes video SSMs more robust, resolution-agnostic, and efficient across diverse action recognition scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes StretchySnake for solving the spatio-temporal inflexibility, enabling seamlessly handle videos ranging from short, fine-grained clips to long, complex activities. On 6 action video benchmarks, StretchySnake outperforms vanilla VideoMamba by up to 28%, while simultaneously delivering 3$\times$ speedups and a 90% reduction in GFLOPs.

### Strengths
1. The motivation of this work is clear and easy to understand, especially spatio-temporal inflexibility.
2. This work has a detailed theoretical analysis.
3. The workload is full and the experiments are comprehensive.

### Weaknesses
1. From most of works for Mamba improvement, they always focus on global modeling of long sequence and computational complexity. This work seems also focus on this. Therefore, I recommend authors highlight their own design and note the corresponding spatio-temporal inflexibility, enhancing their novelty.
2. The writing of the formula is not standardized. For example, conv, concat, and temp in equation 4,5, and 6 are non-variables. It should be written with normal fonts instead of Italic.
3. I hope to see more implementation details, such as how the dataset is processed. Because the SSv2 dataset is direction-sensitive, using a horizontal flip is not appropriate.
4. Given the different capacities of these datasets, are the averages in Table 2 rigorous? Of course, this is just a friendly discussion and does not affect my judgment of this work.
5. The size of Table A3 is too large and looks a little strange.
6. From several figures of t-SNE, the advantages of the proposed method are not well highlighted.

### Questions
Please refer to the weaknesses. Overall a good paper.

### Soundness
3

### Presentation
3

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
This paper proposes STRETCHYSNAKE, a video SSM model with spatio-temporal flexibility (aka st-flexibility) that can be generalized to various spatial and temporal resolutions as well as different patch sizes with improved video understanding performance. 
The core contribution is the realization of st-flexibility, which is achieved by interpolating the weights of spatial/temporal positional embeddings and convolution weights during training and inference. 
The authors conduct many experiments across different inference configurations and show that st-flexibility is possible and effective in video SSMs.

### Strengths
1. This paper is well written and easy to follow.
2. The figures are clear.
3. The authors conduct sufficient experiments to validate the effectiveness of the proposed st-flexibility in video SSMs.
4. Improved performance compared to its counterparts, i.e., the VideoMamba.

### Weaknesses
My major concern is the **novelty issue**, because there is a problematic statements regarding existing video models in the core motivation:

L86-L88: *(transformers) reliance on learning explicit token-to-token relationships usually constrains them to fixed input sizes, preventing generalization across diverse spatio-temporal scales.*

**Using 2-D bi-cubic interpolation or 1-D linear interpolation on spatial/temporal positional embeddings and convolution weights (L319-L322) is not first proposed by the authors but a widely adopted technique in video transformers.** Actually, interpolating spatial/temporal and convolutional weights at finetuning or inference stages for various spatio-temporal scales has been a standard practice in the past few years. There are several well-known implementations:

1. TimeSformer: see `forward_features` (L249) at https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py. It exactly uses spatial/temporal interpolation (same as the paper proposed) during inference to adapt for various scales.

2. The timm repo: see `_load_weights` (L1091) at https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py. It not only interpolates the weights of the positional embeddings but also the convolution weights, i.e., the `resample_patch_embed` and `resample_abs_pos_embed` functions.

Besides, **the proposed st-flexibility does not really resolve the generability issue because the model is still trapped in the pre-set training scales**, i.e., $R^t=[8,16,32,64]$ and $R^s=[96,128,224,384]$. No experiments are provided to show the performance w/ scales beyond training. For instance, the performance if $R^t=110$ and $R^s=768$.

I believe the above issue is not mean because **frontier video models already reach real generability through Any Resolution techniques, which is totally missing in this paper.** Please see the below papers:

1. Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution: 1D-Rope for vision understanding.

2. LLaVA-Video: Video Instruction Tuning With Synthetic Data: 1D-Rope for any-scale video understanding.

3. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution: 2D-Rope for any-scale video understanding.

4. Qwen2.5-VL Technical Report: 3D-Rope for any-scale video understanding.

Besides, **some important baselines that also pre-trained on Kinetics-400 are missing.** For instance, VideoMAE and VideoMAE-V2. Please conduct a rigorous literature review and include them in Table2.

### Questions
1. Can you show some results when testing at an unseen scale beyond the pre-set ones, e.g., $R^t=110$ and $R^s=768$?
2. Can you discuss the differences between your method and the pre-developed interpolation ones, e.g., in TimeSformer and timm repos?
3. Please include the latest any-resolution works, e.g., NaViT, LLaVA-Video, Qwen2-VL and Qwen2.5-VL and discuss the merits of your approach over theirs.

### Soundness
2

### Presentation
3

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
On the recent progresses of SSMs image classification, video understanding and 3D vision, this paper aims to adapt the recent SSM and training to fine-grained action recognition. It implements and incrementally improves the training on video learning and representation on fine-grained action recognition benchmarks. The purposes of the experiments are to identify good type of st-flexibility of SSM for video representation on fine-grained action videos, demonstrate the improvement over vanilla baseline, and compare to SOTA on limited training dataset. The reported progresses have shown a certain level of extension of existing SSMs on additional tasks.

### Strengths
Investigating to apply recent progresses on SSM to video representation and fine-grained action recognition task.

### Weaknesses
The novelty of the paper is still weak, and the evaluations are not completed and concrete enough to show the significance of the progresses. First, the technical descriptions present the existing approaches and implementation details. Not clear what are novel method, model architecture, or learning function and algorithms beyond existing approaches. This paper presents a few incremental progresses and extension of existing SSMs. May need to focus on deeper study and big jump to show significant progresses. Second, the purposes of the experiments are not clear and strong. To show the optimal type of st-flexibility for SSMs in a concrete background, experiments on a wide range of vision tasks would be convincing. In the paper, only experiment for video retrieval on 4 action recognition datasets, lack formal benchmarking on related research topics. On second experiment, only compared with one baseline model. On the third part, the protocol of benchmarking is unclear. If it wants to focus on performance on OOD datasets of fine-grained action recognition with limited training set, it may need to follow a formal protocol and comparison to SOTA on leaderboard benchmarks.

### Questions
What are the novel model modules or learning approaches beyond existing SSM technology proposed in this paper? What are the formal protocol, training set and evaluation sets used in the previous benchmarking in Table 2?

### Soundness
2

### Presentation
3

### Contribution
2
