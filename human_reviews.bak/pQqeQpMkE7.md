# On Scaling Up 3D Gaussian Splatting Training

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
3D Gaussian Splatting (3DGS) is increasingly popular for 3D reconstruction due to its superior visual quality and rendering speed. However, 3DGS training currently occurs on a single GPU, limiting its ability to handle high-resolution and large-scale 3D reconstruction tasks due to memory constraints. We introduce Grendel, a distributed system designed to partition 3DGS parameters and parallelize computation across multiple GPUs. As each Gaussian affects a small, dynamic subset of rendered pixels, Grendel employs sparse all-to-all communication to transfer the necessary Gaussians to pixel partitions and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training with multiple views. We explore various optimization hyperparameter scaling strategies and find that a simple sqrt(batch-size) scaling rule is highly effective. Evaluations using large-scale, high-resolution scenes show that Grendel enhances rendering quality by scaling up 3DGS parameters across multiple GPUs. On the 4K ``Rubble'' dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPU, compared to a PSNR of 26.28 using 11.2 million Gaussians on a single GPU. Grendel is an open-source project available at: https://github.com/nyu-systems/Grendel-GS

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces GrendelGS, which eliminates the bottleneck of 3DGS in parallel training and scaling up. GrendelGS mixes Gaussian-wise and pixel-wise parallelism, so as to fit computation requirements of different stages. It also makes an effort to minimize communication among GPUs and workload imbalance. And it reveals hyperparameter scaling law for 3DGS parallel training.

### Strengths
1. Originality.  A reasonable parallel training strategy with novel modification for 3DGS based on valuable insights.
2. Quality. Sufficient experimental validation on effectiveness of the method.
3. Significance. Pioneering and well-optimized parallel training implementation for 3DGS, resolving critical challenges in synchronized parallel 3DGS training. On the one hand, GrendelGS avoids tedious model pertaining or data and primitive partitioning in prior works like VastGaussian or CityGaussian. On the other hand, this work enables efficient training when multiple GPUs are available, especially under instant requirements or large-scale scenes.

### Weaknesses
1. While the superior efficiency over the original 3DGS is clearly and sufficiently illustrated through experiments, it would be beneficial to show if this synchronized parallel training mechanism shows higher efficiency over existing open-sourced asynchronized paradigms, such as HierarchyGS, OctreeGS, or CityGS. Note that for a fair comparison, the batch size and overall iterations should be aligned. Considering the limited time for rebuttal, a comparison with one existing method on 2-3 representative large-scale scenes is enough.
2. The Independent Gradients Hypothesis is introduced to support the hyperparameter scaling law. It's reasonable that the hypothesis is true on Rubble or even other large-scale scenes, since the views are extremely sparse and approximately independent. However, for condensed views over small scenes or single objects, the hypothesis may not well suit the reality, especially when the batch size grows.  It seems more reasonable to limit the hypothesis to cases with significant view sparsity, or check if the scaling law brings worse results on small scenes.

### Questions
1. The legends in Figure 1 are put in inversed order. A modification seems to be required.
2. In Algorithm 1, the meaning of ET_j is confusing and I didn't find a clear definition. An annotation in the graph can optimize the reading experience.
3. In the "Independent Gradients Hypothesis" section, why can we assume E(g_k)=0 over all views? This hypothesis is not that obvious.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Grendel, a parallelized computation system for 3D Gaussian Splatting. Grendel introduces two parallelization strategies designed for different stages of the 3DGS rendering pipeline: partition by Gaussians and partition by pixels. The partition by Gaussians method distributes Gaussians within the point cloud across multiple GPUs, while partition by pixels assigns pixels across GPUs. Communication between these parallelization schemes is conducted through NCCL all2all. To enhance batch optimization, a learning rate scaling rule is proposed. Experiments demonstrate performance improvements through the utilization of more Gaussians compared to 3DGS approaches in Large scene datasets.

### Strengths
1. Proposal of parallelization with respect to different stages of the rendering pipeline of 3DGS is interesting.

2. With the proposed method, training 3DGS on high-resolution images is possible.

3. Batch processing can significantly increase the training speed.

### Weaknesses
1. This paper lacks a conclusion and limitations, making it difficult to understand its overall contribution contribution and appear incomplete.

2. Similar to the proposed learning rate scaling, RAdam [1] also suggests a method for variance rectification. It also rectifies the learning rate using variance to enable adaptive learning and, I believe, takes batch size into account. However, the advantage of the proposed method remains unclear.

3. The rendering speed is expected to be slow due to the time required for Gaussian transfers between GPUs, however, the paper does not provide details regarding rendering speed.

4. Based on Fig. 14, it appears that increasing batch size does not decrease the maximum number of Gaussians significantly.

5. As shown in Fig. 11, for datasets that are not high-resolution like Rubble, the performance gain becomes marginal even if the number of Gaussians exceeds what can be handled by a single GPU.

[1] Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond*, ICLR 2020

### Questions
1. How much does the rendering speed change?

2. By how much does the actual GPU memory requirement per GPU reduce when using batch processing?

3. What is the storage requirement based on the number of Gaussians?

$~$

Minor comments:

The figure reference is inconsistent in line 173.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a parallel training pipeline for 3D Gaussian Splatting to be trained on up to 32 GPUs, instead of 1 GPU only. This significantly improves the limit of 3DGS in terms of primitive count, scene scale, scene detail, and training speed. This paper splits the 3DGS training process into per-Gaussian stage and per-pixel stage. In per-Gaussian stage, each GPU has its own Gaussians to calculate the Gaussian transformations, mostly 3D to 2D transformation and color evaluation. In per-pixel stage, each GPU is in charge of a set of pixels for rendering, and performs a sparse communication to fetch the required Gaussians from other GPUs. Lastly, a backward propagation is carried out in a similar manner to pass back the gradients. As a result, the proposed method demonstrates a significantly higher number of Gaussians supported, much faster training speed, and a moderate increase in rendering quality.

### Strengths
**Motivation**
* This paper has a strong motivation of expanding the training of 3DGS into multiple GPUs/machines. The vanilla 3DGS has a limited number of primitives supported due to the GPU memory limit, but the exceptional performance of 3DGS indicates its great usefulness in large-scale reconstruction. This paper is monumental in providing support in terms of parallel training.

**Method**
* This paper correctly identifies the per-Gaussian and per-pixel stages of the training pipeline of 3DGS. Since it split the set of Gaussians onto each GPU, the first per-Gaussian stage is relatively simple. For the per-pixel stage, the paper provides intuitive and empirical support for the locality property of 3DGS for each pixel. The paper cleverly leverages this property to design a sparse communication to transport the 3DGS across GPU at a low speed and bandwidth sacrifice. This enables the distributed training of 3DGS.
* This paper rigorously analyzed the scaling property of the 3DGS training process, and carefully designed a hyper-parameter setting paradigm based on the independent gradient assumption and empirical support. 
* To further improve the utilization of all GPUs, the load balancing algorithm is introduced to ensure each GPU can adaptively adjust their workload during training, because the number of Gaussians required for each region of the image cannot be predetermined.

**Experiments**
* This paper conducts extensive experiments across various scales of scenes under different experiment settings. The experiment result are well presented to demonstrate the improvement in primitive count, training speed, and rendering quality.

### Weaknesses
**Method**
* The paper follows the vanilla 3DGS design and does not consider the Level of Detail support. Since this paper focuses on the large-scale reconstruction, LOD is essential for downstream applications. It is not clear how the proposed method can support LOD.

**Experiment**
* The quality improvement is not as significant as the number of primitives or the training speed. However, I would like to argue that this is because of the lack of an appropriate dataset to evaluate a very large-scale reconstruction, which limit the performance of this paper. 
* Although the training speed improvement is very significant with this parallel training, large scenes require very time-consuming pose estimation to be done first. The overall training time including the pose estimation might not be improving as much as the reported improvement. This is not the problem of this paper, but does weaken the impact of this paper.

### Questions
1. Can the proposed method be applied to 3DGS with LOD design?
2. How does the training across different machines instead of different GPUs affect the performance of the proposed paper?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper, named Grendel, presents a system to enable the distributed training of 3DGS. By leveraging the spatial locality and discovering the Gaussian intersection on image pixels, Grendel enables transferring Gaussians with sparse all-to-all communication while balancing the communication overloads. The load balancing can be simply adjusted according to the time estimation of rendering a pixel in previous iterations. While increasing the batch size needs adjusting the hyperparameters for optimizers, Grendel proposed an automatic hyperparameter scaling rule to enable the hyperparameter-tuning-free batched 3DGS training. Experiments conducted on the small-scale Mip-NeRF360, Tansk-and-Temples, and DeepBlending dataset, the large-scale Mega-NeRF dataset, and the large-scale MatrixCity data validated the effectiveness of the proposed method.

### Strengths
(1) This work enables the distributed training of 3DGS, which can be very useful to accelerate the training of 3DGS.

(2) The paper is well-written and easy to understand.

(2) This work did exhaustive experiments on very large-scale datasets (e.g., the MatrixCity dataset)

### Weaknesses
(1) The related works are put into the last section of the paper, which is weird to me. It lacks discussion of some existing distributed methods, such as **DOGS** (Chen and Lee, NeurIPS 2024), and RetinaGS(Li, et, al, arXiv 2024).

(2) In Sec.6 Related Works, The discussion of VastGaussian, CityGaussian, and Hierarchical Gaussian is wrong. They do not merge the resulting images. Instead, they merge the sub-models. And the authors also claimed that "None of these systems can consider a full large-scale scene directly and achieve the same quality as ours". This is ***over-claimed and does not respect existing works since they did not compare to any of these methods*** but only tried different configurations of their Grendel methods.

(3) Baselines are missing in the current version. It is a good system work to me. However, over-claiming something in the paper won't make it a better work. I would like to see a fair comparison of the method to more baselines (original 3DGS, VastGaussian, CityGaussian, DOGS, etc.) and raise my score if they do so.

### Questions
(1) At line 206, the method splits an image into 16x16 pixel blocks. I'm curious whether the patch size of image blocks can influence the performance of image rendering quality.

(2) Since this method enables training 3DGS from multiple views, it can gather gradients from multiple views and then better constrain the training of 3DGS (as has been shown in the paper *MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis*). However, I saw no performance gains using the distributed training. In lines 312-323, the paper claimed that "even though some cameras share similar poses, a set of random cameras generally observe different parts of a scene, hence the gradients in a batch are mostly sparse and can be thought of as roughly independent.". I think that might be the reason that the distributed training does not improve the rendering quality. Can the authors try to explain/discuss this further? For example, are there any non-random batch-splitting methods that can be applied to improve the performance?

### Soundness
3

### Presentation
4

### Contribution
4
