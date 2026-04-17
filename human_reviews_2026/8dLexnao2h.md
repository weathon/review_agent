# A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration

- Decision: Accept (Oral)
- Scores: 2, 10, 6, 8

## Abstract
In this work, we propose FFDP, a set of IO-aware non-GEMM fused kernels supplemented with a distributed framework for image registration at unprecedented scales. Image registration is an inverse problem fundamental to biomedical and life sciences, but algorithms have not scaled in tandem with image acquisition capabilities. Our framework complements existing model parallelism techniques proposed for large-scale transformer training by optimizing non-GEMM bottlenecks and enabling convolution-aware tensor sharding. We demonstrate unprecedented capabilities by performing multimodal registration of a 100μm ex-vivo human brain MRI volume at native resolution – an inverse problem more than 570× larger than a standard clinical datum in about a minute using only 8 A6000 GPUs. FFDP accelerates existing state-of-the-art optimization and deep learning registration pipelines by upto 6 − 7× while reducing peak memory consumption by 20 − 59%. Comparative analysis on a 250μm dataset shows that FFDP can fit upto 64× larger problems than existing SOTA on a single GPU, and highlights both the performance and efficiency gains of FFDP compared to SOTA image registration methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces FFDP (Flash Fused Distributed Primitives) as a scalable and distributed framework for large-scale image registration. The main idea is to fuse non-GEMM operations using IO-aware kernels and distributed tensor sharding, enabling registration of multi-billion-voxel images. The proposed framework includes fused kernels for single-GPU efficiency and GridParallel plus RingSampler modules for distributed multi-GPU scalability without full-image allgather operations. Experimental results show that FFDP significantly accelerates registration and reduces memory consumption for both iterative optimization-based and deep learning-based methods.

### Strengths
1. System contribution: 
FFDP elegantly adapts concepts like ring topology and sharded synchronization to a domain with boundary dependencies.

2. Significant scalability results on iterative methods: 
Demonstrates unprecedented registration (11 B + parameters) within realistic compute budgets.

### Weaknesses
1. The introduction part does not demonstrate the motivation for full-resolution registration:
The introduction does not sufficiently justify the necessity of performing registration at full resolution. While image acquisition capabilities have improved, in many biomedical and clinical applications registration on downsampled images is adequate. The authors should provide stronger evidence or references showing that downsampling significantly degrades registration accuracy or downstream analysis.

2. Limited benefit for deep learning pipelines:
The proposed framework provides modest gains (16.5% and 24.7% memory reduction) for deep learning networks comparing with it in iterative methods. It would be helpful if the authors could analyze whether further optimizations could narrow this gap.

3. Combined Introduction and Related Work:
The paper merges the introduction and related work into a single section, which reduces readability and makes it harder to follow the motivation versus prior context. 

4. Poorly organized experiments:
The experimental section is difficult to follow and lacks clear structure.

### Questions
1. Could you provide evidence or references that shows the necessity of full-resolution registration?

2. In Table 1, what do the three “Baseline” rows represent, what do “Top” and “Bottom” refer to, and which dataset was used to generate Table 1? Besides, Table 1 is not referenced in the main paper.

3. Regarding the faux-OASIS dataset experiment, you register the images at a downsampled resolution and perform patchwise registration followed by mosaicing of the final deformation for deep learning methods. This setup may adversely affect their performance, as many hyperparameters proposed in their paper(such as the weighting of the smoothness term) are designed for whole-image registration rather than patchwise processing. Moreover, you did not report the performance of these methods on the original OASIS dataset. If the deep learning methods achieve better performance on the original dataset, it would suggest that your experimental setup is flawed.

4. In Figure 5, you denote “fire-ants with the proposed method” as Ours and compare it with other methods. However, this comparison appears unfair since you did not include the performance of fire-ants itself. It is highly likely that fire-ants alone could outperform many of the other baselines. You should include the performance of the original fire-ants for a fair comparison. If memory limitations are a concern, you may apply the same two modifications used for the deep learning–based methods.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper proposes a set of IO-aware fused kernels and a distributed framework enabling deformable image registration at extremely large scale. It significantly reduces the memory usage for interpolation, and 2 very common loss functions. Experimental results show significant acceleration for both classical iterative optimization and deep learning–based registration pipelines, and demonstrate the first native-resolution multimodal registration at extremely high spatial resolutions.

### Strengths
1. Enables giga-voxel registration in its native resolution that previously was infeasible.
2. Identifies and proposes solutions for major memory bottlenecks speeding up both iterative and learning based registration.
3. Demonstrates performance on multimodal registration of 250μm to 100μm images completing in 1 minute running on 8 gpus.
4. Ablates each components contribution.
5. Benchmarks the performance of registration algorithms on clinical scale and very high resolution brain images.
6. The language of the paper is clear. 
7. I really enjoyed reading the paper.

### Weaknesses
**Weaknesses and questions**

1. I was wondering why the affine transformations is denoted as φ(x)=Ax+t. Isn’t it the case normally that the translation is incorporated in A along with rotation and scaling? 
2. Although the scalability demonstrated on ex-vivo MRI is impressive, it is unclear why the method was not evaluated on multi-gigapixel histopathology datasets, which seem like an ideal target application given their fine-scale structural alignment requirements. Such experiments would directly validate the paper’s stated motivation and better highlight FFDP’s advantages over existing patch-based histology workflows.
3. While the paper emphasizes improvements for multimodal registration, LNCC is widely used for monomodal scenarios too, so the proposed optimizations appear broadly applicable. I am therefore unsure why the authors chose to position their contribution primarily around multimodal registration and not a general purpose registration with more speedup in the multi-modal case.
4. Not sure if I missed it, but are all methods (learning-based and iterative optimization ones) trained with the same objectives?
5. Multi-resolution strategies are commonly used in high-resolution registration to manage memory and computation, and while they suffer from similar limitations as those discussed in the paper, acknowledging them would improve completeness and situate the work more clearly within existing practice.
6. Regarding the learning-based baselines, their encoders are not designed for extremely high-resolution inputs, and while patchification is clearly suboptimal, it seems that alternative architectural designs (e.g., improved downsampling kernels in the feature space, larger capacity models, more spatially aware context handling) might yield stronger performance even if at higher memory cost. It would be helpful for the authors to comment on whether such adaptations were considered and how FFDP compares in those scenarios. I would be interested in the authors’ perspective on whether the limitations observed are inherent to deep approaches at scale or simply due to the baselines being used outside their intended operating regime.
7. No codebase or link to code is available.

### Questions
Please see the section above.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, we propose the FFDP framework, which aims to solve the problem of insufficient algorithmic scalability in ultra-high resolution multimodal image alignment.FFDP designs three I/O-aware non-GEMM fusion kernels (composite implicit grid sampler, implicit Parzen window mutual information estimation, and efficient fusion inter-correlation) as well as distributed architectures (grid parallelism, ring sampler, and distributed loss computation) to realize the fast processing of tasks at the level of hundreds of billions of voxels. Experiments show that FFDP outperforms existing methods on both synthetic and real datasets, increasing the runtime speed by 6-7 times, reducing peak memory by 20-59%, and significantly reducing GPU computational overhead while maintaining higher alignment accuracy (e.g., Dice coefficient of 89.5% at 250µm resolution).

### Strengths
- Motivation is clear: It precisely targets the core contradiction between high-resolution acquisition capabilities and the insufficient scalability of existing algorithms in medical imaging, with real scenarios like 100µm ex-vivo human brain MRI clearly illustrating the necessity.
- Experiments are rich and comprehensively prove the effectiveness of its method: It covers multiple types of datasets and various dimensions of experiments, from component-level ablation to system-level comparison with mainstream baselines, fully demonstrating the method's validity.
- Performance improvement is significant

### Weaknesses
While this paper targets at improving efficiency, it would be better provide experimental evaluations on accuracy against counterparts.

### Questions
Please refer to weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper makes large 3D medical image registration run fast and fit in GPU memory. 
Key idea the reviewer finds: real bottlenecks at giga-voxel scale are not matrix multiplies but ops like grid sampling, LNCC, MI. They write fused, IO-aware CUDA kernels plus a ring sampler so GPUs don’t need full image copies. Result: 6–7× speed, up to ~59% memory saved, and a demo of 11.8B-parameter multimodal brain registration in about a minute on 8×A6000. This work is quire impressive.

The reviewer is familiar with registration but not with optimization work. But this work appears impressive and practical. Therefore, the reviewer will assign a low confidence score but high rating.

### Strengths
- Writing: Problem importance is well motivated: when the image is 100–1000× larger, registration algorithms did not keep up. It is an important gap.
- Distributed story is novel for registration.
- Empirical section is broad: classical (ANTs-like), learning (TransMorph, SynthMorph, VFA, UniGradICON), and purpose-built (CLAIRE, ITK-DReg). The 250µm setting, where every other method degrades while FFDP improves, is persuasive.
- The accelerations for existing code (TransMorph training 6× faster, big memory drops, and also for FireANTs) mean this work is really a critical progress.

### Weaknesses
- This work enables faster/larger registration, but it lacks a clear section on how to apply this. It does not specify which registration methods are directly supported, how to integrate the kernels, or which methods are not supported and why.
- Only mentioned related work on GPU/system acceleration. The author does not survey other GPU acceleration outside registration.

### Questions
See weakness.

### Soundness
4

### Presentation
3

### Contribution
4
