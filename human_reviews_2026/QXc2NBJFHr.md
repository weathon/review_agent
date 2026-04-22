# The Less You Depend, The More You Learn: Synthesizing Novel Views from Sparse, Unposed Images without Any 3D Knowledge

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 8

## Abstract
Recent advances in feed-forward Novel View Synthesis (NVS) have led to a divergence between two design philosophies: bias-driven methods, which rely on explicit 3D knowledge, such as handcrafted 3D representations (e.g., NeRF and 3DGS) and camera poses annotated by Structure-from-Motion algorithms, and data-centric methods, which learn to understand 3D structure implicitly from large-scale imagery data. This raises a fundamental question: which paradigm is more scalable in an era of ever-increasing data availability? In this work, we conduct a comprehensive analysis of existing methods and uncover a critical trend that the performance of methods requiring less 3D knowledge accelerates more as training data increases, eventually outperforming their 3D knowledge-driven counterparts, which we term “the less you depend, the more you learn.” Guided by this finding, we design a feed-forward NVS framework that removes both explicit scene structure and pose annotation reliance. By eliminating these dependencies, our method leverages great scalability, learning implicit 3D awareness directly from vast quantities of 2D images, without any pose information for training or inference. Extensive experiments demonstrate that our model achieves state-of-the-art NVS performance, even outperforming methods relying on posed training data. The results validate not only the effectiveness of our data-centric paradigm but also the power of our scalability finding as a guiding principle.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors study generalizable novel view synthesis (NVS) from sparse, unposed 2D images and show that methods with less explicit 3D knowledge improve faster with larger data; they propose a framework that eliminates 3D inductive bias and pose annotations and learns implicit 3D awareness from 2D images, reporting photorealistic and 3D-consistent novel views comparable to methods that use posed inputs

### Strengths
1. Clear empirical trend discovery, identifies and quantifies the data-scaling trend: models that depend less on explicit 3D knowledge improve faster as data scales, motivating a data-centric alternative to heavy 3D priors.

2. The framework removes both explicit 3D architectural bias and pose supervision, which, if robust, simplifies pipelines and broadens applicability to unposed datasets.

3. Practical impact in terms of enabling NVS without pose labels lowers annotation cost and allows training on large, in-the-wild photo collections.

4. Empirical results claim parity with posed-input methods, suggesting the approach is not only elegant but effective on benchmarks.

5. The paper reframes the trade-off between inductive bias and data scale, providing a clear hypothesis for future work and dataset collection strategies

### Weaknesses
1. The method’s advantages hinge on data scaling; performance in low-data regimes or niche domains may degrade relative to 3D-aware methods.

2. Without explicit 3D representations or poses, ensuring geometric consistency across large camera motions or severe occlusions can be fragile; failure modes and long-range consistency are likely under-explored.

3. Reported parity with posed methods may depend on specific datasets or evaluation metrics; results on highly diverse, real-world scenes or metric 3D accuracy may be weaker.

4. Removing explicit 3D structure reduces model interpretability and makes diagnosing geometric errors or dataset biases harder.

5. Achieving strong performance via data scaling may require substantial compute and careful curricula; the practical resource requirements may be high.

6. Learning implicit 3D from 2D alone risks encoding photographic regularities that do not generalize to different capture conditions or object categories.

### Questions
1. At what dataset size and diversity does the “less 3D knowledge” model overtake 3D-driven counterparts? Is there a measurable crossover point per dataset type?

2. Which camera motions, occlusion patterns, or scene topologies cause the model to produce inconsistent geometry or view synthesis artifacts?

3. How does the method perform on metric 3D/geometry benchmarks (e.g., depth/pose consistency) compared with explicit 3D methods?

4. Does training on large web-scale data produce robust synthesis on specialized domains or is fine-tuning with some 3D bias still necessary?

5. Can hybrid approaches (weak 3D priors, self-supervised pose signals) such as HawkI reduce data needs while retaining benefits of the minimalist design?

6. What are the training compute, memory, and inference-time costs relative to NeRF-style or other 3D-aware baselines?

7. Which minimal inductive elements (if any) materially help (positional encoding, multi-view consistency losses, rendering modules) and which are redundant?

8. Does reliance on large web data introduce unwanted biases in scene content, lighting, or demographic representation that affect downstream uses?

9. If we train on large in-the-wild data and evaluate on narrow-domain datasets to measure generalization and need for adaptation, how does it work?

10. It will be good to add experiments with light-weight 3D cues (approximate poses, sparse depth) to quantify trade-offs between inductive bias and data size.

11. It will be good to publish FLOPs, GPU hours, and inference latency to clarify practical deployment feasibility

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the trade-off between 3D bias-driven and data-centric design philosophies for Novel View Synthesis (NVS). The central hypothesis is "the less you depend, the more you learn": methods with weaker 3D dependencies exhibit superior scalability, accelerating in performance as training data increases.

The paper tests this hypothesis by conducting a systematic scalability analysis comparing existing methods (MVSplat, LVSM, NoPoSplat) on training subsets of varying sizes. They find that data-centric methods show greater performance gains with more data.

Building on this insight, the paper proposes UP-LVSM, a novel, data-centric framework that operates in a fully unposed setting. This method learns to synthesize novel views from sparse images without any camera pose information or explicit 3D representations during training. The core technical contribution is the Latent Plücker Learner, a component that learns a latent pose space in a self-supervised manner, enabling viewpoint conditioning without ground-truth poses.

Experiments show that UP-LVSM, when trained on a large dataset (66K scenes), achieves state-of-the-art performance, even outperforming methods that require ground-truth pose annotations. The paper argues that this validates their hypothesis and demonstrates that noisy 3D knowledge (such as SfM poses) can be a "performance bottleneck" at scale.

### Strengths
1. The paper's core finding—that at scale, it's better to learn 3D from 2D data than to rely on noisy, explicit 3D knowledge (like SfM) —is a significant and impactful claim, which is well-supported by the UP-LVSM results.

2. The analysis in Section 3, which isolates the effects of 3D inductive bias and pose annotation dependence, is a valuable contribution in its own right. The use of dataset subsets to show performance trends (Figures 2, 4, and 5)  is very well done.

3. The proposed UP-LVSM achieves SOTA performance on RealEstate10K, impressively outperforming LVSM (28.82 vs 27.60 PSNR)  despite LVSM having access to ground-truth input poses. This is a very strong result that validates the paper's hypothesis.

4. The Latent Plücker Learner is a novel and well-designed component for learning a latent pose space without supervision. The design thoughtfully considers and addresses the risk of information leakage from the target view.

5. The appendix is not an afterthought but contains crucial, high-quality analysis. Appendix E shows that LVSM are highly sensitive to pose noise, whereas UP-LVSMs are immune. Appendix D solves a key practical limitation of unposed methods.

6. The paper is also exceptionally well written, structured and the figures and tables are all very informative.

### Weaknesses
1. The paper's claim to remove "3D inductive biases" and operate without "any 3D knowledge"  is inaccurate. The Plücker ray embedding, which is foundational to the Latent Plücker Learner, is a strong 3D geometric prior. It encodes a line in 3D space. The claim should be more precise, e.g., "without explicit 3D scene representations (like meshes or 3DGS) or camera pose annotations. 

2. A major practical drawback of unposed methods is the inability to control the camera for rendering. The paper presents a very simple and effective solution (fine-tuning a linear mapper) in Appendix D. For me personally, this is a crucial insight for making the method practical, yet it is absent from the main paper, potentially leaving readers with the impression that UP-LVSM is not controllable.

### Questions
Do you have a sense of the upper bound? For instance, how does UP-LVSM (trained on 66K scenes) compare to a per-scene optimized 3DGS (a different kind of "upper bound")? 

In Table 8, the 3D correspondence probing shows UP-LVSM (31.9) performing slightly worse than the off-the-shelf DINOv2 (36.8) at 0-15°. This is counterintuitive, as your model's encoder is fine-tuned on this 3D-aware task. Do you have an explanation for why the specialized model would be worse than the general-purpose one?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper analyzes existing Novel View Synthesis (NVS) methods and discovers a core trend: methods that depend less on explicit 3D knowledge (poses, handcrafted 3D representations) benefit more from data scaling and eventually outperform 3D-biased approaches.

- Based on this finding, the authors propose UP-LVSM, a feed-forward, fully data-centric NVS framework that learns implicit 3D structure directly from large-scale 2D images—without camera pose supervision or predefined 3D representations.

- The method introduces a Latent Plücker Learner to infer camera geometry implicitly, enabling state-of-the-art performance in novel view synthesis from sparse, unposed images.

### Strengths
- This paper demonstrates that removing explicit 3D priors allows performance to scale significantly with data, outperforming pose-supervised 3D knowledge-driven methods.

- This paper has a pose-free & explicit-3D-free pipeline, a fully feed-forward transformer architecture that works without SfM poses, NeRF/3DGS priors, or handcrafted 3D structures—simplifying training and deployment.

- Extensive experiments show state-of-the-art results and confirm the central hypothesis, providing both conceptual insight and practical contribution to scalable NVS.

### Weaknesses
- I have watched the supplementary video. (1) The zoom-in and zoom-out distance is too short. (2) Also, compared to existing models, the method seems to only improve visual quality.

- For datasets where GT is provided, the resolution appears to be very low. I am curious how the method performs on higher-resolution datasets. (If this dataset is the best available choice, I expect the authors to justify that in the rebuttal.)

- The process of simply combining a Transformer with DINOv2 raises concerns regarding novelty. The authors should demonstrate where the novelty lies in this work. What is the novelty contribution of the Latent Plücker Learner?

- The ablation study section is not intuitive. Merely describing components with text makes it difficult to understand what was included or excluded in each ablation setting.

### Questions
Mentioned in the weaknesses

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors posit the principle “the less you depend, the more you learn”—arguing that reducing reliance on explicit 3D priors (like NeRF/3DGS representations or SfM-derived camera poses) improves scalability and generalization as training data scales. 

This message extends upon LVSM and removes the training- and test- time pose requirements to make the framework even more data centric. The authors introduce a Latent Plücker Learner to infer camera poses in a self-supervised manner.

The paper presents empirical analysis across datasets (RealEstate10K, DL3DV, ACID, Objaverse) and compares UP-LVSM to bias-driven and pose-dependent methods (MVSplat, NoPoSplat, LVSM). Results show that UP-LVSM scales better with data and even surpasses 3D-supervised models in rendering fidelity and generalization.

### Strengths
- The paper is easy to follow and the authors promise to release code for reproducibility.

- The paper evaluates across diverse datasets and metrics, providing ablations, scalability curves, and qualitative examples. The performance gains are consistent and very decent (e.g., +1 PSNR improvements over LVSM on large-scale data).

- Removing the requirement of training- and test- time pose annotations have huge potential of scaling to much larger-scale datasets; hence this work (along with RayZer) opens up a lot of new possibilities.

### Weaknesses
- UP-LVSM seems to have separately trained models on the RealEstate10k scene data and the Objaverse object data. It remains unclear to me if the latent plucker learning component can be trained on a mix of scene and Objaverse datasets. From the scalability perspective, it makes sense to have a method capable of ingesting all available data sources. 

- Tab. 4 is presented in a confusing way; it's unclear what baseline the performance gain is evaluated against. Moreover, I think it makes more sense to also include the absolute PSNR/SSIM/LPIPS values, rather than just providing the relative changes.

### Questions
- in Tab. 8, it seems to me that DINOv2, though not explicitly trained on 3D tasks, seem to provide more accurate correspondence estimation than UP-LVSM which is trained on the 3D NVS task. I guess I find it a bit hard to understand. 

- in Tab. 5 and Fig. 7, how is the target pose provided to different methods? Did all the methods use the same ground-truth target pose? Or they consume poses estimated by UP-LVSM?

- Line 1105 mentions that "However, for the experiments reported in the main paper, we standardize all evaluations to the 224 ×224 setting, including all baseline comparisons." I wonder if this is fair to baselines if their provided checkpoint was trained only on 256x256.

### Soundness
4

### Presentation
4

### Contribution
3
