# NGTTA: Non-parametric Geometry-driven Test Time Adaption for 3D Point Cloud Segmentation

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Previous Test Time Adaption (TTA) methods usually suffer from training collapse when they are transferred to complex 3D scenes for point cloud segmentation due to the significant domain gap between the source and target data.
 To solve this issue, we propose NGTTA, a stable test time adaption method guided by non-parametric geometric features.  In NGTTA, we leverage the distribution of non-parametric geometric features on target data as an “intermediate domain” to reduce the domain gap and guide the stable learning of the source model on target data. 
 Specifically, we use the source domain model and a non-parametric geometric model to extract the embedding features and geometric features of the point cloud, respectively. Then, a category-balance sampler is designed to filter easy samples and hard samples in the input data to address the class imbalance issue in semantic segmentation. Inspired by previous work, we use easy samples for entropy minimization loss and pseudo-label prediction to fine-tune the source domain model. The difference is that we refine the pseudo labels not only by considering the soft voting among their nearest neighbors in the model embedding feature space but also in the geometric space, which can prevent the accumulation of errors caused by model feature shifts.  Furthermore, we believe that hard samples can effectively represent the distribution differences between the source domain and the target domain. Therefore, we propose to distill the geometric features of hard samples into the source domain model in the early stages of training to quickly converge to an "intermediate domain" that is similar to the target domain. By taking advantage of the ability of the non-parametric geometric feature to represent the underlying manifolds of the target data, our method efficiently reduces the difficulty of the domain adaption.
We conduct the main experiments on the more challenge \textbf{\textit{sim-to-real}} benchmark about synthetic dataset 3DFRONT and the real-world datasets ScanNet and S3DIS for 3D segmentation task. Results show that our method can efficiently improve the mIOU by over \textbf{3\%} on 3DFRONT$\rightarrow$ ScanNet and \textbf{7\%} on 3DFRONT$\rightarrow$ S3DIS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of the domain gap between source and target data for the task of 3D point cloud segmentation. The proposed NGTTA is a Test Time Adaptation (TTA) method that leverages a non-parametric geometric model. This model is used for producing a so-called intermediate domain, guided by the intuition that the geometric features in the source data and target data could be utilized to bridge the domain gap. Furthermore, the introduced methodology addresses the problem of category imbalance by a category-balance sampler filtering easy and hard samples. The evaluation on the sim-to-real benchmarks shows that the proposed method performs better than the competitors.

### Strengths
1. The method introduced by the authors addresses the challenging problem of the domain shift for the task of 3D semantic segmentation and shows that geometric priors could bridge this gap.
2. Various experimental setups and ablations (seem to) speak to the effectiveness of the proposed approach (however, the details should be clarified — see Questions). The reporting of the computational overhead introduced by the method is also appreciated.

### Weaknesses
1. One drawback of the paper is the lack of clarity in writing. The paper could significantly benefit from polishing the text. A *non-exhaustive* list of issues is given in the questions.
2. The prior work PointNN by Zhang et al. (2023) appears to play the central part of the proposed method, since it is the non-parametric geometric model; yet, it needs to be recapped sufficiently in the main part of the paper or in the appendix.
3. The hyperparameter selection is biased to the target domain:
Many components of the proposed method rely on hyperparameters, e.g., $\sigma$ and $\gamma$ in the sampling (4), $\alpha_i$ in the total loss (A.2 eq. (1) ), the $k$ in kNN, as shown by the deviation of the results in the respective tables. 
The authors use the 3DFRONT→S3DIS benchmark to find the hyperparameters. 
Question: Were these selected hyperparameters also used for the 3DFRONT→ScanNet benchmark?
(See also Q2 in Quesitions.)
If so, in the best case, there’s only one true setup, where the target data is not accessed to modify the method — 3DFRONT→ScanNet.

### Questions
1. Could the authors address the following:
- “non-parametric geometric” (e.g., L032, L069, L098, L157, and others)— is “non-parametric geometry” meant or “non-parametric geometric model”?
- grammar in L070-071?
- “data of geometric.” in L156?
- L263, “source data” —> “target data”.
- the citations need to be adjusted — no need for parentheses when in text, only a single pair of parentheses otherwise.
2. In Tables 1-3 in the Appendix, there are $4^3 = 64$ different combinations of $\alpha_1$, $\alpha_2$, and $\alpha_3$.
- Which 12 combinations are presented in the tables? 
3. Does the *j*-th sample in (5) mean a *j*-th point in the *i*-th point cloud (scene)?
4. How is $\beta$ in (11) chosen in practice?
5. Did the authors perform the comparison only for a single initialization? I.e., only one test run?
6. Does the non-parametric model have a decoder as visualized in Figure 2?
7. Comment: The arrow between $\Gamma^{ps}$ and $L_{cls}$ n Figure 2 should be fixed.
8. Comment: The paper would benefit from having additional qualitative results, especially for the 3DFRONT→ScanNet  benchmark, for which they are currently missing.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes NGTTA, a method for test-time adaptation in 3D point cloud segmentation. NGTTA leverages the distribution of non-parametric geometry as an “intermediate domain” to facilitate model adaptation. The approach involves using a source model alongside a non-parametric geometric model to extract features from point clouds. To handle class imbalance, a category-balance sampler separates easy and hard samples. For easy samples, pseudo-labels and entropy minimization are applied, with soft voting used for correction among neighboring points. Hard samples, on the other hand, are distilled into the source model, allowing the model to quickly converge to the intermediate domain that resembles the target.

### Strengths
- The proposed model shows meaningful performance gain for sim-to-real and cross-domain scenarios.

- Extensive experiments and analysis are provided in the paper.

### Weaknesses
- It seems that the explanation regarding the motivation for using the non-parametric geometric model for domain generalizability is insufficient. Is there a way to demonstrate that the non-parametric geometric model is generalizable?

- If I understand correctly, the non-parametric geometric model ultimately allows for good clustering without the need for learning, but I’m not sure if applying it at the feature level is the best approach. If the non-parametric model is truly domain generalizable, wouldn’t it be a better approach to use it simply as a refinement method in the target domain as a form of post-processing, or to utilize it when generating pseudo-GT in a pseudo-labeling approach?

- The paper lacks sufficient qualitative results, which makes it difficult to fully understand or visualize the practical effectiveness of the proposed method. 

- There are writing errors, such as the sudden introduction of "S2T" in the caption of Figure 1 without prior explanation.

### Questions
- Could you clarify which point in time the features in Figure 3 represent? Additionally, while (b) seems to demonstrate promising results, there is no accompanying explanation. Could you provide further details regarding this?

- Is this methodology applicable to LiDAR data as well?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a Test-Time Adaptation (TTA) method for point cloud segmentation on indoor datasets. The method utilizes easy samples to minimize entropy loss and employs pseudo-label predictions to fine-tune the source domain model by considering both semantic and geometric k-nearest neighbors (non-parametric PointNN). By leveraging the ability of non-parametric geometry to represent the underlying manifolds of the target data, this approach effectively reduces the challenges associated with domain adaptation. Experimental results demonstrate improved performance on 3D segmentation tasks.

### Strengths
The paper is well-structured and provides a robust methodology that is rigorously evaluated on challenging sim-to-real benchmarks (e.g., 3DFRONT→ScanNet and 3DFRONT→S3DIS). The experimental results demonstrate a good improvement on indoor datasets. The proposed category-balance sampler and soft voting mechanisms are well-motivated and backed by quantitative results. Additionally, the authors conducted thorough ablation studies to evaluate different components of the framework, such as the impact of soft voting, entropy threshold, and loss weights.

### Weaknesses
The technical contribution of this paper appears relatively weak, as several components are directly adapted from existing 2D TTA methods. For instance, entropy selection and moment update mechanisms are borrowed directly from established 2D TTA techniques [1] without significant innovation in the 3D domain (only add a hard samples selection). Additionally, components such as entropy minimization and pseudo-labeling have been widely explored in related fields, including 3D semi-supervised segmentation and unsupervised learning[3]. The authors could strengthen the paper by either justifying the unique adaptations of these methods to 3D point cloud segmentation or by exploring alternative techniques tailored to the specific challenges of the 3D domain.
Furthermore, it would be valuable to discuss why other geometric features commonly used in 3D point cloud analysis, such as Fast Point Feature Histograms (FPFH), were not considered. Comparing the performance of the non-parametric geometric approach against more established geometric features would provide stronger evidence of its efficacy and clarify the specific benefits of the proposed method.

[1] Wang, Dequan, et al. “Tent: Fully test-time adaptation by entropy minimization.” International Conference on Learning Representations (ICLR). 2021.
[2] Xie, Qizhe, et al. “Self-training with noisy student improves imagenet classification.” IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020.
[3] Hou, Yikang, et al. “3D-SST: Self-Supervised Pretraining for 3D Scene Understanding.” European Conference on Computer Vision (ECCV). 2022.

### Questions
1. The statement "We can see that as the sample count decreases, the threshold will gradually increase" is reasonable. However, it’s stated that tail categories with fewer samples will have a larger entropy threshold and "We can see that as the sample count decreases, the threshold will gradually increase", but the reasoning behind this isn't obvious from the equation itself. 

2. Whether Eq.~(4) is applied individually to each point cloud or collectively across all point clouds. Does each point cloud have its own unique $\sigma_c$, or do all point clouds share the same $\sigma_c$?

3. For Eq.~(13), the notation could be confusing for readers. Since $\mathbf{f}_i$ represents a vector, and $\mathbf{f}_{ij}$ also appears to be a vector, the mathematical symbols could be refined to avoid potential misinterpretation.

4. Lastly, it would be insightful to discuss the model’s performance when transitioning from indoor datasets to outdoor datasets. Are there any notable differences in performance, or specific challenges encountered when adapting to outdoor data?

5. Although the method’s applicability to multiple architectures is partially demonstrated, more rigorous evaluation on additional 3D segmentation backbones (e.g., MinkowskiNet, RandLA-Net) would help generalize the findings. Providing results for diverse architectures could highlight the robustness of the NGTTA framework, particularly if model performance varies significantly with architecture.

6. Visualization of Intermediate Domain Effects: While the paper presents the concept of an “intermediate domain,” it would benefit from visualizations that illustrate this intermediate representation’s impact on the distribution alignment between source and target domains. For example, t-SNE or UMAP plots showing the evolution of feature space alignment as the model adapts would provide more concrete evidence of the intermediate domain’s role in reducing domain gaps.

7. Comparing the performance of the non-parametric geometric approach against more established geometric features such as FPFH would provide stronger evidence of its efficacy and clarify the specific benefits of the proposed method.

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
3

### Summary
This paper proposes a non-parametric geometry-based test time adaption approach, named NGTTA,  to improve the domain adaption ability for 3D point cloud semantic segmentation. NGTTA mainly introduces a no-parametric model to extract features as intermediate domain and proposes a category-balance sampler to address class imbalance. Evaluations on 3DFRONT to ScanNet and D3DIS settings show the practicality of the proposed method.

### Strengths
- The motivation is clear and the paper is easy to follow.
- The proposed method outperforms all other TTA methods on four data domain adaptation settings.
- The experiments and ablation analysis are extensive.

### Weaknesses
- The paper lack of reports of runtime related results (time/memory consumption) to demonstrate the cost of introducing NPATT.
- The performance gains on real2real setting are not significant.
- It lacks insightful analysis for some results, e.g., why most TTA methods drop greatly on 3DFRONT→ScanNet setting compared with source only, while not on 3DFRONT→S3DIS setting.?
- The paper lacks of category-wise IoU results which could better demonstrates how the proposed method improve the overall performance compared with other methods.

### Questions
- How is the visualization made with features in Figure3? It’s better to also mention what x-axis and y-axis means.

### Soundness
3

### Presentation
3

### Contribution
3
