# Equivariant Flow Matching for Point Cloud Assembly

- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
The goal of point cloud assembly is to reconstruct a complete 3D shape by aligning multiple point cloud pieces. This work presents a novel equivariant solver for assembly tasks based on flow matching models. We first theoretically show that the key to learning equivariant distributions via flow matching is to learn related vector fields. Based on this result, we propose an assembly model, called equivariant diffusion assembly (Eda), which learns related vector fields conditioned on the input pieces. We further construct an equivariant path for Eda, 
which guarantees high data efficiency of the training process. Our numerical results show that Eda is highly competitive on practical datasets, and it can even handle the challenging situation where the input pieces are non-overlapped.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proposes an equivariant flow matching framework for multi-piece point cloud assembly tasks. The key idea is to employ a vector field parameterized by equivariant networks on an invariant base distribution to ensure the output distribution is equivariant to SO(3) rotations and permutations. Additionally, the training efficiency is enhanced by considering modified samples and random noises with minimum distance across all possible rotations. Overall, the experimental results show that the proposed framework achieves better results than existing baselines, and the ablation studies well validate the effectiveness of the proposed network.

### Strengths
- This work proposes a clear formulation of the base distribution and vector field network assumptions to ensure the output distribution is equivariant to the group. The formulation of the base distribution ($\left(U_{\mathrm{SO}(3)} \otimes \mathcal{N}(0, \omega I)\right)^N$) and equivariant layers appear to be well-suited for these assumptions.
- The experimental results demonstrate strong performance of the proposed framework, achieving better results in both pair-wise registration and multi-piece assembly.
- Additionally, the manuscript validates the proposed components through the ablation study in Table 4, which further justifies the need for rotation correction and equivariant networks.

### Weaknesses
There is only one minor concern in the current manuscript:
Regarding the ablation study, it is mentioned that the proposed equivariant network is replaced with a non-equivariant counterpart. It would be beneficial to provide more details on the non-equivariant network and describe the exact changes in the Appendix to ensure the experiment's fairness.

### Questions
See the weakness section.

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
This paper proposes Eda (Equivariant Diffusion Assembly), a novel correspondence-free, multi-piece point cloud assembly model built on equivariant flow matching. The key theoretical contribution is to show that learning equivariant distributions can be reduced to learning related vector fields, provided the initial noise distribution is invariant. Building on this, the authors design an SE(3)^N-equivariant flow-matching framework where the equivariance of the learned distribution is guaranteed by construction. Eda parametrizes vector fields through an equivariant neural network and introduces an equivariant path construction that improves data efficiency during training. Experiments on 3DMatch, BB, and KITTI demonstrate strong quantitative improvements over state-of-the-art baselines, including robust performance even on non-overlapping fragments (3DZeroMatch). The results support both theoretical soundness and empirical effectiveness.

### Strengths
- The paper provides a solid theoretical framework by reducing equivariant distribution learning to learning related vector fields. The derivations appear rigorous and consistent, although I did not verify every step in detail.

- The use of E3NN-based equivariant attention and Croco blocks makes the approach practical.

### Weaknesses
- Although the paper provides theoretical guarantees for SO(3)^N-equivariance, the empirical validation is mostly indirect. The ablation studies show performance drops when removing the equivariant backbone or path, suggesting that equivariance helps, but this only demonstrates effectiveness, not faithful equivariant behavior. A more rigorous validation would involve explicitly applying controlled rotations to input fragments (right-multiplication in SO(3)^N), or global rotations (left-multiplication), and verifying whether the predicted poses transform accordingly. Such experiments would directly confirm that the learned flow v_X(g) satisfies the claimed equivariance relation v_{rX}(rg)=r\,v_X(g).

- The authors claim that their model achieves permutation equivariance; however, no direct experiment is provided to validate this claim. It remains unclear how the predicted poses change when the input order of point clouds is permuted.

- Weak Experimental Validation. The experimental section is relatively weak and limited in scope. The first experiment focuses on pairwise registration, which does not align with the paper’s stated goal of multi-piece point cloud assembly. The comparison set is also narrow and excludes strong, widely recognized metrics commonly adopted in the pairwise registration literature and also missing many prevalent pairwise methods such as FCGF, Predator and BUFFER. The multi-piece assembly evaluation is further constrained, only 2–8 fragments on synthetic datasets, which makes it difficult to assess the method’s scalability or robustness in realistic settings. Fig. 4 further indicates limited generalization capacity, performance degrades notably on unseen fragment lengths, revealing the model’s fragility. Including results on more comprehensive datasets such as Fantastic Breaks or FRACTURA would significantly strengthen the empirical claims.
Finally, the KITTI experiment appears loosely connected to the main task, and its relevance to point cloud assembly is not clearly justified, leaving the overall empirical validation unconvincing.

Overall, the experimental validation is quite weak and does not convincingly demonstrate the real effectiveness of the proposed Equivariant Flow Matching. The experiments are limited in scope, and key claims, such as equivariance and invariance, are only indirectly supported. These issues collectively make me lean toward rejecting this paper at its current stage.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Eda, an equivariant flow-matching framework for assembling 3D point cloud fragments. It combines E(3)-equivariant layers with a flow-based architecture to enable efficient SE(3)-equivariance learning . Experiments on 3DMatch, 3DLoMatch, and Breaking Bad show over 50% lower rotation error than baselines. While theoretically elegant and empirically strong, the paper lacks ablation on equivariance, efficiency analysis, and tests on noisy real-world point clouds.

### Strengths
- The paper provides a solid theoretical foundation for framing point cloud assembly as a flow matching problem. 
- On 3DMatch and 3DLoMatch, Eda achieves >50% lower rotation errors than GEO/ROI/AMR baselines. It also handles non-overlapping fragments (3DZeroMatch) where correspondence-based methods fail entirely.
- The paper provides a good ablation study on varying different settings.

### Weaknesses
- How does the method work on an untrained category of assembly?
- How does the method work if equivariance is ablated? The author might want to consider comparing with the same architecture but only lack of equivariance for paper completeness. 
- While the theory side is useful, a better native like intuitive diagram would help aid readability of the paper. 
- While the paper claims that learning related vector fields provides a more efficient alternative to full equivariant flow modeling, the evidence remains largely qualitative. The only quantitative indicator is a reduction in assembly runtime (≈ 19 minutes per object versus ≈ 34 minutes for diffusion-based baselines). A more detailed analysis of the computational efficiency would benefit the paper’s completeness (e.g. training convergence, FLOPs, memory footprint, scalability curve, etc.)

### Questions
- How does the method generalize to real world noisy point clouds? - As one of the benefits of flow base method is its generalization ability to messy real world applications.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an equivariant solver for assembly tasks based on flow matching. Theoretically, the authors demonstrate that learning equivariant distributions via flow matching requires learning corresponding equivariant vector fields. Building upon this result, this paper proposes the Equivariant Diffusion Assembly (EDA) model, which learns these vector fields conditioned on the input pieces. Furthermore, they construct an equivariant sampling path for EDA, a design that ensures high data efficiency during training.

### Strengths
1.	The motivation is clear.
2.	Mathematical descriptions are sufficient.

### Weaknesses
1. Inadequate literature review on key related works, especially patch-based registration and non-overlap registration methods.
2. Unsubstantiated claim of solving the multi-piece problem, as the method and experiments primarily focus on two-piece problem with one experiment for multi-piece problem on BB dataset.
3. Unclear novelty and contribution, as the method heavily builds upon established components without a clear clarification.
4. Incomplete experimental validation, due to an insufficient number of compared methods, limited datasets, and a lack of rigorous testing on multi-piece cases.

### Questions
1.	This paper claims to address the multi-piece assembly. There are lots of patch-based point cloud registration methods that have not been carefully discussed: [1] Zhao, T., Tian, T., Zou, X., Yan, L., & Zhong, S. (2025). Robust Point Cloud Registration via Patch Matching. IEEE Transactions on Geoscience and Remote Sensing. [2] Zhao, T., Li, L., Tian, T., Ma, J., & Tian, J. (2023). Patch-guided point matching for point cloud registration with low overlap. Pattern Recognition, 144, 109876. [3] Qin, Z., Yu, H., Wang, C., Guo, Y., Peng, Y., Ilic, S., ... & Xu, K. (2023). Geotransformer: Fast and robust point cloud registration with geometric transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(8), 9806-9821. 
2.	This paper is also targeted at non-overlap assembly, which is also not a new problem. For example, the following strengths and weaknesses should also be discussed: [1] Xu, J., Dai, H., Hu, X., Fan, S., & Ke, T. (2024). SCREAM: Scene rendering adversarial model for low-and-non overlap point cloud registration. IEEE Transactions on Geoscience and Remote Sensing. [2] Xu, J., Zhang, Y., Zou, Y., & Liu, P. X. (2023). Point cloud registration with zero overlap rate and negative overlap rate. IEEE Robotics and Automation Letters, 8(10), 6643-6650. 
3.	This method seems to be constructed on the basis of Ryu et al. (2024). The main difference between this method and Ryu et al. lies in the Brownian diffusion on SO(3), while the method proposed by Ryu et al. solves the more general multi-piece problem.
4.	The related work is not organized very well; readers can not catch the differences between this paper and existing methods. It is recommended to separate this section into several related subsections.
5.	The preliminaries section is too long; it is suggested to shorten it and only put the important information into the main text.
6.	Although this paper claims the use of multiple pieces, the definition and experiments mainly focus on a two-piece problem with one dataset to validate the multiple pieces problem.
7.	SO(3)-equivariant networks are also widely used in 3D. The introduction of this network in the main text needs to be shortened.
8.	Moreover, the vector field in flow matching is also a well-known definition, whose introduction also needs to be shortened.
9.	If all experiments are conducted on only one multi-piece problem, then it is not very suitable to claim that solving a multiple-piece problem.
10.	Equivariant flow is widely used in 2D computer vision and 3D molecular generation. It is hard to find the main contribution when compared with existing equivariant flows. It is recommended to re-emphasize your contribution in Section 4.2.
11.	Sampling with the RUNGE-KUTTA method is also not a novel technique in flow matching.
12.	In the implementation, the vanilla Transformer is employed. Why not use a point transformer with permutation-equivariant or a transformer with SO(3)equivariant in recent years? They might obtain better performance. Most importantly, there are lots of innovations in point cloud process. Building your methods on existing practices will be better.
13.	There are also existing blocks employed in your architecture. The main contributions can not be clearly understood. Moreover, it is not recommended to rename self-attention and cross-attention with a new name croco block. They can be easier to understand than giving a new name,
14.	As for experiments, there are lots of pages for introducing existing details. Therefore, fewer pages are left for experiments, which leads to incomplete experimental validation.
15.	In Figure 3, if the 8-piece assembly process is to be displayed, 8 different colors should also be given. It is highly suggested to validate your method on multiple piece problems.
16.	Moreover, the compared methods in point cloud registration are not comprehensive enough. It is a widely studied area, and more up-to-date methods should be compared.
17.	Importantly, only two datasets are limited. There are also many datasets related to point cloud registration.

### Soundness
2

### Presentation
2

### Contribution
2
