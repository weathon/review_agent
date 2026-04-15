# NF-ICP: Neural Field ICP for Robust 3D Human Registration

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5

## Abstract
Aligning a template to 3D human point clouds is a long-standing problem crucial for tasks like animation, reconstruction, and most supervised learning pipelines. Recent data-driven methods leverage predicted surface correspondences; however, they are not robust to varied poses or distributions. In contrast, industrial solutions often rely on expensive manual annotations or multi-view capture systems. In this work, we present NF-ICP, a method that, for the first time, generalizes well on a large set of challenges, including complex poses, clothed humans, and noisy scans. Leveraging the large MoCap dataset AMASS, we learn a neural field model to predict the direction towards the localized SMPL vertices on the target surface. Such neural field leads to a reasonable initialization, but the resulting template often does not overlap with the target surface. NF-ICP exploits a classical Iterative Closest Point objective adapted to our model to quickly fine-tune the model, resulting in a significantly improved template to target surface overlap. NF-ICP constitutes a simple and computationally efficient registration method that significantly improves over public benchmarks and solidly surpasses the state of the art. We will release code and checkpoints in \url{link}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a method to perform 3D human registration from point cloud input. The central idea is to conduct ICP-style iterative refinement between the neural field and the target point cloud after the initial prediction by LVD. Both qualitative and quantitative evaluations on public 3D human registration datasets are presented.

### Strengths
- This paper presents a complete working system of 3D human registration from input point cloud to LVD, followed by Neural Field ICP and then final refinement. It also demonstrates interesting applications such as converting the initial scan from Luma AI into an animation-ready avatar driven by the SMPL skeleton.

- The paper presents a wide range of qualitative results from different datasets such as D-FAUST, BEHAVE, and HuMMan.

### Weaknesses
- From the perspective of the system, the major technical contribution proposed in this paper, i.e. NF-ICP, is used as a post-processing component in the system. More concretely, the previous pipeline would be LVD (Corona et al. 2022) -> template fitting -> refinement; with NF-ICP added, the pipeline becomes LVD -> NF-ICP -> template fitting -> refinement. While LVD provides the most important initialization, the effect of NF-ICP is relatively incremental. The significance of the contribution in terms of scope seems to be on the minor side.

- From the methodology point of view, the current formulation of NF-ICP evaluates the correspondences and offset only at the target scan points. However, the neural field does not guarantee consistent prediction over different points in the space. Since the vertices are initialized at (0, 0, 0), having an ICP error over the target scan does not guarantee the correctness of offset for points far from the scan surface.

- Since this NF-ICP is a add-on post-processing step, I believe that the central question to discuss is the necessity of NF-ICP in the LVD -> template fitting -> refinement pipeline. Essentially the question here is how much benefit does the additional NF-ICP step brings in comparison with directly using SMPL for the template fitting & ICP, which is the traditional model fitting approach with LVD initialization. On the one hand, the paper gives some quantitative evidence of this by comparing "LVD + R" with "Ours + R" in Table 4 (1.11 -> 1.06, 2.48 -> 2.26) and 6 (6.27 -> 6.13), and Table 3 (3.26 -> 3.08) in the supplementary document, where the improvement seems to be relatively minor.

- On the other hand, some qualitative results are presented beside Table 6, and in Figure 2 & 3 of the supplementary material. However, these examples show that the SMPL model-fitting is under-regularized in the experimental setup of the paper. SMPL encodes a strong prior of human body shape and pose. A well-regularized model fitting process should prevent many of the artifacts from happening. In addition, it is possible to use the LVD initialization together with ICP, instead of completely relying on the LVD correspondences. For such reasons, I believe that the some of the artifacts presented by LVD + R may not necessarily happen under a proper implementation of classical SMPL model fitting.

### Questions
- Since my major concern is about the necessity of NF-ICP in a registration pipeline, my central questions for the author is whether more evidence of the advantage of Ours + R over LVD + R can be provided. Here LVD + R should have proper regularization, for example shape regularization by penalizing beta and pose regularization by setting proper prior weight, and balancing between LVD initialization term and ICP term.

- In the FAUST experiment, why is there a setting where input shape is the registration (FaustR)? What is the purpose of this setting?

- Why does this method use 690 vertices instead of 6890 vertices in the original LVD experiment?

- In the LVD paper, their Eq (1) wrote that the neural field g maps a point $\mathbf v_i \in \mathbb R^3$ to $\Delta \mathbf v_i \in \mathbb R^3$, while in their implementation it is actually predicting offset of all the vertices as in this paper (maps $\mathbf v_i \in \mathbb R^3$ to $\Delta \mathbf v \in \mathbb R^{6890 \times 3}$). Although it is more of a fault of the LVD paper, maybe a footnote can be added in this submission to clarify this to avoid confusion for the audience.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper is about registration of human meshes. Every vertex of a given a query mesh has to be assigned to a vertex of some target mesh. That is a crucial step in many pipelines for high-quality 3D reconstruction of human bodies. The authors propose a cross-over between a neural field (NF) and the classical iterative closest point (ICP) algorithm. A variant of a previous algorithm (LVD) is proposed. This variant uses multiple output heads that are specialized for different areas of the meshes which requires prior shape segmentation.

### Strengths
- registration is a very hard problem that is very important
- the loop between NF and ICP is sound
- a working algorithm for test-time adaption is pretty cool

### Weaknesses
My main objections against this paper are in the experimental evaluation: 
- There is no evidence how well the algorithm works or whether the performance stems from a better trained NF. The authors should show results of just  s single correspondence estimation (via the NF) without any test-time adaption. 
- Similarly, the impact of the multiple output heads is not evaluated. Does the proposed algorithm work with just a single output head? 
- The results show shapes after SMPL fitting. This raises the question how much important this last step is? How do results look before this fitting? Are the error metrics computed only on the 690 sample points or on all points of the SMPL shapes?
- What happens if the shape segmentation fails? 
- Supplementary, Fig 3: While many of the red points are outlying, the set of points used for optimization might not be same as outlying. Which of the red points have been selected for optimization? How susceptible is the algorithm to outliers?

### Questions
See above

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a neural field (NF) refinement method using ICP principles and a pipeline for human body registration using a new variant of learned vertex descent (LVD). Experimental results validate the design and robust performance.

### Strengths
- The overall design is intuitive and straightforward, and the method is easy to replicate.

- Ablation experiments quantitatively demonstrate the design of this method.

- The experiments show good generalization to real challenging data.

### Weaknesses
- By reading the method part of Section 4, including the main content of the method in Section 4.1 and the implementation in Section 4.2, I feel that there are limited new and engaging contributions.
This method is like a new application of existing methods, Neural Fields (NF) and Learning Vertex Descent (LVD).
This reliance on existing works may raise concerns about the substantial contribution of the paper.

- In Table 3, the performance of this method after refinement is greatly improved. What would be the result if the refinement was not performed in Tables 4 and 5? What would be the performance if other baseline methods also used the refinement step?
This makes me concerned about the real performance of the method.

- Why is there no comparison with more methods in the BEHAVE data set in Table 6?

- How efficient is this method compared to baseline methods?

- The symbols NFICP and NF-ICP are mixed, which is not standardized.

### Questions
Please address the concerns in the Weaknesses part.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
