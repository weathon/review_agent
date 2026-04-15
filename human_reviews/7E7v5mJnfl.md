# PuzzleFusion++: Auto-agglomerative 3D Fracture Assembly by Denoise and Verify

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 8

## Abstract
This paper proposes a novel “auto-agglomerative” 3D fracture assembly method, PuzzleFusion++, resembling how humans solve challenging spatial puzzles. Starting from individual fragments, the approach 1) aligns and merges fragments into larger groups akin to agglomerative clustering and 2) repeats the process iteratively in completing the assembly akin to auto-regressive methods. Concretely, a diffusion model denoises the 6-DoF alignment parameters of the fragments simultaneously,
and a transformer model verifies and merges pairwise alignments into larger ones, whose process repeats iteratively. Extensive experiments on the Breaking Bad dataset show that PuzzleFusion++ outperforms all other state-of-the-art techniques by significant margins across all metrics In particular by over 10% in part accuracy and 50% in Chamfer distance. We will release code and model.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper looks at the task of object reassembly in which a set of fractures are given and the goal is to assemble them into an object. This is a challenging problem and has applications in computer vision, computer graphics and robotics. The paper proposes a method called puzzlefusion++ which solves the problem by aligning and merging the fragments into larger groups (think of this as a clustering processing) and then iteratively completing the assembly. Results show that the proposed method make some improvements over the competing methods.

### Strengths
1. the proposed method makes sense and I believe this is the way to solve this problem as in doing some clustering first and then merging them (unlike prior work which tries to predict a pose for each fragment in one single pass).

2. performance is good compared to state-of-the-art methods. this means that the proposed method is effective

3. presentation is good and the paper is easy to follow

### Weaknesses
1. speed is significantly slower than some methods (see table 1)

2. can the proposed method solve the reassembly task when the number of fractures is big (eg 100)

3. how well can one expect the proposed method to work on unseen objects?

### Questions
good paper, but i still have quesitons. please see the weaknesses above 

minor question: which software, library is used to render images? the rendering is very professional

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
5

### Summary
This paper introduces PuzzleFusion++, which addresses the 3D fracture assembly problem. It employs a diffusion model enhanced by a proposed autoencoder, optimized denoise scheduling, denoiser structure, and an auto-agglomerative verifier. These advancements collectively contribute to performance that surpasses that of previous work.

### Strengths
The paper demonstrates performance improvements over previous work. I like the idea of using a verifier to assess the quality of the assembly results.

### Weaknesses
Despite significant efforts on ablation studies of the proposed method, the paper does not clearly differentiate from prior works. A valuable contribution would be for the authors to provide (component level) comparisons with similar methods, such as PuzzleFusion and DiffAssemble. It would be beneficial to highlight that what limits the existing approaches and what contributes to the success of the proposed one. For now, the uniqueness of this paper is not clear.

### Questions
1. With the name of this paper, it’s hard not to compare it with PuzzleFusion that have a similar motivation in tackling jigsaw puzzle related problem. Could PuzzleFusion be adapted to handle 3D tasks as a baseline for comparison? What distinguishes PuzzleFusion++ from PuzzleFusion, particularly in features that enhance its suitability for 3D tasks?

2. The matching from Jigsaw is not perfect, is there a scheme to handle such situation? Meanwhile, the provide upper bound based on perfect matching is only 34 degree error. This is still very high. If Jigsaw can have such perfect matching, the error can be reduced to less than 10 degrees. This raises questions about the suitability of using a diffusion model for this type of problem.

3. Even for comparison among diffusion models, why PuzzleFusion++ can significantly outperforms DiffAssemble? Can the authors describe the key difference that make your model works better?

### Soundness
2

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
This paper proposes PuzzleFusion++, a framework for 3D fracture assembly. A fully neural auto-agglomerative design is proposed that simulates human cognitive strategies for puzzle solving. Moreover, a diffusion model enhanced with feature embedding designs is devised that directly estimate 6-DoF alignment parameters.

### Strengths
1.	This paper is well-written.
2.	The motivation is clear enough.
3.	The organization of this paper is great.
4.	The alignment verifier is well designed.

### Weaknesses
1.	The Figure 3 is a little puzzled. A better format is recommended.
2.	The showed 3D objects seems relatively simple. More complicate objects from Objaverse are commended.

### Questions
1.	The main difference between PuzzleFusion and PuzzleFusion++ should be clarified in the paper, since it is a future work from PuzzleFusion.
2.	Are 25 points enough to represent a fragment in Sec.3.1?
3.	Does the pairwise alignment verifier work well for all objects?
4.	In auto-agglomerative inference, are 6 iterations enough for merging? Have you try more iterations?
5.	Please discuss could reinforcement learning whether is help for this task.
6.	As mentioned in Weakness, could you provide more complicate objects during rebuttal?
7.	Please discuss the possible solution for your limitations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Authors improve over existing work on 3d fracture assembly by proposing an iterative approach. The proposed method iteratively denoised poses for current fragments and merge aligned fragments into clusters until all fragments are merged into a single 3D object. They build over existing work (Scarpellini et. al, Wu et al.) regarding the diffusion model formulation for alignment. The novelty regards adopting a transformer for clustering fragments.

### Strengths
- Authors' proposal is easy to follow and well-thought
- Noise scheduler improves performances by a surprising margin. Since different noise schedulers do not change training dynamics and effects only MC estimate of the loss (Kingma et al. 2024), my guess it's that this scheduler makes training more efficient. I'd like to see more comments on that, either in main paper or supplementary. Right now it's relegated to a small section in supplementary and not discussed in main paper.
- Good ablations section
- Method is somewhat novel. There's no technical novelty since it adopts existing methods and approaches (iterative refinement has been adopted by Ken et al, diffusion model formulation was in Scarpellini et al). Still the combination of these modules make outperform the proposed baselines so I would not consider this a weakness

Kim, Jinhyeok, Inha Lee, and Kyungdon Joo. "Fracture Assembly with Segmentation And Iterative Registration." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024.

Kingma, Diederik, and Ruiqi Gao. "Understanding diffusion objectives as the elbo with simple data augmentation." Advances in Neural Information Processing Systems 36 (2024).

### Weaknesses
- L158  anchor fragments were introduced in Jigsaw (Lu eta al) and are not adopted in original BreakingBad. Comparing anchor-based methods to methods that do not adopt anchors is a bit unfair. A fair comparison would entail re-training those baselines with anchors (e.g., L212-214 regarding alignment could be adopted in all other baselines). I believe this point makes the experimental section weaker and does not reflect the actual performances of the baselines.
- Missing baselines: authors should also compare to other existing methods that were published before ICLR deadline: PHFormer: Multi-Fragment Assembly Using Proxy-Level Hybrid Transformer (Cui et al, AAAI2024), Fracture Assembly with Segmentation And Iterative Registration (Kim et al., ICASSP2024). Both achieve impressive results and are not cited by the authors--especially the latter achieves RMSE R 23.8, RMSE T 7.30, PA 74.50.

### Questions
- L150-L154 what guarantees that the input representation are equivariant (aka "sensitive to rotation"). Normalizing wrt center mass achieves invariance, but not necessarily equivariance.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes PuzzleFusion++, an "auto-agglomerative" method for the task of 3D fracture assembly.
Specifically, the proposed pipeline undergoes a iterative process of using a diffusion model to predict a 6-DoF alignment for each fragment, followed by a transformer model which verifies and merges pairwise alignments into larger ones. 
This is much alike how humans assemble fragments - hypothesizing how two fragments fit together, and checking if it the alignment is indeed ture. 
To encode fragments into latent vectors suitable for diffusion training, the authors integrate PointNet++ and VQVAE.
PuzzleFusion++ achieves state of the art on the Breaking Bad dataset by a large margin, and comprehensive analysis clarifies the significance of each introduced module.

### Strengths
- Novel approach ideated as 'auto-agglomerative', which has been implemented through the integration of a diffusion model (SE3 denoiser) and a transformer model (pairwise alignment verifier). 

- The manuscript is overall well-written and easy to follow.

- Comprehensive analysis experiments, which validates the design choices of PuzzleFusion++ and clarifies how each design choice affects the performance.

- Strong performance on the Breaking Bad benchmark, outperforming existing methods by a large margin.

### Weaknesses
- While Figure 3 provides valuable details into how the SE3-denoiser and Pairwise Alignment Verifier work, it is not very visual, and not straightforward to understand as-is. I believe providing Figure 3 as a form of an algorithm may improve its readability.

- The authors mention that "different anchor initialization does not affect the quality of the assembly results (L482)"; however, it can be seen that the results in Table 7 is closer to the results for Ours(#ite=1) in Table 2. I believe it is an overstatement to mention that anchor initialization 'does not affect' the quality of the assembly results - unless a single iteration (#iteration=1) was used for the results in Table 7. It would be informative to provide the results for varying number of iterations for random initializations of the anchor fragment.

- The paper is missing a recent baseline for 3D assembly, which seem to show strong performances on the Breaking Bad benchmark as well: 
Nahyuk Lee et al, 3D Geometric Shape Assembly via Efficient Point Cloud Matching, ICML 2024.


- Minor writing mistakes: 
1) Table 4: Autoncoder -> Autoencoder 
2) L 454: Denioser -> Denoiser

### Questions
- Why was the number of iterations set to 6? Table 2 shows that the results are best at # iterations = 6. At what number of iteration does PuzzleFusion++ achieve the best results, without further increase in performance with increasing number of iterations?

- In Table 4, what are the results when the scheduler is x (i.e., not the proposed scheduler), while the Autoencoder and Anchor fragment are being used? It would be helpful to include this result in the ablation, for improved clarity of the ablation results.

- Figure 9 visualizes the difference between 3 noise schedulers - could the authors include a quantitative/qualitative comparison of all 3 noise schedulers? The proposed rationale of "locating more denoising budgets to getting precise alignments than moving fragments to the rough locations" sounds intuitive, and it would certainly help to have additional results to validate this claim, beyond the results in Table 4. 

The idea of the proposed PuzzleFusion++ is interesting and novel, and also shows strong empirical results. While the paper can be still improved by including more recent baselines and providing more comprehensive results, I believe that the strengths of the paper outweigh its weaknesses as-is. I am leaning towards accept, and am willing to improve my score if my questions / weaknesses are addressed.

### Soundness
3

### Presentation
3

### Contribution
3
