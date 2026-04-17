# Adversarial Encoding Perturbation and Synthesis for Set Representation Auxiliary Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Sets are a fundamental data structure, and learning their vectorized representations is crucial for many computational problems. Existing methods typically focus on intra-set properties such as permutation invariance and cardinality independence. While effective at preserving basic intra-set semantics, these approaches may be insufficient in explicitly modeling inter-set correlations, which are critical for tasks requiring fine-grained comparisons between sets. In this work, we propose SRAL, a Set Representation Auxiliary Learning framework for capturing inter-set correlations that is compatible with various downstream tasks. SRAL conceptualizes sets
as high-dimensional distributions and leverages the 2-Sliced-Wasserstein distance to derive their distributional discrepancies into set representation encoding. More
importantly, we introduce a novel adversarial auxiliary learning scheme. Instead of
manipulating the input data, our method perturbs the set encoding process itself and
compels the model to be robust against worst-case perturbations through a min-max
optimization. Our theoretical analysis shows that this objective, in expectation,
directly optimizes for the set-wise Wasserstein distances, forcing the model to
learn highly discriminative representations. Comprehensive evaluations across
four downstream tasks examine SRAL’s performance relative to baseline methods,
showing consistent effectiveness in both inter-set relation-sensitive retrieval and
intra-set information-oriented processing tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a new method, SRAL, for set representation learning. To encode a set, the authors first extract distributional features from it using sliced wasserstein distances rank-matching to a learned canonical set. These set representations are then inserted into a contrastive loss for learning the canonical set embeddings. To choose the positive pairs, the authors augment each set by adding noise to its elements and then extracting the new noisy set embeddings. Specifically, to increase the augmentations effect, they optimize an adversarial noise direction in an inner loop.

### Strengths
- While writing clarity could be improved, the proposed method appears simple and intuitive for set representation.

- The analsys presented in remark 1, showing the connection between the proposed loss metric and a sliced wasserstein based contrastive loss, is interesting.

### Weaknesses
- **Writing clarity of method sections.** Sections 2-3 would benefit from simplified presentation. Although the proposed method appears simple, it is hard to follow. Specifically, it is notation-heavy (the notation table in the appendix is almost a page long) which further reduces clarity. Rewriting these sections in a simplified manner can greatly improve the paper's clarity.

- **Limited baseline comparisons.** The experimental section is missing several key comparisons to more recent methods. Moreover, when SRAL is compared to a more recent method, CrossCBR, it shows only marginal gains. The paper would be strengthened by adding comparisons to [1,2,3] on tasks 1 and 4 to evaluate SRAL against more recent approaches.

- **Marginal gains in tasks 2 & 3.**  The results of SRAL on tasks 2 and 3 show only marginal gains over the baselines. Additionally, on Task 2, SRAL is integrated into CrossCBR rather than evaluated standalone. Providing standalone results for Task 2 would help clarify SRAL's direct contribution.


Minor remarks:

- **Figures and tables sizing.** Most figures and tables are quite small and difficult to read, particularly in printed form. Enlarging key results would improve readability.


[1] Lee, Dong Bok, et al. "Self-supervised set representation learning for unsupervised meta-learning." The Eleventh International Conference on Learning Representations. 2023.

[2] Guo, Dandan, et al. "Learning prototype-oriented set representations for meta-learning." arXiv preprint arXiv:2110.09140 (2021).


[3] Lee, Geon, Chanyoung Park, and Kijung Shin. "Set2box: Similarity preserving representation learning for sets." 2022 IEEE International Conference on Data Mining (ICDM). IEEE, 2022.

### Questions
- How does the authors view an integragtion of a powerful LLM into this problem? A discussion on that could be helpful. 

- Except the embeddings of the canonical set V_o, what other parameters are learned in SRAL?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an set representation learning approach that uses optimal transport to compare input sets and learnable references. The optimal transport is approximated via sliced Wasserstein distance (SWD) where the number of slices R is a hyperparameter (that trades between accuracy and computation time). In addition to the sliced Wasserstein distance used to encode sets (thus achieving permutation invariance and cardinality-independence), the authors also use adversarial-robust learning (in a min-max formulation) to be robust against the perturbations of encoded set features. The authors show that in a variety of tasks, using SWD to encode sets together with the adversarial-robust learning exceeds state of the art results.

### Strengths
This is a nice experimental paper, that extensively evaluates their proposed method and shows good results in all the tasks considered. The ablations also support the conclusion that both the OT-based encoding and adversarial-robust learning help to improve the performance in the tasks considered.

### Weaknesses
Unfortunately, OT in set-representation learning is not new, see Skianis et al. for what seems to be the first OT-approach. The authors do cite this paper in passing but not in detail. In that paper the OT approach is presented as a bipartite graph optimization, which is equivalent to the linear programming formulation for solving OT problems. The authors should discuss that paper in detail and compare their method against theirs (in a fair way, e.g. using the same encoders etc.) Interestingly, whereas the authors use the sliced Wasserstein distance to approximate the OT, the authors in that paper take another approach: by dropping one of the constraints, Skianis et al. reduces OT to the semi-relaxed case (unbalanced OT with one of the penalties going to infinity). The authors should also extend their comparisons to Skianis et al. by comparing the various approximations.

### Questions
- I found the intro to be hard to crack, given that I have a background in OT and deep learning but not set representation learning (SRL). I had to look into some of the author's references to understand the topic better. Perhaps the problem description in Preliminaries could be extended with a figure or a motivational example could be given to introduce SRL to a wider audience.
- how do you define a CDF in d-dimensions (line 159) ? You can only do this after projecting into 1-d (unless you introduce 'copulas'!)
- not clear from intro how the 'reference distribution' O is learned!
- For Proposition 1 it is enough to cite a standard book on OT, e.g. Peyre and Cuturi's book. As it is a well known result, no need to cite it as a Proposition, because it's not the author's contribution.
- Similarly for Proposition 2, sorting and matching the quantiles is well known in the literature as a solution, no need to state it as a proposition.
- It is not clear how the features z_{i,k} of the sets are obtained from the intro.
- Besides Skianis et al., the papers by Mialon et al. (2021) [OTKE] and Kim (2022) both discuss OT-approaches in a kernel-method context. These and other OT-based methods should be discussed in detail and compared against.
- Are all datasets in the experiments supervised? Discussing the experiments a bit better in the main text would be better.
- Figure 3: A and B captions should be swapped
- very hard to read most of the figures, please revise
- not clear what is the difference between AEPO and AL in the ablation study.

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
4

### Summary
This paper proposes an optimal transport-based approach for learning cardinality- and permutation-invariant representations for set-structured data. A set encoder based on sliced Wasserstein distance is used, and a contrastive-style adversarial loss is added to the objective to learn more informative set embeddings based on worst-case perturbations of the element-wise embeddings. Numerical results on four different tasks show the superiority of the proposed approach over several other set learning methods.

### Strengths
- Learning representations of sets is a critical problem in many application areas of machine learning, so the considered problem is timely and important.
- The paper is written clearly and relatively straightforward to read and follow, even for an audience with minimal background in set representation learning.
- The scope of the presented experiments is broad and acceptable in my view.

### Weaknesses
- It is unclear how the optimal adversarial perturbations are applied to the sets. On line 249, the authors mention that the perturbations are shared between all sets and both views of each set. However, the two views of each set will be identical if the perturbations are equal. Is one view unperturbed and the other view perturbed by $\epsilon^+$? Is $\epsilon^+$ the perturbation used for all sets in a given training batch?
- In the abstract and introduction section, the authors mention that perturbations are not applied to the input features but occur during the encoding process. However, as presented, it does seem that the perturbations are indeed applied to the features *before* the encoding process. If my understanding is correct, the claims in the abstract and introduction need to be revised. Otherwise, if the encoding process itself is to be perturbed, one option is to perturb the projection parameters $\Theta=[w_r]_{r=1}^R$; would that be feasible with your adversarial approach?
- If I understand correctly, the SFE module introduced in Section 3.2.1 is now new and is the same as the one introduced by PSWE [A]. Could you please explain if there are any distinctions between SFE and PSWE?
- The notation at the beginning of Section 2 is confusing. Is the universe $\mathcal{E}$ finite and contains $n$ elements, or is $n$ the cardinality of the set $S_i$? What is the difference between $e_i$'s and $z_i$'s?
- The contribution of Remark 1 is unclear to me. SFE and PSWE both imply that the Euclidean distance between embeddings approximates the Sliced Wasserstein distance before the encoding process, which is unrelated to the used InfoNCE loss function. Also, could you please explain more precisely how you go from Eq. (8) to Eq. (9) (from summation to integration)? Do you need assumptions on, e.g., an infinite number of projections $R$?

[A] Navid Naderializadeh, Joseph F Comer, Reed Andrews, Heiko Hoffmann, and Soheil Kolouri. Pooling by sliced-wasserstein embedding. NeurIPS, 34:3389–3400, 2021.

### Questions
- Is it possible to simplify Eq. (13) by setting $\epsilon^+ = \frac{\pi}{\\|g_{\epsilon^+}\\|} g_{\epsilon^+}$?
- Could you please compare your method's performance with some more recent approaches, such as FSW [B]?

[B] Amir, Tal, and Nadav Dym. "Fourier Sliced-Wasserstein Embedding for Multisets and Measures." In The Thirteenth International Conference on Learning Representations, 2025.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes SRAL (Set Representation Auxiliary Learning), a novel framework for set representation learning that explicitly captures inter-set distributional relationships rather than focusing solely on preserving intra-set structures. SRAL introduces a learnable reference distribution O and encodes each set according to its 2-Sliced-Wasserstein distance to this common reference, thereby aligning all sets within a shared geometric space. Moreover, SRAL incorporates a self-supervised objective with adversarial perturbations applied in the aligned feature space to obtain more robust and discriminative set embeddings.

### Strengths
- Proposes a theoretically grounded approach to set representation learning based on the Wasserstein distance.
- Achieves superior performance compared to existing methods across various benchmark datasets.

### Weaknesses
Motivation for introducing reference set O is a bit unclear to me. In the manuscript, it is only briefly explained: “To learn stable and discriminative set representations, our encoder leverages the distributional distance between an input set and a learnable reference distribution O.” For instance, if my understanding is correct, one could compute an L_wd-like loss without introducing a reference distribution, by directly calculating the sliced Wasserstein distance between each pair of P_i and P_j (with perturbations), and replacing the term -||v_i′ - v_j″||_2 with that sliced Wasserstein distance. Could the authors clarify this design choice or point out what I might be missing?

### Questions
- How were the hyperparameters chosen, and how computationally costly is the proposed method? The approach requires tuning the hyperparameters λ1 and λ2, and according to Table E.1, the optimal values of λ1 vary considerably across different tasks (from 5e-1 to 1e-3). For comparison, do the baseline methods such as FSPool and CrossCBR in the benchmark tables also require similar levels of hyperparameter tuning to achieve the reported performance? It would be interesting to see how the proposed method performs in terms of computational efficiency and cost.

- In Eq. (5), the mapping g+ appears to be non-differentiable. Did you use any soft approximation of the sorting operation, or how do you propagate the gradients with respect to the parameters.

- Typos and minor notes
  - L.107: g: R^d -> R^d
  - L.155: should be "true data distribution P_i" (without a hat)
  - L.159: (perhaps clear from context, but to confirm) inside delta, it should be theta(x) >= theta(z), or x and z should be one-dimensional scalars.
  - L.257: the correct reference should be https://arxiv.org/abs/1412.6572


I will adjust the score based on replies from the authors.

### Soundness
3

### Presentation
3

### Contribution
3
