# Policy-Based Trajectory Clustering in Offline Reinforcement Learning

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We introduce the task of clustering trajectories in offline reinforcement learning (RL) datasets to address the multi-modal nature of offline data. Such datasets often contain trajectories from diverse policies, and treating them as a single distribution can obscure structure and increase distributional shift. We formalize trajectory clustering by linking the KL-divergence of offline trajectory distributions with mixtures of policy-induced distributions. 

To solve this, we propose Policy-Guided K-means (PG-Kmeans) and Centroid-Attracted Autoencoder (CAAE). PG-Kmeans iteratively trains behavior cloning policies and assigns trajectories based on generation probabilities, while CAAE adopts a VQ-VAE style objective to guide latent representations toward codebook entries. We prove finite-step convergence of PG-Kmeans and analyze the ambiguity of optimal solutions caused by policy-induced conflicts. Experiments on D4RL and GridWorld show that PG-Kmeans and CAAE partition trajectories into coherent clusters and offer a framework for structuring offline data, with applications in data selection, curriculum learning, and policy transfer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes policy-based trajectory clustering as a formal and principled research problem within offline RL. The authors argue that standard offline RL methods, value-based, uncertainty-based, and behavior cloning, ignore the heterogeneity of mixed-policy datasets, leading to instability and degraded performance. They formalize the problem by connecting trajectory clustering to mixtures of policy-induced distributions and relate its computational complexity to the K-coloring problem, showing NP-completeness. Two complementary algorithms are introduced: 1) Policy-Guided K-Means (PG-Kmeans, which iteratively trains policy centroids via behavior cloning and assigns trajectories based on likelihood under each policy, ensuring convergence in finite steps. 2) Centroid-Attracted Autoencoder (CAAE), which is a representation learning approach that uses a VQ-VAE-inspired codebook regularization to attract latent trajectory embeddings toward cluster centroids while maintaining reconstruction fidelity. Experiments on GridWorld and D4RL benchmarks demonstrate better NMI scores compared to deep clustering baselines (DEC, VAE+KMeans, VQ-VAE) and policy-based alternatives (SORL, Return+KMeans).

### Strengths
The authors introduce 1) PG-Kmeans, a mixture-model interpretation of clustering via policy likelihoods, bridging the EM framework and behavior cloning, and 2) CAAE, which merges deep representation learning with policy semantics, addressing quantization errors inherent in VQ-VAE.

### Weaknesses
1) Experiments are limited to small synthetic and mid-scale D4RL tasks. The applicability of the proposed methods to high-dimensional, real-world offline RL data (e.g., robotics or vision-based control) remains untested.

2) PG-Kmeans, while conceptually sound, may face computational bottlenecks as it requires iterative policy retraining per cluster. There is limited analysis of computational complexity relative to dataset size or cluster count.

3) Reliance solely on NMI might obscure nuances in policy similarity. Complementary metrics such as behavioral KL-divergence or policy return similarity could provide a richer evaluation.

### Questions
Q1: The reduction to K-coloring highlights non-uniqueness. In practice, how do PG-Kmeans and CAAE behave when multiple near-equivalent solutions exist?

Q2: How would the proposed clustering framework behave if the reward function varies over time or across subsets of the dataset (e.g., due to non-stationary objectives or environment drift)?

Q3: The best-of-N selection and merging procedure in PG-Kmeans improves robustness, but what is the asymptotic complexity in N (dataset size) and K (cluster count)?

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
3

### Summary
This paper introduces two policy-based trajectory clustering algorithms, Policy-Guided K-means (PG-Kmeans) for maintaining central policies per cluster and Centroid Attracted Autoencoder (CAAE) for dataset-level clustering.

### Strengths
This paper is well-written and easy to follow, and explores an interesting field, trajectory clustering in offline RL. The paper conducts various analyses to validate the proposed models in experiments and the appendix.

### Weaknesses
As the authors mentioned, the paper's completeness is limited by a lack of validation through experiments with large-scale datasets and by further theoretical analysis. Please see the questions.

### Questions
**Questions**

Q1. What is the strength of the proposed model compared to GMMs and Diffusion-based policy-based clustering?

Q2. The authors assume deterministic behavior policies $\{\pi_i\}_{i=1}^k$. Do you mean policy should be deterministic, i.e., $\pi(s)$? Is the proposed method not applicable to a stochastic policy $\pi(.|s)$?

Q3. It is worth including the trajectory clustering methods in MARL, such as TRAMA [1], for a comprehensive literature review, as it explores policy-conditioned agents utilizing VQ-VAE in MARL settings. 

[1] Na, H., Lee, K., Lee, S. and Moon, I.C., 2025, March. Trajectory-Class-Aware Multi-Agent Reinforcement Learning. In The Thirteenth International Conference on Learning Representations.

Q4. The variable $\mathcal{P}(\tau|\theta_j)$ is not explicitly defined. 

Q5. $K$ and $k$ are interchangeably used. Why not use the same one to avoid confusion? What happens when $k*> k$ in Algorithm 1? Is it valid to assume that $k$ can always be selected so that $k > k^*$, as the authors mentioned in the overparameterization-and-merging approach?

Q6. What are the $\alpha$ and $m$ in Eq. (4)? And how to determine them?

**Minor Comments**

C1. In page 3, “by repeating playing$\pi_i$” has missing space.

C2. Please check citation format at “SORL Mao et al. (20924) in page 5.

C3. Tables 1,3,6,7,8 should be adjusted so that it does not intrude into the margin.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose two methods, Policy Guided Kmeans (PG-Kmeans) and Centroid-Attached AutoEncoder (CAAE), that can cluster trajectories in an offline reinforcement learning dataset. PG-Kmeans adopts an expectation–maximisation algorithm that alternates between the M-step, where each policy $\pi_j$ is updated through behaviour cloning on the trajectories assigned to cluster $j$, and the E-step, where each trajectory is assigned to the cluster $j=\arg\max_i\sum_{(s, a)\in\tau}\log\pi_i(a\mid s)$. CAAE utilises a GRU to encode each trajectory into a single latent vector and trains a decoder $D(z, s)$ to predict $a$ for each $(s, a)$ in $\tau$. The latent vectors are regularised towards the closest centroid in the codebook of size $k$. The paper conducts experiments on multiple heterogeneous offline RL datasets, demonstrating the effectiveness of their algorithms.

### Strengths
The authors make an interesting connection between trajectory clustering and the colouring problem, and point out the inherent ambiguity of the formulation. Additionally, empirical analysis demonstrates that both proposed algorithms can cluster with high accuracy across a range of heterogeneous offline RL datasets.

### Weaknesses
Trajectory clustering by itself has very limited usage in real-world settings, as it is challenging to gather high-quality data. Although the authors do provide several meaningful applications of trajectory clustering in lines 49-67, they do not validate their claims with empirical results, which raises some concerns about the significance of the work. Although, Section D presents some empirical evidence where clustering improves the performance of the downstream algorithm, CQL and IQL are pretty outdated and there are currently a lot of algorithms that perform much better than the clustered variant of CQL and IQL. Finally, the PG-KMeans algorithm can be viewed as a repackaging of the EM algorithm to policy clustering with a few minor changes, such as the usage of discrete weights, making its contribution very incremental.

### Questions
Could you please elaborate on the VQ-VAE baseline used in Section 6.2?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
To tackle the challenge in offline RL with multi-modal data, the work introduced an approach to capture shifting trajectory distributions using policy-guided k-means and autoencoder-based methods. The proposed approach shows superior performance in tasks of D4RL and GridWorld compared to some clustering methods.

### Strengths
- The paper investigated an important challenges in offline RL, to handle the shifting trajectory distributions in data.
- The experimental results on a set of tasks in D4RL and GridWorld show the superior performance of proposed method in terms of  normalized mutual information, compared to clustering baselines.

### Weaknesses
- Motivations are not very clear to me. I believe the investigated problem is important, but the discussions and claims in paper are not closed linked to the investigated problem (multi-modal or heterogeneity of offline datasets). For example: 1) The multi-modality is indeed a challenge in RL, but I don’t fully agree with the authors when they simple discussed the policy distribution-shifting scenario when introducing multi-modality, that could usually be also considered as mixture types of input data (audio, video, etc.). 2) Why trajectory clustering is needed in the offline RL scenarios? If we say stability in policy learning and data efficiency, why trajectory clustering is necessary instead of existing offline RL approaches? I would like to see either why all existing offline RL approaches cannot handle this challenge, or comparison between the proposed approach and state-of-the-art offline RL methods in D4RL and GridWorld. If we say interpretability, how does the advances of the proposed approach in terms of interpretability? Maybe this could be further demonstrated via concrete case studies.
- Experimental design seems insufficient and may not well support the claimed contributions. 1) Besides some my comments above in linking the motivation to proposed methods that may need further experimental justifications, it’s great to see the authors designed to combine clustered trajectories with downstream RL policy training. However, only traditional CQL and IQL are considered, while I would encourage a thorough evaluation over varied types of popular RL algorithms to demonstrate the effectiveness of using the proposed method instead of simply using the existing RL work. 2) The work only selected a small part of tasks (medium-expert) in D4RL, while the authors discussed that they hope multiple deterministic policies and I appreciate they admit the use of small scale of datasets in their limitation statement, given the proposed work is targeting a general purpose and claimed contributions,  I would still highly encourage a broader testing (maybe medium, medium-reply of your tasks on D4RL, and other testbeds such as Adroit, RL Unplugged, NeoRL, etc.) of the proposed method to justify its claimed superiority and necessary in offline RL.  
- While lacking of comparison to directly using existing offline RL algorithms on the heterogenous datasets, it seems not very encouraging of incorporating the clustering method into downstream RL policy learning (according to table 9 in appendix).

### Questions
- First, please see weaknesses above
- It’s also unclear to me how you implement the baseline methods in your experiments to achieve a comprehensive comparison. Further details are appreciated.

### Soundness
2

### Presentation
2

### Contribution
2
