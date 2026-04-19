# Multi-Objective Molecular Design through Learning Latent Pareto Set

- Decision: Reject
- Scores: 5, 3, 6

## Abstract
Molecular design inherently involves the optimization of multiple conflicting objectives, such as enhancing bio-activity and ensuring synthesizability. Evaluating these objectives often requires resource-intensive computations or physical experiments. Current molecular design methodologies typically approximate the Pareto set using a limited number of molecules. In this paper, we present an innovative approach, called Multi-Objective Molecular Design through Learning Latent Pareto Set (MLPS). MLPS initially utilizes an encoder-decoder model to seamlessly transform the discrete chemical space into a continuous latent space. We then employ local Bayesian optimization models to efficiently search for local optimal solutions (i.e., molecules) within predefined trust regions. Using surrogate objective values derived from these local models, we train a global Pareto set learning model to understand the mapping between direction vectors (called ``preferences'') in the objective space and the entire Pareto set in the continuous latent space. Both the global Pareto set learning model and local Bayesian optimization models collaborate to discover high-quality solutions and adapt the trust regions dynamically. Our work is the first endeavor towards learning the Pareto set for multi-objective molecular design, providing decision-makers with the capability to fine-tune their preferences and thoroughly explore the Pareto set. Experimental results demonstrate that MLPS achieves state-of-the-art performance across various multi-objective scenarios, encompassing diverse objective types and varying numbers of objectives.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents a multi-objective optimization algorithm designed for molecular structure design. The algorithm introduces an approach by encoding discrete molecular structures into a continuous latent space, subsequently optimizing the latent Pareto set through a combination of global optimization and local search techniques. Experimental results demonstrate that MLPS achieves state-of-the-art performance.

### Strengths
1. The integration of local search and global search is interesting.
2. The proposed algorithm achieves the state-of-the-art performance.

### Weaknesses
1. The ablation study is not enough to demonstrate the effectiveness of each component in the proposed algorithm. It is unclear what is meant by "without local models." Was a single surrogate model built for performance prediction? Additionally, the number of evaluations in Figure 2 is 2000, while in Table 1 is 5000. However, the HV value in Figure 2 is larger than the HV value in Table 1, which is very confusing. 

2. The idea of optimizing in a continuous latent representation has been extensively explored in the field of neural architecture search (NAS). Many algorithms exist that optimize discrete graphs by encoding them into continuous space and employing surrogate models for performance prediction (e.g., [1][2][3][4]). Since molecular optimization and NAS are very similar, it is important to mention and compare these existing works.

3. This paper aims to learn the latent Pareto set; however, the experiments only showcase a limited number of solutions. It would be beneficial to sample a large number of $\lambda$ to provide an accurate approximation of the Pareto front.

[1] Neural Architecture Optimization with Graph VAE, NeurIPS 2018

[2] NSGANetV2: Evolutionary Multi-Objective Surrogate-Assisted Neural Architecture Search, ECCV 2020

[3] Bridging the Gap between Sample-based and One-shot Neural Architecture Search with BONAS, NeurIPS 2020

[4] BRP-NAS: Prediction-based NAS using GCNs, NeurIPS 2020

### Questions
1. Why do you use multiple local surrogates? We can use a single global surrogate to perform local search, which is much cheaper and more straightforward.
2. Why do you choose Tchebycheff scalarization instead of linear scalarization?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Through this paper, the authors aim to establish a method that can efficiently handle multi-objective molecular design scenarios by learning the comprehensive Pareto set in a latent space. To accomplish this, the authors propose to utilize an encoder-decoder model to transform the chemical space into a continuous latent space.

### Strengths
- The writing is easy to follow. The concept figure also aids the understanding.

### Weaknesses
I will combine the *Weaknesses* section and the *Questions* section. My concerns and questions are as follows:
- The authors did not provide the codebase to reproduce the results.
- The main weakness of the paper is that the experiment results are weak and insufficient.
   - The effectiveness of the proposed model is verified on a single kind of tasks introduced in [1]. I highly recommend reporting results with more benchmarks such as the multi-property objective (MPO) tasks of the PMO benchmark [2].
   - The proposed model is a generative model, but there is no visualization showing the molecules generated.
   - There are other models that used the same benchmark, such as MolEvol [3] and RetMol [4]. I recommend to include those methods to the baselines and report the performance (Table 1) and visualize the solution space (Figure 6).
- The authors claimed the existing Pareto set learning models like P-MOCO and PSL have limitations in Related Work, but they were not selected as baselines in the experiments. All the baselines in Table 1 do not utilize Pareto optimization. I highly recommend to include those methods to the baselines and rigorously show the specific advantage of the proposed method compared to existing Pareto set learning methods.

For now, I’m leaning toward reject, but I’ll be glad to raise the score when all my concerns are fully resolved.

---

**References:**

[1] Xie et al., Mars: Markov molecular sampling for multi-objective drug discovery, ICLR 2021.

[2] Gao et al., Sample efficiency matters: a benchmark for practical molecular optimization, NeurIPS 2022.

[3] Chen et al., Molecule optimization by explainable evolution, ICLR 2021.

[4] Wang et al., Retrieval-based controllable molecule generation, ICLR 2023.

### Questions
Please see the *Weaknesses* part for my main questions.

---

**Miscellaneous:**
- Section 4.2, the first paragraph and in Table 1, *RationalRL* -> *RationaleRL*
- Figure 2, title, *JNK* -> *JNK3*

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose MLPS, a new approach for learning the latent Pareto set targeting multi-objective molecular design.
The proposed scheme utilizes widely used encoder-decoder model that casts discrete and high-dimensional chemical space to a continuous and low-dimensional latent space.
MLPS tries to train a "global" Pareto set learning model that can map preference vectors to optimal points in the Pareto set.
Under several multi-objective molecular design scenarios (ranging from the optimization of two objectives to the optimization of up to four objectives), the proposed MLPS is shown to have advantages over other popular molecular design schemes based on the hypervolume of the generated molecules.

--- [Added after the rebuttal/discussion phase ] ---

The authors have provided a comprehensive rebuttal to the review comments, which has sufficiently clarified several ambiguities in the original manuscript and addressed many of the concerns raised in the original review comments.
Furthermore, the additional results, elaborations, and discussions provided by the authors have further improved the manuscript and present stronger support for the main contributions made in this work as well as their significance.
The evaluation scores have been updated accordingly to reflect this.

### Strengths
Overall, the manuscript is written well in a clear and easy-to-understand manner.
The proposed MLPS is well-motivated and the proposed approach is reasonable.
An effective global Pareto set learning model, as proposed in this current work, that can efficiently map preference vectors to the corresponding Pareto optimal solutions would enable fast exploration of the Pareto optimal molecular set to identify optimized molecules that meet the multi-objective design criteria and balance the trade-offs between multiple properties as desired.

### Weaknesses
While the proposed MLPS scheme is well-motivated, interesting, and reasonable, there are several major & minor concerns regarding the current manuscript.
These are elaborated below.

1. Throughout the manuscript, the authors make strong claims that may be misleading or unsubstantiated.

For example, it is claimed that "the work is the first endeavor towards learning the Pareto set for multi-objective molecular design", which is clearly not true unless they substantially narrow down the definition of "learning the Pareto set" and specify what type of "learning problem" they are focusing on.

It is claimed, MLPS is a general framework that is compatible with "any plug-in encoder-decoder with continuous latent representations. 
Although the reviewer agrees that this is reasonable speculation, the authors do not show whether the proposed MLPS scheme can indeed be easily applied to different generative molecule design models with such architecture and latent molecular representation.
Furthermore, there are experimental results showing whether the incorporation of MLPS would indeed lead to enhanced molecular design for different types of models.

2. At the core of training effective local surrogate models lie two central issues: how to set the center for each local trust region and how to reinitialize.
The proposed strategy appears to be reasonable, but in-depth discussion and empirical analysis are missing.
For example, it is not discussed how frequently different trust regions may intersect or collapse with one another (effectively reducing the number of distinct trust regions) and how often trust regions need to be reinitialized in practice (e.g., for the multi-objective scenarios considered in the results section).

3. The paper compares MLPS to several other existing methods (GA+D, JT-VAE, GCPN, RaionaleRL, MARS, and MolSearch) but it is unclear how these baselines were used for "multi-objective" molecular optimization since they are not necessarily inherently designed for multi-property optimization.
Was each baseline used to randomly sample novel molecules?
Was scalarization based on a specific (or randomly sampled) preference vector used to turn MOO into SOO?
Or was multi-objective BO used for optimization in the latent space?
This choice will have a tremendous impact on the performance of each baseline (both in terms of the property of the optimized molecules as well as the sample efficiency), hence needs to be clearly described.

4. As the performance metric, only the HV (hypervolume) of the optimized molecules was used.
However, in addition to such "aggregated" performance metric, it would be meaningful to compare the individual properties as well (e.g., average property score, average property of top molecules, scatter plot - at least for the two-objective scenario, etc.)

5. There is no discussion on the impact of various hyper parameters.
For example, what is the impact of varying the number of trust regions n_tr, minimum edge length L_min, or number of iterations Tg for training?
This needs to be clearly discussed and empirically tested.

6. There is currently no discussion on the computational cost of MLPS and its scalability with respect to the number of objectives.
At least a high-level discussion and some empirical results need to be provided.

### Questions
Please see the comments above.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good
