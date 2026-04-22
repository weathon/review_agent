# Perturbed Flow Matching for Structure-Based Drug Design

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Generating 3D molecules that bind to specific protein targets via generative models has shown great promise in structure-based drug design. Recent diffusion-based approaches are constrained by marginals known in closed form and lack a design of the conditional probability path, which may hinder improved molecular generation. To address this issue, we introduce a flow-matching–based method—Perturbed Flow Matching (**PFM**)—which introduces a unique *perturbed conditional probability path design* that incorporates pocket binding site information and atom type–coordinate coupled information to enhance molecular generation performance. Experiments on the CrossDocked2020 dataset show that our model generates molecules with competitive 3D structures and state-of-the-art (SOTA) binding affinities toward protein targets, achieving an average score of -7.83. Validation on broader molecular datasets further confirms the consistent effectiveness of the proposed method. The code is available at https://anonymous.4open.science/r/pfm_ev_rv.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Perturbed Flow Matching (PFM) is a generative modeling technique that combines ideas from flow matching and perturbation-based methods. It aims to introduce a perturbed conditional probability path between data and noise distributions. A Full-atom Multi-stage Equivariant Network architecture is proposed to model the vector fields required for PFM. The authors present a loss to train the network composed by different terms, Flow Matching Loss, Local Loss (incorporates local structure information), and Clash Loss (prevents atomic clashes). The model is trained on 99,900 selected complexes from CrossDocked2020 dataset and 100 novel complexes for testing.

### Strengths
- the paper formulate how to construct perturbed conditional probability path (PFM) but it is not very clear what are the theoretical benefits of this compared to diffusion based methods
- results show that PFM performs well on the task of protein-ligand complex generation
- the paper provides results on a challenging benchmark for pocket-conditioned ligand generation
- the paper is overall well written even though some parts in the method section need to be simplified for better clarity (especially on the method's notation)
- qualitative figures on the generated ligands are provided

### Weaknesses
The comments here focus on providing constructive feedback to improve the paper.

- The method "perturbed conditional probability path" sound somewhat an incremental modification of existing flow-matching paths rather than a fundamentally new concept, where the path is perturbed in a self-conditioning manner (see [1][2]) during training. 

- The authors do not provide any theoretical justification on benefits of this PFM framework over pre-existing flow matching or diffusion based methods. In Equation 6 is not provided any theoretical reasoning on why this term ξt(1−t)\hat{X}(...) should improve the generation performances nor how it is affecting the flow.

- The final loss is the combination of different losses, but the paper does not provide any ablation on the effect of weighting differently the clash and local loss. In table 5 they provided the results of removing or adding the loss components. I quite don't understand in table 5 why by removing separately the clash and local loss the performances on the vina score (avg) do not improve much, but together they provide a boost of -0.64. Can you provide more insights on this?

- Please specify in the Table 1 that cfg and cg means classifier free guidance and classifier guidance this notation is hard to infer.

- In Table 1, I quite don't understand why you are reporting the performance of PFM with classifier free guidance (cfg) and classifier guidance (cg). Since you are not reporting the results on the baselines with cfg and cg, it is not a fair comparison. Either you report all the baselines with cfg and cg or you just report PFM without any guidance and remove the cg and cfg results from the table.

- In Table 1, it is not clear the advantage of PFM (fair comparison without cfg) over the baselines, it seems even that the overall performances of the baselines looks better on different metrics.

- The author claim a 21x speedup, not reporting the hyperparameters and inference sampling steps for the baselines used. Without this information, the comparison is not very solid.

[1] Ting Chen, Ruixiang ZHANG, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In The Eleventh International Conference on Learning Representations, 2023.

[2] Ross Irwin, Alessandro Tibo, Jon Paul Janet, Simon Olsson. SemlaFlow - Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching, AISTATS, 2025.

### Questions
- Since are not provided any theoretical insights on the benefits of PFM, what is the advantage of PFM compared to diffusion based methods? Can you provide any simple experimental intuition on why PFM is better than diffusion based methods? what is the difference apart from the fact that PFM is flow matching based?

- PFM without classifier free guidance it doesn't seem to perform well on the presented baselines. Can you provide more insights on why this is happening? Why is the classifier free guidance so crucial for the model to work well?

- Can you provide an explanation on why several strong 2023–2025 baselines (e.g. AliDiff (Oct 2024), MolCraft (May 2024), FlowSBDD (Dec 2024), EQGAT (Nov 2023)) are missing? This would significantly reduce the validity of SOTA claims.

I provide here an example
| Method   | Vina Score (↓) |        | Vina Min (↓) |        | Vina Dock (↓) |        | High Affinity (↑) |        | QED (↑) |        | SA (↑) |        | Diversity (↑) |        |
|----------|----------------|--------|--------------|--------|---------------|--------|------------------|--------|---------|--------|--------|--------|----------------|--------|
|          | Avg.           | Med.   | Avg.         | Med.   | Avg.          | Med.   | Avg.             | Med.   | Avg.    | Med.   | Avg.   | Med.   | Avg.           | Med.   |
| PFM      | -7.12          | -7.18  | -7.58        | -7.32  | -8.32         | -8.26  | 72.1%            | 76.5%  | 0.49    | 0.50   | 0.51   | 0.51   | 0.73           | 0.71   |
| ALIDIFF  | -7.07          | -7.95  | -8.09        | -8.17  | -8.90         | -8.81  | 73.4%            | 81.4%  | 0.50    | 0.50   | 0.57   | 0.56   | 0.73           | 0.71   |





- I would suggest to move Table 2 to the Appendix and provide more space to add more qualitative results of generated ligands in the main paper. Also, I don't find useful Table 3, since I don't see relevant improvements over the baselines reported. Please consider to move it to the Appendix as well.

- Sometimes in the paper's notation a clear explanation of some symbols is completely missing (e.g. ξ (Equation 6), µ, ν (Equation 11)). Can you please double check that all the symbols reported have a clear explanation of their meaning?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce a new perturbed flow matching paradigm for structure-conditioned molecule generation and achieve strong results.

### Strengths
The perturbed flow matching is a novel paradigm for structure-conditioned molecule generation.

The results appear to be much better than existing baselines.

### Weaknesses
It’s hard to tell if PFM performs statistically significantly better than the baseline methods. Could the authors report error bars/CIs?

Adding some figures on the model architecture and example model outputs would be helpful.

### Questions
See weaknesses above.

Could the authors compare to DrugFlow, another flow matching method for ligand design? It would be nice to compare two flow matching methods to show the efficacy of PFM.

Have the authors done an ablation on the effect of the classifier guidance with the Vina scores? I wonder if that has a large role to play in the performance of PFM compared to other methods.
Correct me if I’m wrong, but none of the other methods employ classifier guidance with Vina scores to guide generation. This could be a big factor in the strong Vina scores but could lead to over-reliance and overfitting on Vina. In addition, it could be useful to compare to methods such as AliDiff (Xu et al), which explicitly optimize Vina scores with preference optimization. It’s quite different from the classifier guidance of PFM, but the overall setup of explicitly using Vina scores in the training is similar.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Perturbed Flow Matching (PFM), a modification to flow-matching for target-conditioned structure-based drug design. The PFM technique integrates information about the binding pocket to produce a better probability path, leading to better molecular generations with higher binding affinities. As I understand it, this is done via a learned bias towards chemically plausible positions for ligand atoms in the binding site. The results show that PFM produces better quality ligands with higher binding affinities than many previous SBDD models.

### Strengths
1. The PFM technique is reasonable, I think, but it's hard to understand how exactly it is done and the intuitive motivation for doing so
2. The results seem strong, and the change in Vina score is significant between baselines
3. The selected baselines represent a good range of SBDD techniques, especially the SOTA diffusion methods

### Weaknesses
1. To me, the paper was very hard to read and the concept was difficult to understand. I don't really understand the intuitive motivation behind PFM, and much of the approach is hidden in math instead of written out
2. I don't see any comparisons to other flow-matching techniques, for example [1] and [2]. Are these baselines relevant?
3. I think adding MOOD [3] as a baseline would be helpful, since it has strong performance and is a good diffusion model baseline.
4. The BFM row in the ablation table, which I believe is base flow matching (?), seems surprisingly bad and underperforms all the other baselines. Any ideas for why that is?

[1] Cremer et al. "Flowr – Flow Matching for Structure-Aware De Novo, Interaction- and Fragment-Based Ligand Generation." 2025.
[2] Huang and Zhang. "MolFORM: Multi-modal Flow Matching for Structure-Based Drug Design". 2025.
[3]  Lee et al. "Exploring Chemical Space with Score-based Out-of-distribution Generation." ICML 2023.

### Questions
1. What is the intuitive motivation behind PFM?

### Soundness
2

### Presentation
1

### Contribution
2
