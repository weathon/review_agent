# Towards Novel Metamaterial Discovery via Latent Space Regulation and Exploration

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Metamaterials are artificially engineered structures whose unique mechanical and physical properties arise from geometry rather than composition, enabling applications in wave control, energy absorption, and soft robotics. To capture this structural programmability in a unified form, voxel representation provides a natural choice: it can express diverse classes of metamaterials including truss, shell, and porous metamaterials within a single cubic discretization. However, existing voxel-based generative models face severe limitations. The vast design space, combined with sparse and costly datasets, leads to a generalization dilemma: models tend either to memorize known designs, sacrificing novelty, or to produce invalid, low-quality structures. To address this, we propose VOXPLORER, a generative framework that couples voxel representation with latent space regulation and guided exploration. VOXPLORER introduces a repel-and-sink (RAS) mechanism to smooth and densify the latent distribution of valid structures, and a short-range repulsion (SRR) guidance during diffusion to promote exploration beyond memorized regions while preserving validity. We further contribute a systematic benchmark for voxel-based metamaterials and develop an evaluation module that jointly assess quality, novelty, and diversity. Extensive experiments show that VOXPLORER outperforms state-of-the-art baselines, achieving +8.9% in quality, +46.4% in novelty, and +128.6% in diversity on average across two datasets, establishing a principled pathway toward systematic discovery of next-generation metamaterials.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces both a benchmark for evaluating generative models generating metamaterials, and their own technique for performing this generation.

I think the paper is limited as it only considers methods based on the voxel-representation, and I also think there are plenty of things that can be improved regarding clarity in the paper. I also have some concerns regarding the numerical evaluation.

### Strengths
Ambitious with both new data and new model.

### Weaknesses
The paper focuses on voxel-based representations only. A benchmark should ideally not rely on the representations by the model, but be agnostic to such design choices.

I think the clarity can be improved (see Questions)

I think the paper is missing a discussion about the computational overhead with their model, and how the computation compares with other models.

### Questions
Why can’t you use graph-based methods, then convert their output to the voxel-based data (like you did when constructing the training dataset)?

Line 199: how do you measure distance to determine the “nearest neighbor” in the training set?

Line 239: how is the synthesization of negative samples performed? How large is this set of negative samples compared to the positive samples?

Eq 5+6: I guess you mean by pos/neg that this is either pos or negative, and then you have two terms (one for positive, one for negative)?

Regarding the evaluation, it is a bit concerning to me that multiple models in the MetaTruss benchmark have 0 novelty, and additionally, in those cases the quality scores are not similar. I do not understand exactly how the novelty score works, it seems to me that if this is 0, the generated samples should essentially overlap some sample in the training set. But still, for these models, the quality scores are very different. Any ideas why? Do you have some unseen real data to see what the metrics are for this data, and therefore what numbers one should expect from a good model?

Table 3: I get the feeling that the ablation is somewhat inconclusive as novelty is very poor with only SRR. I could understand it if the latent space has a lot of overlap, but shouldn’t SRR still be able to find something? To me, it seems like RAS is the bigger reason for improved novelty: without a good latent space, novelty will be poor. Also, could improve readability of the table if instead writing in the left column + RAS, + SRR Diff, + RAS + SRR Diff (full framework). 

The ablation is missing the baseline no regularization, no SRR diffusion.

Line 462: you say the related methods provide a solution that is insufficient, but are there any empirical results that show that, or is this your own feeling/speculation? As far as I can see, you never try no regularization, no SRR diffusion. Or did someone else do this? In that case, a reference that backs up this statement is needed.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents VoxPlorer, which is a generative model for creating collections of diverse, high-quality metamaterial designs over a voxel representation. To overcome the hurdles of the voxel design space -- which contains large swaths of invalid designs -- this paper first reduces the dimensionality by using an autoencoder to project the voxel grid into a small latent space. This latent space is carefully regularized to encourage smooth sample distributions while enforcing separability between the valid and invalid subspaces. The curated latent space makes it possible for stable diffusion-based exploration of valid structures, even in regions that are not populated by known samples. This stability allows us to use a bolder sampling strategy guided by SRR, which explicitly pushes the sampling process away from known samples in latent space, into novel yet feasible design regions. The authors also offer a set of metrics for evaluating the current and future collection(s) of voxel-based metamaterial designs.

### Strengths
This paper addresses an interesting and challenging problem in metamaterial design -- namely, the generation and evaluation of large, valid metamaterial datasets. The regularization applied to the latent embedding space is insightful and grounded, while seemingly effective and widely generalizable. The ablation results are also compelling, particularly in the qualitative examples showing the impact of different regularization schemes (Table 2). The text is also very well written and illustrated.

### Weaknesses
1. The authors position the unified representation as a major motivating factor, but the experimental results reflect experiments over trusses and shells separately. Does VoxPlorer continue to function as designed in the presence of distinct classes? I'm particularly curious whether the latent space regularization and diffusion sampling can effectively perform their roles over a combined dataset.

2. Since the negative voxel samples are critical to your latent space construction, it would be nice to include more information about their definition and construction process in the main paper. The current supplemental section is also somewhat sparse. Did your noising process include any consideration for symmetry or periodicity? Did you perform any checks on the resulting "negative" samples to ensure that they are invalid as desired? What criteria are used to determine (in)validity? Relatedly, do you track the relative frequency of (in)valid designs generated by your models and/or other baselines? 

3. The constraints of symmetry and periodicity are discussed several times (and feature prominently in your evaluation metric suite), but they do not seem to factor into the system design in any meaningful way -- not in the latent space, regularization, diffusion, or negative sample creation. I'm particularly curious about the relationship between symmetry/periodicity constraints and the latent space, as the former attributes offer two principled axes for dimensionality reduction, yet it seems that they are not being used. It also seems as if this choice is made in opposition to the approach of MetaShell [Yang et al. 2024], since they're explicitly enforcing cubic symmetry (and with it, periodicity). Is there a reason that you chose not to (or were unable to) leverage these structural constraints?

4. Methods like 3D-CDM are specifically designed for performance-guided inverse design, so I wonder how meaningful it is to compare their novelty/diversity metrics in an unconditional generation setting. Could the authors comment on this choice? 

5. There are a few lines of related work that are not currently represented, but came to mind while I was reading. These specific citations are not required, but I share them here for the authors' consideration:
	- l. 54 -- A few recent works introduced other class-spanning metamaterial representations, including Xue et al. (2025, "MIND: Microstructure INverse Design with Generative Hybrid Neural Representation") and Makatura et al. (2023, "Procedural Metamaterials: A Unified Procedural Graph for Metamaterial Design")
	- l. 74 - There are large datasets/generative frameworks for shell-type metamaterials, including "Parametric Shell Lattices" [Liu et al. 2022] and "Data-Efficient Discovery of Hyperelastic TPMS Metamaterials with Extreme Energy Dissipation" [Perroni-Scharf et al. 2025].
	- l. 150 - there are quantitative metrics for evaluating metamaterial design diversity, such as those proposed by METASET [Chan et al. 2020] and others discussed in a recent survey by Lee et al (Data-Driven Design for Metamaterials and Multiscale Systems: A Review", 2023).
These papers do not invalidate the current submission, but they do temper some of the claims/contributions made in the paper.

### Questions
Please see weaknesses.

1. If symmetry is an integral part of metamaterial design, why should rotational augmentation as in l. 190 be relevant or useful?

## Minor comments
- l. 292 -- clapse --> collapse
- l. 344 -- exploartion --> exploration
- throughout (e.g., 4.2 header, Table 2 caption) -- "regulation" and "regularization" seem to be used interchangeably, but it should be exclusively the latter.
- Table 2 -- Consider reordering the rows so that rows 1-3 are in the same order as case 1-3 as discussed in the corresponding paragraph

### Soundness
3

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
4

### Summary
The paper introduces, a generative framework aimed at advancing metamaterial discovery using voxel representations. Metamaterials are artificially engineered structures whose remarkable properties—such as negative Poisson’s ratio or high stiffness-to-weight ratio—stem from their geometry rather than composition. They are vital in applications ranging from soft robotics to energy absorption. The voxel representation, which discretizes 3D space into solid or void cells, is particularly advantageous for unifying diverse metamaterial classes like truss, shell, and porous designs. However, voxel-based generative models face the generalization dilemma, where the vast design space and limited datasets lead to models that either overfit to known designs (losing novelty) or generate invalid structures (losing quality).

To tackle this, integrates two key innovations: latent space regulation and guided exploration. First, it encodes complex voxel structures into a low-dimensional latent space using an autoencoder enhanced with a Repel-and-Sink (RAS) mechanism. RAS ensures a smooth, dense latent distribution while clearly separating valid and invalid regions, mitigating mode collapse. It consists of three parts: Inter-Class Repulsion (IeR), which simplifies the decision boundary between valid and invalid samples; Intra-Class Repulsion (IaR), which prevents latent clusters from collapsing; and a Central Sink (CS), which maintains distribution compactness by pulling latents toward the origin. Together, these elements enhance robustness and maintain structural validity.
In the generation phase, the work employs a diffusion model for sampling new designs, augmented by Short-Range Repulsion (SRR) guidance. SRR introduces a force that repels the current latent sample from known training latents, encouraging exploration of new but still feasible design regions. The repulsion decays rapidly with distance, ensuring exploration remains within valid boundaries defined by RAS. This combination allows VOXPLORER to balance the competing goals of novelty and validity.
To address the lack of benchmarks in voxel-based metamaterial design, the authors developed a comprehensive dataset and evaluation suite. They contributed MetaTruss, the first large-scale voxel dataset for truss-based metamaterials containing 60,000 samples discretized into a 48³ grid, alongside the existing MetaShell dataset for shell-type structures. Their evaluation framework employs five metrics across three dimensions: quality (symmetry, periodicity, and connectivity scores), novelty (IoU distance from nearest training sample), and diversity (variety of unique nearest neighbors among generated samples).

Experimental comparisons with state-of-the-art baselines like DiT-3D and 3D-CDM showed that VOXPLORER substantially outperforms existing models, achieving on average +8.9% improvement in quality, +46.4% in novelty, and +128.6% in diversity across datasets. On MetaShell, it matched or exceeded quality benchmarks while more than doubling diversity. Ablation studies confirmed that RAS effectively separates valid from invalid latents, while SRR significantly enhances novelty and diversity without degrading structure quality.
Overall, VOXPLORER presents a principled and scalable approach for discovering next-generation metamaterials. By uniting latent space regulation through RAS with guided exploration via SRR, it successfully achieves a robust balance between quality, novelty, and diversity in voxel-based generative metamaterial design.

### Strengths
The paper demonstrates remarkable strengths through its innovative generative framework, VOXPLORER, and its major contribution to the systematic benchmarking of voxel-based metamaterial design. Its first strength lies in offering a novel technical solution to the generalization dilemma (C1)—a persistent challenge in voxel-based generative models where methods often compromise between quality and novelty. VOXPLORER overcomes this trade-off with two ingenious mechanisms: the Repel-and-Sink (RAS) latent regulation, which introduces Inter-Class Repulsion, Intra-Class Repulsion, and a Central Sink to refine latent space organization and improve model robustness; and the Short-Range Repulsion (SRR) guidance, which ensures the exploration process avoids overfitting by pushing sampling away from memorized data while still generating valid, high-quality designs. These mechanisms collectively enhance generalization, producing novel yet feasible metamaterial structures.

### Weaknesses
Despite its impressive contributions, the paper also faces several weaknesses and practical challenges related to both the underlying voxel-based design field and the implementation of the VOXPLORER framework. A major limitation lies in the inherent challenges of voxel representation and data sparsity. The research domain itself suffers from an enormous design space—on the order of (2^{64^3}) possible configurations—while available training datasets remain extremely limited. Although the authors introduce the 60,000-sample MetaTruss dataset, voxel data remain costly to build and store, meaning that even the expanded dataset still represents a tiny and sparse portion of the possible design space. Consequently, VOXPLORER continues to operate under conditions of limited data coverage, which constrains the model’s ultimate generalization potential.

Another challenge arises from trade-offs in performance metrics. While VOXPLORER achieves a strong balance between quality, novelty, and diversity, these objectives are inherently competing. The Short-Range Repulsion (SRR) guidance successfully pushes the model toward unexplored regions of the latent space, boosting novelty and diversity; however, this exploration naturally comes with a modest deterioration in quality compared to standard diffusion-based methods. The paper acknowledges that this reduction is expected, as conventional Denoising Diffusion Probabilistic Models (DDPMs) tend to reproduce training samples—thus, by seeking novelty, VOXPLORER inevitably sacrifices a degree of structural refinement.

Overall, these weaknesses highlight both the technical challenges of scaling voxel-based metamaterial generation and the computational sensitivity of VOXPLORER’s design, underscoring that while the framework marks a substantial step forward, further refinement and domain adaptation remain essential for broader practical deployment.

### Questions
Thanks for the paper, it does seem like a good work to me, although I have some important questions which needs to be addressed. Kindly go through them.

1. [Section 1] : The authors should also note other method of material generation in general, they have only noted graph and voxel representation. I recommend the authors to include other representation space methods which involve grouping techniques. (Although maybe limited to crystals, but worth noting in this paper as related work). A few of these works can include 1. Xie, Tian, et al. "Crystal diffusion variational autoencoder for periodic material generation." arXiv preprint arXiv:2110.06197 (2021)., 2. Sinha, Anshuman, Shuyi Jia, and Victor Fung. "Representation-space diffusion models for generating periodic materials." arXiv preprint arXiv:2408.07213 (2024)., 3. Luo, Youzhi, Chengkai Liu, and Shuiwang Ji. "Towards symmetry-aware generation of periodic materials." Advances in Neural Information Processing Systems 36 (2023): 53308-53329.

2. [Line 71-92] : kindly include a bullet-wise details of the contribution of this work, simultaneously with each bullet refer the section where you have addressed your claim. This will make it more readable.

3. Kindly add motivation for RAS and SRR, has such a regularization technique never been seen in latent space models? If yes, then kindly also cite them properly. 

4.  Generalization Dilemma (C1): What precisely is the Generalization Dilemma (C1) that VOXPLORER seeks to solve, and how does the combination of the vast design space (e.g., $2^{64^3}$ configurations) and limited dataset size (around 10,000 samples) cause this issue?

5. I have a couple of question regarding Table 1. Seems like the authors have not mentioned why 3D-CDM and various other methods have score poorly on Snov and Sdiv, while Voxplorer does so much better. But that's not seen on the Quality Score? Also what's the contrast between MethaTruss and MetaShell.  Why other methods are more competitive for MetaShell. While lag behind heavily in MetaTruss. Does it have anything to do with the inductive biases your current approach has? Which is likely the case, when you specifically introduced the repulsion function.

6. Regularization of latent space is definitely not a new topic of research, although all those regularization have come up with some limitation or the other. I request the authors to kindly add a good section on limitations, since it's quite evident with the results.

7. I strongly encourage the authors to provide a detailed study of hyper-maters in order to control the latent space regularization. I couldn't find explicit detail on this topic. Do the latent space regularization and sampling have any associated hyper-parameters with which they can produce controllable results on Novelty and diversity? If no, then I think this should be included (I did not had the time to go over the appendix, particularly because it is not in the main paper and as a reviewer I am only assigned the time to review the main section).

The only talk on hyper-parameters which I saw is on eq 8 and 9, however there is no ablation study on that, Is the ablation in table 4 addressing that? If yes, then it should've been properly mentioned.

8. I do understand that this paper does not necessarily need experimental validation to support the claim that there method does generate materials which are valid though first principle methods. Although I would subject that to other reviewers who are expert in ab-initio calculations to check whether this would require any experimental validation or not.

Thanks, kindly address the above questions. Happy to discuss further.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of generating novel yet valid 3D metamaterial designs in voxel space, where existing generative models either overfit or fail to maintain validity. The authors propose VOXPLORER, a two-part framework consisting of Repel-And-Sink (RAS) latent regulation to disentangle valid and invalid latent regions and Short-Range Repulsion (SRR) guidance during latent diffusion to promote exploration without sacrificing quality. They also introduce MetaTruss, a new voxel-based dataset, and unified metrics evaluating quality, novelty, and diversity. Experiments on MetaTruss and MetaShell datasets demonstrate that VOXPLORER achieves superior performance, improving quality by 8.9%, novelty by 46.4%, and diversity by 128.6% on average compared to state-of-the-art baselines.

### Strengths
1. The proposed method demonstrates superior performance compared to baseline approaches across two benchmark datasets.

2. The authors introduce a new voxel dataset for trusses (MetaTruss).

### Weaknesses
1. Lack of physics-based validation. The quality / validity evaluation relies only on voxel heuristics but not effective property evaluation (e.g., energy absorption, poisson's ratio). 

2. Insufficient details on negative sample synthesis. The authors do not clearly explain how the negative samples are synthesiszed. The negative samples might not reflect realistic invalid metamaterial failure modes. 

3. Incomplete hyperparameter analysis. The proposed method contains numerous hyperparameters. The authors do not describe how the hyperparameters are selected or provide a study to evaluate the sensitivity of the method's performance to the hyperparameters.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
