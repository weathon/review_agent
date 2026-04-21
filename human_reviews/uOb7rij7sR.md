# CryoGEN: Generative Energy-based Models for Cryogenic Electron Tomography Reconstruction

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Cryogenic electron tomography (Cryo-ET) is a powerful technique for visualizing subcellular structures in their native states. Nonetheless, its effectiveness is compromised by anisotropic resolution artifacts caused by the missing-wedge effect. To address this, IsoNet, a deep learning-based method, proposes iteratively reconstructing the missing-wedge information. While successful, IsoNet's dependence on recursive prediction updates often leads to training instability and model divergence. In this study, we introduce CryoGEN—an energy-based probabilistic model that not only mitigates resolution anisotropy but also removes the need for recursive subtomogram averaging, delivering an approximate *10*$\times$ speedup for training. Evaluations across various biological datasets, including immature HIV-1 virions and ribosomes, demonstrate that CryoGEN significantly enhances structural completeness and interpretability of the reconstructed samples.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
CryoGEN's author(s) proposed a new method called CryoGEN, aimed at addressing the "missing wedge" problem and the challenge of low signal-to-noise ratio in cryo-electron tomography (Cryo-ET). CryoGEN combines generative adversarial networks (GANs) with energy-based models to produce more consistent and high-quality 3D reconstructions. Compared to traditional weighted back-projection (WBP) and existing deep learning methods like IsoNet, CryoGEN effectively fills in missing information and reduces blurring artifacts.

Key contributions :

1. The introduction of CryoGEN, which uses energy-based models to handle multiple possible solutions, avoiding the blurring effect caused by simple averaging.

2. The inclusion of consistency loss and noise injection to ensure that generated images maintain fidelity to the original data while preserving diversity in the results.

3. Experimental validation on multiple datasets, such as HIV viral particles and neural synapses, where CryoGEN outperforms existing baseline methods in metrics like peak signal-to-noise ratio (PSNR), structural similarity index (SSIM), and Fourier shell correlation (FSC).

In all, the authors announces a more effective solution for reconstruction in Cryo-ET, showing significant improvements in quality and stability. Future work includes improving training efficiency, automating parameter adjustments, and extending its applications to other cryo-electron microscopy domains.

### Strengths
Originality: The originality of the paper lies in its attempt to improve and combine existing methods by using generative adversarial networks (GANs) and energy-based models to address the missing wedge problem in cryo-electron tomography (Cryo-ET). 

Quality: The quality of the research is validated through experiments on multiple datasets, including synthetic samples and real biological samples (e.g., HIV viral particles and neural synapses). However, the experimental design lacks in-depth comparison with other state-of-the-art methods, especially the absence of sufficient ablation studies to demonstrate the importance of each model component.

Clarity: The paper is generally well-written with a logical flow, but some parts are explained too briefly, particularly the mathematical description and implementation details of the energy model, which might make it difficult for non-expert readers to fully understand the working principles.

Significance: The potential impact of CryoGEN in the field of cryo-electron tomography is limited. Although the method improves reconstruction quality to some extent, the degree of improvement is relatively modest, and challenges may arise when applied to more complex datasets.

### Weaknesses
Lack of Novelty: Although CryoGEN combined GANs and energy-based models, this architecture still relies on existing methods and lacks truly groundbreaking innovation. For instance, the combination of GANs and energy-based models has already been widely applied in other fields, and the paper does not sufficiently demonstrate the unique contribution of this combination in Cryo-ET. We recommend citing some recent literature, such as "Your GAN is Secretly an Energy-based Model and You Should Use It" by Tong Che et al. (2020), which discusses the connection between GANs and energy-based models and their effectiveness in data reconstruction. Furthermore, it is suggested to elaborate on the specific aspects where this combination lacks novelty in Cryo-ET applications, such as similarities in loss functions, model architecture redundancy, or training strategy limitations, and clarify how CryoGEN distinguishes itself from existing methods to enhance the contribution statement of the paper.

Incomplete Theoretical Explanation: The mathematical description and implementation details of the energy model are overly simplified. It is recommended to provide a more detailed description of the energy model's training process, including key equations and their roles in optimizing the model. Adding intuitive examples or visual aids, such as diagrams illustrating the training dynamics of the energy model, would make the content more accessible to non-expert readers. Specifically, expanding on equations (3) and (4) would help clarify how the energy model interacts with the GAN during training. These additional details would help address potential confusion and make the feedback more actionable.

Lack of In-Depth Result Analysis: Although the experimental results show performance improvement, the authors do not provide in-depth discussion on the reasons behind these key performance improvements. It is recommended to provide a more detailed analysis of why CryoGEN performs better on specific datasets compared to other methods, which would better support the validity of its contributions.

### Questions
1. Lack of Comparative Experiments: It is suggested to add comparisons with other recent methods, particularly with more challenging and complex models, to better validate the advantages of CryoGEN.

2. Supplemental Theoretical Explanation: Could the authors provide more detailed explanations regarding the implementation details of the energy model? Adding intuitive examples and diagrams would help readers understand its complexity.

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
3

### Summary
The paper addresses the limitations of Cryogenic Electron Tomography (Cryo-ET) in reconstructing 3D structures of cellular components due to the missing wedge problem, which creates anisotropic resolution in tomograms. The authors introduce CryoGEN, a generative energy-based model designed to tackle the missing wedge problem more effectively and stably, without requiring recursive prediction.

### Strengths
1. The use of energy-based models for addressing the missing wedge problem is novel and shows promise in improving reconstruction quality.
2. CryoGenic is computationally efficient, achieving significant runtime reductions compared to IsoNet, which would be valuable in large-scale biological studies
3. The paper is well-written and easy to follow

### Weaknesses
1. Couldn't find big technical issues.
2. A more detailed discussion would be needed (e.g., limitations, what would be the future work).
3. Typo: Page 1, line 32 – Repeated phrase "insights insights into."

### Questions
1. What is the SNR of the data shown in Figure 7? I’m curious to see 3D reconstruction results (comparison with IsoNet) with higher noise (SNR 0.01 or 0.001) levels.
2. On page 6: "At the inference stage, we begin by cropping the complete tomogram into multiple overlapping subtomograms" – Could you specify the number of crops used?

### Soundness
3

### Presentation
3

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
This paper introduces a new method for reconstructing 3D structures from cryo-electron tomography (Cryo-ET) data. It aims to address a common issue in Cryo-ET reconstruction known as the missing wedge problem. The proposed algorithm, CryoGEN, tackles this by using an energy-based model that learns to generate realistic 3D structures from incomplete information. The model includes two main components: an energy model, which scores how realistic a structure is, and a prediction model, which reconstructs the missing regions. CryoGEN also adds noise to the input data during training to handle the challenges posed by the one-to-many mapping issue. This approach enables the model to create accurate and clear reconstructions by filling in missing areas based on learned patterns from data. The paper uses four datasets to evaluate the effectiveness of CryoGEN, particularly in comparison to the baseline solution, IsoNet. Results show that CryoGEN produces clearer and more complete reconstructions with better preservation of structural details. On three of the datasets, quantitative evaluation is performed, demonstrating that CryoGEN surpasses IsoNet.

### Strengths
1. This paper addresses the missing wedge problem, a major issue in Cryo-ET that causes incomplete and blurred 3D reconstructions.

2. The paper proposes a novel deep learning solution for Cryo-ET reconstruction. Using GANs and energy models is a reasonable approach.

3. Diverse datasets are considered for evaluation, including both simulated and real tomograms.

### Weaknesses
1. The techniques used in CryoGEN are fairly standard in general computer vision. There doesn’t appear to be any part of the algorithm specifically motivated by the unique characteristics of Cryo-ET data, such as the missing wedge. For example, the motivation discussed in Section 3 is very common in general generative models like VAE and EBM. Also, EBM and GAN models have been widely used in general inpainting problems but not discussed as related work. 

2. It's unclear why Equation (8) represents the posterior distribution. Is Equation (8) an implementation of Equation (3)?

3. This paper improves upon IsoNet by incorporating the energy model and noise perturbation on the input. Which of these additions is more essential to the strong performance of CryoGEN?

4. Regarding computational complexity, the paper provides inconsistent information. Section 5.3 reports that CryoGEN is much faster than IsoNet. However, CryoGEN is clearly more complex than IsoNet in terms of architecture. The authors also mention that the energy model converges more slowly in Section 4.1.

5. As mentioned in related work, DeepDeWedge can be used to address denoising and missing wedges simultaneously. Why is DeepDeWedge not included in the quantitative evaluation?

### Questions
See Weaknesses

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
3

### Summary
This paper proposes a novel deep learning based method to solve the missing wedge problem, which is an inverse problem that arises in cryogenic electron tomography (cryo-ET). 

In cryo-ET, the goal is to reconstruct the 3D density of a biological sample (e.g. a cell) from a set of 2D projections of the density. Due to limitations during acquisition, there is a wedge-shaped region of viewing directions where no projections can be recorded. As a result, the set of projections does not uniquely determine the 3D density of the sample. 

The authors propose a method called CryoGEN that aims to fill in the missing data of the missing wedge. The method is self-supervised (does not require clean ground truth for training) and fits a missing wedge reconstruction network directly to the tomogram(s) whose missing wedge is to be filled. The fitting of this reconstruction network is regularized by an energy function, represented as another neural network, which is learned together with the reconstruction network in an adversarial manner.

Based on a theoretical motivation, and experiments on synthetic data and two real-world tomograms, the authors argue that the energy function yields improved missing wedge reconstruction performance over two closely related state-of-the-art missing wedge methods (IsoNet and DeepDeWedge). 


**Recommendation:** I do not recommend publishing the paper at ICLR *in its current form*. This is mainly because I found parts of the paper (especially the motivation and training of the energy function and the construction of the losses) unclear. However, since the missing wedge problem is important, and since CryoGEN performs better than state-of-the-art baselines in some cases. 
I recognize the potential of the method and am willing to raise my score if the authors better explain their method and why it outperforms the baselines (IsoNet and DeepDeWedge).

**Post rebuttal:** I have increased my score from **5 to 6**, as the authors addressed most of my concerns (see discussion below). Overall, I think that the most important contribution of the paper is the observation that equipping methods that work like IsoNet or DeepDeWedge with GAN-like energy nets to can improve the missing wedge reconstruction performance.

### Strengths
- CryoGEN outperforms IsoNet on the clean synthetic data (Section 5.2) and gives a cleaner reconstruction of the (real) HIV capsids (Figure 10).
- The missing wedge problem addressed in the paper is important and challenging.
- The authors applied CryoGEN to two real-world tomograms, demonstrating the practical applicability of the method.

### Weaknesses
- I cannot really follow the motivation in Section 3, where the authors argue that IsoNet produces an average of two 3D densities $x_1$ and $x_2$ that produce the same measurement $y_0$ under the forward map $\mathcal{T}_M$. The following is unclear to me:
	- Why do the authors "choose" a Gaussian mixture prior for IsoNet (lines 170 - 171)? If I understand correctly, IsoNet does not use a prior for regularization.
	- What is the concrete energy function ($E(x) = - \log p(x)$) that gives a better solution?
	- Why does adding noise to the measurements $y$ solve the problem that there are many $x$ that map to the same $y$ (lines 194 - 199)?

- I do not understand the motivation and intuition behind the consistency loss (equation (7)). Why is the reconstruction mesh $g_\theta$ applied to y and then again after $\mathcal{T}_M$?

- The authors give no intuition as to why the energy function (which looks like a prior to me) can be learned directly on the data one wants to reconstruct. I would also appreciate more explanation as to why the energy function is learned in an adversarial manner (equation (9)).

- The authors argue that CryoGEN outperforms IsoNet on the purified ribosome tomogram (Figure 9), but I find the CryoGEN tomogram visually much less appealing. It seems to contain less detail and the contrast of the ribosomes looks too strong to me. (Since there is no ground truth, it is impossible to measure which reconstruction is actually better). 

- The experiments on synthetic data (Section 5.2) are done on noiseless data. However, the low signal-to-noise ratio (as low as <10%) is one of the main challenges in cryo-ET. Therefore, it would be good to include an experiment that demonstrates how CryoGEN handles strong noise. Synthetic data is suitable for such an experiment because clean ground truth is available.

- In the abstract, the authors state that IsoNet suffers from model collapse as it updates its own training data. This aspect is not discussed in the paper.

### Questions
- Why is the energy function learned in an adverserial way (Equation (9))?
- Does the adverserial training of the energy function lead to any instabilities during training?
- Why does the "Posterior" loss (Equation 8) involve random rotations?

### Soundness
2

### Presentation
2

### Contribution
3
