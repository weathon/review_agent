# Quantum-Inspired Structure-Aware Diffusion for Efficient 3D Molecular Generation

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
The high computational cost of classical diffusion models can limit their use in large-scale 3D molecular generation for drug and material discovery. We introduce $\textbf{S}$tructure-aware $\textbf{Q}$uantum $\textbf{Diff}$usion (SQ-Diff), the first full quantum diffusion model for this task, designed to leverage potential quantum advantages in the Noisy Intermediate-Scale Quantum era. Structural priors (e.g., inter-atomic distances) are encoded into the initial quantum state via a novel state preparation procedure that yields a unified normalization scheme dependent only on the number of atoms. The denoising process is driven by a Quantum U-Net, a fully quantum architecture that combines learnable variational quantum circuits with parameter-free operators. Training is guided by these structural priors enforced through a graph-based objective function to maintain structural consistency. Experimentally, SQ-Diff generates valid and diverse 3D molecules and shows improved performance over existing quantum-based methods. While a gap in generation quality compared to leading classical models remains, our model matches the inference speed of the fastest classical approaches with only a few quantum parameters, setting a new benchmark for pure quantum generative models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce the first quantum diffusion model for 3D molecule generation. Its empirical results compared to both classical and quantum baseline methods are promising. Overall, this work points to the potential of quantum generative modeling for efficiently learning high-dimensional data distributions in the physical sciences, though a few key concerns regarding writing and model evaluation remain.

### Strengths
1. The authors' writing and descriptions are clear and easy to follow. I appreciate such details, especially as someone not very familiar with quantum machine learning.
2. SQ-Diff's empirical results for QM9-based unconditional molecule generation are promising, suggesting it is the best-in-class quantum generative model for this task. Further improvements in quantum hardware may yield competitive results compared to classical algorithms in the near future.
3. I haven't seen the classifier guidance strategy in Section 3.3.2 used in other works, and it seems like a clever trick to make conditional generative molecule generation simple for quantum neural networks.

### Weaknesses
1. The authors haven't reported a quantitative comparison of their method's conditional molecule generation results to those of baselines such as EDM and GeoLDM (in terms of property prediction mean absolute error using an external property regression model). Please refer to the EDM paper (Table 2) to see what such a benchmark might look like for this paper.
2. The authors should report additional metrics assessing the quality of their unconditionally generated molecules using the PoseBusters software suite [1]. This has become common practice in the field, including in recent works such as All-Atom Diffusion Transformers (ADiTs) for molecule and material generation [2].
3. It'd be nice to see results for datasets of larger 3D molecules such as GEOM-Drugs.

**References:**

[1] Buttenschoen, M., Morris, G. M., & Deane, C. M. (2024). PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chemical Science, 15(9), 3130-3139.

[2] Joshi, C. K., Fu, X., Liao, Y. L., Gharakhanyan, V., Miller, B. K., Sriram, A., & Ulissi, Z. W. (2025). All-atom diffusion transformers: Unified generative modelling of molecules and materials. arXiv preprint arXiv:2503.03965.

### Questions
1. In Line 178, shouldn't Euclidean distances be denoted as E(3)-invariant, not SE(3)-invariant, since they are also invariant to 3D reflections?
2. In Line 321, is the citation for TorchQuantum correct?
3. In Line 340, how do you evaluate 2D quantum molecular graph generation baselines for 3D molecule generation?

### Soundness
3

### Presentation
2

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
The paper introduces the first full quantum diffusion model for 3D molecular generation. It encodes structural priors such as inter-atomic distances into the quantum state, and adapts a Quantum U-Net style architecture to perform denoising with variational quantum circuits and parameter-free operators. The proposed approach generates valid, diverse molecules, outperforming prior quantum models, and achieving inference speeds comparable to SOTA classical methods, though with some remaining quality gap.

### Strengths
- The paper is generally well-written and well-structured. 
- The authors propose the first pure quantum diffusion model for 3D molecule generation. They employ a quantum-parametrized U-net denoiser, consisting of a series of VQCs. The proposed method achieves inference speeds comparable to the fastest classical methods [1], and shows improved generation quality over prominent quantum models [2]. 
- The authors include comprehensive experiments and ablation studies. In particular, it offers detailed comparisons against classical, hybrid, and quantum baselines, underscores the benefits of incorporating VQCs, and analyzes the contribution of different components within the loss function.

### Weaknesses
- While the proposed approach outperforms prior hybrid/quantum methods in terms of quality and efficiency, there is still a clear performance gap in the generation quality compared to state-of-the-art classical methods. 
- The experiments are limited to the QM9 benchmark. It would be nice if the authors could extend the evaluation to other datasets. 
- I am a bit concerned about the methodological novelty of the paper, as it employs a U-net style architecture similar to [3] that is constructed from a series of VQCs. 

[1] Hong, Haokai, Wanyu Lin, and Kay Chen Tan. "Accelerating 3D Molecule Generation via Jointly Geometric Optimal Transport." arXiv preprint arXiv:2405.15252 (2024).

[2] Wu, Huanjin, Xinyu Ye, and Junchi Yan. "Qvae-mole: The quantum vae with spherical latent variable learning for 3-d molecule generation." Advances in Neural Information Processing Systems 37 (2024): 22745-22771.

[3] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Cham: Springer international publishing, 2015.

### Questions
- For the conditional generation experiments (Figure 3), how does the distribution of chemical properties of the generated molecules by other baselines compare to SQ-Diff?
- In addition to the effective time $T_\text{eff}$, could you provide the time for the generation and compare it to other methods? 
- Small typo: in figure 1, the noisy quantum state $|x_T>$ should be $|x_t>$ instead.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces the first ever quantum diffusion model for 3D molecular generation, SQ-Diff. The method is aimed at addressing two main challenges: the computational burden of classical diffusion models, and the ability of these method to explore novel chemical space. The authors demonstrate that SQ-Diff outperforms all other hybrid/quantum methods that utilize trainable quantum parameters. They additionally perform an ablation study to show that replacing Variational Quantum Circuits with MLPs results in a significant performance drop compared to a fully quantum SQ-Diff, highlighting the value of this addition. They also show the value of each loss component in contributing to the model's total performance. The authors additionally demonstrate that the model can be effectively tuned to produce molecules within a specific distribution, highlighting the controllability of SQ-Diff.

### Strengths
The paper introduces a novel method for molecular generation, leveraging unique model components such as Quantum Variational Circuits and a Quantum-UNET. The paper clearly explains the model and demonstrates its performance across certain capabilities, such as improved control to generate molecules within a distribution. The paper additionally does a good job in ablating different components of the model to assess their relative impact on model performance.

The paper is written well, and the figures and tables clearly illustrate the results.

### Weaknesses
The paper unfortunately does not substantiate one of its two primarily motivators, that methods like SQ-Diff should allow for more thorough exploration of chemical space. Additionally, while from first principles the model should be lighter weight than a classical method, it would be important to further explore the efficiency of this method, beyond comparing Teff in Table 1.

### Questions
Table 1 comparing SQ-Diff's performance against classical methods misses several SOTA methods like EQGAT-diff and SemlaFlow. As the table stands now, it seem as if SQ-Diff achieves median performance compared to classical methods as well, which is slightly misleading. Can you possibly add these results?

Is it possible to add an additional result demonstrating something akin to the similarity of N generated molecule to molecules in the training set for SQ-Diff versus classical methods, or any result that highlights the models ability to explore novel chemical space?

A performance analysis of this method relative to classical methods would also make a stronger story.

### Soundness
2

### Presentation
2

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
The submission proposed a diffusion based method for molecule generation in the latent space. Though it claims that it is inspired by 'Quantum', it is actually close to the previous works such as GEOLDM [1].

[1] https://arxiv.org/pdf/2305.01140

### Strengths
I think the method is easy to follow and it follows the popular design: encoding-decoding + diffusion models.

### Weaknesses
Though the paper claim it is inspired by 'Quantum', but the whole process is similar to GEOLDM. And the authors call it 'quantum diffusion' in 3.2, but equation (2) (3) (4) are exactly the same as general diffusion process. The encoder and unet are also the same as previous works. I am not sure where 'quantum' refers to. And it has been emphasized that the 'The high computational cost of classical diffusion models', but there is no any comparison of computation cost, and training/inference time. I feel like the whole process is the same as GeoLDM.

### Questions
1. Did the whole process run on torch+gpu or any quantum computation?
2. Could you give the time comparison with geoldm?
3. what is the difference of 'quantum diffusion' with classical diffusion? The equations are exactly the same from the paper.

### Soundness
3

### Presentation
3

### Contribution
2
