## Human Reviewer 1

### Summary
This work presents a novel machine learning framework for predicting Kohn-Sham Hamiltonians on large-scale molecular systems. The proposed novel method includes a novel loss function WALoss based on basis change, and a novel SO(3)-equivariant model WANet adopting eSCN operation and MACE based many-body interaction in its architecture. A large-scale molecule dataset PubChemQH is also curated from PubChemQC benchmark for experiments. The proposed method is experimentally shown to achieve good performance on QH9 and PubChemQH benchmarks.

### Strengths
Originality:  
The originality contribution of this work is excellent by proposing a physics-inspired loss function and model for Hamiltonians prediction problem. Also, the curated dataset is a significant originality contribution by providing a large-scale testbed in the field.

Quality:  
The quality of this work is evidenced by solid theoretic analysis and strong performance on benchmark experiments.

Clarity:  
The writing of this work is good, clear and well-organized.

Significance:  
This work makes significant contribution in not only proposing an insightful machine learning model, but also having benefits in reducing computational cost for broad quantum physics and chemistry community.

### Weaknesses
(1) In Table 2, it is observed that if using WALoss on WANet, while Hamiltonian prediction performance improves, the prediction performance on the two quantum properties degrades. Could authors give explanations why this phenomenon happens on small-scale QH9 dataset but not on  large-scale PubChemQH dataset (Table 1)? Also, authors are encouraged to report the results of QHNet with WALoss on QH9 datasets.

(2) For practical application targets, both prediction accuracy and speed are important for an effective method. Authors are encouraged to give analysis about computational complexity or report inference speed of the proposed method, and compare them with QHNet [1].

[1] Haiyang Yu, Zhao Xu, Xiaofeng Qian, Xiaoning Qian, and Shuiwang Ji. 2023. Efficient and equivariant graph networks for predicting quantum hamiltonian. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 1692, 40412–40424.

### Questions
See Weaknesses part.

### Soundness
4

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper identifies a problem with training Hamiltonian models using only element-wise loss, which can lead to unphysically large errors in ground-state properties, especially for larger systems. To address this, the authors propose a WALoss function that aligns the eigenspaces of the predicted Hamiltonian with the ground truth, along with a new architecture, WANet. Additionally, they introduce PubChemQH, a new Hamiltonian dataset containing substantially larger molecules than previous datasets like QH9. Experiments demonstrate that the proposed WALoss significantly improves the accuracy of properties derived from the predicted Hamiltonian.

### Strengths
Originality:
- The observation and formalization of the Scaling-Induced MAE-Applicability Divergence (SAD) phenomenon is both insightful and important, representing a novel identification of a fundamental limitation in current methods.
- The WALoss approach offers an original solution to maintaining physical accuracy in Hamiltonian predictions, particularly for larger molecular systems.

Quality:
- The theoretical analysis is rigorous, supported by eigenvalue perturbation theory and clear mathematical derivations.
- The experimental validation is comprehensive, with thorough ablation studies and detailed comparisons with baseline methods.

Clarity:
- The technical content is well-organized and clearly presented, with supporting diagrams and mathematical formulations.
- The experimental results effectively demonstrate the practical benefits of the proposed methods.

### Weaknesses
The paper's main weakness lies in its insufficient differentiation from prior work, particularly regarding the use of eigenvalues in Hamiltonian prediction loss functions. The methodology section lacks critical details about computational costs and efficiency analysis of the proposed method, particularly regarding the training overhead of WALoss compared to traditional approaches.

### Questions
1. The computational cost of WALoss could be substantial for large molecular systems, especially those containing heavy elements, as it involves eigendecomposition and matrix operations for each training step. Could you provide details about the wall-clock training time comparison between models with and without WALoss, the GPU memory requirements (particularly for the largest molecules), and how the computational overhead scales with molecule size and atomic number?

2. How were the loss term weights (λ1, λ2, λ3) in Equation 20 determined? The balance between these terms seems crucial for the method's performance, yet the paper doesn't discuss the process of selecting these hyperparameters or their sensitivity analysis.

3. There appears to be a discrepancy in Table 4's Hamiltonian MAE results, where Naive Loss shows a notably lower error (0.0412) compared to other methods. Could you explain this unexpected result and its implications?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper tackles the problem of predicting the Kohn-Sham Hamiltonians for molecules. It first proposed a new Hamiltonian dataset for large molecules derived from the PubChemQC dataset. The paper then identifies the problem with the robustness of prior models when applied to molecules with large numbers of atoms. The cause of the problem is explained both empirically and theoretically. To improve the robustness of Hamiltonian prediction models to larger molecules, the authors introduce a new loss called WALoss that explicitly penalizes deviations from the expected eigenstructure of the ground-truth Hamiltonian. To further improve the quality of Hamiltonian prediction, the paper proposed a novel architecture dubbed WANet. The experiments demonstrate that WALoss significantly improves the quality of the baseline model (QHNet) and the proposed model (WANet) on downstream tasks such as System Energy,  $\epsilon_\text{LUMO}$ and $\epsilon_\text{HOMO}$ prediction. The proposed WANet model outperforms the baseline model.

### Strengths
- The problem with robustness is explained nicely. The authors demonstrate it empirically in Figure 1 and then provide two theoretical results that explain poor System Energy prediction quality even when relative MAE is low.
- The effect of the proposed WALoss on the performance of models on downstream tasks such as System Energy,  $\epsilon_\text{LUMO}$ and $\epsilon_\text{HOMO}$ prediction is remarkable.

### Weaknesses
### **Major weaknesses**
- The WANet model's architecture is hard to understand and poorly motivated. There are no ablations for various building blocks of the model, and no architecture hyperparameter values are provided.
- The results in Table 1 do not provide enough evidence that WANet architecture is superior to baselines. A recent benchmark $\nabla^2$DFT [1] compares Hamiltonian prediction models on a dataset of molecules with more atoms than QH9. The results in [1] show that PhisNet [2] performs significantly better than QH9. A comparison to other Hamiltonian predicting models, such as [2] and [3], is necessary to validate the claims about the proposed architecture.
- One of the paper's main claims is that improved Hamiltonians can be used for downstream regression tasks and to accelerate DFT computations. Claiming that WANet is superior to regression-based models in the HOMO-LUMO gap prediction task seems a little ambitious, given that only one baseline (Equiformer v2) was used. To give a better perspective, it might be worth comparing WANet with SOTA models, such as UniMol+ [4] or other models from [5]. The System Energy MAE is also ~50 kcal/mol, almost 50 times larger than metrics published in [1] and [6]. Moreover, the acceleration rate seems relatively small (18%), and most SCF iterations are still required when starting from the predicted Hamiltonian.

### **Minor weaknesses**
- Relatively large molecules (40 - 100 atoms) are used, but the functional (B3LYP-Def2TZV) does not contain the dispersion correction. The absence of the dispersion correction can potentially lead to inaccurate Hamiltonians.
- The notation across the paper is unclear. A matrix with a subscript (i.e., $C^*_i$) usually defines the i-th row of the matrix, whereas in this paper it defines the $C^*$ matrix for the i-th molecule in the batch. Moreover, $C^*$  often denotes the conjugate transpose of matrix $C$. This notation complicates understanding Algorithm 1, as the Shur algorithm for complex-valued matrices operates with conjugate transposition. Undefined symbols in the equation on page 4. In equation (2), $n$ denotes the batch size but was never introduced.
- Related work is in Appendix.

### Questions
### **Questions and remarks**
- Equation (3) might benefit from additional details. Energies associated with occupied molecular orbitals (and LUMO) make up the most energy of the molecule.
- The $\epsilon_\text{LUMO}$ and $\epsilon_\text{HOMO}$ results in Table 1 seem strange. Why is the difference between QHNet and WANet so significant only for these targets? Moreover, the $\epsilon_{\Delta}$ MAE for WANet is larger than  $\epsilon_\text{LUMO}$ and $\epsilon_\text{HOMO}$ MAE. This implies that the model is consistently mistaken in opposite "directions." Could you please double-check the results or explain why this is happening?
> The theorem highlights that the difference between the predicted and actual Hamiltonian matrices when only considering the element-wise norm, can lead to unbounded differences in eigenvalues/eigenvectors due to a significant $\frac{\kappa(S)}{||S||_2}$ratio
- It would be interesting to see the distribution of these values for the PubChemQH/QH9 datasets. If I understand correctly, $||\Delta \mathbf{H}||_{1, 1}$ is the $B^2 * \operatorname{MAE}(\mathbf{H}, \hat{\mathbf{H}})$ between real and predicted Hamiltonians, and its values can be relatively small.
- line 219: mistake in "spectral"
- Is the Algorithm 1 novel? If not, please provide clear citations. Also, it is better to include the final version of the algorithm, including the changes from the last paragraph on page 5.

### **Closing remark**
Overall, the paper presents a valuable technique that greatly improves the applicability of Hamiltonian prediction models for downstream tasks. However, the WANet architecture seems poorly motivated, has no ablations, and is not properly compared with baselines. I would consider raising my score if the concerns with the WANet model are resolved.

[1] Khrabrov, K., Ber, A., Tsypin, A., Ushenin, K., Rumiantsev, E., Telepov, A., ... & Kadurin, A. (2024). ∇ DFT: A Universal Quantum Chemistry Dataset of Drug-Like Molecules and a Benchmark for Neural Network Potentials. CoRR.

[2] Unke, O., Bogojeski, M., Gastegger, M., Geiger, M., Smidt, T., & Müller, K. R. (2021). SE (3)-equivariant prediction of molecular wavefunctions and electronic densities. Advances in Neural Information Processing Systems, 34, 14434-14447.

[3] Zhong, Y., Yu, H., Su, M., Gong, X., & Xiang, H. (2023). Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids. npj Computational Materials, 9(1), 182.

[4] Lu, S., Gao, Z., He, D., Zhang, L., & Ke, G. (2023). Highly accurate quantum chemical property prediction with uni-mol+. arXiv preprint arXiv:2303.16982.

[5] Hu, W., Fey, M., Ren, H., Nakata, M., Dong, Y., & Leskovec, J. OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

[6] Eastman, P., Pritchard, B. P., Chodera, J. D., & Markland, T. E. (2024). Nutmeg and SPICE: Models and data for biomolecular machine learning. Journal of Chemical Theory and Computation.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This work encompasses 4 contributions. First, it introduces a new, larger dataset for predicting Kohn-Sham Density Functional Theory (KS-DFT) Hamiltonians, surpassing previous datasets in size. Second, it finds that training with Mean Absolute Error (MAE) on the Hamiltonian is inadequate for larger datasets, often producing results inferior to traditional minao guesses. Third, the authors propose a wave function alignment loss as an alternative to MAE, focusing on optimizing the eigenspace rather than the Hamiltonian itself. Finally, they present a new model that achieves state-of-the-art performance.

### Strengths
* The paper is very well written and easy to understand.
* The empirical analysis of SAD is valuable and motivates this work well.
* The WALoss appears like a natural choice and comes at essentially no additional cost.
* Both the WALoss and WANet significantly improve upon previous works.

### Weaknesses
* The asymptotic scaling in Collary 1 for the lowest eigenvalue seems unfit to describe Gaussian-type orbitals. Due to their fast decay, the overlap is spatially limited. After a certain system size, more atoms are unlikely to affect the lowest eigenvalue.
* l.62/63, "complex transformation" sounds overly complicated given the symmetric nature of the Hamiltonian.
* Unfortunately, the real-world DFT speed-ups remain limited.

### Questions
* While the comparison to a regression model is laudable, the comparison to equiformer has many confounding variables. Could the authors compare to their WANet but with a regression head?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3