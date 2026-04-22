# Physically Valid Biomolecular Interaction Modeling with Gauss-Seidel Projection

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Biomolecular interaction modeling has been substantially advanced by foundation models, yet they often produce all-atom structures that violate basic steric feasibility. We address this limitation by enforcing physical validity as a strict constraint during both training and inference with a unified module. At its core is a differentiable projection that maps the provisional atom coordinates from the diffusion model to the nearest physically valid configuration. This projection is achieved using a Gauss-Seidel scheme, which exploits the locality and sparsity of the constraints to ensure stable and fast convergence at scale. By implicit differentiation to obtain gradients, our module integrates seamlessly into existing frameworks for end-to-end finetuning. With our Gauss-Seidel projection module in place, two denoising steps are sufficient to produce biomolecular complexes that are both physically valid and structurally accurate. Across six benchmarks, our $2$-step model achieves the same structural accuracy as state-of-the-art $200$-step diffusion baselines, delivering ${\sim}10\times$ wall-clock speedups while guaranteeing physical validity. The code is available at https://github.com/chensiyuan030105/ProteinGS.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the physical validity challenge in generative modeling for protein structures. It proposes a first-order constraint and enforces it during both fine-tuning and inference processes. The constraint is formulated as an optimization problem, which is approximately solved using Gauss-Seidel updates. Additionally, the paper introduces implicit differentiation to enable backpropagation during fine-tuning. Compared to baseline methods, this approach achieves faster inference, requiring only 2-step denoising, while maintaining competitive accuracy.

### Strengths
1. The manuscript is well-written, and easy to follow.
2. The performance in terms of inference acceleration is impressive.
3. The proposed method is reasonable and offers valuable insights to the field of AI4S. In many scientific applications, strict physical laws must be upheld. When training data is limited, models often struggle to learn these constraints in a data-driven way; thus, explicitly enforcing the constraint is a reasonable and effective solution.

### Weaknesses
1. While the main focus of the paper is on inference performance under a small number of sampling steps, I believe it would be beneficial to also include results under larger-step settings. This would provide a more comprehensive understanding of the method's performance when the computational budget is less constrained, and offer a clearer comparison with existing approaches in such scenarios.
2. It would be beneficial to discuss the relationship between proposed method with reinforcement learning (RL)-based fine-tuning approaches for diffusion models (e.g., DPOK, DRaFT). Compared to inference time guidance methods, they also involve fine-tuning to optimizing the reward. I’m curious about how the fine-tuning cost of RL compares to that of your iterative optimization algorithm.
3. How is the scalability of your method? Since your method involves enforcing constraints on all substructures, as the complexity or size of the system grows, does the method remain computationally feasible during training and inference?
4. Alphafold3 is an important baseline in this domain and is suggested to be included in experiment section for a more complete comparison. Additionally, the evaluation could benefit from incorporating more comprehensive metrics such as TM-score and pb-valid.
5. To facilitate reproducibility and enable further research based on this work, I strongly encourage the authors to release the code (such as in an anonymous form during the review period).

### Questions
1. In Equation (2), Is there a redundant factor of 0.5 in the definition of E(x)? It seems to be inconsistent with Equation (3).
2. The iterative linearization of eq.4 (Theorem D.1) relies on multiple approximations (replacing K by I; assume g = 0). Could you please discuss how these approximations affect the convergence speed and solution quality of the optimization problem? Maybe by numerical simulations or tests on real protein datasets) could help validate the affect of these approximations.
3. Could there be additional textual description to clarify the "invalid" structures in Figure 4? It is not always obvious which structural issues are presented: Are all cases attributed to atomic clashes, or are there other problems (e.g., bond length violations, steric hindrance)?
4. How does the method perform without relying on the Protenix-mini sampling strategy and compare it to Boltz-1?
5. While the paper emphasizes reducing the number of sampling steps from 5 to 2, it is unclear why this results in a wall-clock time reduction significantly greater than 2.5× in Figure 5 (right) when compared to Protenix-mini.

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
5

### Summary
This paper tackles the critical issue of physical invalidity (steric clashes, distorted geometry) in structures generated by deep learning models for biomolecular interactions, particularly all-atom diffusion models. The authors propose a method to enforce physical validity as a hard constraint during both training and inference. The core idea is a differentiable projection module that maps the provisional atomic coordinates generated by a diffusion model onto the nearest physically valid configuration . This projection is efficiently implemented using a Gauss-Seidel iterative scheme, exploiting the locality of physical constraints . Crucially, the module is made differentiable via implicit differentiation, allowing it to be integrated seamlessly into existing diffusion frameworks (like Boltz-1) for end-to-end finetuning . A key result is that incorporating this module enables the generation of physically valid and structurally accurate complexes using only 2 denoising steps, achieving accuracy comparable to 200-step SOTA baselines while offering ~10x speedup and guaranteeing validity . The method is evaluated on six diverse benchmarks against strong baselines.

### Strengths
- Generating physically plausible structures is a prerequisite for the reliability and utility of biomolecular models. This paper directly confronts the common failing of deep generative models in this regard , offering a principled solution.
- The use of the Gauss-Seidel method for the projection step is well-suited for the problem, leveraging the local nature of physical constraints (bond lengths, angles, clashes) for fast and stable convergence compared to methods like gradient descent.
- Making the iterative Gauss-Seidel solver differentiable via implicit differentiation  is a key technical contribution, enabling end-to-end training and allowing the diffusion model to adapt to the projection step. This integration is crucial for maintaining high accuracy, as shown in the ablation study.

### Weaknesses
- The evaluation of the Protenix baseline appears to be an underestimation. According to the Protenix technical report, as well as anecdotal user feedback, its performance at 200 steps (e.g., in terms of DockQ and validity metrics) is reportedly not as low as presented in this paper. I recommand the authors to either provide the detailed configuration (config) files used for their Protenix experiments or, preferably, release the raw prediction files generated by their baseline models to ensure a fair and reproducible comparison.
- Regarding the Protenix mini tech report, the original authors claim that 'increasing the ODE sampling steps to 10 effectively mitigates this issue (clash)'. Given this, the comparison in Table 1 might be suboptimal. I strongly suggest that the authors re-evaluate the baseline using 10 ODE steps in Table 1, as this seems to be the recommended setting for mitigating structural clashes.
- The paper states that GS projection achieves good physical constraint effects within just 2 steps. This raises a question: why did the authors not experiment with applying the projection for more steps? Furthermore, the results in Table 1 indicate that sampling for 2 steps with constraints still underperforms the original 'boltz2' baseline. The comparison is incomplete. The authors should also include experiments comparing their method against baselines like 'boltz2' when restricted to a similar small step budget (e.g., 10 steps) for structure prediction.
- Although the 2-step process is much faster overall, the paper could provide more details on the computational cost (time and memory, forward and backward) of the Gauss-Seidel projection module itself, especially as the system (complex) size grows larger. The backward pass involves solving a linear system using CG, which could become expensive.

### Questions
- Can the authors provide metrics (e.g. RMSD) quantifying the magnitude of coordinate changes introduced by the projection step? Are there cases where projection significantly alters key interface interactions or secondary structure elements?

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
This paper introduces a physically constrained diffusion framework for biomolecular structure generation that enforces local geometric and energetic constraints while maintaining structural accuracy. The method integrates a Gauss–Seidel based differentiable projection that iteratively enforces constraints to ensure the physical feasibility of generated structures. The framework supports backpropagation through conjugate gradients, enabling stable gradient-based training and inference. Empirical results demonstrate improved geometric validity, structural stability, and fast inference across complex biomolecular benchmarks.

### Strengths
- The motivation is clear and well grounded in the need for physically valid biomolecular generation.
- The method enforces atomically realistic outputs compared to unconstrained baselines, a practically important contribution.
- The algorithmic formulation is sound. The iterative Gauss–Seidel projection and penalty method are mathematically principled and differentiable.

### Weaknesses
- The PoseBusters results show a notable drop in docking-related metrics, but the paper does not analyze the cause, possibly due to a trade-off between hard constraint enforcement and ligand flexibility.
- Some evaluation metrics are missing. Including measures such as the PoseBusters-valid success rate, TM-Score, and iLDDT (in Table 2) would make the empirical validation more complete and convincing.

### Questions
1. Since the Gauss–Seidel scheme guarantees uniqueness only for each linearized subproblem, can different orderings or initialization seeds lead to non-unique final projections? Have the authors observed multiple feasible solutions in practice?
2. The drop in docking-related metrics on PoseBusters is notable. Can the authors provide analysis or ablation evidence on whether this degradation stems from the specific constraint choices or the limited number of denoising steps?
3. Why is metrics such as iLDDT excluded from Table 2? Including complementary metrics such as TM-Score or PoseBusters-valid success rate would make the evaluation more comprehensive.
4. Could the authors clarify how $\alpha$ is chosen in practice, and whether convergence depends sensitively on this value?

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
3

### Summary
This paper presents a Gauss–Seidel–based projection module that enables stable and rapid convergence to physically plausible structures using only a few denoising steps. By explicitly enforcing geometric constraints during sampling, the method significantly accelerates inference while maintaining a high degree of physical validity.

### Strengths
1. The use of a Gauss–Seidel projection mechanism to enforce physical validity is novel and appears highly effective. It provides a computationally efficient way to ensure that generated structures adhere to key geometric constraints.

2. Both quantitative and qualitative results demonstrate that the proposed method is effective and efficient across various evaluation metrics.

### Weaknesses
1. The paper does not release its code, project page, or any runnable repository. This raises concerns regarding reproducibility and the ability of other researchers to validate or extend the results.

2. The experiments are conducted by finetuning the method on an existing co-folding model (Boltz-1). However, the reported two-step sampling results do not consistently surpass the original Boltz-1 baseline. Although this may be attributable to the reduced number of denoising steps, it remains unclear whether the proposed method can outperform Boltz-1 when using a comparable number of steps.

3. The comparison with Proteinix-mini appears somewhat unfair, as Proteinix-mini uses a significantly reduced architecture and removes the MSA module, placing it at an inherent disadvantage.

4. Since the second-stage projection refines the output from the first denoising step, the method improves physical validity but may have limited capacity to further enhance structural accuracy. It is unclear whether the approach can simultaneously improve both accuracy and validity, or whether it inherently trades one for the other.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
2
