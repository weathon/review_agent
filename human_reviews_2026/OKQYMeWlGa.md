# Align Your Structures: Generating Trajectories with Structure Pretraining for Molecular Dynamics

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8, 2

## Abstract
Generating molecular dynamics (MD) trajectories using deep generative models has attracted increasing attention, yet remains inherently challenging due to the limited availability of MD data and the complexities involved in modeling high-dimensional MD distributions. To overcome these challenges, we propose a novel framework that leverages structure pre-training for MD trajectory generation. Specifically, we first train a diffusion-based structure generation model on a large-scale conformer dataset, on top of which we introduce an interpolator module trained on MD trajectory data, designed to enforce temporal consistency among generated structures. Our approach effectively harnesses abundant structural data to mitigate the scarcity of MD trajectory data and effectively decomposes the intricate MD modeling task into two manageable subproblems: structural generation and temporal alignment. We comprehensively evaluate our method on the QM9 and DRUGS small-molecule datasets across unconditional generation, forward simulation, and interpolation tasks, and further extend our framework and analysis to tetrapeptide and protein monomer systems. Experimental results confirm that our approach excels in generating chemically realistic MD trajectories, as evidenced by remarkable improvements of accuracy in geometric, dynamical, and energetic measurements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces EGInterpolator, a novel diffusion-based model designed for modeling MD distributions. To address the data sparsity of MD trajectories, the model adopts a two-stage training strategy: it is first pretrained on a large conformer dataset and subsequently fine-tuned on MD data. A key component of the framework is a temporal interpolator, which captures temporal dependencies by integrating the pretrained unconditional generative model with a temporal network through linear interpolation. Experiments on GEOM-QM9 and GEOM-Drugs demonstrate that EGInterpolator achieves strong consistency with MD reference trajectories across unconditional generation, forward simulation, and interpolation tasks.

### Strengths
- The paper presents a clear research motivation and a well-organized logical flow. Considering the scarcity of MD data, the strategy of pretraining on large-scale conformer data followed by fine-tuning on MD data effectively reduces the complexity of spatio-temporal modeling of MD distributions.
- The temporal interpolator is designed in a concise and effective manner, enabling efficient temporal modeling while preserving the structural generation capability acquired during pretraining.
- The experiments on organic molecules thoroughly investigate tasks including unconditional generation, forward simulation, and transition path sampling, providing a comprehensive demonstration of the model’s capability to capture MD distributions.

### Weaknesses
- This work is evaluated only on small-molecule systems. However, for such systems with relatively few atoms, MD simulations using empirical force fields can achieve high accuracy at an acceptable computational cost, which may limit the practical advantages of the proposed model. The authors are encouraged to further justify the benefits of their approach over traditional MD methods or provide additional experiments on more complex biomolecular systems.

### Questions
1. For small-molecule systems, MD simulations using empirical force fields already offer a good balance between efficiency and accuracy. The authors should further clarify the advantages of their model over traditional MD methods or include additional experiments on more complex biomolecular systems.
2. Line 171 defines the so-called conformer distribution $p^{cf}(x)$, which is a potentially misleading definition. It can only be well-defined if it is based on an empirical data distribution (e.g., a dataset) or a known distribution (e.g., the Boltzmann distribution). The authors should clearly specify the definition.
3. In the MD finetuning stage, are the model parameters $\theta$ pretrained on the conformer dataset fixed or further optimized? This point does not appear to be clearly specified in Equation (3). If the parameters $\theta$ continue to be updated during finetuning, then the premise of Theorem 4.1 may not hold, since it would be unclear whether $\epsilon_{\theta}^{cf}$ remains consistent with the pretrained approximation of $p^{cf}$ during finetuning. This would undermine the practical validity of the theorem.
4. Lines 234–235 assume an extreme case, $\hat{p}^{md} = p^{md}$, to illustrate the advantage of the parametrization. I find this assumption inappropriate, as $\hat{p}^{md}$ assumes all conformations are i.i.d., while there exist temporal dependencies between consecutive MD samples, making it nearly impossible for their joint distribution to coincide with that of an i.i.d. case. Could the authors provide a reasonable justification for this assumption?
5. From the experimental results in Figure 3(A), the recall of the coverage and matching metrics on the QM9 dataset still falls short of SOTA performance. Could the authors provide possible explanations for this gap?
6. I have concerns about the setup of the interpolation task. Even though the model can generate a trajectory from the initial to the target state through the conditioning mask, the actual transition time corresponding to this process may be longer than 0.52 ns, which could result in the generated trajectory failing to capture the true underlying dynamics. Could the authors provide an explanation for this?
7. A typo: line 458 states that the generated MD trajectory spans 0.52 ns, whereas Figure 5D labels it as 1 ns.

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
This paper presents EGInterpolator, a framework for generating molecular dynamics (MD) trajectories by leveraging structure pretraining. The key idea is to first pretrain a diffusion model on large-scale conformer datasets, then introduce a temporal interpolator module trained on limited MD data to enforce temporal consistency.

### Strengths
1. The authors test their method across multiple tasks (unconditional generation, forward simulation, interpolation) and provide extensive metrics including JSD for various metrics. The ablation studies are quite thorough.
2. Using structure pretraining to leverage abundant conformer data is a sensible solution that makes practical sense.
3. The method outperforms baselines and performs impressive results on the interpolation.

### Weaknesses
1. The experiments are restricted to small organic molecules (QM9 has molecules with ≤9 heavy atoms). While the authors acknowledge this in limitations, it's unclear how well this approach would scale to larger, more practical systems like proteins or protein-ligand complexes. The method's utility for real-world drug discovery applications remains uncertain without demonstration on larger molecules (As far as I knew, there are also large molecules in GEOM dataset).
2.  In several metrics, the model performs worse than even short MD oracle trajectories. This suggests the generated dynamics may be too fast or not physically accurate enough for practical applications. The authors should discuss this gap more critically.
3.  Theorem 4.1 shows the interpolator induces an intermediate distribution, but why this particular interpolation strategy is optimal, and how the choice of α affects the bias-variance tradeoff.

### Questions
1. How much conformer data is actually needed? What if you pretrain on a smaller, more targeted set of conformers? This would help understand the data efficiency of your approach.
2. You train separate models for QM9 and Drugs. Have you tried training a single model on both datasets? What prevents the method from generalizing across different molecular systems in a unified way?
3. What about long-time stability - do generated trajectories eventually diverge or produce unphysical configurations?
4. Table 5 shows some degradation in later blocks for forward simulation. How many blocks can you roll out before quality becomes unacceptable? Is there a way to prevent this deterioration?

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
The paper proposes a two-stage framework for generating molecular dynamics (MD) trajectories using diffusion models. First, they train a large structure diffusion model on static molecular conformers (like QM9, GEOM-Drugs). This model learns how valid 3D molecular structures look like. Subsequently, they freeze it and train a smaller temporal interpolator on a limited set of MD trajectories. This temporal module learns to align the static structures in time, adding smooth and physically consistent motion between them. The combined system, called EGINTERPOLATOR, can generate realistic molecular trajectories even when there’s very little MD data available. It also keeps SE(3)-equivariance and can handle different generation modes (simulation, interpolation, etc.).

### Strengths
- The idea to separate spatial structure learning from temporal dynamics is elegant and practical. It reduces data requirements and improves generalization.
- The temporal interpolator design is well conceived — it works as a learned guidance or adapter that connects independent conformer frames into a coherent trajectory.
- Experiments are convincing, with good results on QM9 and GEOM-Drugs datasets. The model seems to generate smoother and more realistic bond and torsion distributions than baselines.
- It’s well grounded in symmetry (SE(3) equivariance), which is crucial for molecular data.
- Conceptually it’s similar to the trend in video diffusion models, but nicely adapted to the molecular domain.

### Weaknesses
- The physical validation is lacking — results are mostly on geometric statistics. No tests about energy conservation, temperature stability, or realistic MD physics.
- The method depends a lot on the pretrained conformer model. If that model is not good, the whole system might fail.
- It’s not clear how this would scale to bigger molecules (like proteins) or longer trajectories.
- The temporal module is still a bit of a black box. The paper doesn’t show much intuition about what it actually learns.
- Fine-tuning with small MD datasets could overfit, and the paper doesn’t really study that.
- Fundamentally, the architecture setup is still sequential when generating whereas BioEmu for example targets the equilibrium distribution directly.

### Questions
1) How stable are the generated trajectories over long time horizons? Do they drift away from realistic energy basins?
2) Could the temporal interpolator be trained or conditioned on energy or force information to improve physical consistency?
3) Could the method handle non-equilibrium or biased simulations?
4) What’s the computational cost compared to training a full end-to-end trajectory diffusion model?
5) Have you run experiments that in the limit sample from all meta-states from a single starting red point in appendix E6 and E7?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Continuing on previous work in the field authors propose a diffusion model for molecular dynamics (MD) trajectories of small molecules. Specifically, they propose a decomposed approach where they divide the task into modeling valid 3D structures (per frame conformers) as the first pre-training task based on larger conformer datasets, ensuring structural validity and generalizability, and then learning a temporal interpolator to realize valid MD trajectories. Introducing a structure pre-training task for a dynamics model seems well-motivated.

### Strengths
The paper is technically well-written and derivations seem correct. Broadly speaking the proposed decomposition is also well-motivated. I particularly like introducing a marginal distribution over 3D structures (conformers) as a pre-training task so as to ensure structural validity across trajectories and transfer across molecules. The approach is also in principle extendable to larger molecules such as (small) proteins. 

The proposed temporal interpolator seems to capture some measures of temporal dynamics (autocorrelation) better than alternatives such as GeoTDM as shown in Figure 4 E/F.

### Weaknesses
The per-frame marginal distribution over conformers could be perhaps more cleanly separated as a learning task. As a pre-training task, the model could be estimated from larger conformer datasets and then separately fine-tuned based on subsampled frames from MD trajectories prior to learning any temporal dynamics. It seems unnecessary to further adjust this part. Indeed, a product distribution over the frames would serve as a proper tilting function for learning a light temporal interpolator (tilted analogously to reward guided sampling). It is unclear to me why authors adopted a more convoluted convex combination that is learned from trajectories (with some frozen layers).

While theorem 4.1 appears correct, it also highlights potential issues with the approach. The resulting intermediate target distribution seems undesirable with the 1/(1-alpha) exponent.

The primary comparison results in Table 1 pertain to per-frame metrics except TICA. For this reason, authors' own pre-trained per-frame model (or any other per-frame model) would do well for most of these metrics, requiring no temporal interpolator. It would be helpful to focus primarily on evaluation of dynamics since this extension is the key contribution in the paper, not conformer generation.

### Questions
Could you elaborate on the justification for the convex combination in comparison to a simpler approach that uses per-frame marginals (pre-trained with all the data, including sampled MD frames) to adjust interpolator scores? The architecture that the authors use for the interpolator, equivariant temporal attention network, already includes analogous alternating per-frame and temporal updates. The temporal interpolator could still take ${\hat\epsilon}^{md}$ estimates as input, ensuring that its role would remain similarly light, offering (only) temporal corrections, aligning well with the motivation start at lines 231. I understand that the authors do freeze pre-trained ES layers in their approach.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to learn generative models of MD trajectories by fine-tuning conformer generative models. This is done by adding additional temporal layers to the conformer generation model and fine tuning on dynamics trajectories. The model is evaluated on simulations of small druglike molecules from QM9 and GEOM-DRUGS, where it roughly reproduces static and dynamic observables.

### Strengths
The approach is very sensible - developing full-trajectory models by fine-tuning static structure models is an approach that many later works will likely follow.

### Weaknesses
**Novelty**
* The authors' contribution amounts to the incorporation of conformer pretraining for MD trajectory generation, which in my opinion is not significant or non-obvious enough for a conference paper in the absence of compelling results.

**Significance**
* The task of MD trajectory generation for small molecules is of unclear utility. I am sympathetic that the authors are following prior precedent, where small molecule conformations have historically served as testbeds for modeling larger systems. However, as the AI for science field matures, it is important that the community stays focused on forward-looking applications.

**Experiments**
* The authors write "In contrast, our method generalizes more readily across arbitrary molecular systems," yet do not show experiments on peptides or proteins, which would allow proper comparisons with previous work.
* The result that the model outperforms AR baselines is not surprising, given prior work (MDGen).
* From the results shown in Figure 4, it appears that the model has a lot of trouble matching ground truth distributions of bond lengths, torsion angles, and autocorrelation decays.

**Method**
* The linear interpolation of the temporal module output seems rather contrived, especially the thereotical justification. There are no ablation studies showing why this additional complexity is necessary.

### Questions
No specific questions

### Soundness
2

### Presentation
2

### Contribution
1
