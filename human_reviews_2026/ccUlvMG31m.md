# Learning Materials Interatomic Potentials via Hybrid Invariant-Equivariant Architectures

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Machine learning interatomic potentials (MLIPs) can predict energy, force, and stress of materials and enable a wide range of downstream discovery tasks. A key design choice in MLIPs involves the trade-off between invariant and equivariant architectures. Invariant models offer computational efficiency but may not perform as well, especially when predicting high-order outputs. In contrast, equivariant models can capture high-order symmetries, but are computationally expensive. In this work, we propose HIENet, a hybrid invariant-equivariant materials interatomic potential model that integrates both invariant and equivariant message passing layers. Furthermore, we show that HIENet provably satisfies key physical constraints. HIENet achieves state-of-the-art performance with considerable computational speedups over prior models. Experimental results on both common benchmarks and downstream materials discovery tasks demonstrate the efficiency and effectiveness of HIENet. Finally, additional ablations further demonstrate that our hybrid invariant-equivariant approach scales well across model sizes and works with different equivariant model architectures, providing powerful insights into future MLIP designs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents HIENet, a machine learning interatomic potential (MLIP) that (1) satisfies O(3)-equivariance for stress and force predictions and (2) integrates both invariant and equivariant layers in the architecture.

### Strengths
- The authors conduct diverse experiments that not only include standard benchmarks but also downstream evaluations, such as ab initio molecular dynamics (MD) simulation and phase diagram predictions.
- The paper is well written and easy to follow, with clear description of its methodologies.

### Weaknesses
- **Inaccurate performance claims and missing baselines:** While the paper claims state-of-the-art performance, it does not include recent baselines like eSEN and GRACE, which are currently the top-performing models on the [Matbench leaderboard](https://matbench-discovery.materialsproject.org/).
- **Limited contribution:** While HIENet enforces equivariance by design, this approach is adopted in other works like eSEN. Also, the proposed architecture is a straightforward combination of invariant and equivariant layers, which is very incremental.

### Questions
Have the authors considered comparing with leading models like eSEN, Nequip, and GRACE?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HIENet, a machine learning interatomic potential (MLIP) that integrates both invariant and equivariant message passing layers to predict energies, forces, and stresses of crystalline materials. The model is designed to balance computational efficiency (from invariant layers) and physical expressivity (from equivariant layers), while provably satisfying physical constraints such as O(3)-equivariance, force conservation, and stress tensor symmetry. Extensive evaluations on Matbench Discovery, Materials Project Trajectory (MPtrj), and several downstream tasks (phonons, bulk moduli, AIMD, alloy phase diagrams) show improved accuracy and faster inference relative to strong baselines such as EquiformerV2, MACE, and SevenNet.

### Strengths
**High throughput with improved accuracy** : The proposed HIENet achieves higher computational efficiency (up to 90–140% faster inference) while showing slightly better accuracy than prior equivariant models such as SevenNet and EquiformerV2.

**Comprehensive evaluation on diverse physical tasks** : The paper goes beyond standard benchmarks and includes downstream physical evaluations such as phonon frequency prediction, bulk modulus estimation, and dynamic simulations. These results strengthen the paper’s claim that HIENet can be applied to real-world materials discovery pipelines.

### Weaknesses
**Lack of conceptual novelty**: While the paper is technically well-executed, the core idea of combining invariant and equivariant message-passing layers is not novel. Similar hybridization strategies—where invariant layers are used to initialize features and feed rank-0 tensors into equivariant layers—have been explored and used in previous works. Although HIENet implements this idea through an attention-based invariant architecture, this modification appears incremental rather than conceptually new. The approach mainly reformulates an existing pipeline in a more structured way rather than introducing a fundamentally new mechanism for coupling invariant and equivariant representations.

**Lack of comparison with eSEN (Fu et al., ICLR 2025)**: The paper does not compare HIENet with eSEN, which currently achieves state-of-the-art performance on Matbench-Discovery (RMSD:0.075). While eSEN uses around 30M parameters, it would be valuable to include a comparison where HIENet is scaled to a similar parameter count to evaluate whether the proposed architecture remains competitive in the high-capacity regime.

Fu, Xiang, et al. "Learning smooth and expressive interatomic potentials for physical property prediction." arXiv preprint arXiv:2502.12147 (2025).

### Questions
**Hybrid invariant–equivariant design**
(1.1) How does your combination of invariant and equivariant layers differ from previous invariant feature initialization?
(1.2) What is the concrete role of the E(3)-invariant layer? Could you discuss how performance changes if you replace it with a rank-0 tensor product, a plain MLP, or other comparable invariant architectures?

**Equivariant operator choice**
Is it possible to substitute the current O(3)-equivariant message passing with SO(2) convolution–based layers, as used in eSEN or EquiformerV2, while maintaining your physical guarantees? If not, which part of your theoretical framework or design would fail?

**Scaling and capacity**
How does model performance scale with parameter count or $L_{\text{max}}? Could you provide scaling analyses to separate architectural improvements from pure capacity effects? In particular, how would HIENet perform if trained with a parameter budget similar to eSEN (≈30M)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces HIENet, a hybrid invariant–equivariant MLIP. HIENet enforces physical constraints by calculating forces and stresses as the derivative of energy. The authors argue that this design(combine one invariant message-passing layer with multiple O(3)-equivariant layers ) keeps high-order fidelity compared to fully invariant MLIPs. In Matbench-Discovery evaluation, HIENet achieves higher accuracy than EquiformerV2 and ORB. Also shows higher inference throughput than equivariant MLIPs such as EquiformerV2 and SevenNet-l3i5. Ablations show the hybrid approach outperforms invariant-only and equivariant-only variants.

### Strengths
Downstream task’ evaluation: Strong phonon (MAE/MSE/RMSE) and bulk modulus results highlight the practical value of HIENet.

### Weaknesses
Incomplete baselines: The paper does not compare against several recent MLIP baselines that have been displayed on Matbench-Discovery (e.g., EqNorm, reported F1 ≈ 0.786). Please refer to the official Matbench-Discovery leaderboard: https://matbench-discovery.materialsproject.org/
Lack of organic molecule evaluation: Results focus on inorganic crystalline settings. Supporting a broader dataset, including tests on molecular benchmarks such as SPICE datasets, is necessary.
Overemphasis on “physically consistent formulation.”: Enforcing E(3)-invariant energy with O(3)-equivariant forces/stresses via energy gradients is common sense and standard practice in modern MLIPs. Section 3.4’s proof is naïve. Consider moving the proof to the appendix and using the main text to discuss nontrivial design choices (e.g., the design of HIENet model).
Add specific citations: The authors state that invariant models are computationally efficient but struggle to produce physically meaningful and robust predictions, especially on downstream tasks. Please provide detailed citations to further specify under which type of downstream tasks this claim holds.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents HIENet, which integrates invariant and equivariant message passing layers, enabling efficient prediction of material energy, forces, and stress while maintaining physical constraints. On the Matbench-Discovery benchmark, HIENet outperforms models such as EquiformerV2 and SevenNet-l3i5 in both accuracy and computational efficiency.

### Strengths
The paper introduces a new hybrid invariant-equivariant message passing network, HIENet, which effectively balances expressive power and computational efficiency.

### Weaknesses
There are shortcomings in the selection of baseline models. Models that integrate both equivariant and invariant information already exist (e.g., GeoMFormer[1] ), yet the paper does not compare with these models, failing to demonstrate the effectiveness of HIENet's fusion approach. Additionally, models trained on the MPTrj dataset are not rare, but the paper only compares against models like EquiformerV2, ORB, and SevenNet-l3i5. Neither in terms of accuracy nor speed does it compare with more recent models (e.g., DPA3, EScAIP, etc.), which significantly weakens the persuasive power of the paper's claim of "SOTA performance."
[1] GeoMFormer: A General Architecture for Geometric Molecular Representation Learning

### Questions
1.There are many models that hybrid equivariant and invariant information (e.g., GeoMFormer). Could you compare HIENet with these models to further validate the effectiveness of this hybrid approach?
2.In the evaluation of the paper, all models are trained on the MPTrj dataset. However, regarding the selection of baselines, why are only models like EquiformerV2 and ORB included, and why are models like Eqnorm, DPA3 and EScAIP not considered? These models also do not use any auxiliary data or training objectives, and thus should be included as comparison. Could you add their data to Table 1, Table 3, Table 4, and Table 5 (if the data is accessible)?
3.Since the MPTrj dataset does not provide an official train-validation-test split, are the results reported in Table 2 based on the same training and validation sets?
4.In Table 4, why is only the bulk modulus metric compared? Could you provide results for other metrics in MatCalc, such as Shear modulus, Constant volume heat capacity, and Off-equilibrium force?
5.The authors claim that HIENet is a model for materials, but in the evaluation, only the MPTrj dataset is used for training. Could you test the performance on other materials datasets, such as OMat24 or MatPES?

### Soundness
2

### Presentation
3

### Contribution
1
