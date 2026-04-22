# Symmetry-Guaranteed Prediction of High-Order Tensor Properties for Crystalline Materials via Irreducible Decomposition

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Predicting high-order tensor properties for crystalline materials is crucial for various scientific and engineering applications. Crystal symmetry is one of the primary factors influencing high-order tensor properties, such as elasticity and piezoelectricity, making strict adherence to symmetry constraints essential. However, exactly guaranteeing symmetry compliance remains challenging. Recent approaches rely on enforcing symmetry but often fail to strictly preserve symmetry. In this work, we propose a novel method that guarantees exact symmetry compliance by predicting symmetry-constrained irreducible components of high-order tensors. Specifically, we first develop a computational procedure to identify the basis tensors corresponding to symmetry-constrained irreducible components under various symmetry conditions. This symmetry-constrained basis guarantees that the assembled full tensor strictly adheres to the required symmetry constraints. To predict the numerical values for these irreducible components, we then propose a spherical-harmonic convolutional neural network designed to effectively capture essential high-order tensor information. Extensive experiments validate that our method achieves exact symmetry compliance without compromising prediction accuracy, thereby outperforming state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes selecting the correct irreducible spherical harmonics channels to account for the symmetry constraints in crystalline materials tensorial property prediction tasks, including dielectric, piezoelectric, and elastic tensors. The authors first determine the unconstrained set of channels based on a group of ℓ values, and then remove those that do not satisfy the symmetry constraints. However, this approach is nearly identical to the symmetry enforcement module introduced in previous work, GMTNet, where a mask is applied to exclude channels that violate the symmetry constraints. Furthermore, the results presented in Table 2 for GMTNet might be problematic due to the improper use of this mask-based symmetry enforcement mechanism. This paper also introduces an equivariant graph neural network for the same task, but it exhibits a notable level of similarity to GMTNet.

### Strengths
## Strengths

- The proposed method achieved good performance compared with previous methods across three tensor prediction tasks, including dielectric, piezoelectric, and elastic tensors.

- Ablation studies are provided to verify the effectiveness of each component.

- The symmetry enforcement idea is solid but shares a level of similarity with the previous GMTNet symmetry enforcement module.

### Weaknesses
## Weakness

The major issues of this paper are the missing discussions and citations of what has been done by previous methods.

- The authors first determine the unconstrained set of channels based on a group of ℓ values, and then remove those that do not satisfy the symmetry constraints. However, this approach is nearly identical to the symmetry enforcement module introduced in previous work, GMTNet, where a mask is applied to exclude channels that violate the symmetry constraints. 

- The property prediction block is similar to GMTNet, obtaining tensor properties using gradients.

- The results presented in Table 2 for GMTNet might be problematic due to the improper use of this mask-based symmetry enforcement mechanism. 

- This paper also introduces an equivariant graph neural network for the same task, but it exhibits a notable level of similarity to GMTNet. These similarities need to be discussed.

### Questions
As listed above in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces IrredNet, a framework that predicts symmetry-constrained irreducible components of high-order crystal tensors and then reconstructs the full tensors so that point-group symmetry is satisfied exactly by construction. The theory (SO(3) irreducible decomposition plus point-group filtering with character tables) underpins an algorithm to enumerate admissible components, and a spherical-harmonic equivariant network (using efficient SO(2) channel operations) regresses their magnitudes. Experiments on JARVIS-DFT dielectric (2nd-order), piezoelectric (3rd-order), and elastic (4th-order) tensors claim exact symmetry preservation and improved accuracy (Fnorm/EwT) over MEGNet, ETGNN, and GMTNet.

### Strengths
- Predicting in irreducible spaces and reconstructing guarantees point-group symmetry compliance irrespective of regression error; the algorithmic details (character sums, parity factor for improper operations) are clearly specified.
- The work targets 2nd/3rd/4th-order tensors on JARVIS-DFT with explicit symmetry constraints and Voigt notation background for elasticity (21 dofs).
- Tables report lower Fnorm and higher EwT versus baselines, and perfect “zero/equality” scores for elastic tensors (as expected if symmetry is enforced).
- The spherical-harmonic/eSCN design aligns with the SO(3) structure and reduces computational cost relative to full SO(3) convolutions.

### Weaknesses
- The paper emphasizes "zero-element" and "equality" accuracies, but these are guaranteed once outputs are reconstructed from symmetry-filtered irreps; hence Table-2 perfect scores are not informative about predictive quality. Please de-emphasize these or replace with checks the model could actually fail (e.g., physical constraints not enforced by irreps). 
- The introduction stresses that elastic tensors must obey properties linked to physical consistency (e.g., positive definiteness), but the experiments do not evaluate or guarantee SPD/thermodynamic admissibility of the reconstructed elasticity. Add SPD checks (eigenvalue spectra, Born stability) and compare to methods that enforce such constraints.
- All results are on DFT-computed JARVIS-DFT tensors with an 80/10/10 random split; there is no evidence on experimental data, no cross-dataset transfer, and no crystal-system-aware or composition-aware splits to mitigate leakage. Report stratified or leave-system-out splits and external validation. 
- The paper re-implements MEGNet, ETGNN, GMTNet "with default hyperparameters", which risks under-tuning baselines relative to the proposed model. Provide rigorous tuning budgets for all methods and include stronger modern equivariant baselines (e.g., higher-degree transformers) to match the spherical-harmonic capacity.
- EwT (error-within-threshold based on Fnorm ratio) can be dominated by tensor scale; ensure unit handling across dielectric/piezo/elastic is comparable and add normalized, component-wise metrics. The tolerance for "success" in symmetry checks (10⁻⁴) is stated only once—justify this and test sensitivity.
- The model caps the maximum spin at `=3/3/4 for 2nd/3rd/4th-order tensors; it is unclear how this choice affects approximation fidelity, especially for complex systems or higher-order targets. Provide sensitivity to the maximum degree and timing breakdowns (the "Time (s)" row lacks clear definition—training vs inference, hardware).

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

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
This paper introduces a symmetry-constrained framework for predicting high-order tensor properties of crystalline materials. The method leverages group theory to decompose each tensor into a minimal set of symmetry-constrained irreducible components. A spherical-harmonic neural network then predicts scalar magnitudes for these few components, which are subsequently used to reconstruct the full tensor—ensuring physical consistency by construction. This approach not only provides a theoretical guarantee of symmetry adherence but also significantly improves prediction accuracy compared to prior state-of-the-art models.

### Strengths
1. The method is firmly grounded in rigorous group theory and high-order tensor decomposition, making the approach robust and theoretically sound.
2. By decomposing high-order tensors into a small set of irreducible representations, the proposed method strictly enforces symmetry and improves training efficiency. Given that high-order tensors are common in engineering applications, this framework has strong potential for broader applicability.
3. The proposed model clearly outperforms the three baseline methods, and the ablation study is comprehensive and well-executed.

### Weaknesses
1. While the paper focuses on predictive accuracy and symmetry compliance, it lacks an explicit comparison of training/inference speed and memory consumption between the proposed method and the baselines. Including this analysis would provide a more complete performance evaluation.
2. The paper is hard to follow with unclear terminology. For instance, the definition of irreducible components/representations (from Line 147) is not clearly explained, which makes it hard for readers to understand how it relates to tensor decomposition techniques such as proper generalized decomposition. Also, Figure 1, which is central to understanding the method, is unclear. The relationships between the sub-blocks and the main blocks (a and b) are ambiguous. The caption could be expanded to provide clearer and more informative descriptions.
4. The architectural novelty of the paper is not clear. The main innovation appears to lie in reducing high-order tensors into a few irreducible representations and using a neural network to predict the corresponding scalars. The spherical-harmonic convolutional design itself seems standard.

### Questions
1. Please provide a complexity analysis (runtime and memory) for the basis generation step in Algorithm 1.
2. Since the model predicts scalar coefficients, do these coefficients correlate with any known physical or geometric scalar invariants of the tensor (e.g., the bulk modulus in elasticity)? Exploring this connection could enhance the interpretability of the model’s predictions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper predicts symmetry-constrained SO(3) irreducible components and reconstructs crystal tensors to guarantee exact symmetry; IrredNet implements this, outperforming MEGNet/ETGNN/GMTNet on JARVIS-DFT (especially elastic) and improving downstream moduli, with ablations confirming irreducible-space prediction plus explicit symmetry are essential.

### Strengths
1. IrredNet predicts SO(3) irreducible components and reconstructs tensors only from symmetry-compatible bases, guaranteeing exact crystal/index symmetries without post-hoc fixing.
2. IrredNet shows the gains specifically come from irreducible-space prediction and explicit symmetry constraints.

### Weaknesses
### Majors

1. **Contributions of this work is incremental**. Computing higher-order tensor decompositions and invariant bases under SO(3)/crystal symmetries is routine and well-known; the paper mainly engineers these known tools into an ML pipeline to guarantee symmetry rather than introducing new theory.
2. **Presentation can be improved**. It is recommended to add a concrete running example (e.g., rank-4 elastic) walking through your algorithm and include a brief contrast explaining why prior methods (e.g., GMTNet/ETGNN) can still violate high-order symmetries without explicit projection. Also, please redo tables with self-contained, consistent captions including dataset, property, baselines, and metrics+units.

### Minors

1. Line 314, 317, 776 use "nonzero" but line 412, 759, 950 use "non-zero". Please make them consistent.
2. Line 105 "Jarvis" -> "JARVIS"
3. Line 401: "GMTNET" -> "GMTNet"

### Questions
1. In what precise way does predicting SO(3) irreducible components differ from prior equivariant GNNs (like GMTNet) that already operate in irreps?
2. How sensitive are results to $\ell_{max}$ and channel counts?

### Soundness
3

### Presentation
1

### Contribution
2
