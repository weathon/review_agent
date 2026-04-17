# UNAAGI: Atom-Level Diffusion for Generating Non-Canonical Amino Acid Substitutions

- Decision: Reject
- Scores: 2, 4, 0

## Abstract
Proposing beneficial amino acid substitutions, whether for mutational effect prediction or protein engineering, remains a central challenge in structural biology. Recent inverse folding models, trained to reconstruct sequences from structure, have had considerable impact in identifying functional mutations. However, current approaches are constrained to designing sequences composed exclusively of natural amino acids (NAAs). The larger set of non-canonical amino acids (NCAAs), which offer greater chemical diversity, and are frequently used in in-vivo protein engineering, remain largely inaccessible for current variant effect prediction methods.

To address this gap, we introduce \textbf{UNAAGI}, a diffusion-based generative model that reconstructs residue identities from atomic-level structure using an E(3)-equivariant framework. By modeling side chains in full atomic detail rather than as discrete tokens, UNAAGI enables the exploration of both canonical and non-canonical amino acid substitutions within a unified generative paradigm. We evaluate our method on experimentally benchmarked mutation effect datasets and demonstrate that it achieves substantially improved performance on NCAA substitutions compared to the current state-of-the-art. Furthermore, our results suggest a shared methodological foundation between protein engineering and structure-based drug design, opening the door for a unified training framework across these domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces UNAAGI, an **$E(3)$-equivariant diffusion-based generative model** designed for the reconstruction of residue identities from atomic-level structure. A notable feature is its ability to explore both **canonical and non-canonical amino acid substitutions** within a unified generative paradigm. The method is evaluated on several experimentally benchmarked mutation effect datasets. Overall, the work addresses an important problem in protein design, but improvements in clarity, novelty exposition, and experimental validation are necessary.

### Strengths
1.  **Novel Generative Task**: The inclusion of non-canonical amino acid substitution as a generative task is a significant and novel contribution. This method expands the scope of computational protein design beyond the traditional 20 canonical residues, opening promising new avenues for exploring protein functionality, stability, and therapeutic applications using synthetic or engineered residues.

### Weaknesses
1.  **Questionable Methodological Novelty**: The proposed UNAAGI model appears to be a combination of existing methodologies, and the manuscript does not sufficiently elucidate the key, specific technical innovations that distinguish this model from its predecessors in the context of the presented tasks.
   
2.  **Insufficient Experimental Baselines**: The experimental evaluation is limited by a narrow selection of baseline methods. it omits comparisons with several highly relevant state-of-the-art models for side-chain prediction and mutation effect prediction, which often handle atomic-level or torsional representations of side-chains. 

3.  **Lack of Clarity in Method Description**: The paper suffers from a lack of sufficient detail regarding the architectural specifics and the training regimen of the UNAAGI model. This absence of clarity. The authors should improve the methodological description.

### Questions
### Novelty and Methodological Clarification

1.  Could the authors explicitly clarify the specific, novel methodological contributions of UNAAGI? As the current description suggests a combination of pre-existing components.

2.  The statement regarding the novelty of "modeling sidechain in full atomic detail rather than as discrete tokens" requires refinement. Previous works, such as those employing methods like FAMPNN [1] or torsional diffusion models [2], have also modeled side-chains using atomic coordinates or torsion angles. Could the authors provide a more precise explanation of how UNAAGI's approach to atomic-level side-chain modeling differs fundamentally or offers an advantage over these existing methods?

3.  Could the authors include and compare their work with contemporary side-chain modeling approaches specifically tailored for protein mutation effect prediction in the Related Work section, such as those referenced in [3] and [4].

4.  In the Method section, clarification is needed regarding the processing of coordinates for masked tokens. Are these coordinates zero-padded, initialized to a specific value, or handled in an alternative manner during the forward and reverse diffusion processes?

### Experimental Design and Evaluation

5.  The rationale behind several choices in the Dataset Curation section must be clarified:
    * Why was the PDB subset restricted to only 1000 proteins? Is this sufficient for robust model training, particularly for a diffusion model?
    * What specific procedures were implemented to address and mitigate potential data overlap between the training, testing, and external evaluation sets, particularly concerning the NCAAs and PDBBind datasets?

6.  Regarding the evaluation on DMS datasets, why was the model's performance estimated using a sample frequency derived from 100 iterations rather than utilizing the likelihood or score function directly provided by the diffusion model? Clarification on the statistical justification for this sampling approach is required.

7.  Could the authors justify the decision to only evaluate on assays containing fewer than 100 residues? This size constraint may limit the generalizability of the reported performance to larger or more complex proteins.

8.  In the Comparison on the ProteinGYM dataset, the evaluation is incomplete. The authors must include a comparison with recent high-performing methods on this benchmark, such as SaProt [5] and other relevant models, to provide an up-to-date and authoritative performance assessment.




References:

[1] Widatalla, T., Shuai, R.W., Hie, B. and Huang, P., Sidechain conditioning and modeling for full-atom protein sequence design with FAMPNN. In Forty-second International Conference on Machine Learning.

[2] Zhang, Y., Zhang, Z., Zhong, B., Misra, S. and Tang, J., 2023. Diffpack: A torsional diffusion model for autoregressive protein side-chain packing. Advances in Neural Information Processing Systems, 36, pp.48150-48172.

[3] Liu, S., Zhu, T., Ren, M., Yu, C., Bu, D. and Zhang, H., 2023. Predicting mutational effects on protein-protein binding via a side-chain diffusion probabilistic model. Advances in Neural Information Processing Systems, 36, pp.48994-49005.

[4] Luo, S., Su, Y., Wu, Z., Su, C., Peng, J. and Ma, J., Rotamer Density Estimator is an Unsupervised Learner of the Effect of Mutations on Protein-Protein Interaction. In The Eleventh International Conference on Learning Representations.

[5] Su, J., Han, C., Zhou, Y., Shan, J., Zhou, X. and Yuan, F., SaProt: Protein Language Modeling with Structure-aware Vocabulary. In The Twelfth International Conference on Learning Representations.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UNAAGI, a model based on E(3)-equivariant diffusion that predicts the impact of amino acid substitutions by generating side chains at the atomic level. Its core innovation lies in the ability of the model to operate on a continuous molecular structure representation, thereby unifying the prediction space for natural amino acids (NAA) and non-natural amino acids (NCAA). This work addresses an important gap in computational protein engineering, where most state-of-the-art models are limited to the 20 natural amino acids. However, these advantages are significantly compromised by a lack of experimental rigor, particularly small training datasets, limited and opaque coverage of benchmark NCAAs, and the need for more comprehensive comparative analyses.

### Strengths
1. Shifting from discrete symbols (tokens) prediction to continuous, atom-by-atom side-chain generation is an interesting work. Traditional inverse folding models (such as ProteinMPNN and ESM-IF1) are fundamentally limited by their fixed output vocabulary. UNAAGI cleverly circumvents this constraint by modeling the underlying atomic coordinates directly.
2. Choosing an E(3)-equivariant Graph Neural Network architecture is a methodologically sound and principled choice. Since molecular data is inherently three-dimensional, respecting Euclidean symmetries (rotations, translations) is a crucial inductive bias that helps build more data-efficient and robust models.
3. The model in this paper showed a meaningful positive correlation on the NCAA DMS benchmark test.

### Weaknesses
1. The experimental evaluation strategy has significant weaknesses, which undermine the credibility of UNAAGI's performance conclusions on natural amino acids. The practice of subset selection based on protein length lacks sufficient justification and may introduce systematic bias. This leads to selection bias, as small proteins are more likely to consist of globular proteins with a single domain, whose properties are mainly determined by local interactions. UNAAGI is a model that relies on local environments. Therefore, this carefully selected subset may have precisely amplified the model's strengths, making its performance appear better than its actual performance on more complex large proteins or multi-domain proteins.
2. The baseline comparison in the paper is unreasonable. UNAAGI is itself a structure-based model, so structure-based inverse folding models (such as ProteinMPNN, ESM-IF1) and hybrid models (such as MIF-ST, SaProt) should be used as baselines for comparison, whereas comparing with mainly sequence-based models would greatly reduce the informativeness and persuasiveness. From Figure 2, it can be seen that the Spearman correlation coefficients of UNAAGI are mostly between 0.1 and 0.4, which is likely much lower than current SOTA structure models.
3. Data scarcity is a major challenge in NCAA modeling. The model was trained using only a small number of PDB structures, which limits the generalization ability of the model. The success on a few NCAAs may be because these NCAAs are structurally simple or chemically similar to natural amino acids (i.e., they are "interpolated" results). If the model performs poorly on NCAAs with more distinctive chemical properties, then the paper's claim about the model's "generalization ability to NCAAs" is unfounded.
4. The paper uses the results of NAA on ProteinGym as a reasonableness check and points out that "it has not fully reached the state-of-the-art level." However, as can be seen from Figure 2, the average Spearman correlation coefficient of ProteinGym replacing the SOTA models on benchmark tests can reach about 0.70 or even higher. In contrast, the performance of UNAAGI is below 0.4 on many test sets, showing significant inadequacy. The authors have not provided a reasonable analysis.
5. Lack of Argument: whether a model jointly trained on SBDD and side-chain generation tasks can learn a more general "protein-ligand interaction grammar," is lacking theoretical support and argumentation.
6. Lack of a section on qualitative analysis, presenting the generated 3D structure, and discussing its chemical feasibility. Adding some qualitative examples can improve the quality of the paper. For instance, an NCAA side chain successfully generated in a protein environment with a high adaptability score; a side chain that failed to generate or is poorly positioned; a novel NCAA generated by the model that has not been seen in the training set; etc.. This will provide more intuitive information for understanding the model's behavior.

### Questions
1. 100 samples are clearly insufficient to estimate the probability density of a vast chemical space. Could the authors provide an analysis of sampling stability (e.g., by repeating the 100-sample
experiment multiple times and observing the variance in the correlation score)?
2. The paper admits that the coverage of NCAA is "limited to a small subset". Could the authors quantify which specific NCAA types the model successfully sampled and the proportion of these in the 20 types in the NCAA benchmark (e.g., CP2 and PUMA)? Was the high correlation in Figure 4 calculated based only on this small subset? This is crucial for evaluating the model’s generalization ability and avoiding selection bias.
3. The mixed strategy of training data (PDB, PDBBind, isolated NCAAs) is confusing. Can the authors provide ablation experiments? Specifically: (a) How does the model perform when trained only on 1000 PDB structures (only NAA)? (b) What is the contribution of the PDBBind ligand data and isolated
NCAA data?
4. Regarding the trade-off in NAA performance: The model’s performance on NAA (ProteinGym) is far below SOTA. Is this a fundamental cost for generalizing NCAA, or is it merely due to the model/data scale being too small (3.6M parameters, 1000 PDB structures)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper proposes UNAAGI, a diffusion-based approach that generates amino acid side chains at the atomic level to enable variant effect prediction for non-canonical amino acids (NCAAs). The method uses E(3)-equivariant molecular diffusion to reconstruct side chains from structure, allowing continuous generation beyond the standard 20 amino acids. The authors evaluate on standard benchmarks (ProteinGym) and a small NCAA dataset, showing some correlation with experimental measurements.

However, the work appears highly preliminary. The experimental scale is inadequate (1,000 training structures, 2 NCAA benchmark complexes, evaluation limited to small proteins). The method underperforms existing approaches on standard benchmarks and fails to demonstrate convincing NCAA prediction, with the authors acknowledging it "tends to interpolate between canonical-like structures." Critical methodological details are missing, the evaluation protocol has statistical weaknesses, and figures lack the quality and analysis expected for a venue like ICLR. The work requires substantial development in scale, rigor, and completeness before it can be properly evaluated.

### Strengths
1. The paper addresses an important and largely unexplored problem of extending variant effect prediction to non-canonical amino acids.
2. The atomic-level generation approach is conceptually sound.
3. The connection between structure-based drug design and protein engineering is insightful. Recognizing that both domains involve modeling non-covalent interactions in protein contexts and could share methodological tools is a valuable observation that may inspire future work.
4. The virtual node padding strategy of handling variable-sized side chains is elegant.

### Weaknesses
1. **The method fails to demonstrate meaningful NCAA prediction capability despite being its core contribution**. The model only successfully samples a small fraction of the 20 NCAAs in the benchmark and admits it "tends to interpolate between canonical-like structures" rather than generating chemically distinct non-canonical amino acids. This undermines the entire premise of the work.


2. **The experimental scale is too small to evaluate the approach**. Training on only 1,000 PDB structures with 3.6M parameters is orders of magnitude below modern protein models. The NCAA benchmark contains only 2 protein complexes. Evaluation is restricted to proteins under 100 residues due to computational constraints. These limitations make it impossible to determine whether poor results reflect the method or simply insufficient resources.


3. **The method underperforms on standard benchmarks where comparisons are possible**. UNAAGI fails to match state-of-the-art on ProteinGym and even struggles against weak baselines. NCFlow shows negative correlations on NCAAs, and PepINVENT barely recovers wild-type residues. This suggests the field lacks any reliable NCAA prediction method, not that UNAAGI solves the problem.


4. **Critical methodological details are missing throughout**. The graph isomorphism algorithm for matching generated structures to amino acids is not described. Architecture specifications (layer counts, hidden dimensions, etc.), hyperparameters (learning rate, batch size, etc.), and training procedures are largely absent. The choice of 100 sampling iterations is unjustified. These omissions prevent reproduction and evaluation of design choices.


5. **The evaluation protocol has fundamental statistical flaws**. Using sampling frequencies from only 100 iterations as probability estimates is statistically weak. The method can only evaluate positions where both wild-type and mutant appear in samples, creating severe selection bias. No confidence intervals, error bars, or significance tests are provided. High variance across assays with no predictable pattern further limits reliability.


6. **Figures are of poor quality and provide minimal insight**. Figure 1 is simplified to the extend that it bears no information. The 24 nearly identical scatter plots in the appendix are cluttered and unreadable. Critical visualizations are missing (examples of generated structures, which NCAAs can actually be produced, sample quality assessment). The paper lacks any qualitative analysis of what the model learns or why it fails.


8. **The work lacks sufficient NCAA training data in relevant contexts**. Most NCAA examples are isolated amino acids without protein environments or come from protein-ligand complexes that may not capture constraints relevant for variant effect prediction. This data scarcity likely explains why the model cannot generalize to chemically distinct NCAAs.

### Questions
1. What is the relationship between generative sample quality and predictive performance? Do positions with higher geometric accuracy or chemical validity correlate with better mutational effect predictions?
2. Why should atomic-level generation outperform discrete vocabulary models for variant effect prediction? The theoretical motivation is unclear. Inverse folding models like ProteinMPNN achieve strong performance despite discrete sampling. What property of the continuous approach provides an advantage?
3. What determines which amino acids (canonical or non-canonical) the model successfully recovers? Is there a pattern based on chemical properties (size, polarity, aromaticity), frequency in training data, or structural context? 
4. Have you analyzed whether including NCAA-protein structures from the PDB improve results?
5. Could guidance or conditioning during sampling improve NCAA generation?

### Soundness
1

### Presentation
1

### Contribution
2
