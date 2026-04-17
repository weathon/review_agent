# Design of Ligand-Binding Proteins with Atomic Flow Matching

- Decision: Reject
- Scores: 4, 6, 4, 4, 8

## Abstract
Designing novel proteins that bind to small molecules is a long-standing challenge in computational biology, with applications in developing catalysts, biosensors, and more. Current computational methods rely on the assumption that the binding pose of the target molecule is known, which is not always feasible, as conformations of novel targets are often unknown and tend to change upon binding. In this work, we formulate proteins and molecules as unified biotokens, and present AtomFlow, a novel deep generative model under the flow-matching framework for the design of ligand-binding proteins from the 2D target molecular graph alone. Operating on the positions of biotokens, AtomFlow captures the flexibility of ligands and generates ligand conformations and protein backbone structures iteratively. We consider the multi-scale nature of biotokens and demonstrate that AtomFlow can be effectively trained on a subset of structures from the Protein Data Bank, by matching the flow vector field using an SE(3) equivariant structure prediction network. Experimental results demonstrate that our method generates high-fidelity ligand-binding proteins, matching or surpassing the performance of RFDiffusionAA across multiple metrics—without requiring bound ligand structures. As a general framework, AtomFlow can be readily extended to diverse biomolecule design tasks in the future.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces AtomFlow, a flow-matching framework for protein-ligand complex generation condition on molecular ligand graphs. design. Protein and ligand representations are unified via "biotokens". The papers claims that the proposed model can perform de novo protein binder design without requiring a predefined ligand conformer, achieving comparable or superior binding affinity to RFDiffusionAA and faster inference speed.

### Strengths
- Biotoken representation: a unified feature framework for ligand atoms and protein residues within an SE(3)-equivariant space is elegant and facilitates joint modeling of multiple biomolecular types.

- Speedup: AtomFlow achieves over 5x inference speed compared to RFDiffusionAA.

### Weaknesses
1. The model predicts only C$-\alpha$ coordinates for proteins. Therefore, it is unclear to me how it can achieve high binding affinity when full protein information is not generated directly by the model but instead depends on downstream pipelines. It is also unclear whether, when using AutoDock Vina, the generated complexes were redocked or if the Vina score was simply computed on the generated structures. If the former is true, such discrepancies would weaken the claims regarding physical binding accuracy. Finally, some of the reported metrics, such as PoseBuster and PoseCheck, are mentioned as being performed but are not actually presented in the paper (see Appendix A.6, page 18). These metrics are essential for assessing the quality of the generated structures.

2. Equation 5 bins continuous distance values before applying flow matching, introducing a discontinuity that may conflict with the continuous-space assumptions of the flow ODE formulation.

3. The handling of categorical variables (e.g., residue types) is not discussed, even though flow matching operates in a continuous space.

4. The Vina score distributions for AtomFlow, reported in Equation 5, are quite similar to those of the competing method, RFDiffusionAA. These results, combined with the absence of key metrics such as PoseBuster and PoseCheck, weaken the overall contributions of the paper. However, I acknowledge that the reported speed-up remains a valid and valuable contribution.

5. The relationship of $(r_i, t_i)$ with respect to $a_i$  in the set  {$T_i = (r_i, t_i) \ | \ a_i \in \mathcal{P}$}  on page 3 line 135, is not specified. Also $t$ is used for both time and $t_{i,j}$ in Eq. 5.

### Questions
1. How are the categorical residue or atom types incorporated into the continuous flow-matching framework?

2. Can you provide explicit mathematical expressions for how $f^{pair}$ and $f^{token}$ are computed and used by the structure prediction network?

3. When using AutoDock Vina, are the generated complexes redocked or if is Vina simply computed on the generated structures?

4. Why is the PoseBusters/PoseCheck evaluation missing?

5. Could you discuss why continuous variables are discretized in Eq. (5), while still using a flow matching that handles continuos variables and whether this affects training stability?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses an important challenge in molecular AI: designing protein binders for small-molecule ligands. The proposed method, **ATOMFLOW**, generates binding coordinates for protein-ligand complexes using only the 2D molecular graph of the target ligand. This is achieved through biotokenization of ligand atoms and protein residues, pairwise relationship embedding, and a flow-matching framework. The method is supported by thorough experiments across diverse ligands.

However, the novelty of the work is limited. The problem formulation and methodological components—such as flow matching and SE(3)-equivariant modeling—are largely adapted from prior literature. While the results are competitive, they do not demonstrate significant improvements over existing models such as RFDiffusionAA or Chai-1.

### Strengths
- Clear problem formulation and modular architecture.
- Robust evaluation across ligands.
- Faster inference compared to RFDiffusionAA.
- Unified representation of protein and ligand tokens.

### Weaknesses
- Limited methodological novelty.
- Benchmarking setup may introduce bias due to self-generated reference structures.

### Questions
1. **Terminology clarity**: Please expand acronyms such as ODE and SAM. Given the diversity of the AI4Science community, not all readers share the same technical background.

2. **Line 205**: What does the symbol \( Q \) represent? Clarifying this would improve readability.

3. **One-hot binned distance map**: What motivated this design choice? How does it compare to alternatives such as continuous embeddings or radial basis functions?

4. **Choice of LigandMPNN and ESMFold**: Why were these models selected for sequence recovery and structure validation? A brief overview of their accuracy and relevant references would be helpful.

5. **Benchmark fairness (Table 2)**: The reference structures are derived from your own pipeline using LigandMPNN and ESMFold. Does this introduce bias when comparing against other models? Please clarify.

6. **Feature redundancy**: Figure 2 shows that the distance map is passed to the Feature Embedder. If pairwise features already encode distance, is there redundancy between `pair feat` and the distance map? Clarifying their distinct roles would strengthen the architectural rationale.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AtomFlow, a flow matching model for designing a protein structure to bind a small molecule ligand. The model jointly denoises the structure of the protein and ligand and thus does not require knowledge of the ligand pose, unlike RFDiffusionAA. The architecture is based on AlphaFold and predicts denoised structures from a distance map input of the input structure. The method is evaluated on (1) the set of 4 ligands studied in RFDiff-AA, and (2) an expanded set of ligands curated by the authors. AtomFlow is shown to have comparable or better designability than RFAA as well as similar Vina score.

### Strengths
* The work is solidly executed, with sensible architectural choices, strong initial evaluations, and clear and concise writing.
* The task tackled is significant and the competitive results signify well-executed model engineering and training practices.
* The authors re-derive the quotient-space flow matching fromework from AlphaFlow with more solid theoretical justification.
* The paper is quite clearly written. The figures are well made, visually appealing, and informative.

### Weaknesses
**Originality**
* The methodology can be described as a flow-matching version of RFDiff-AA and does not score high on originality / novelty from a ML perspective. Further, the flow model architecture and noising process are based on AlphaFlow, with different justification but no difference in practice as far as I can tell. To improve on this axis, while it's not clear that more methodological novelty is needed for its own sake, the authors could focus on novel evaluations or applications of the proposed method.

**Quality** 
* The computational evaluations are well executed, but limited in scope. Most of the analysis focuses on only 4 ligands, raising concerns about sample size and statistical significance.
* The diversity and novelty evaluations are nice, but only AtomFlow is evaluated, not RFDiff-AA or the other baselines.

**Significance**

* The overall significance of the contribution is unclear as it represents an incremental methodological advance over RFDiff-AA with more or less the same model capabilities. The authors argue that not needing to specify the ligand pose is a big plus, but no meaningful evidence or use case is provided for this distinction. After all, RFDiff-AA has been experimentally validated, whereas AtomFlow-generated poses have not. It is of course not expected for a ML submission to experimentally validate the proteins, but it should be made a bit clearer why the main point of difference with RFDiff-AA is important to tackle as a ML problem.

### Questions
No specific questions.

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
4

### Summary
The paper proposes ATOMFLOW, which uses a unified biotoken representation to jointly generate protein and ligand structures
by learning the distribution of token positions conditioned on a ligand chemical graph. It uses flow matching to learn the structure prediction model based on both token features and pair features. The proposed method is evaluated on the PDBBind dataset, with a focus on four ligands (FAD, SAM, IAI, OQO).

### Strengths
The paper targets at an important and valuable task and achieves comparable performance to RFDiffusionAA.

### Weaknesses
The weaknesses of this paper are listed as follows:

1. The evaluation of the method just focuses on several ligands, making the generalization of the method unknown. Basically, most of the experiments are conducted on four specific ligands. I'm not sure if the designed methods can be well adapted to broader applications, like enzyme-substrate complex structure prediction, which is even harder as the transition state of the enzyme is generally hard to capture.

2. It seems the proposed method is limited to generate shorter proteins. All the designed proteins are limited to a length shorter than 300.

3. The performance of the original ATOMFlOW-N is worse than RFdiffusionAA, and the variant ATOMFlOW-H is comparable to RFDiffusionAA. 

4. There are some newer models like RFDiffusion2, which the proposed method should be compared with.

5. In Figure 8, when showing the novelty and diversity of the proposed method, the paper doesn't provide the performance of baseline methods like RFDiffusionAA, which makes it hard to capture the comparison with previous SOTA methods.

### Questions
The ATOMFLOW-N performs worse than RFDiffusionAA, while ATOMFLOW-H which uses an auxiliary hint input of the pairwise distance matrix of the bound structure achieves better performance. Can the authors explain in detail how they implemented ATOMFLOW-H and why it helped?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method for protein binder design. Both protein residues and ligand are modeled in a joint representation ("biotokens"), which enables to jointly generate protein-ligand complexes in an SE(3)-equivariant flow matching framework.

### Strengths
The paper is well written and the presentation is clear. The approach of representing both ligand and residues in a joint feature representation is elegant and enables to use the same flow matching generative process for both, leading to a generative model for protein-ligand complexes. The method is on par and sometimes outperforms RFDiffusionAA in binder design quality, and (through flow matching) offers faster inference.

### Weaknesses
No major weaknesses. Great paper!

Technical remarks:

Typo in Fig2: Piror Distribution

### Questions
'

### Soundness
4

### Presentation
4

### Contribution
4
