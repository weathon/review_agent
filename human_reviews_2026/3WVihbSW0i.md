# TetraGT: Tetrahedral Geometry-Driven Explicit Token Interactions with Graph Transformer for Molecular Representation Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Molecular representations that fully capture geometric parameters such as bond angles and torsion angles are crucial for accurately predicting important molecular properties including enzyme catalytic activity, drug bioactivity, and molecular spectral characteristics, as demonstrated by extensive studies.
However, current molecular graph representation learning approaches represent molecular geometric parameters only indirectly through combinations of atoms and bonds, neglecting the spatial relationships and interactions between these higher-order geometric structures.
In this paper, we propose \textbf{TetraGT} (\textbf{Tetra}hedral \textbf{G}eometry-Driven Explicit \textbf{T}oken Interactions with Graph Transformer), a novel architecture that directly models molecular geometric parameters.
Based on the spatial solid geometry theory of face angle and dihedral angle inequality, TetraGT explicitly represents bond angles and torsion angles as structured tokens for the first time, directly reflecting their intrinsic role in determining the molecular conformational stability and properties. 
Through our designed spatial tetrahedral attention mechanism, TetraGT achieves highly selective direct communication between structural tokens.
Experimental results demonstrate that TetraGT achieves superior performance on the PCQM4Mv2 and OC20 IS2RE benchmarks. 
We also apply our pre-trained TetraGT model to downstream tasks including QM9, PDBBind, Peptides and LIT-PCBA, demonstrating that TetraGT delivers excellent results in transfer learning scenarios and shows scalability with increasing molecular size.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work focuses on traditional molecular modeling, aiming to learn effective representations from 3D molecular configurations and then fine-tune them for specific downstream tasks (e.g., property prediction). The authors identify that previous methods often fail to incorporate higher-order geometric structures. To address this, they enhance Graph Transformer (GT) architectures by introducing bond angle and torsion angle information as tokens and designing a spatial tetrahedral attention mechanism. Additional techniques include a specially designed “directed cycle angle loss” and a hierarchical virtual node for molecule-level information aggregation. Experimental results demonstrate strong performance.

### Strengths
1. **Strong benchmark results.** The proposed method achieves excellent performance on several benchmarks (LIT-PCBA, PCQM4MV2, OC20 IS2RE, QM9), reaching or surpassing state-of-the-art results in nearly all cases. These results effectively demonstrate the transfer learning capability of the approach across diverse downstream tasks.
2. **Novel tetrahedral constraints.** The introduction of Tetrahedral Constraints and the Tetrahedral Interaction Module shows a novel perspective on modeling geometric relationships. The ablation study further confirms that this module contributes meaningfully to overall performance.
3. **Chirality structure analysis.** The authors conduct a statistical analysis of chirality structures (46.61% of the total 3M molecules). Table 9 compares errors with TGT, supporting the rationale behind their more complex design choices (e.g., attention mechanism and loss functions). The ablation study on the Directed Cycle Loss also validates the effectiveness of this component.
4. **Hierarchical virtual node.** The hierarchical virtual node design provides a certain degree of novelty and contributes to molecular-level information aggregation, though the innovation is not particularly substantial compared to other components.

### Weaknesses
1. **Efficiency concern.** As a work focused on model architecture design, the paper lacks analysis or discussion of model efficiency in the main text. Although Appendix E presents promising results, a more comprehensive discussion of efficiency in the main body would strengthen the paper’s overall contribution and practical relevance.
2. **Discussion of prior works on higher-order geometric structures.** Works such as [1,2] have also considered interactions among higher-order geometric structures, albeit in **implicit** ways (in contrast to the token-based approach proposed in this paper), as mentioned and discussed in Appendix J. Although the *“Implicit Modeling of Geometric Structures”* section in the Introduction provides some relevant statements, it would be helpful to clearly specify **which prior works** do not explicitly model higher-order geometric structures and **how** these structures are indirectly represented through combinations of atomic positions or pairwise relationships. Including this clarification in the main Introduction would better emphasize the motivation and novelty of the proposed approach.

[1] Wang Z, Liu G, Zhou Y, et al. Efficiently incorporating quintuple interactions into geometric deep learning force fields[J]. Advances in Neural Information Processing Systems, 2023, 36: 77043-77055.

[2] Wang Y, Wang T, Li S, et al. Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing[J]. Nature Communications, 2024, 15(1): 313.

### Questions
1. In Table. 12&13, the training and inference times of UniMol+ and TGT are missing. Since inference time is typically more important in practical applications, could you clarify why these results were omitted?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
TetraGT is a molecular representation model that explicitly incorporates 3D geometric parameters (bond angles and torsion angles) into graph transformer architectures. TetraGT treats angles and dihedrals as structured tokens and models their interactions based on tetrahedral geometry constraints. The method defines mathematical relationships among face and dihedral angles to ensure physically valid spatial configurations and introduces a multi-level attention mechanism that hierarchically updates representations from atoms to bonds, bond angles, and torsion angles. A specialized “tetrahedral interaction module” enables efficient communication among geometrically related triplets and quadruplets while reducing complexity via local sampling. Additionally, TetraGT introduces a directed circular angle loss to handle periodicity and chirality in angle prediction and employs hierarchical virtual nodes to integrate multi-level structural information for final molecular property prediction.

### Strengths
- Explicit “angle and torsion tokens” + “tetrahedral attention” are new combinations not seen in previous 3D graph Transformers.
- The physical constraints (Lemma 1) add rigor, not just architectural novelty.
- The directed cycle angle loss (DCA loss) for handling 2π-periodicity is conceptually sound and addresses chirality.
- The experiments show strong performance gains over baselines across multiple benchmarks, but clearer comparisons with recent equivariant models and ablations on key components would better substantiate the claimed improvements.

### Weaknesses
- While the model’s use of tetrahedral geometry constraints is mathematically grounded, the repeated use of the term “tetrahedral” may be confusing in a chemical context, since most molecular sites are not tetrahedral in bonding geometry. The paper should clarify that “tetrahedral” here refers to the geometric configuration of any four non-coplanar atoms (a 3D simplex), rather than chemically tetrahedral centers (sp³ atoms).

### Questions
There have been recent higher-order geometric message passing and equivariant Transformers (e.g., GemNet, Equiformer, TFN, SE(3)-Transformers, MACE) that already model angular and dihedral information. Does this explicit token-based formulation yield substantial conceptual or empirical advantage beyond existing geometric message passing frameworks?

### Soundness
3

### Presentation
2

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
This paper proposes TetraGT, an attention-based model designed to accurately predict molecular properties from both 2D molecular graphs and 3D conformations. Existing methods often fail to directly model key molecular geometric parameters, such as bond angles and torsion angles, leading to limitations in capturing high-order structural relationships and local molecular chirality. TetraGT addresses these shortcomings through four core innovations: (1) The direct modeling of geometric parameters as structured tokens to prevent error accumulation from indirect atomic and bond representations; (2) A spatial tetrahedral attention mechanism, informed by tetrahedral geometry theory, to facilitate information exchange between these parameters; (3) An improved directed cycle angle loss function to handle geometric parameters and identify local chirality; (4) A hierarchical virtual node aggregation architecture that captures sub-structural information for a comprehensive molecular representation.
Experimental results demonstrate that TetraGT achieves superior overall performance on upstream datasets such as PCQM4Mv2 and OC20 IS2RE compared to models like Uni-Mol+. It also shows strong generalizability, outperforming models like EquiformerV2+NN on downstream tasks including QM9 and PDBBind.

### Strengths
1. The work is innovative in its direct representation of molecular geometry as structured tokens and the introduction of a spatial tetrahedral attention mechanism and a directed cycle angle loss function.
2. The model design is comprehensive and methodologically sound, effectively integrating geometric parameter interaction, local chirality identification, and graph sub-structure aggregation.
3. The model's performance and generalizability are rigorously validated across multiple upstream and downstream benchmark tasks.
4. This research provides a novel paradigm for molecular property prediction, with considerable potential for application in drug design and materials discovery.

### Weaknesses
1. The analysis of experimental results is insufficient. While Table 4 shows that TetraGT achieves state-of-the-art performance on 5 out of 12 metrics on the QM9 dataset, it is outperformed on the remaining 7 metrics by models like EquiformerV2+NN. The manuscript lacks a systematic discussion to explain this performance disparity.
2. The organization of the related work section is non-standard. Placing the "Related Work" section after the "Experiments" and immediately before the "Conclusion" deviates from conventional academic structure.
3. The introduction is poorly articulated. The summary of key contributions in the "Introduction" section is not itemized, which hinders readability and fails to clearly preview the paper's innovations.

### Questions
1. A systematic explanation should be provided in the main text to account for the performance gap on the specific QM9 metrics where TetraGT does not achieve optimal results.
2. The "Related Work" section should be relocated to follow the "Introduction" and precede the "Method" section. Content in the introduction that details core challenges and innovations (e.g., the rationale for the spatial tetrahedral attention mechanism) should be integrated into this restructured "Related Work" section to better contextualize the research.
3.  The key contributions within the "Introduction" section should be presented as a clear, itemized list to enhance readability and provide a straightforward overview of the paper's novel elements.

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
4

### Summary
This paper proposes TetraGT, a graph Transformer framework for molecular representation learning. The key idea is to treat not only atoms and chemical bonds as tokens, but also higher-order geometric structures, specifically bond angles and torsion angles, as explicit tokens that can directly interact through an attention mechanism constrained by tetrahedral geometric consistency, rather than relying on traditional models to infer such information implicitly from pairwise distances. The paper further introduces a chirality-aware angular learning objective, the Directed Cycle Angle Loss, which models angles as directed periodic variables over (0,2π); this is intended to distinguish local chirality and avoid discontinuities around cases that occur in standard angle regression or classification. In addition, the model employs a hierarchical virtual node design that aggregates information separately at the atom, bond, angle, and torsion levels before passing it to a global node, aiming to alleviate the information bottleneck of a single global aggregator. The authors argue that these components together enable more faithful modeling of stereochemistry, local chirality, and conformational stability, thereby improving representation quality and transferability.

### Strengths
1. The paper directly addresses three known gaps: explicit modeling of chirality, explicit treatment of higher-order geometry (bond and torsion angles), and explicit handling of geometric consistency constraints. 

2. The proposed tetrahedral interaction module constrains attention among angle/torsion tokens to local tetrahedral units and encodes geometric consistency, aiming to retain physical validity while avoiding naive O(N^3)high-order connectivity. 

3. The Directed Cycle Angle Loss treats angles as directed periodic variables over (0,2π), which is intended to distinguish local chirality and avoid instability at angular wrap-around. Most prior models do not handle it directly. 

4. The model promotes atoms, bonds, bond angles, and torsions to first-class tokens and aggregates them via hierarchical virtual nodes, rather than relying on a single global node. This is meant to reduce information bottlenecks. 

5. The method is evaluated on diverse and competitive benchmarks (PCQM4Mv2, OC20 IS2RE, QM9, PDBBind, Peptides, LIT-PCBA) and reports state-of-the-art or near state-of-the-art results against strong baselines, suggesting transferability beyond a single task

### Weaknesses
1. The method treats local groups of atoms as “tetrahedral units” and uses these units as the fundamental template for constrained attention and geometric consistency. However, the paper does not demonstrate how well this assumption holds in other common chemical settings (e.g., conjugated rings, metal coordination sites) where local geometry is not tetrahedral. It is unclear whether this inductive bias could introduce systematic errors in such non-tetrahedral regimes, or whether the model adapts automatically?

2. The tetrahedral geometric inequalities are injected as attention biases/gates, and the authors argue this both reduces complexity from 𝑂(𝑁3) to approximately 𝑂(𝑤𝑁2) and enforces physical consistency. However, the paper does not provide a formal characterization of this mechanism. In particular, it does not report metrics such as the fraction of geometrically invalid local angle/torsion configurations before vs. after training, nor does it prove that predicted angles/torsions are always physically realizable. It remains unclear whether this acts as a true constraint or mainly as an inductive bias.

3. The method relies on a local sampling window of size 𝑤 to decide which angle/torsion tokens are allowed to interact. If 𝑤 fails to cover an important substructure, or only partially covers it, the model could miss critical geometric couplings. The paper does not present sensitivity studies on different 𝑤 values or analyze the impact of under-coverage on accuracy, nor does it clarify how 𝑤 should scale with molecular size.

4. Training still requires on the order of tens of A100 GPU-days, comparable to other top-performing large models, which is a high barrier for many labs. The authors assert that the approach is scalable to larger systems, but they do not analyze how resource usage grows when moving to substantially larger molecules or protein–ligand complexes. In particular, it is not clear whether the number of tokens and pairwise interactions will scale roughly linearly or blow up faster, so the scalability claim is not yet quantitatively supported. 

5. In the abstract, Figure 1, and the conclusion, the authors use the alternate name “TDGT” for the proposed architecture but do not provide a proper definition.

### Questions
1. The method treats local groups of atoms as “tetrahedral units” and uses these units as the fundamental template for constrained attention and geometric consistency. However, the paper does not demonstrate how well this assumption holds in other common chemical settings (e.g., conjugated rings, metal coordination sites) where local geometry is not tetrahedral. It is unclear whether this inductive bias could introduce systematic errors in such non-tetrahedral regimes, or whether the model adapts automatically?

2. The tetrahedral geometric inequalities are injected as attention biases/gates, and the authors argue this both reduces complexity from 𝑂(𝑁3) to approximately 𝑂(𝑤𝑁2) and enforces physical consistency. However, the paper does not provide a formal characterization of this mechanism. In particular, it does not report metrics such as the fraction of geometrically invalid local angle/torsion configurations before vs. after training, nor does it prove that predicted angles/torsions are always physically realizable. It remains unclear whether this acts as a true constraint or mainly as an inductive bias.

3. The method relies on a local sampling window of size 𝑤 to decide which angle/torsion tokens are allowed to interact. If 𝑤 fails to cover an important substructure, or only partially covers it, the model could miss critical geometric couplings. The paper does not present sensitivity studies on different 𝑤 values or analyze the impact of under-coverage on accuracy, nor does it clarify how 𝑤 should scale with molecular size.

4. Training still requires on the order of tens of A100 GPU-days, comparable to other top-performing large models, which is a high barrier for many labs. The authors assert that the approach is scalable to larger systems, but they do not analyze how resource usage grows when moving to substantially larger molecules or protein–ligand complexes. In particular, it is not clear whether the number of tokens and pairwise interactions will scale roughly linearly or blow up faster, so the scalability claim is not yet quantitatively supported. 

5. With respect to the overall pipeline: in the multi-stage training process, could you clarify the relationship between the “conformation predictor” and the “task predictor”? It would also be helpful to more explicitly describe the full loss composition across stages — is it primarily pairwise atomic distance regression together with the Directed Cycle Angle Loss for angles, or are there additional property-prediction heads contributing to the objective? 

6. (Minor question) In the abstract, Figure 1, and the conclusion, the authors use the alternate name “TDGT” for the proposed architecture but do not provide a proper definition.

### Soundness
2

### Presentation
2

### Contribution
3
