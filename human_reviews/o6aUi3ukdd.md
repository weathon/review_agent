# An Open Quantum Chemistry Property Database of 120 Kilo Molecules with 20 Million Conformers

- Decision: Reject
- Scores: 3, 3, 3, 1

## Abstract
Artificial intelligence is revolutionizing computational chemistry, bringing unprecedented innovation and efficiency to the field. To further advance research and expedite progress, we introduce the Quantum Open Organic Molecular (QO2Mol) database — a large-scale quantum chemistry dataset designed for researches on organic molecules under an open-source license. 
The database comprises 120,000 organic molecules and more than 20 million conformers, encompassing 10 different elements (C, H, O, N, S, P, F, Cl, Br, I), with heavy atom counts exceeding 40. Each conformation was computed at B3LYP/def2-SVP level of theory to derive quantum mechanical properties, including potential energy and forces. The molecules included in the dataset are based on fragments from compounds in ChEMBL, ensuring their structural \textit{relevance to real-world compounds}.
The extensive variety of molecular structures and elemental compositions represented in the dataset can facilitate construction of potential energy surface and various downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a benchmark dataset, named the Quantum Open Organic Molecular (QO2Mol) database, which contains 120,000 organic molecules and approximately 20 million conformers. The authors highlight that this dataset covers 10 different elements, (C, H, O, N, S, P, F, Cl, Br, I) with heavy atom counts exceeding 40. The quantum mechanical properties of the organic molecules were computed using the B3LYP/def2-SVP calculation method, which is claimed to provide accurate computations of properties such as potential energy and forces. The paper presents benchmark results for potential energy prediction, evaluating the performance of both invariant and equivariant models.

### Strengths
The strength of this paper may come from the large number of organic molecules included in the QO2Mol database and its diverse coverage of elements. Specifically, the database includes molecules with heavy atom counts exceeding 40, representing an improvement over existing datasets like QM9 and ANI-1. Another positive aspect is the inclusion of tasks related to conformer generation.

### Weaknesses
The most significant weakness of this paper is its fundamental lack of novelty when compared to the existing dataset in this chemistry field. The existing nabla^2DFT dataset [1] contains 2 million molecules, which is more than the 120,000 molecules in this QO2Mol database, and includes 16 million conformers, a number comparable to this work. The comparison becomes even more unfavorable when considering the depth of information provided. The nabla^2DFT dataset offers 17 different molecular properties, along with Hamiltonian and overlap matrices, and a wavefunction object - features that are not provided in this QO2Mol database. Additionally, the nabla^2DFT dataset includes relaxation trajectories for a significant number of organic molecules, another feature not present in this paper. Given that the nabla^2DFT dataset was made available online in June 2024, well before the ICLR submission deadline, it is unclear what distinct advantages the QO2Mol database offers over the existing nabla^2DFT dataset. Furthermore, this paper presents a significant lack of benchmark results, only providing potential energy prediction benchmarks in the main manuscript, which limits its scope and impact.

[1] Khrabrov, Kuzma, et al. "$\nabla^ 2$ DFT: A Universal Quantum Chemistry Dataset of Drug-Like Molecules and a Benchmark for Neural Network Potentials." arXiv preprint arXiv:2406.14347 (2024).

### Questions
Major Issues:
1. The authors need to explain the absence of citation to the nabla^2DFT dataset, which is a significant oversight given its relevance and prior availability.
2. The authors should clearly articulate the unique advantages of the QO2Mol database compared to the nabla^2DFT dataset
3. The limited scope of benchmark results requires further explanation. The focus solely on potential prediction experiments in the main manuscript seems insufficient for a benchmark dataset paper of this nature.
4. Although the authors emphasize the high precision of B3LYP, the choice still requires justification. B3LYP's performance may degrade with non-covalent and long-range interactions, which can be present in organic molecules. The authors should explain why B3LYP was chosen over ωB97X-D, which potentially offers better precision.

Minor Editorial Issues:
1. In the appendix section, there are typos such as 'inffuence' (should be 'influence') and 'signiffcant' (should be 'significant'), which diminishes the professionalism of the paper.
2. The spelling of ChEMBL is inconsistent throughout the paper, appearing as both "ChemBL" (appendix page 16) and "ChEMBL" (elsewhere). This inconsistency should be corrected for clarity.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a new dataset and benchmark with quantum chemical (QC) properties of 120,000 organic molecules and over 20 million conformers. The dataset uses moderately accurate physical theory B3LYP/def2-SVP. Moreover, the paper demonstrates the performance of 7 modern Neural Network Potentials on the proposed benchmark.

### Strengths
- A new dataset with QC properties and a new scheme for conformer generation. 
- A new benchmark for NNPs.

### Weaknesses
- Most recent datasets with the same or higher level of theory are not cited and compared with, for example, QMugs [1], SPICE [2], nablaDFT [3], and QM1B [4]. Moreover, the cited PubchemQC [5] dataset contains 86 million structures, while PCQM4Mv2 [6] is its subset.
- The comparison between theory levels is unclear: for example, the level of theory in ANI-1 has more complex exchange correlation potential than in other listed datasets.
- The conformer generation procedure seems undermotivated. Why were 30-degree increments for the angles chosen instead of random or more physically based?
- The dataset composition scheme that involves fragmentation may not be optimal since the molecules that are included in the dataset are the most frequent fragments of bigger molecules. Such dataset may not cover the compound space well enough as it does not contain molecules that are not fragments of larger molecules. This may explain the performance drop between the subset A test set and the subset B.
- Finally, it is not clear how the subset B was generated.  Were these molecules chosen from the initial dataset or taken from another source? Atom and scaffold analysis may reveal severe divergence between A and B.

### Questions
- Listed in Cons.
- Provide a comparison with other datasets.
- Provide a detailed comparison between subsets A and B. 
- Add the description of the molecular fragmentation method.
- Typo in line 185: We provides.

[1] Isert, C., Atz, K., Jiménez-Luna, J., & Schneider, G. (2022). QMugs, quantum mechanical properties of drug-like molecules. Scientific Data, 9(1), 273.
[2] Eastman, P., Pritchard, B. P., Chodera, J. D., & Markland, T. E. (2024). Nutmeg and SPICE: Models and data for biomolecular machine learning. Journal of Chemical Theory and Computation.
[3] Khrabrov, K., Ber, A., Tsypin, A., Ushenin, K., Rumiantsev, E., Telepov, A., ... & Kadurin, A. (2024). ∇ DFT: A Universal Quantum Chemistry Dataset of Drug-Like Molecules and a Benchmark for Neural Network Potentials. CoRR.
[4] Mathiasen, A., Helal, H., Klaser, K., Balanca, P., Dean, J., Luschi, C., ... & Masters, D. (2023). Generating QM1B with PySCF $ _ {\text {IPU}} $. Advances in Neural Information Processing Systems, 36, 55036-55050.
[5] Nakata, M., Shimazaki, T., Hashimoto, M., & Maeda, T. (2020). PubChemQC PM6: Data sets of 221 million molecules with optimized molecular geometries and electronic properties. Journal of Chemical Information and Modeling, 60(12), 5891-5899.
[6] Hu, W., Fey, M., Ren, H., Nakata, M., Dong, Y., & Leskovec, J. OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a computational dataset of small to medium-sized molecules. For 120k molecules, energies and atomic forces were computed for ~170 conformers per molecule. The molecules were generated through fragmentation and recombination of drug-like compounds in the CHEMBL library. The process of database construction (molecule library enumeration, conformer generation, electronic structure calculations) is described. Eventually, the study includes a comparison of established molecular machine learning models applied to in-domain and out-of-domain prediction tasks within this dataset.

### Strengths
1. High-level reference data on molecular energies and forces are necessary for creating more generalizable ML potential energy surfaces and force fields, with applications e.g. in drug discovery. 
2. The fragmentation–recombination strategy is chemically sensible, ensuring that the included molecules are realistic and feasible.

### Weaknesses
The main weaknesses of the paper are significant misinterpretations of core chemistry concepts and a lack of clarity around the dataset’s intended applications, leading to technical ambiguities and misleading claims. 

1. The paper does not clearly articulate the goal of developing the dataset, or potential use cases. Although the data is well-suited for potential energy surface modeling, the introduction implies applications outside of this scope (e.g., structure–activity relationships), which is strongly misleading. Section 2.2 should be re-written to clarify these use cases, and clarify the unclear “conformer generation” task. Moreover, the paper would strongly benefit from repositioning the use cases earlier in the paper to align the introduction with actual practical applications. A number of vague statements regarding applicability should be removed. 

2. The discussion (and application) of quantum chemistry methodologies is severely flawed. 
  * The high-level introduction to DFT misses a number of key points: The main goal is to approximate solutions to the electronic Schrödinger equation for obtaining energies and, possibly, estimate further molecular property labels from the computed solution. 
  * Figure 2 and the accompanying discussion makes strong statements about computation accuracy using different basis sets. However, both the Pople basis sets (6-31G(d) / 6-31G(2df,p)) and the Karlsruhe basis sets (def2-SVP) are double-zeta basis sets with additional polarization functions. To the best of my knowledge, for the molecules studied, the accuracy differences between these basis sets should be negligible. Any claims of significant accuracy differences must be rigorously supported with citations. 
  * Statements like “B3LYP/def2-SVP as one of the highest precision levels achievable“ are strongly misleading. In fact, in computational chemistry, double-zeta basis sets are considered a rather low-level method. Established workflows, such as optimizing geometries with cheap semi-empirical methods, followed by single-point DFT for energies and gradients, are entirely overlooked in generating the dataset. 
3. The discussion of chemistry concepts contains a number of inaccuracies and misinterpretations which weaken the perceived credibility of the study. Examples (non-comprehensive selection) include:
  * Force fields and potential energy surface are often discussed as two disjoint entities – rather than explaining that force fields are formally the derivative of the potential energy surface. 
* SMILES and InChI representations are introduced without clarity on their primary function as molecular graph notations. 
* “Heavy atom is any atom other than hydrogen, typically used in molecular studies to focus on more complex atomic interactions” (p. 3) This statement fails to address why distinguishing heavy atoms is relevant to real-world applications.
* “Labeled as important phosphate groups by our chemistry expert” (p. 5) What is the rationale behind labeling phosphate groups (which are one specific out of many relevant functional groups)? 
* “while the significance of other conformations is also noteworthy” (p. 6) This statement is highly ambiguous. What is the relevance of further (non-minimum) conformations? 

4. The choice of MAE as the error metric for model evaluation overlooks an essential consideration. Since energy scales with molecular size (i.e., number of heavy atoms), the MAE will naturally increase in an out-of-domain prediction task that includes larger molecules. The (in)ability to generalize across chemical space is not reflected in this metric. 

5. The paper uses overly promotional language, particularly in the abstract and introduction. Examples include “unprecedented innovation and efficiency”, “professional and transformative research”, “high-precision […] quantum mechanical level”, “meticulously computed”, “comprehensive studies of structure–property relationships”. These sections should be thoroughly revised.

### Questions
1. The paper provides insufficient detail on the fragmentation / recombination strategy to construct the molecule library. What constitutes a fragment? How were the fragments generated and selected? 
2. Similarly, the rationales and workflows of conformation generation and selection of “stable” conformations should be explained in more detail (at minimum, in the Appendix). 
3. A notable portion of the paper blend descriptions of the dataset and comparisons to existing datasets, with overlapping content in section 2.2 and section 3. A clearer separation of these aspects would improve clarity and aid reader comprehension. 
4. A better distinction between *molecules* and *conformations* would be beneficial for readability of the paper, and for understanding the structure of the dataset. E.g. the analyses in Fig. 5a and Fig. 6 would be more informative on the molecule level (rather than on the conformation level). A plot of average conformation number against heavy atom count, for example, would help contextualize the dataset structure. Similarly, plots of further simple-to-compute topological molecular descriptors (molecular weight, ring counts, …) would add further information. 
5. The purpose of Fig. 5b, showing energy as a function of dihedral angle without specifying the molecule(s) is unclear. From my perspective, this plot, as it stands, is elementary, and provides no insights into the dataset. 
6. An estimate of the computational resources required for dataset generation (in terms of CPU hours), would provide readers with a realistic perspective on the cost, and feasibility of expanding such datasets.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
QO2Mol, an open-source quantum chemistry dataset, is presented for organic molecular science research. It contains 120,000 molecules and 20 million conformers (10 elements, >40 heavy atoms) with potential energy and forces calculated at the B3LYP/def2-SVP level. Derived from ChEMBL fragments, the dataset is divided into three subsets: A (20 million conformations, >110,000 molecules, equilibrium and near-equilibrium), B (2.4 million conformers, ~1,400 drug molecules, numerous heavy atoms), and C (dissimilar molecules). Benchmark results for potential energy prediction using various models are included. The dataset and code are available on GitHub.

### Strengths
This paper demonstrates several strengths. It addresses a recognized need for a large and diverse quantum chemistry dataset for training and evaluating machine learning models in molecular sciences. The inclusion of benchmark results using established models provides a baseline for future research. The open-source availability of the dataset and code further encourages community engagement and broader application.

However, the paper's strong focus on quantum chemistry methodologies and detailed discussions of molecular properties, conformations, and potential energy surfaces might be more suitable for a computational chemistry or materials science audience. While leveraging machine learning for benchmarking, the core contribution and primary impact lie within the domain of chemistry, potentially making it less central to the interests of a machine learning conference focused on algorithmic advancements or theoretical foundations of AI.

### Weaknesses
This manuscript and the accompanying database should be submitted to a chemistry journal. The work here is predominantly chemistry focused, primarily quantum physical chemistry. Details of DFT calcuations, the construction of dataset, and the utility of the dataset requires peer-review from experts in quantum chemistry. To highlight this, ANI-1 and QM9 are both published in Scientific Data. PCQM4v2 is published in NeurIPS, however the work is part of a larger body of benchmarks for OGB-LSC, and the DFT datasets were curated from a peer-reviewed chemistry work by Nakata et al. (2017) in Journal of Chemical Information and Modeling. 

The paper provides a lot of details about how the dataset was assembled, but it is unlikely that the anonymous reviewers of this venue would be able to give proper feedback on this; it would be irresponsible to publish this work without proper peer-review. Only the final section 5.3, with Table 2, are there relevant machine learning benchmarking results on some deep models on the dataset. There is also a lot of importance placed on the size of the dataset (Figures 1, 2, 3), but it is not clear whether the additional conformations/fragmentation methods are physically sound or useful for chemical modelling.

Additional issues:
- Table 2 should have caption before the table.
- Figure 6 caption should be below the figure.
- There are gramatical errors that need to be addressed: for example, line 359, "The main subset, referred to as subset A, encompassing the most extensive conformation data, containing 20 million conformations from more than 110,000 molecules."
- A double period in line 350.

Additionally there are other works, not mentioned, that have explored something similar; with conformations, higher precision calculations, property annotations etc. (also published in chemistry related venues): GEOM set by Axelrod et al. (2022), QCDGE by Zhu et al. (2024), and Nabla2DFT and NablaDFT by Khrabrov et al. (2024), VQM24 by Khan et al. (2024).

### Questions
For example, Figure 2, the precision levels seem to be some linear scale of various basis sets. While I am not an expert in computation chemistry, I believe this figure paints a deceptively simple measure of precision in DFT. Are the choice of level of theory and basis sets not dependent on the system being looked at; and are they appropriately chosen? Are there any dispersion corrections (as done in GEOM, and QCDGE)? Is the choice of DFT level of theory appropriate for properties that are used as targets (ie. entropy, enthalpy, heat capacity, etc.)? 

Is the fragmentation method deployed to generate the structures common used? There are no details given on the fragmentation algorithm used. The authors claim that this approach would give "a diverse range of significant and complex chemical space," with no evidence of this.

The conformation generation through cheminformatics software like RDKit seems low-precision, given the computational cost required for DFT at B3LYP/def2-SVP. GEOM for example uses CREST, which is based on semi-empirical tight-binding model. Is RDKit sufficient? What are convergence criteria typically used for optimization process of these structures?

Note that most of these questions would likely not require an explanation if reviewed by an expert in quantum computational chemistry.

### Soundness
2

### Presentation
1

### Contribution
1
