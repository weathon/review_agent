# bio2token: all-atom tokenization of any biomolecular structure with mamba

- Decision: Reject
- Scores: 8, 5, 5, 5, 6

## Abstract
Efficient encoding and representation of large 3D molecular structures with high fidelity is critical for biomolecular design applications. Despite this, many representation learning approaches restrict themselves to modeling smaller systems or use coarse-grained approximations of the systems, for example modeling proteins at the resolution of amino acid residues rather than at the level of individual atoms. To address this, we develop quantized auto-encoders that learn atom-level tokenizations of complete proteins, RNA and small molecule structures with reconstruction accuracies well below 1 Angstrom. We demonstrate that a simple Mamba state space model architecture is efficient compared to an SE(3)-invariant IPA architecture, reaches competitive accuracies and can scale to systems with almost 100,000 atoms. The learned structure tokens of bio2token may serve as the input for all-atom generative models in the future.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper describes mamba-based approach to representation and reconstruction of atomic configurations of small molecules, proteins, and RNA. The paper reports ability of the presented appoach to scale to 10^5 atoms while reaching competitive accuracies.

### Strengths
The paper describes a meaningful effort in modeling mechanistically-motivated molecular structure, such as Cartesian coordinates of the atoms in covalent systems, instead of various serialized descriptors. This is a strong point towards originality and significance.

The paper reports application of the model to biochemically relevant molecular systems, including small molecules, proteins, and RNA, described in publicly available datasets. Ability to treat configurations of covalent systems up to 10^5 atoms is significant.

Adaptation of mamba to the problem enables efficiency of all-atom modeling, including small size of the model, fast inference, and attractive scaling with molecular size. The referee is aware of several ongoing mamba-based developments for computational chemistry, the reported one is definitely original and useful.

### Weaknesses
While paper explicitly claims ability to "reach competitive accuracies" there are no mentions of accuracies to compare with. In other words, the model performance does not seem to be evaluated against any alternatives.

Atomic configurations of molecules are a staple of computational chemistry. There is a literature on AI modeling of Cartesian coordinates of molecules in computational chemistry. It would be fair for the authors to cite such contributions, even those limited to small molecules.

There's a statement and evidence of scaling to large system size, but no scaling curves reported.

### Questions
What is the nature of scaling with the molecule size? Is improved accessibility of large systems a consequence of improved scaling or improved prefactor?

What are existing approaches to representation and reconstruction of atomic configurations in computational chemistry?  

What are performance metrics related to the chemical correctness of the reconstructed configurations? It is straightforward to create bonding patterns based on interatomic distances of reconstructed configurations and compare them with the ground truth patterns.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
It provides an all-atom level VQVAE for different modalities with Mamba, but the writing of this paper should be improved and additional experiments are needed for the evaluation.

### Strengths
- It provides an all-atom level VQVAE for different modalities, i.e., protein, RNA and ligands, which is the first work in this field.
- With mamba architecture, it is more efficient than transformer-based methods and maintains similar to superior performance.

### Weaknesses
- It is more like a technical report than a research paper. Modeling biomolecules in an all-atom resolution typically requires many complex operations such as the **broadcasting** (token index to atom) and **aggregation** (atom index to token) in AlphaFold3. Additionally, the model architectures will become complex either, such as **AtomAttentionEnconder** and **AtomAttentionDeconder** in AlphaFold3. However, this paper lacks details on these components and instead frequently mentions “Mamba” without substantive explanation.
- For evaluation of structure reconstruction, **pLDDT** (or, preferably, **pAE**) should be set as output heads .
- I question whether a codebook size of 4096 is sufficient to capture an all-atom vocabulary effectively. A comparison across different codebook sizes should be included.
- The low-quality reconstruction samples should also be visualized to help learn the issues of tokenizer.
- For complex structure reconstruction, multi-chain permutation alignment (as in AF2-Multimer) is usually necessary. However, this paper does not include details on that.
- It is impressive that the model achieves comparable results to CASP14/15 benchmarks with ESM3, despite being trained on only 18k CATH 4.2 dataset entries, as opposed to larger datasets like PDB, AFDB, or ESMAtlas used in ESM3. **Additional training details must be provided** to clarify how this performance was achieved.

### Questions
- **Efficiency IPA Transformer versus Mamba** section is overly brief. No tables or figures are provided.
- All-domain tokenizing Table could be moved to the main text.
- No code is available.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a novel approach for efficient encoding and representation of large 3D molecular structures at the all-atom level. The authors develop quantized auto-encoders that learn atom-level tokenizations of complete proteins, RNA, and small molecule structures with high reconstruction accuracy. The Mamba state space model architecture employed is shown to be computationally efficient, requiring less training data, parameters, and compute compared to transformer-based methods, while maintaining similar or superior performance. The authors demonstrate the ability to scale to biomolecular systems with up to 95,000 atoms, which is beyond the capabilities of existing transformer-based models. The learned structure tokens from this approach, called bio2token, may serve as the input for future all-atom language models.

### Strengths
- For the first time, Mamba is used to construct an all-atom discrete representation of multiple biological structures.
- The generalization capability of bio2token for complexes is also quite impressive.

### Weaknesses
- The definition of biological structures in the article only involves coordinate point clouds, which is incomplete; information on atomic types is also crucial.
- The paper only discusses discrete tokenization without demonstrating the advantages of this tokenization through downstream applications. Moreover, bio2token does not reduce the number of atoms, raising doubts about whether it can truly support language model development in the relevant field.

### Questions
- The excellent generalization ability of bio2token for complexes might simply be due to its replication of the input coordinates.
- There are many other types of compounds that were not covered in this paper, and furthermore, no corresponding quantitative analysis was included.
- There is also no quantitative analysis in the discussion regarding computational efficiency.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a novel architecture for training quantized auto-encoder of 3D molecular structures. It leverages the mamba architecture and an all atom aligned MSE loss. 
The authors perform several trainings of the framework from several datasets, of diverse nature, such as RNA – small molecules – proteins the proposed method achieved competitive reconstruction accuracy compared to established baselines.

### Strengths
The authors proposition is straightforward and fairly well motivated for leveraging the mamba architecture. 
The proposed models seem effective in the sense that the optimized quantity (i.e.  the all atom RMSE) is low on their test sets. Moreover, the authors proposition seem data efficient notably in the protein setting compared to competitors.

They showcase their proposition in a variety of settings. 
Interestingly they show that when training on all data sources at once they do not observe a significant boost or decrease in performances. This seem to illustrate that there is only a little transfer between tasks / datasets given the authors design choices.
The authors also provide an interesting discussion on the limitation of their work.

### Weaknesses
**Architecture**

While the authors develop a paragraph dedicated to mamba based SSM, (and since the authors’ proposition heavily relies on the mamba architecture) I would have enjoyed a thorougher description of the design choices, and hyperparameter selection. Indeed, since to the best of my knowledge it is the first work leveraging a SSM deep architecture for 3D structure encoding, it is important for practitioners to understand the rationale behind the design choices.

**Performance comparison**

The authors implement an "all-to-all" atom autoencoding approach, assigning a unique integer code to each atom in a point cloud of NN atoms. This strategy substantially increases the information density compared to other models like ESM-3 or InstaDeep’s quantized autoencoder, which encode only a single integer per residue in protein structures. 
While encoding every atom individually this procedure enable (very) fine grain resolution, the authors achieve a much finer level of detail at the expense of a lower compression, therefore I find it difficult to understand the relative advantage of the author’s proposition compared to competitors.

**Invariance**
 To the best of my understanding the authors provide an unprocessed point cloud (centered) suggesting that the rotated point cloud can have a different representation compared to the original one. This remark might require further investigation. Indeed, it could be interesting to understand whether the learned decoder is a surjection.


**POST REBUTTAL**
 I appreciate the authors' response but have decided to maintain my score. While the work is interesting, I find it borderline in the current state and believe it falls short of ICLR's standards for publication.

### Questions
1- You report all-to-all RMSD for proteins and only TM on C-alpha, can it be computed all atoms ? 

2- When reconstructing proteins how difficult is it to attribute an atom to a residue ? 

3- Invariance: As highlighted in the above paragraph, it would be interesting the see if the tokens / output changes when the input point cloud is rotated since the encoding do not seem to be invariant to rotation ? And also what is the reconstruction error distribution of a molecule given a set of rotation.

4 -  Do you expect to obtain significant better results when scaling your datasets ? For instance moving from CATH to pdb or increasing using AF db ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to train a mamba-based auto-encoder on biomolecular structures to allow for accurate tokenization (i.e. conversion to discrete tokens). The authors compare training several domain-specific tokenizers vs one shared one, and investigate scalability.

### Strengths
The modelling choices are generally sound and practical, with the choice to go for mamba over transformers well-justified for the domain. Bio2token and other tokenizers proposed in this work appear to be highly scalable, allowing for an all-atom representation to be used for a range of chemical objects.

### Weaknesses
From reading the paper, I am not sure how useful the tokenizer is in itself, and how exactly it enables new models that could build on top of it. Would the main downstream models making use of the pretrained tokenizer be generative or predictive in nature? Is the tokenizer at all useful on its own? Moreover, I am wondering how can we know the models that build on top of bio2token would be useful in the face of compounding errors (i.e. errors stemming from the tokenizer itself adding up with errors of the downstream model)?

=== Update 02/12/2024 ===

During the discussion period the authors have argued that structure tokenizers such as the one proposed in this work are widely used in the field, for example by generative models. While they do not present results showing downstream improvements, they do compare with other tokenizers which were already evaluated in downstream tasks, so it is reasonable to expect the improved tokenizer would also lead to improvements downstream, even though one can't be sure. The authors also included several new results and expanded the discussion. To reflect this, I raise my score. However, I leave my confidence as low, as it's not clear to me how confident we can be that the improved tokenization leads to improvements in downstream tasks without testing it directly.

### Questions
See the "Weaknesses" section above for specific questions.

### Soundness
3

### Presentation
3

### Contribution
3
