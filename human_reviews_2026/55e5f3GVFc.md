# Protein Structure Tokenization via Geometric Byte Pair Encoding

- Avg Score: 7.50
- Decision: Accept (Poster)
- Scores: 10, 8, 6, 6

## Abstract
Protein structure is central to biological function, and enabling multimodal protein models requires joint reasoning over sequence, structure, and function. A key barrier is the lack of principled protein structure tokenizers (PSTs): existing approaches fix token size or rely on continuous vector codebooks, limiting interpretability, multi-scale control, and transfer across architectures. We introduce GeoBPE, a geometry-grounded PST that transforms continuous, noisy, multi-scale backbone conformations into discrete ``sentences'' of geometry while enforcing global constraints. Analogous to byte-pair encoding, GeoBPE generates a hierarchical vocabulary of geometric primitives by iteratively (i) clustering Geo-Pair occurrences with k-medoids to yield a resolution-controllable vocabulary; (ii) quantizing each Geo-Pair to its closest medoid prototype; and (iii) reducing drift through differentiable inverse kinematics that optimizes boundary glue angles under an $\mathrm{SE}(3)$ end-frame loss. GeoBPE offers compression ($>$10× reduction in bits-per-residue at similar distortion rate), data efficiency ($>$10× less training data), and generalization (maintains test/train distortion ratio of $1.0-1.1$). It is architecture-agnostic: (a) its hierarchical vocabulary provides a strong inductive bias for coarsening residue-level embeddings from large PLMs into motif- and protein-level representations, consistently outperforming leading PSTs across $12$ tasks and $24$ test splits; (b) paired with a transformer, GeoBPE supports unconditional backbone generation via language modeling; and (c) tokens align with CATH functional families and support expert-interpretable case studies, offering functional meaning absent in prior PSTs.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
The authors present GeoBPE, a protein structure tokenizer which produces hierarchical tokenizations of proteins using a byte-pair encoding-like technique. In brief, residue-level backbone geometric features are clustered into a vocabulary, and then iteratively merged to reduce the size of the vocabulary and make it less sensitive to noise. This induces distortions in the resulting geometry of a protein, which they resolve by making corrections at each stage of the tokenization. The approach is remarkably effective, resulting in increased compression and reconstruction accuracy over strong baselines. ProToken and GeoBPE form a Pareto front, trading off reconstruction accuracy for compression, with GeoBPE performing roughly 10x better on BPR (compression) than ProToken.

### Strengths
Originality: I think the authors have come up with a creative and quite distinct solution to the problem of structure tokenization--while many others have tried to improve on the VQ-VAE idea, they have not broken out of that fundamental paradigm. GeoBPE is entirely different and at the same time simpler and more intuitive. 

Quality: The analysis is thorough, covering structure reconstruction, codebook usage, and several downstream tasks of high biological relevance. 

Significance: GeoBPE substantially outperforms most other structure tokenizers in nearly all evaluations, and provides clear trade-offs between compression and accuracy. I imagine that it will be of great utility to researchers modeling proteins, and hopefully will find use in a protein LM soon.

### Weaknesses
Clarity: The writing in section 3 could use a lot of improvement. I found it difficult to follow, and I'm still not entirely sure I know what all symbols mean. Many terms are only properly defined _after_ they've been used multiple times. The notation section is clear, but takes a lot of space that might be better given to (clearer) descriptions of the method itself.

It would be very helpful to visualize the hierarchy of structure tokens/motifs. I imagine, for instance, that there should be a large group of loop-like structures, with a smaller number of tokens devoted to common structures like alpha-helices and different kinds of beta sheets. Given the local nature of the representation, I would expect high level clusters corresponding to the 8 basic secondary structure elements.

### Questions
In this emerging field, there is some debate over what level and type of structural information each token should have. That is, each token can represent not just information internal to the residues represented, but more global information about how the residue is positioned in the protein. GeoBPE is essentially backbone-local--it does not use information about nearest spatial neighbors at all. By contrast ESM3's tokenizer does contain substantial information about residues which can be very distant in the linear sequence of the protein, yet GeoBPE does better on global reconstruction tasks. This is perhaps surprising and I wonder if the authors could comment on it.

Generally the peptide backbone bond lengths are very close to fixed (in fact there are usually restraints on these when solving structures by X-ray diffraction, NMR, or Cryo-EM). I wonder whether these are actually needed in GeoBPE; do the authors find clusters of motifs where these are very different, or are the bond lengths similar between all motifs? My guess is that the clusters differ primarily in their backbone dihedral angles

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper “Protein Structure Tokenization via Geometric Byte Pair Encoding (GEOBPE)” proposes a hierarchical, geometry-aware tokenizer that discretizes protein backbones (N, Cα, C) into interpretable structural “words.” Inspired by text BPE, it merges frequent geometric motif pairs using clustering and inverse-kinematics “glue” corrections to preserve structural consistency. GEOBPE achieves superior compression–distortion tradeoffs, strong out-of-distribution generalization, and improved downstream task performance over VQ-VAE-based baselines. Its tokens align with functional protein motifs (e.g., CATH domains) and enable structure language modeling and backbone generation, offering a scalable and interpretable discrete representation of 3D protein geometry.

### Strengths
- The GEOBPE idea is elegant: motif clustering + quantization + glue angle correction giving both local motif reuse and global structural consistency, and iteratively building a vocabulary. That hierarchical tree/merge structure is meaningfully analogous to BPE and advantageous in representing protein structure at multiple resolutions.
- On compression vs distortion, GEOBPE traces a favorable Pareto front.
- Out-of‐distribution generalization appears very good compared to VQ-VAE
- Downstream tasks benefit from the tokenization
- The tokens learned are not just “latent vectors” but actual observed motifs (medoids) with real structure, and they show alignment with functional families (CATH) and interpretability in case studies.
- The authors highlight that they trained on much less data (~7% or even ~1% of typical size) and yet got strong results (Sec 5).

### Weaknesses
- The method involves many steps (clustering motifs, quantizing, global inverse kinematics to correct glue angles, iterative merging). It may be computationally heavy (they discuss complexity in Appendix H). The practicality at very large scale (hundreds of millions of structures) may still be challenging, and training time/compute cost details may be less emphasized.
- The method depends on choices: how to cluster, how many medoids (K), when to merge, how to set glue‐IK weights, how to quantize, etc. These may introduce sensitivity and hyperparameter tuning cost. The robustness across many protein fold types or extremely large or unusual proteins may vary.
- The method tokenizes only backbone geometry (ignores side-chains, ligands, post-translational modifications). For many functional tasks side-chains or ligand contacts matter a lot.

### Questions
- What is the wall-clock/time/compute cost to build the vocabulary (i.e., cluster motifs, compute glue corrections) for e.g., the full PDB (hundreds of thousands of chains)?
- How sensitive is the performance (compression vs distortion) to choices of K (number of medoids), merge iterations, and glue‐IK weights? Is there a guideline for selecting these hyperparameters for a new domain (e.g., membrane proteins or antibody loops)?
- The work focuses on backbone only. Do the authors envision an extension to include side-chain geometry (Cβ, other atoms) or ligand binding sites?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GEOBPE, a geometric analog of byte-pair encoding (BPE) for protein backbones. Instead of tokenizing amino acid sequences, GEOBPE tokenizes 3D structures into discrete motifs, building a hierarchical vocabulary through iterative clustering and merging of “Geo-Pairs.” And it enforces geometric consistency using rigid-body optimization during BPE process. This method does not reply on neural networks to "automatically" learn the discretized tokens like VQ-VAE-like protein structure tokenizers.

GEOBPE supports downstream applications like function prediction and shows strong interpretability by aligning the tokens with CATH functional families. The tokenizer is also used to train generation tasks, though it's not the focus on this paper's experiment section.

### Strengths
- **Novel conceptual framing**: this paper continues an "old fashion" way to heuristically tokenize structures. GEOBPE extends BPE from linguistic to protein structure data, introducing a hierarchical, interpretable tokenizer for protein structures. This is process is far complicated than applying BPE on protein sequence data. And the authors seem to design a way to solve it.

- **Interpretability**: A highlight difference for GEOBPE with popular VQ-VAE-like structure tokenizers is the interpretability: (1) the vocabulary construction process is interpretable, unlike VQ-VAE relies on automatically learning from data; (2) the hierarchical vocabulary offers intuitive geometric primitives that align with known structural motifs in CATH.

- **Empirical strength**: As shown in Table 1, tokens from GEOBPE contain rich functional information, and performs the best on related benchmarking tasks.

### Weaknesses
- **Per-residue task ambiguity**: Since tokenization is no longer residue-level, how can we make GEOBPE's tokens aligned for per-residue benchmarks? This is a crucial design and evaluation gap.

- **Severe Presentation issues**: Excessive information density makes the paper difficult to parse; algorithmic pseudocode dominates over conceptual clarity. And referring readers to appendix for very detailed algorithms in appendix and replying on these algorithms to inform what is happening inside the method is a very bad way to deliver ideas. The paper would benefit from improved figure explanations and appendix referencing.

### Questions
1. Will all the code be released?

2. Line 150 has notation conflicts with Line 161, \phi_{i+1} should be \phi_{i}?

3. Even though GEOBPE can trained with only 1% PDBs to get good performance. The algorithm complexity seems to be high and prevents GEOBPE to scale further to large scale datasets (current training data is around 48k proteins). Could you please analyze the complexity of your algorithm and the empirical time used for GEOBPE?

4. Suggestion to modify Figure 2: based on my understanding, the top-right two motifs in “step 1” subfigure stand for “prototypes” and the arrows mean that each motif is mapped to those prototypes. And then this pair of prototype is selected to be merged in BPE process.

5. How does GEOBPE tokenize a new unseen structure once the vocabulary is learned? Is there a decoding or segmentation step described explicitly? Not sure if I missed anything.

6. Concerns for GEOBPE to generate good structures when coupled with LMs. For example, in ProteinAE [1] Table 2, they can perform well on "unconditional protein backbone generation" when compared with competitive.

### Soundness
3

### Presentation
1

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
This paper introduces Geometric Byte-Pair Encoding (GeoBPE), a novel algorithm for protein structure tokenization. The authors identify key limitations in existing protein structure tokenizers, particularly the popular VQ-VAE variant. They argue these methods suffer from fixed token sizes that prevents multi-scale resolution, and produce vector-based tokens that lack the interpretability and hierarchical structure of methods like Byte-Pair Encoding (BPE) in natural language processing. GeoBPE is designed as a geometric analog to BPE. It works by iteratively building a hierarchical vocabulary of structural motifs, which involves: iteratively identifying the most frequent pairs of structural motifs and clustering their occurrences using k-medoids, then replacing these occurrences with a representative prototype from the new vocabulary. This local quantization inevitably introduces geometric errors, and the authors correct this by optimizing the boundary glue angles through differentiable inverse kinematics under SE(3) loss. Experimental results demonstrate that GeoBPE achieves high compression, highly data-efficient, and generalizes well to unseen structures. Furthermore, its tokens are shown to be interpretable, aligning with CATH functional families.

### Strengths
1. GeoBPE produces a hierarchical vocabulary of discrete structural motifs, which is a significant advantage as the resulting tokens have functional meaning. Experimental results show that the tokens align with CATH functional domain annotations and are validated by expert case studies.
2. GeoBPE addresses the fixed token size limitation of VQ-VAE. Its hierarchical BPE-like approach allows it to capture multi-scale features, and its performance can be tuned to trade compression (BPR) for distortion (LDDT/RMSD), which other tokenizers with fixed dimensions cannot do.
3. The paper's explicit handles the geometric drift. The use of differentiable inverse kinematics to optimize glue angles is a principled solution to the problem of maintaining global structural integrity while performing local quantization. Ablation study shows this step is essential, as omitting it causes reconstruction RMSD to jump from 1.66 to 4.39.
4. GeoBPE shows excellent out-of-distribution generalization. It maintains a test/train distortion ratio of 1.16-1.28, whereas the VQ-VAE baseline reportedly degrades by as much as 6.4x on the same test sets.

### Weaknesses
I am not very familiar with structural biology and thus may not be able to provide deep insights into the algorithmic details of GeoBPE. My primary concerns are focused on the experimental validation and its implications:
1. The SSLM experiments (Table 7) indicate that the language model trained on the GeoBPE vocabulary struggles significantly with unconditional generation. This is evidenced by the low scTM (less than 0.3) and designability (less than 5%). Performance at this level is insufficient for most practical applications. This finding leads to serious concerns about using GeoBPE as a foundational component for practical development (e.g., for building multimodal structure and sequence language models, like ESM3).
2. The paper demonstrates GeoBPE's high data efficiency, with appendix results showing that using only 1% of the training data will not introduce performance loss. While this is an advantage, it could also be interpreted as a potential risk. This finding might suggest that the model has a low capacity and hits a performance ceiling almost immediately, making it incapable of capturing the complex, subtle, and diverse structural patterns that can only be learned from large-scale datasets. 

Given these points, I hope the authors will provide a detailed discussion about these concerns.

### Questions
See above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
