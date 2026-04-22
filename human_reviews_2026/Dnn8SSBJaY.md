# Towards Understanding the Shape of Representations in Protein Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
While protein language models (PLMs) are one of the most promising avenues of research for future de novo protein design, the way in which they transform sequences to hidden representations, as well as the information encoded in such representations is yet to be fully understood. Several works have attempted to propose interpretability tools for PLMs, but they have focused on understanding how individual sequences are transformed by such models. Therefore, the way in which PLMs transform the whole space of sequences along with their relations is still unknown. In this work we attempt to understand this transformed space of sequences by identifying protein structure and representation with square-root velocity (SRV) representations and graph filtrations. Both approaches naturally lead to a metric space in which pairs of proteins or protein representations can be compared with each other.

We analyze different types of proteins from the SCOP dataset and show that the Fréchet radius and effective dimension of the SRV shape space follows a non-linear pattern as a function of the layers in ESM2 models of different sizes. Furthermore, we use graph filtrations as a tool to study the context lengths at which models encode the structural features of proteins. We find that PLMs preferentially encode immediate as well as local relations between amino acids, but start to degrade for larger context lengths. The most structurally faithful encoding tends to occur close to, but before the last layer of the models, indicating that training a folding model ontop of these layers might lead to improved folding performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the geometric and structural properties of representations learned by PLMs, specifically focusing on ESM2 models. The authors propose two approaches: (1) a shape-space analysis using Square-Root Velocity (SRV) representations to compare protein structures and PLM embeddings in a rotation-invariant Riemannian space, and (2) a graph filtration method to analyze the context length at which PLMs encode protein structural information. The main findings include: (1) PLM representation shape spaces exhibit an expansion-contraction pattern in effective dimensionality across layers. (2)PLMs encode 3D protein structure most faithfully at very short (2 residues) and slightly longer (8 residues) context lengths. (3)The optimal structural encoding occurs in intermediate layers, not the final layer, suggesting potential improvements for folding models. The authors argue that their approach provides a novel way to understand how PLMs transform the space of protein sequences and encode structural information without explicit supervision.

### Strengths
- The paper introduces a novel application of shape analysis (SRV) and graph filtrations to study PLM representations. This is a fresh perspective compared to existing interpretability methods that focus on individual sequences.
- The SRV and graph filtration approaches are well-motivated and technically sound, providing a principled way to compare protein structures and embeddings.
- The expansion-contraction pattern in effective dimension and the bimodal structural encoding at specific context lengths are intriguing, which facilitates understanding of the context lengths at which models encode the structural features of proteins.
- The findings have practical implications for improving protein folding models and protein design, e.g., by selecting intermediate layers for downstream tasks.
- The paper is generally well-written, with clear motivations and a logical flow. The figures help illustrate the methods and results effectively.

### Weaknesses
- The study is restricted to ESM2 models and eight classes of the SCOP dataset. It would be beneficial to include other PLMs and datasets to assess generalizability.
- The paper lacks comparisons with simpler baselines (e.g., random embeddings, non-PLM representations) to better contextualize the reported metrics (e.g., Fréchet radius, effective dimension).
- While the graph filtration method is innovative, the interpretation of the “graph filtration moment” is somewhat abstract. More intuition or a qualitative analysis (e.g., visualizing which residues are captured at different context lengths) would strengthen the claims.
- The paper implies that encoding protein structure emerges as an important intermediate processing step for unmasking, but no causal experiments are provided to support this.
- The impact of interpolation methods (quadratic splines) and SRV parameterization choices is not explored.  
- No statistical tests or confidence intervals are reported for the Fréchet radius or effective dimension estimates.
- No implementation details or hyperparameters are provided in the current draft. While the ESM2 model and SCOP dataset are publicly available, releasing the code of this work would greatly improve reproducibility. Although some scripts appear to be included in the supplementary .zip file, it is recommended that the authors formally release a public code repository link in the paper.

### Questions
- How sensitive are the SRV and graph filtration results to the choice of interpolation method (quadratic splines) and the number of interpolation points?
- Could the authors provide a qualitative analysis or visualization of the structural features captured at the 2- and 8-residue context lengths, and discuss whether the observed bimodal context-length sensitivity is associated with specific secondary structure motifs?
- Have the authors experimented with using the identified optimal layers to initialize a folding model, did it improve performance over using the final layer?
- How do the results generalize to other PLM architectures or training regimes (e.g., models trained with different objectives or data)?
- The effective dimension is computed via tangent PCA. How does this compare to intrinsic dimension estimators (e.g., MLE, TWO-NN) applied directly to the PLM embeddings?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors compare spaces induced by protein language models to those induced by protein structures. By applying various representation schemes and dimensionality analysis techniques they derive some insights about the way PLM embeddings are organized. In particular, we learn that protein structure is mirrored in PLM embeddings at highly localized neighbourhoods, and that the effective shape space of embeddings is quite limited. These are highly valuable insights for future PLM model development.

### Strengths
* This work provides a highly unique and novel view on PLM work and generates valuable insights. The framework for understanding PLMs is also novel and will be a valuable asset for the field.
* Modeling the 'shape' of embeddings is a very interesting proposition. I am also very surprised by the finding that embedding shape can mirror protein shape at all.
* The paper is very well-written and flows nicely from the initial premise and questions to be explored (e.g. "If structure determines function, then it is reasonable to assume that similar structure determines similar function. Furthermore, if protein representations in PLMs characterize structural, evolutionary and functional features, then one might expect that similar representations share such features. ")

### Weaknesses
* There is a lack of simpler methods than the one proposed. The authors begin by asking whether PLM embeddings for structurally similar proteins would be similar themselves. Of course this can be tackled by mapping the embeddings to shape spaces and comparing shapes, but one could also try to directly measure correlation for example between TM scores and PLM embeddings.

### Questions
* How sensitive is the analysis to the curve fitting degree? How do you measure the quality of a curve fitting for a given protein? As far as I know, this representation approach is not well-established for proteins so its soundness should be explored further.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to unravel the geometry of hidden representations in Protein Language Models (PLMs), a critical gap in understanding how PLMs encode protein structure/function for de novo design. Using two novel analytical frameworks—square-root velocity (SRV) shape spaces and graph filtrations—the authors analyze multiple sizes of ESM2 (a leading PLM) on 1098 proteins from the SCOP dataset (spanning 8 protein classes). Key findings include: (1) PLM representation shape spaces exhibit non-linear patterns in Karcher mean and effective dimension across layers, with larger models showing "dimension expansion" in early layers and "contraction" in later layers; (2) PLMs optimally encode protein structure at short context lengths (~2 amino acids) and moderately short lengths (~8 amino acids), but degrade at larger contexts; (3) structural encoding is strongest in layers just before the last layer, suggesting potential improvements for folding models.

### Strengths
1. Adapts shape analysis (SRV) and topological data analysis (graph filtrations)—previously used for 3D protein structure—to PLM representation spaces, enabling rigorous metric comparisons between protein structures and their PLM embeddings.
2 Identifies a universal "expansion-contraction" pattern in effective dimension across ESM2 layers, where early layers explore diverse shape variations and later layers converge to a low-dimensional subspace of biologically relevant shapes.
3. Quantifies that PLMs prioritize local residue relationships (contexts of 2 or 8 amino acids) over global ones, explaining why PLMs implicitly encode structural features despite being trained only on masked sequence prediction.
4. Highlights that the last layer of PLMs is not optimal for structural encoding; using pre-last layers could improve downstream folding models (e.g., ESMFold).

### Weaknesses
1. The study only uses ESM2. It is unclear if the "expansion-contraction" pattern or context length bimodality (2, 8) applies to other popular PLMs (e.g., ProtTrans, AlphaFold’s embedding layers) or PLMs trained on different datasets (e.g., UniProt vs. Pfam).
2 The authors use Euclidean distance for graph filtrations but do not test if alternative metrics (e.g., cosine similarity, Manhattan distance) alter the bimodal context pattern. Euclidean distance may not be optimal for high-dimensional PLM embeddings, where cosine similarity is often preferred.
3. The paper suggests using pre-last layers to improve folding models but provides no experimental evidence (e.g., comparing ESMFold performance with pre-last vs. last layers). This makes the suggestion speculative. It is interesting to see such findings, and I am also curious whether the pre-last layers can improve other downstream tasks, like GO, EC prediction? 
4. The paper notes that shape space effective dimension is much lower than flattened PLM representations but does not identify which specific shape features (e.g., helix curvature, beta-sheet spacing) drive this low dimensionality.
5. Most of the findings or conclusions of this theory are conforming to cognition.

### Questions
Does sequence length correlate with structural encoding quality?
The authors find PLMs preferentially encode immediate as well as local relations between residues, but start to degrade for larger context lengths, does this mean that PLMs cannot learn global relations?

### Soundness
3

### Presentation
2

### Contribution
3
