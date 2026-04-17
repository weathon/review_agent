# Representing local protein environments with machine learning force fields

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The local structure of a protein strongly impacts its function and interactions with other molecules. Representing local biomolecular environments remains a key challenge while applying machine learning approaches over protein structures. The structural and chemical variability of these environments makes them challenging to model, and performing representation learning on these objects remains largely under-explored.  In this work, we propose representations for local protein environments that leverage intermediate features from machine learning force fields (MLFFs). We extensively benchmark state-of-the-art MLFFs—comparing their performance across latent spaces and downstream tasks—and show that their embeddings capture local structural (e.g., secondary motifs) and chemical features (e.g., amino acid identity and protonation state), organizing protein environments into a structured manifold. We show that these representations enable zero-shot generalization and transfer across diverse downstream tasks. As a case study, we build a physics-informed, uncertainty-aware chemical shift predictor that achieves state-of-the-art accuracy in biomolecular NMR spectroscopy. Our results establish MLFFs as general-purpose, reusable representation learners for protein modeling, opening new directions in representation learning for structured physical systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper explores the use of MLFFs as general-purpose representation learners for local protein environments. Instead of relying on sequence-based or handcrafted descriptors, the authors repurpose latent embeddings from pretrained MLFFs (AIMNet, MACE, OrbNet, Egret) as compact, physics-grounded descriptors for atomic neighborhoods in proteins. The study benchmarks MLFF embeddings on diverse downstream tasks—secondary structure and amino acid classification, pKa prediction, and NMR chemical shift regression—and shows that these representations outperform or rival specialized baselines such as PropKa and pKa-ANI. Overall, the paper establishes MLFFs as reusable, physics-informed foundation models for structural biology.

### Strengths
1. The central insight, repurposing MLFF embeddings as general-purpose, transferable protein environment representations, is both novel and timely. Most prior MLFF applications focus on energy or force prediction for small molecules; extending them to protein representation learning is original and valuable. The paper effectively bridges quantum-chemistry-based potentials and protein machine learning.

2. The experimental setup is comprehensive. The authors curate 165 k environments from 1048 proteins and evaluate four MLFF families on four biologically relevant tasks. Comparisons include classical and ML baselines (PropKa, pKa-ANI, UCBShift2-X). Results demonstrate meaningful improvements in both accuracy and interpretability. The inclusion of uncertainty quantification and physical consistency tests (e.g., ring-current effects) adds rigor.

3. The methodology that extracting embeddings from pretrained MLFFs and mapping them to canonical residue-centered environments is sound and clearly motivated. Statistical reporting (mean absolute errors, standard deviations) is adequate, and all experiments appear well-controlled.

4. The paper is well written and pedagogically organized. The introduction clearly motivates the challenge of representing local protein environments; figures (e.g., Fig. 1 and 2) effectively illustrate how embeddings are constructed and used. Terminology (canonical environment, focus residue, MLFF feature extraction) is consistent and accessible even to readers outside computational chemistry.

5. This work could substantially influence both computational biology and machine-learning communities by providing a physics-consistent alternative to sequence-only protein language models. MLFF embeddings encode quantum-derived information unavailable in existing representations and show transferability to tasks requiring local chemical precision. The idea of using pretrained MLFFs as “foundation models for atoms” could be broadly significant.

### Weaknesses
1. The study restricts environments to 5 Å radius regions; while suitable for local chemistry, it omits long-range electrostatic or conformational effects. For tasks like folding or binding prediction, this locality may be insufficient. A discussion or experiment extending to multi-scale contexts would strengthen generality claims.

2. MLFFs such as MACE and OrbNet require quantum-level pretraining on millions of molecules. Although embeddings are reused, the computational barrier to obtaining them limits accessibility compared to pretrained sequence models (e.g., ESM, ProtT5). The paper could better address scalability and efficiency trade-offs.

3. Qualitative case studies (e.g., helix → strand unfolding) are insightful but anecdotal. Quantitative metrics (e.g., correlation between embedding distances and RMSD/chemical similarity) would solidify interpretability claims.

4. While the authors compare to physics-based predictors, they do not benchmark against modern structure-aware geometric encoders (e.g., GVP-GNN, ProteinNeRF, FrameFold). Including such baselines would clarify whether MLFF embeddings provide advantages beyond standard geometric message-passing.

### Questions
1. Could the same MLFF-based representation transfer to nucleic acids or protein–ligand complexes, where local environments include non-canonical atoms and charges?

2. MLFFs have multiple internal layers encoding different orders of interaction (0th, 1st, 2nd). Did the authors investigate which layer yields the most informative embeddings for downstream tasks?

3. Since MLFF embeddings originate from networks with different scales and symmetries, how are they aligned or normalized across model families?

4. The authors mention uncertainty-aware predictions. Is uncertainty derived from ensemble variance, likelihood width, or another calibration technique?

5. Could MLFF embeddings serve as complementary features to pretrained protein language models, bridging physics-based and sequence-based representations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Local protein environments contain highly relevant chemistry that affects its function and interactions. This paper benchmarks existing machine learning force field (MLFF) methods on their ability to understand local protein structures. They evaluate a suite of MLFFs on their ability to predict the protonation state, secondary structure, and amino acid types.

### Strengths
This paper rigorously evaluates multiple popular MLFF methods on across benchmarks and connect them to some protein structure related tasks.
They perform an interesting analysis on the chemical shift prediction.

### Weaknesses
The evaluation benchmarks such as secondary structure, amino acid type prediction are a bit straightforward.

While MLFF naturally learn the physics of local structural environments, there are other machine learning based approaches that reason over the local structure and are suitable to predict protonation state, secondary structure, and amino acid types. This work does not benchmark MLFF versus these methods on representing local structure.

[1] Simulating 500 million years of evolution with a language model. Hayes et al.
[2] 3D deep convolutional neural networks for amino acid environment similarity analysis. Torng et al.
[3] Distilling Structural Representations into Protein Sequence Models. Ouyang-Zhang et al.

### Questions
Are there any other interesting local protein structure properties to evaluate on?

How well do non-FF based local structure models, such as ESM3, perform on these benchmarks?

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
3

### Summary
This paper proposes repurposing intermediate embeddings from pre-trained Machine Learning Force Fields (MLFFs) as "physics-grounded" feature representations for local protein environments. The authors benchmark embeddings from MACE, AIMNet, and others, claiming state-of-the-art (SOTA) results in $pK_{a}$ prediction (reportedly outperforming pKa-ANI) and NMR chemical shift prediction (reportedly outperforming UCBShift2-X). The method's physical realism is validated via case studies, such as aromatic ring current effects.

### Strengths
1. Repurposing MLFFs as "foundation models" for structural biology is in nnovative and valuable.   
2. The design of the validation experiments (e.g., ring current effect , helix unfolding ) is a commendable standard for physical realism.   
3.The paper effectively demonstrates the general-purpose nature of the embeddings for zero-shot clustering and generative guidance.

### Weaknesses
1.The paper's central claims rest on comparisons against pKa-ANI and UCBShift2-X that appear to be factually incorrect or based on flawed implementations.

2.The $pK_{a}$ evaluation is missing the entire 2024/2025 SOTA, invalidating its performance claims.

3.The paper compares its structural embeddings against sequence (ESM) embeddings. It critically fails to benchmark against the most obvious and relevant competitors: other structural embeddings, namely those from the AlphaFold2 or ESMFold structure modules.

### Questions
see weakness

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes to leverage pretrained ML interatomic potential embeddings as canonical descriptors of local protein environments. Embeddings are extracted from a residue-centered neighborhood. Various applications including pKa prediction of titrable residues and protein NMR chemical shift prediction are explored, with MLFF-based embeddings show good representative power for downstream tasks.

### Strengths
- Repurposing latent features of MLFFs as canonical protein descriptors is a timely, well-motivated idea that links quantum-level atomistic modeling with biomolecular representation learning.
- The paper covers a solid range of downstream tasks tied to experimental observables and provides thorough analysis of the physical plausibility of its predictions.

### Weaknesses
- Dataset and baselines: For the pKa and NMR shift tasks, the baselines are evaluated in conditions that differ from their intended use, whereas the MLFF-feature models introduced here are trained directly for the target objective. For instance, the pKa baselines are designed to predict experimental values, while the proposed methods are trained to reproduce a cheaper computational reference. This creates a benchmark mismatch, since the baselines are not optimized for the reference chosen in this work. Because the core question is whether MLFF features are useful, a more appropriate primary comparison would be a standard GCN with simple learned embeddings rather than prior task-specific baselines. This issue is compounded by the use of AFDB structures as inputs.
- Experimental design: Some experiments are not very indicative of how the embeddings would be used in practice. Inferring amino acid type or secondary structure from full atomic coordinates (Section 4) is a trivial task under those inputs, and the distribution-shift analysis in Section 5 (Fig. 2) mainly shows that embeddings from energy-relaxed structures are similar, which is expected since both MLFFs and classical force fields approximate the same physical energy landscape. Several experiments currently placed in the appendix appear more compelling and better aligned with the paper’s motivation. I recommend reorganizing the manuscript so that the most informative use cases are in the main text, especially given that all experiments are listed as contributions but not all are presented in the main body.

### Questions
- How sensitive are results to the MLFF layer chosen for embeddings?
- MACE is an architecture, but there are several variants according to the dataset it is trained on. It seems like the MACE-OFF23 is used in this work, could authors elaborate?
- For AF3-guided structure selection in Fig A14, could authors report the optimization statistics similarly to Fig A13? (What fraction of optimization result in the target structure, or whether it always lead to the target structure)

### Soundness
3

### Presentation
3

### Contribution
3
