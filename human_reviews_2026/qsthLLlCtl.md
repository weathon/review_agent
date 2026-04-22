# PRISM: Enhancing PRotein Inverse Folding through Fine- Grained Retrieval on Structure-Sequence Multimodal Representations

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
Designing protein sequences that fold into a target 3-D structure, termed as the inverse folding problem, is central to protein engineering. However, it remains challenging due to the vast sequence space and the importance of local structural constraints. Existing deep learning approaches achieve strong recovery rates, however, lack explicit mechanisms to reuse fine-grained structure-sequence patterns conserved across natural proteins. 
To mitigate this, we present PRISM a multimodal retrieval-augmented generation framework for inverse folding. PRISM retrieves fine-grained representations of potential motifs from known proteins and integrates them with a hybrid self-cross attention decoder. PRISM is formulated as a latent-variable probabilistic model and implemented with an efficient approximation, combining theoretical grounding with practical scalability. 
Experiments across multiple benchmarks, including CATH-4.2, TS50, TS500, CAMEO 2022, and the PDB date split, demonstrate the fine-grained multimodal retrieval efficacy of PRISM in yielding SoTA perplexity and amino acid recovery, while also improving the foldability metrics (RMSD, TM-score, pLDDT).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the protein inverse folding problem: designing a sequence for a target 3D structure. The authors identify that existing methods lack mechanisms to reuse conserved, fine-grained structure-sequence patterns. They propose PRISM, a retrieval-augmented generation (RAG) framework that retrieves fine-grained "potential motif" representations from a vector database. These retrieved embeddings are integrated with global backbone context using a hybrid self-cross attention (MHSCA) decoder. The method is formulated as a latent-variable probabilistic model. Experiments across five benchmarks show that PRISM achieves new state-of-the-art results in sequence recovery and perplexity, while also improving structural foldability metrics.

### Strengths
The paper demonstrates strong empirical results, consistently outperforming existing methods, including its own base model (AIDO.Protein-IF), across five challenging benchmarks (CATH-4.2, TS50, TS500, CAMEO 2022, PDB date split).

The application of a retrieval-augmented generation (RAG) framework to inverse folding at a fine-grained, residue-level ("potential motif") is a novel contribution. This provides an explicit mechanism to reuse conserved local patterns, which is a limitation in prior work

The paper includes an extensive set of ablations that validate the core design choices, including the impact of the number of retrieved entries, the contribution of the hybrid decoder, the effect of aggregation depth, and the saturation of the vector database. The analysis of the recovery-diversity trade-off is also a valuable addition.

### Weaknesses
Presentation needs improvement. Some tables and figures overlap with text body, e.g., tab 3 and fig 3. Sec. 4.2 is missing. ‘Fig.’ and ‘Figure’ are both used in text body.

Some baselines, like SPDesign and VFN-IFE, that report higher metrics (on CATH), are not compared.

The training objective (Eq. 20) optimizes the parameters of the aggregation module ($\theta_Z$) and a structure encoder ($\theta_B$). However, the joint encoder $\mathcal{G}$ (identified as AIDO.Protein-IF) that produces the query embeddings $\hat{\mathcal{E}}^q$ and populates the database appears to be frozen, as the TopK retrieval step is non-differentiable. This prevents end-to-end training and means the model cannot learn to improve its representations for better retrieval; it can only learn to use the results from a fixed, and potentially suboptimal, retrieval set.

The "PRISM (str. enc.)" ablation (Tables 1, 2, 3) is vaguely described. The paper states this variant replaces the joint encoder with a "purely structure-based encoder (ProteinMPNN-CMLM)" and retrieval operates "only over structural embeddings". This seems to contradict the framework's core "multimodal representation" $\mathcal{E} = \mathcal{G}(B,S)$ (Eq. 5) and the definition of the database $D$. It is unclear how the multimodal vector database is constructed or queried in this "structure-only" setting.

### Questions
Let’s denote the training set as A, the RAG database as B. The paper compares models trained on A and augmented with B, with models trained on A. However, since the information in B is also used, could you compare models trained on A and augmented with B, with models trained on A+B?

Could you explain the independences for Eq. 3?

Is the joint encoder $\mathcal{G}$ (AIDO.Protein-IF) kept frozen during training? If so, have the authors considered methods to make the retrieval step differentiable (e.g., via relaxation) to enable end-to-end fine-tuning of the query encoder?

At inference, an "initial sequence estimate $\tilde{S}^q$" is required to generate the query embedding $\hat{\mathcal{E}}^q$. How sensitive is the model's final performance to the quality of this initial sequence? For instance, what is the drop in AAR if $\tilde{S}^q$ is generated by a weaker baseline like ProteinMPNN instead of AIDO.Protein-IF?

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
3

### Summary
The paper proposes a RAG system for the protein inverse folding task. It uses AIDO.Protein-IF as a joint protein sequence-structure encoder to generate residue-level embeddings for every protein in the knowledge base, which are then used to construct a RAG system. During inference, an initial guess of the protein sequence is first generated from a base estimator, then used to index and retrieve top-k candidates from the RAG. The paper claims SOTA performance of their approach on various benchmarks compared to existing baselines.

### Strengths
* The paper addresses the protein inverse folding problem, which is a core challenge in proteomics and biology. 
* The residue-level RAG system allows flexible reuse of local structure and sequence patterns. 
* The performance of the model is impressive on different benchmarks for its SOTA amino-acid recovery rate, perplexity and foldability.

### Weaknesses
* The model relies heavily on the AIDO.Protein-IF model in both RAG system construction and inference. The not-so-significant improvement in performance of the proposed method compared to AIDO.Protein-IF makes the entire work more like a second-stage refinement rather than a completely novel end-to-end framework. 
* I suspect that the performance of the model might rely heavily on the quality of the initial guess from the base estimator AIDO.Protein-IF. This may be a particular problem when the retrieval system is at residue-level. The paper provides little further interpretation or ablation study on this matter.

### Questions
Please address my two major concerns in the “Weaknesses” section first. I will reassess after the rebuttal. Two other miscellaneous questions are as follows: 
* The RAG system likely limits the generalization ability of the model on unseen protein backbones. Have you tested the performance of the model on novel or orphan proteins that are dissimilar to any known protein in the RAG? 
* How much disk space do we need to store residue-level embeddings for every protein in the knowledge base? And what size of RAM did you use for experiments in section 4.6?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes PRISM, a multimodal retrieval-augmented generation (RAG) framework for protein inverse folding. The key idea is to augment a generative inverse-folding model with fine-grained structure–sequence retrieval from a large protein database. Instead of relying solely on end-to-end transformer inference, PRISM retrieves localized structural motifs or fragments and conditions decoding through a hybrid self-cross-attention decoder that integrates both query (target structure) and retrieved (reference) contexts.

### Strengths
- The problem setup and latent-variable formulation provide theoretical grounding for the proposed method.
- The ablation studies and runtime analysis provide clear evidence of the effectiveness of the proposed method.

### Weaknesses
- The author did not provide the database construction process. What structures have been used in the retrieval? Even without exact overlap, similar folds from the same CATH family may leak local sequence priors, effectively making the task easier.
- Equation (13): The $p(Z \mid B, \mathcal{R} )$ should be $p(Z \mid B, \mathcal{R} , \mathcal{E})$, where $\mathcal{E}$ is the encoding of the retrieved fragments.
- The PGM derivation seems to prove something trivial. The actual training process is just a dense retrieval with some next-token prediction loss.

### Questions
- Some of the figures have overlaps with the text, and some tables are too small to read.
- Section 4.2 is empty.
- The figures look too busy to read. You can either extend it to full size or remove some details.

### Soundness
2

### Presentation
2

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
The paper introduces PRISM, a novel framework for protein inverse folding.
The authors identify a key limitation in existing deep learning models: they lack an explicit mechanism to reuse conserved, fine-grained structure-sequence patterns (e.g., motifs) found in known proteins. To address this, PRISM is proposed as a multimodal retrieval-augmented generation (RAG) framework.
The core idea is to supplement end-to-end generation with a memory-based approach. The method works in several steps:
(1) construct a vector database by encoding known protein structures and sequences into contextualized representations.
(2) given a target 3D structure, leverage a base model to generate an initial sequence estimate, which is used to retrieve the vector database to obtain the most similar potential motifs.
(3) a novel hybrid self-cross attention (MHSCA) decoder then integrates the backbone information with the retrieved motifs to generate the sequence.
The method is theoretically grounded as a latent-variable probabilistic model. Empirically, PRISM improves upon all baselines in both sequence accuracy (AAR) and structural foldability (RMSD and TM-score) across five major benchmarks (CATH-4.2, TS50, TS500, CAMEO 2022, and PDB date split). Extensive and thorough ablation and analysis demonstrate the effectiveness of the proposed modules.

### Strengths
1. The core idea of explicitly retrieving fine-grained, residue-level motifs is an intuitive and novel solution to reuse the conserved local patterns which are central to protein function.
2. The method demonstrates clear and consistent improvement over existing models across all five evaluation benchmarks. This improvement is shown not only in sequence recovery metrics but also in more practical foldability scores.
3. The authors show that the significant improvement brought by retrieval-based method are achieved with a negligible runtime overhead (only a 14.3% compared to the base estimator), making PRISM a practical and scalable solution.
4. The authors provide extensive ablation studies to justify their design choices. They successfully prove the contribution of the retrieval mechanism and improvement over varying sequence lengths, optimize the number of retrieved entries, validate the hybrid MHSCA decoder design, and analyze the effect of decoder depth.

### Weaknesses
1. The main results in Table 1 should include structure-related metrics (e.g., scTM and RMSD) instead of relying solely on AAR and PPL for evaluation. Although scTM and RMSD are assessed in Table 4, this comparison is limited to DPLM2-3B and AIDO and lacks a broader set of baselines.
2. Several critical details are missing from the main body text:
- What protein library was the vector database built upon?
- In Figure 2, do the structure embedding and joint embedding originate from the same model? Are they kept fixed during training?
- A new term, "base estimator," appears in line 315. I speculate it is used to estimate the initial sequence during inference. If so, the authors should explicitly clarify it.
3. The sampling process relies on the "base estimator" to generate an initial sequence, which is then used for subsequent retrieval. This raises a key question: does the quality of this initial sequence significantly impact the final generated sequence? The authors have not provided an analysis of this potential dependency.
4. The retrieval-based generation method may cause the generated protein sequences to be overly similar to natural proteins (as suggested by the high AAR of ~70% in Table 2), which will limit the novel protein design. Furthermore, it's uncertain if the method's performance will drop significantly when used on a novel backbone that not similar to any entries in the database.
5. Minor issue: there is a formatting error where Table 3 overlaps with the main text.

### Questions
See above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
