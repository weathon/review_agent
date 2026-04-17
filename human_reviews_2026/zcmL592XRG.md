# VenusX: Unlocking Fine-Grained Functional Understanding of Proteins

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Deep learning models have driven significant progress in predicting protein function and interactions at the protein level. While these advancements have been invaluable for many biological applications such as enzyme engineering and function annotation, a more detailed perspective is essential for understanding protein functional mechanisms and evaluating the biological knowledge captured by models. This study introduces VenusX, the first benchmark designed to assess protein representation learning with a focus on fine-grained intra-protein functional understanding. VenusX comprises three major task categories across six types of annotations, including residue-level binary classification, fragment-level multi-class classification, and pairwise functional similarity scoring for identifying critical active sites, binding sites, conserved sites, motifs, domains, and epitopes. The benchmark features over 878,000 samples curated from major open-source databases such as InterPro, BioLiP, and SAbDab. By providing mixed-family and cross-family splits at three sequence identity thresholds, our benchmark enables a comprehensive assessment of model performance on both in-distribution and out-of-distribution scenarios. For baseline evaluation, we assess a diverse set of popular and open-source models, including pre-trained protein language models, sequence-structure hybrids, structure-based methods, and alignment-based techniques. Their performance is reported across all benchmark datasets and evaluation settings using multiple metrics, offering a thorough comparison and a strong foundation for future research. Our code (https://github.com/ai4protein/VenusX), data (https://huggingface.co/collections/AI4Protein/venusx-dataset), and a leaderboard (https://ai4protein.github.io/venusx/) are provided as open-source resources.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study presents VenusX, a comprehensive benchmark for assessing protein representation learning models with respect to fine-grained, intra-protein functional understanding. VenusX encompasses three primary task categories: residue-level binary classification, fragment-level multi-class classification, and pairwise functional similarity scoring. The benchmark is built from over 878,000 curated samples sourced from InterPro, BioLiP, and SAbDab, featuring both mixed-family and cross-family data splits at multiple sequence identity thresholds to evaluate in-distribution and out-of-distribution generalization. The authors benchmark a diverse range of models—including sequence-only, sequence-structure, and structure-only approaches—across all tasks. Key observations reveal that sequence-based models excel in in-distribution performance, whereas sequence-structure hybrid models exhibit superior generalization to unseen families, with cross-family prediction remaining a significant challenge. The benchmark, along with its code, data, and leaderboard, is released as open-source.

### Strengths
- VenusX represents the first large-scale benchmark specifically targeting fine-grained, sub-protein functional understanding, addressing a critical gap in existing protein benchmarks that largely focus on protein-level properties.
- The benchmark encompasses a diverse set of tasks (residue-level, fragment-level, and pairwise functional similarity), multiple annotation types (active sites, binding sites, epitopes, etc.), and meticulously designed data splits (mixed-family, cross-family, and multiple sequence identity thresholds), enabling a thorough evaluation of model capabilities.
- The study conducts an extensive assessment of state-of-the-art models across different modalities (sequence-only, structure-only, and sequence-structure hybrids), providing valuable insights into their respective strengths and limitations.
- All code, data, and a leaderboard will be released as open-source, facilitating community adoption and future research. The methodology for data curation, task formulation, and evaluation is documented in sufficient detail to ensure reproducibility.

### Weaknesses
- Although a wide range of models is evaluated, the paper provides limited insight into why certain approaches—such as sequence-structure hybrids—perform better on cross-family splits. Incorporating ablation studies or attention visualization could shed light on the learned representations and their functional relevance.
- Performance on the epitope prediction (Epi) task is notably low (AUPR < 0.3), even for top-performing models. A more detailed discussion of the intrinsic challenges of this task, along with potential reasons for the poor results, would strengthen the analysis.
- The benchmark’s scale (e.g., pairwise similarity tasks involving 177 billion negative pairs) and reliance on large models (e.g., ESM2-3B, ProtT5) entail substantial computational resources for full evaluation. Addressing the computational burden and discussing accessibility for researchers with limited resources would be valuable.
- For residue-level tasks on the BioLiP (BindB) and SAbDab (Epi) datasets, only sequence-only models are evaluated (as shown in Table 4). Including structure-aware baselines where feasible would offer a more complete assessment of model performance.

### Questions
- Could the authors provide further analysis or intuition on why sequence-structure hybrid models, such as SaProt, exhibit superior generalization on cross-family splits compared to sequence-only models? Is this improvement primarily driven by the structural vocabulary, the training objective, or a combination of both factors?
- The epitope prediction task (Epi) reports very low AUPR scores. What are the specific challenges underlying this task—such as data quality, label definition, or the conformational diversity of epitopes—that contribute to its difficulty? What potential directions could be explored to improve performance in future work?
- For the pairwise functional similarity scoring task, evaluation is conducted on a fixed sample of 10,000 positive and negative pairs. Was the sensitivity of the AUC score to this particular random sample analyzed? Reporting the variance across different random samples could provide a clearer picture of evaluation robustness.
- The GVP-GNN model, a structure-only approach, is trained from scratch, whereas other models use frozen pretrained checkpoints. Could the authors elaborate on the rationale for this choice? Were experiments conducted using pretrained GVP-GNN models or treating GVP-GNN as a frozen feature extractor to enable a fairer comparison in terms of parameter efficiency and training data usage?
- What are the key innovations and contributions of VenusX compared to prior work such as ProteinBench [1]?

[1] Ye, F., Zheng, Z., Xue, D., Shen, Y., Wang, L., Ma, Y., … & Gu, Q. (2024). ProteinBench: A holistic evaluation of protein foundation models. arXiv preprint arXiv:2409.067.

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
The paper curates data from a few protein databases to create some benchmarking tasks for protein function prediction. The tasks differ from most prior work in this space in that they focus on annotating sub-sequences of proteins corresponding to active sites, domains, etc. Given the evolutionary relationships between protein sequences, they exhibit clustering behavior, and thus non-trivial train test splits are important to evaluate models' extrapolation. They do a good job of guaranteeing distance between training and evaluation sequences.

The performance of variety of deep learning and alignment-based baselines are compared.

### Strengths
This problem is important for analyzing proteins, both for making predictions for the wide variety of uncharacterized natural proteins and for designing new proteins.

The benchmark appears to be well organized and accessible for future researchers.

### Weaknesses
I have key questions about how fragments are defined for certain annotation types. See below.

There are no non-deep-learning baselines for some tasks.

Both the fragment-level classification and fragment similarity tasks don't seem to be simulating a workflow that practitioners would encounter in practice. How practical is it to be presented with a pre-identified sub-sequence of the protein without knowledge of it's functional label? Usually, this is presented as a sequence segmentation problem, where the input is a full sequence and the output is a set of labeled segments.

The discussion of related work should have been in the main paper instead of Appendix.

When doing such a large exploration of baseline deep learning models, it is always hard to understand how much the results would have changed if more hyperparameter tuning had been done for each.

### Questions
I was surprised to see that fragment-level classification and pairwise similarity scoring tasks are done using localized annotation types such as active sites and binding sites. An enzyme has a small number of active site residues, and they are often non-consecutive. What is the corresponding 'fragment' for this task? In Fig 2, the fragments are fairly long. Did you just take the entire subsequence that includes the active site residues? I don't think this is justifiable as a prediction task, since there are no clear semantics to this subsequence. This issue might also serve as a confounder when comparing the alignment-based baselines on these tasks.

For the residue-level annotation task, I'm curious how an alignment-based technique would have worked (you do alignment between the query sequence and the training set). I expect that this would perform well for things like active site residues.

### Soundness
2

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
This paper introduces VENUSX, a pioneering benchmark for assessing deep learning models’ fine-grained functional understanding of proteins, beyond whole-protein tasks. It highlights the limitations of current models in capturing localized mechanisms—crucial for enzyme engineering, annotation, and interpretability—by curating >878,000 samples from InterPro, BioLiP, and SAbDab. VENUSX features three task types: residue-level binary classification, fragment-level multi-class classification, and pairwise similarity scoring, with mix- and cross-family splits at 50/70/90% identity. Baselines (ESM2, SaProt, GVP-GNN, etc.) show global performance does not guarantee local accuracy. All code, data, and leaderboards are open-sourced.

### Strengths
1.	VENUSX introduces a new focus on fine-grained protein functional understanding through novel residue-, fragment-, and similarity-based tasks, with cross-family splits to test generalization, addressing a key gap in evaluating localized biological signals.

2.	The benchmark is well-constructed from over 878k curated samples, with detailed task definitions, appropriate metrics, diverse baselines, and clear documentation, making it practical and accessible for research.

### Weaknesses
1. The evaluation lacks newer protein language models such as ESM3, which models sequence, structure, and function jointly; its absence limits the benchmark’s ability to reflect current state-of-the-art performance on fine-grained functional tasks.

2. Residue- and fragment-level tasks use only frozen residue embeddings passed through two linear layers; the lack of experiments on fine-tuning language models leaves unaddressed whether supervised adaptation can improve capture of localized functional signals.

3. Many of the tasks require training new classification heads on top of each protein representation model, which may be computationally demanding and could limit the benchmark’s accessibility and widespread adoption.

### Questions
1. Given the diversity of annotation sources (InterPro, BioLiP, SAbDab) and task types, could the authors provide an empirical analysis (e.g., correlation of model rankings, overlap in difficult examples, or transfer learning performance) across the seven residue-level and five fragment-level subtasks?

2. The experimental setup uses only frozen encoders with linear probes for all protein language models. Could the authors explain whether fine-tuning was attempted and, if not, provide a rationale (e.e., computational constraints, risk of overfitting on smaller splits, or intent to test pretraining quality)?

3. Proteins are often truncated to meet the input length limitations of the models. How are the protein structures modified in these cases to ensure that each residue is correctly mapped to its spatial location in the sequence+structure models? Providing more details on this process would significantly strengthen confidence in the fairness and rigor of the evaluation.

4.  For many entries, the benchmark relies on predicted structures from the AlphaFold Database rather than experimentally determined structures from PDB. While this is necessary to achieve the benchmark's massive scale, it introduces a potential confounding variable. An analysis of model performance on high-confidence predicted structures versus experimental structures would be a valuable addition.

5. For the classification tasks, GVP-GNN is the primary structure-only model evaluated and is trained from scratch. Including other modern, pre-trained structure-only models in the comparison would provide a more complete picture of that model class.

### Soundness
3

### Presentation
3

### Contribution
3
