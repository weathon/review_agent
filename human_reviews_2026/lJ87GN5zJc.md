# Graph Diffusion Transformers are In-Context Molecular Designers

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2, 4

## Abstract
In-context learning lets large models adapt to new tasks from a few demonstrations, but it has shown limited success in molecular design, where labeled data are scarce and properties span millions of biological assays and material measurements. We introduce demonstration-conditioned diffusion models (DemoDiff), which define task contexts through molecule–score examples instead of texts. These demonstrations guide a denoising Transformer to generate molecules aligned with target properties. 
For scalable pretraining, we develop a new molecular tokenizer with Node Pair Encoding that represents molecules at the motif level, requiring 5.5$\times$ fewer nodes.
We pretrain a 0.7B parameter model on datasets covering drugs and materials. Across 33 design tasks in six categories, DemoDiff matches or surpasses language models 100–1000$\times$ larger and achieves an average rank of 4.10 compared to 6.56–17.95 for 19 baselines. These results position DemoDiff as a molecular foundation model for in-context molecular design.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new paradigm for molecular design, named DemoDiff, which combines in-context learning with graph diffusion models to tackle property-guided molecular generation. The authors' core idea is to use molecule-property score pairs (demonstrations) to define a task, guiding a Graph Diffusion Transformer during the generation process. To support this framework, the paper also designs Node Pair Encoding, which compresses molecules from the atom level to the motif level, significantly increasing the context length the model can handle. This work is novel in its conception and experimentally thorough, demonstrating its potential as a foundation model for molecular design.

### Strengths
- A Novel and Inspired Framework: The idea of using "demonstrations" as a condition to guide a diffusion model is highly inspiring. DemoDiff dynamically defines the task context through molecule-score pairs, which is not only more flexible and scalable but also enables the model to learn and generalize to new, data-scarce tasks. 
- An Efficient and Innovative Graph Tokenizer: NPE is a purely data-driven motif learning algorithm that adaptively discovers high-frequency patterns from the pre-training data. Experiments show that this method achieves an average compression ratio of 5.5x , which directly supports the conclusion that the model performs better with longer contexts.
- Comprehensive and In-depth Experimental Evaluation: The paper conducts a comprehensive test on 33 tasks across 6 major categories, including drug and materials design. It compares DemoDiff against 13 baselines, including traditional optimization algorithms, conditional generation models, and even general-purpose large language models with 100-1000x more parameters.

### Weaknesses
- Regarding Consistency Score: According to the appendix, the core performance metrics (Table 1) are not derived directly from DemoDiff's output but are obtained by first generating 1000 candidates and then using the consistency score to filter for the top 100 for evaluation. This introduces a significant confounding variable, making it impossible to determine the model's inherent generative capability. Also, Figure 5 shows this score has a very low correlation with the true oracle score for key tasks like Drug MPO and Materials Design (correlation coefficient close to 0), which questions its validity as an effective filtering metric.
- NPE Tokenizer's Robustness and Limitations: While the paper showcases the excellent compression efficiency of the NPE tokenizer, one of its core contributions, it fails to discuss its performance when handling rare chemical motifs or molecules that deviate significantly from the pre-training data distribution. As a component intended to be the core of a foundation model, its reliability in out-of-distribution (OOD) scenarios is crucial. Furthermore, the claim of "lossless reconstruction" remains a qualitative description, lacking quantitative reconstruction accuracy metrics on a standard dataset, which makes it difficult to assess its reliability.

### Questions
1. Regarding model scaling (Table 2), the data shows that for the "Structure Constrained" task, the small (78M) and medium (311M) models outperform the large (739M) model (0.59/0.63 vs. 0.56). This contradicts the paper's conclusion that "the benefits of parameter scaling become more evident at the large scale". How do you explain this performance degradation? 
2. The core advantage of ICL is its ability to generalize to new tasks with few shots. The paper states that the 33 evaluation tasks are distinct from the pre-training data. However, given the scale of the pre-training data, I am concerned about the degree of this distinctness. To what extent are the chemical spaces, property targets, or molecular scaffolds in the evaluation tasks truly unseen during pre-training? 
3. In the baseline comparison, the paper includes general-purpose LLMs like GPT-4O but does not compare against LLMs pre-trained on chemical texts or molecular sequences (e.g., SMILES), which possess domain-specific knowledge. Why were these models not included in the comparison?
4. The introduction states that directly applying the autoregressive framework of LLMs is "infeasible", thus justifying the choice of GraphDiT as the backbone. However, molecules can be represented as sequences like SMILES or SELFIES, and some baselines are indeed based on them. Is there any evidence to support that GraphDiT is superior to a powerful Transformer decoder trained on SMILES sequences within this ICL framework?

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
4

### Summary
Through this paper, the authors propose DemoDiff, a demonstration-conditioned diffusion model that has a Graph DiT as the backbone, to construct a molecular generative model under in-context learning. In addition, the authors also propose a molecular tokenizer trained with Node Pair Encoding (NPE) for motif-level representation to support efficient pretraining of DemoDiff.

### Strengths
- The authors provided the codebase.
- The introduction of a motif-level tokenizer using Node Pair Encoding, which reduced the node count by 5.5×, is a reasonable approach in terms of both efficiency and performance.
- The proposed DemoDiff was evaluated across 33 design tasks spanning 6 categories, achieving SOTA performance in most of them.

### Weaknesses
Weaknesses
I will combine the *Weaknesses* section and the *Questions* section. My concerns are as follows:
- There is little discussion on how sensitive the demonstration/context selection is to performance and how well the derived context applies to diverse new tasks.
- SOTA molecular optimization baselines such as GenMol [1] and Genetic GFN [2] are missing. Comparisons with these baselines are necessary for the results to be considered meaningful.
- No computational or memory efficiency was reported. This would be an effective method to demonstrate that the proposed DemoDiff is more efficient compared to larger generalist LLMs.

---

**References:**

[1] Lee et al., GenMol: A Drug Discovery Generalist with Discrete Diffusion, ICML 2025.

[2] Kim et al., Genetic-guided GFlowNets for Sample Efficient Molecular Optimization, NeurIPS 2024.

### Questions
Please see the *Weaknesses* section for my main concerns.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel molecular design framework, DemoDiff, which integrates diffusion models with in-context learning. By introducing a Node Pair Encoding (NPE) tokenizer for molecular graphs, the model achieves efficient pretraining with motif-level representations. A 0.7B-parameter model is pretrained on large-scale, multi-task data covering 155K properties and millions of molecules. DemoDiff achieves an average rank of 3.63 across 33 molecular design tasks, outperforming ten baseline models and several LLMs.

### Strengths
1. DemoDiff implicitly embeds task information into the diffusion denoising process. The proposed Node Pair Encoding (NPE) is a substantive contribution to molecular graph tokenization, eliminating the need for handcrafted reaction rules and enabling automatic motif discovery.
2. Extensive evaluation on 33 tasks demonstrates that DemoDiff significantly outperforms both domain-specific baselines and ICL-based LLMs. Despite having only 0.7B parameters, the model matches or even surpasses general-purpose LLMs.
3. The paper provides detailed descriptions of the pretraining task design, motif tokenizer, context consistency score, and diffusion inversion inference. The experiments offer comprehensive analyses of parameter scale, context length, positive/negative sample ratios, and consistency filtering.

### Weaknesses
1. Lack of theoretical analysis. Although the paper mentions an implicit Bayesian interpretation, it lacks formal proofs or theoretical analysis on how the diffusion trajectories reflect “posterior inference of task concepts.” The comparison with ICL in language models remains largely analogical, without quantitative or interpretable mechanism-level parallels.
2. Potential bias in data and task construction.
- The pretraining tasks mainly rely on ChEMBL and polymer datasets, whose property distributions are highly skewed (Zipf-like). While the authors claim this facilitates ICL ability, no systematic study is provided on the relationship between task frequency and generalization performance.
- The oracle evaluation for generation tasks depends on human scoring and predictor models, which may introduce noise.
3. Insufficient validation of NPE’s chemical semantic consistency. The paper lacks statistical verification of motif interpretability or chemical validity. Although DemoDiff outperforms Graph-DiT, no strictly fair comparison (e.g., controlled parameter size or identical token count) is presented.
4. High pretraining cost. The work lacks small-scale, reproducible experiments or an ablation-only release.

### Questions
1. What does the “Total sum” represent — which scores are included in this total? In Eq. 6, what is the function denoted by T?
2. The authors interpret DemoDiff’s in-context learning as implicit Bayesian inference. Could this interpretation be validated via attention pattern or diffusion trajectory visualizations?
3. NPE is a frequency-driven motif discovery algorithm. Do these automatically learned motifs correspond to known chemical functional groups or reaction fragments? How well does NPE generalize to out-of-distribution (OOD) chemical spaces? PRODIGY [1] also applies in-context learning on graphs — please provide a comparison.
4. The authors mention that the property distribution follows Zipf’s law (Figure 3a). Has the effect of this long-tailed distribution on the model’s ICL ability been evaluated? Does DemoDiff maintain strong performance on low-frequency property tasks? Since the number of examples per assay is imbalanced, could this lead to training bias toward high-frequency tasks?
5. The authors use consistency score and related metrics. Could evaluation be extended to include metrics more closely tied to molecular properties, such as QED or SA score?

Reference:

[1] PRODIGY: Enabling In-context Learning Over Graphs

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose DemoDiff, a generative model that uses molecule-score examples for in-context molecular design. The authors also develop Node Pair Encoding (NPE), a strategy to represent molecules at the motif level, which requires 5.5x fewer nodes on average. DemoDiff achieves impressive performance across 33 molecular design tasks spanning 6 categories.

### Strengths
- The proposed NPE reduces the number of nodes by 5.5x, which is impressive and is a generally useful contribution even outside the context of DemoDiff. 
- The proposed in-context learning approach is novel and a unique contribution to the ML molecular design literature 
- DemoDiff demonstrates promising results on the drug/material design categories and is able to generate more diverse samples than baseline methods
- The authors provide extensive ablation and case studies

### Weaknesses
I am not very convinced by the main experimental results and I think Table 1 is somewhat misleading. Firstly, it is a bit unfair to compare to LLMs given that DemoDiff is trained on >1M molecule-property pairs and LLMs are not designed for this specific task. A more fair comparison would be to finetune the LLMs to do in-context molecule generation in the same way as DemoDiff. Secondly, and more importantly, after looking at the results in the Appendix, it appears that DemoDiff only achieves SOTA performance on 13 out of the 33 tasks (based on the Top-10 Harmonic Mean results). In particular, outside the drug/material design categories, DemoDiff does not reliably outperform baseline LLMs. Given that DemoDiff success is limited to a small subset of tasks it is difficult to support the claim that this is a generalizable method.

### Questions
LLMs are not necessarily trained to do in-context learning or to do molecular design, which makes it difficult to compare to your proposed method which is trained on a large and specialized dataset of molecule-spectra pairs. Did the authors consider finetuning LLMs to do in-context molecular design on the same dataset used for DemoDiff pretraining? Do you think DemoDiff would outperform such a finetuned LLM?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work introduces a conditional generative model for molecules based on a graph diffusion transformer.
To reduce the context size, atoms are grouped together using a BPE-like algorithm adapted to graphs.
Conditioning is done by using pairs of molecules with normalised scores as _property_ inputs to the graph diffusion transformer.
Experiments on a dataset extracted from ChEMBL and a collection of polymer datasets show favourable performance in terms of oracle and diversity scores.

### Strengths
- Overall, the paper is well written.
 - Most reported results come with error bars.
 - Reported results seem to be competitive.
 - Being able to control molecule generation on activity levels seems relevant.

### Weaknesses
- The proposed node pair encoding feels similar in spirit to how (extended connectivity) fingerprints are computed.
   However, the relation to these possibly related methods seems to be underexplored in the current version of the manuscript.
   It would be good to have an explanation how NPE is similar/different from fingerprint computations.
   If there is enough similarity, also an empirical verification of NPE vs fingerprint features would be meaningful.
   I also feel that NPE has some similarities with virtual nodes used in some GNNs (e.g. Hwang et al., 2022).
   A discussion on possible similarities/differences in this regard might be useful as well.
 - There is little to no motivation on why molecular graphs are used instead of e.g. SMILES.
   I have often heard that GNNs bring little to no performance advantage over working with SMILES (e.g. Renz et al., 2024).
   Furthermore, GNNs are typically much more complex to work with than models that work with SMILES.
   Given that the core Graph DiT idea seems transferrable to non-graph architectures as well,
   it would have been interesting to investigate how important the _graph_ aspect of this model is.
 - It is not entirely clear how strong the provided baselines are.
   Most notably, there is little information on how the LSTM was used to operate on the NPE encodings.
   Also, it seems hard to imagine that there are no more specialised models that are able to generate molecules in-context.
   Especially in the context of autoregressive SMILES generation (e.g. Renz et al., 2024; Schmidinger et al., 2025)
 - Figure&nbsp;2 confuses me more than it helps me to understand the method.
   I understand that molecules are encoded by grouping sub-structures,
   but there is no explanation for what $S_\mathrm{ingle}$ is supposed to be,
   or what the digits in the brackets are supposed to mean.
   Furthermore, the inputs ($[?, S0]$ and $[?, S5, S4, S7]$) to the transformer,
   which I would have assumed are on top of the demo tokens,
   seem to be the sub-motifs from the tokenized molecule illustrated to the right.
   However, this tokenized molecule should be part of the demo tokens.
   This all seems to make little sense and is hard to connect to the explanations in the main text.

### Minor Issues
 - The expression "molecule-assay pair" (line 257) seems to be a bit weird.
   Could it be that you mean "molecule-activity pair",
   i.e., a molecule with its activity value for a particular assay (a.k.a. the context)?
 - In the main text (line 302) _10_ novel, unique and valid molecules are mentioned for evaluation,
   but in the appendix, line 1054 mentions _100_ molecules for evaluation.
 - Table&nbsp;1 is claimed to report two scores (oracle and diversity),
   but there is only a single value for each model-task combination.

### Additional References
 - Hwang et al. (2022). [An analysis of virtual nodes in graph neural networks for link prediction](https://openreview.net/forum?id=dI6KBKNRp7). In The first learning on graphs conference.
 - Renz et al. (2024). [Diverse hits in de novo molecule design: Diversity-based comparison of goal-directed generators](https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c00519). Journal of Chemical Information and Modeling, 64(15), 5756-5761.
 - Schmidinger et al. (2025). [Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences](https://openreview.net/forum?id=IjbXZdugdj). International Conference on Learning Representations.

### Questions
1. Is there any/What is the relation between NPE and the computation of molecular fingerprints?
 2. Is there any/What is the relation between NPE and the use of virtual nodes (cf. Hwang et al., 2022) in GNNs?
 3. Could the proposed model architecture also work with non-graph (e.g. SMILES) inputs?
 4. How was the LSTM applied to the NPE-encoded graphs?
 5. Are there no other (e.g. SMILES-based) specialised generative models that could be used as baselines?
 6. Can you explain what is happening in Figure&nbsp;2?

### Soundness
2

### Presentation
3

### Contribution
2
