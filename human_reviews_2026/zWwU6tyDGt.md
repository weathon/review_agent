# Evolution-Aware Positive-Unlabeled Learning for Protein Design

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 0, 4

## Abstract
We consider prediction of protein function, focusing on protein functionalities that enhance survival for one or more organisms. Sequencing these organisms provides plentiful positive training examples due to survivorship bias. In contrast, synthesizing and characterizing a protein with a mutation unseen in nature requires time-consuming wet lab experiments, making negative training examples scarce. Thus, datasets are often imbalanced, hindering classifier accuracy outside the training data.  Positive-unlabeled (PU) learning attempts to address this issue by considering unlabeled protein sequences to be part of the data and modeling them as positive with a probability called the class prior. This class prior is often constant. Our insight is that an understanding of evolution suggests a novel sequence-dependent class prior when learning from sequencing data. We propose Evo-PU, a PU learning framework that integrates our novel class prior to create a likelihood for training classifiers. We evaluate Evo-PU on multiple real-world tasks on influenza hemagglutinin protein. Using influenza genomic surveillance data and held-out laboratory assays of mutants unseen in nature, Evo-PU outperforms state-of-the-art PU learning, one-class classification (OCC), and deep generative model-based methods (DGM) on these real-world problems, demonstrating the benefit of combining evolutionary modeling with data-driven learning for protein design. We further assess Evo-PU on standard ProteinGym benchmarks, focusing on protein overall fitness prediction. Evo-PU outperforms existing PU-learning and OCC baselines, while remains competitive to DGM-based approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents EVO-PU, a positive-unlabeled learning framework for classifying favorable versus disruptive protein variants. The problem it addresses is important but not highlighted enough at machine learning conferences, and the proposed method is novel and interesting. Nevertheless, the work is disconnected from modern machine learning approaches.

### Strengths
- Data scarcity and the lack of negative data are common challenges in machine learning for biology. This work directly addresses these issues.
- The proposed EVO-PU method is well-explained, novel, and well-justified.
- The results show that EVO-PU outperforms other positive-unlabeled learning and one-class classification approaches.

### Weaknesses
The paper's main weakness is its lack of connection to modern machine learning approaches.
- The evaluation is performed on three custom-built datasets based on the Influenza virus. The paper lacks an evaluation on standard benchmarks, such as the well-established ProteinGym [1]. If none of standard datasets used in prior work is suitable, this should be explained.
- The baselines consist mostly of traditional machine learning algorithms. The paper lacks a comparison with modern machine learning models, such as AlphaMissense [2] or even simple ESM2 [3] log-likelihoods [4].
- EVO-PU uses the Wide and Deep (WD) network architecture from 2016 with a simple featurization of protein sequences. The paper would benefit from using a more modern, for example Transformer-based architecture [3]. This would address the issue mentioned on Line 345 (“Directly optimizing the loss in Eq. 7 over discrete amino acid sequences is intractable”), as discrete amino acid sequences can be represented as continuous features [3].

[1] Notin et al., 2023, “ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction” https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1

[2] Cheng et al., 2023, “Accurate proteome-wide missense variant effect prediction with AlphaMissense” https://www.science.org/doi/10.1126/science.adg7492

[3] Lin et al., 2023, “Evolutionary-scale prediction of atomic-level protein structure with a language model” https://www.science.org/doi/10.1126/science.ade2574

[4] Meier et al, 2021, “Language models enable zero-shot prediction of the effects of mutations on protein function” https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1.full

### Questions
1. What is the sequence distance between the training and test data for each of the three datasets?
2. The abstract states, “We consider prediction of protein function, focusing on protein functionalities that enhance survival for one or more organisms.” This suggests that identifying favorable variants is the primary application. If so, would other retrieval-based metrics that account for class imbalance, such as precision and recall, be more informative than the standard AUROC?

### Soundness
2

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
The authors are interested in predicting the effects of mutations from observed sequences from evolution. They cite a biophysical model, suggest a modification based on the choice of negatives and infer models based on this model. They Show their model better predict the effects the effects of mutations from an influenza assay.

### Strengths
* The authors suggest a modification to a biophysical model
* They some some (inconsistent) improvement on an assay.

### Weaknesses
The main weakness of this work is that it does not properly interface with the mainstream methods for learning evolutionary conserved information from protein sequences, that is, generative models trained on sequences seen across life.
* They only cite EVE and EVEscape. What about ESM, Progen, etc...?
* They claim "While generative models effectively capture conserved constraints, they focus on single-point mutations and often struggle to predict activity for sequences with a few mutations from known positives" without citation. On the contrary, according to ProteinGym, these models are also state of the art for multiple mutations as well.
* They only compare to EVE rather than state-of-the art models that do much better on viral sequence inference.

As well, modern models are evaluated on ProteinGym, which contains hundreds of assays each with thousands of measurements. In contrast, the authors only evaluate their model on a single assay with less than 50 measurements.

### Questions
* Why did you train EVE on prevalence data rather than evolutionary data from across life, say by running an alignment. This is another confusion of mine -- in principle your model can be used to perform inference 
* "However, since the true wild types for our test sequences are unknown, we instead calculate this index for each test sequence against the top 20 most frequently observed sequences in Dn and select the minimum index. These minimum indices are then modeled with a two-component Gaussian mixture model (GMM) used to predict the probability that a test sequence possesses the property of interest." Why is this a fair choice?
* Much simpler biophysical models have been used to justify mainstream generative models [example](https://proceedings.neurips.cc/paper_files/paper/2022/file/247e592848391fe01f153f179c595090-Paper-Conference.pdf). Could you compare your theory to these?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces Evo-PU, designed for protein binary function prediction where only positive examples are available. The core methodological novelty is the introduction of a sequence-dependent, "evolution-aware" class prior. This evolutionary prior is integrated into a custom likelihood function to train a classifier. The authors evaluate Evo-PU on three tasks related to influenza virus proteins (fusion, binding, and evasion), comparing it against other PU learning methods, one-class classifiers, and an evolution-based generative model (EVE). They report state-of-the-art performance, which they attribute to their more biologically realistic learning framework.

### Strengths
1. The paper's core assumption that the sampling bias in natural sequence data is not uniform and can be modeled by evolutionary preference is a strong and valid argument. 

2. Strong Ablation (E-GEN vs. RAND): The comparison of baselines using both E-GEN and RAND unlabeled data is a strong control experiment. It successfully suggests that the performance gain of Evo-PU is likely attributable to its unique loss function, not just the quality of the generated unlabeled sequences.

### Weaknesses
1. Critically Incomplete and Potentially Misleading Baselines: The experimental evaluation is fundamentally flawed and ignore the most basic and powerful baselines.

      a. The paper fails to compare against standard similarity-based methods, such as a simple k-NN classifier on either BLAST scores, ESM-2 embeddings, or FoldSeek/SaProt structural alignments. These are the fast, robust, and established first-line approaches for function prediction. 

     b. The comparison against EVE is conceptually questionable. EVE is a generative model rather than classification model. Furthermore, the reported AUC scores of < 0.5 is flawed. 


2. Novelty is Limited and Overstated: The paper's novelty hinges on its "evolution-aware" component. However, this is philosophically very similar to the direct use of scores from pre-trained protein language models (like ESM, AlphaMisense, EVE) to estimate a sequence's plausibility. While the authors integrate this concept into the PU framework in a novel way, the underlying idea is not new. The work feels like a clever recombination of existing ideas (PU learning + evolutionary models) without biological applications and working scenarios. 

3. Lack of Connection to Practical Utility: It never demonstrates that its improved AUC score on this specific binary task leads to any tangible downstream benefit. A successful paper would show that its model, for example, can be used to propose a novel immune-evasive viral epitope that is later validated, or that its functional predictions correlate well with clinically relevant outcomes like disease severity. As it stands, the work is an isolated academic exercise in improving a specific metric.

### Questions
See weakness above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Evo-PU, a positive-unlabeled learning framework for detecting functional protein sequences. Evo-PU introduces a sequence-dependent class prior derived from a probabilistic model of evolutionary emergence at the nucleotide level. The model estimates A(x), the probability that a protein sequence is functional, effectively serving as a fitness predictor. Empirically, Evo-PU achieves superior AUC scores compared to Protein-PU and other baselines on three influenza case studies (fusion, binding, and immune evasion).

### Strengths
- The paper presents a new and well-developed theoretical framework with clear definitions and explanations.  
- The proposed method is shown to outperform directly comparable approaches such as Protein-PU.  
- The paper is well written and logically organized.

### Weaknesses
- Missing comparison to zero-shot protein language model-based fitness (i.e., A(x)) predictors. The paper does not compare Evo-PU to zero-shot fitness predictors derived from protein language models, which have become standard in the field (for example, the ESM-1v paper https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2 or more recent methods for example from the ProteinGym benchmark https://proteingym.org/benchmarks). A comparison on the same influenza benchmark would clarify whether Evo-PU captures additional biological signal beyond what these models already encode implicitly. Without such a comparison, it is difficult to position Evo-PU relative to current state-of-the-art fitness predictors.  

- Limited evaluation. The paper evaluates Evo-PU only on a single custom influenza dataset. While this benchmark is carefully constructed and biologically relevant, it has not been used in previous studies. Evaluating Evo-PU on a well-established dataset such as ProteinGym would enable direct comparison to standard baselines and better demonstrate the method’s generality.

### Questions
- Could the authors comment on the computational efficiency of Evo-PU? If my understanding is correct, the model must be retrained for each protein sequence, and the computational complexity scales exponentially with sequence length through D_n. 

- The theoretical framework operating with both nucleotide and amino acid sequences does not seem to be well justified but introduces a substantial complexity. Would not it be enough to work only on the level of protein sequneces? For example, observability directly operates with protein sequneces in Section 2.2 but is defined on the level of nucleotide sequences.

### Soundness
3

### Presentation
2

### Contribution
2
