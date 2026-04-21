# IgBleng: Unifying 3D structures and sequences in antibody language models

- Avg Score: 3.50
- Decision: Reject
- Scores: 3, 3, 3, 5

## Abstract
Large language models (LLMs) trained on antibody sequences have shown significant potential in the rapidly advancing field of machine learning-assisted antibody engineering and drug discovery. However, current state-of-the-art antibody LLMs often overlook structural information, which could enable the model to more effectively learn the functional properties of antibodies by providing richer, more informative data. In response to this limitation, we introduce IgBlend, which integrates both the 3D coordinates of backbone atoms (C-alpha, N, and C) and antibody sequences. Our model is trained on a diverse dataset containing over 4 million unique structures and more than 200 million unique sequences, including heavy and light chains as well as nanobodies. We rigorously evaluate IgBlend using established benchmarks such as sequence recovery, complementarity-determining region (CDR) editing and inverse folding and demonstrate that IgBlend consistently outperforms current state-of-the-art models across all benchmarks. Furthermore, experimental validation shows that the model's log probabilities correlate well with measured binding affinities.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces IgBlend, a large language model (LLM) specifically designed for antibody engineering, combining 3D structural data with sequential data. IgBlend aims to address the limitations of existing antibody LLMs by incorporating 3D backbone coordinates (C-alpha, N, and C atoms) alongside traditional sequence information. The model leverages a large dataset comprising over 4 million unique structures and 200 million unique sequences, enabling it to perform tasks like sequence recovery, CDR editing, and inverse folding. Empirical evaluations show that IgBlend consistently outperforms state-of-the-art antibody models across several benchmarks, and its log probabilities correlate well with binding affinity.

### Strengths
The proposed antibody language model has biomedical applications, specifically in antibody and therapeutic design, where such capabilities are highly impactful.

The model proposes an novel combination of structure and sequence information, which is theoretically a promising approach for antibody design tasks.

The methodological and architectural descriptions are mostly clear, and the integration of benchmark comparisons supports the claims effectively.

### Weaknesses
Lack of clarification of the benchmark dataset. Given the fact that the model has seen millions of training samples, I worry about the data leak in benchmarking and the author did not mention much about the benchmark construction.

The author compare the model on several CDR infilling tasks. As a language model, the utility in representation learning is unclear.

The HER2 H-CDR3 editing experiment shows a weak correlation (Spearman correlation: 0.24) between model scores and binding affinity, which is also close to the baseline AntiFold of 0.23.

Lack of open source code for the implementation and experimental results.

### Questions
What is the size of your language models?

Antibody sequences and structures are very humongous especially in the constant regions. I wonder if it's worth to train on millions of examples where most of them are similar. In addition, the introduction of predicted antibody structures in the training might introduce biases, and limits the model generalization capabilities. I would like to see some ablations the training data diversity, quantity, and the inclusion of predicted structures.

It would be convincing to also compare the model in representation tasks, such as antibody related downstream tasks.

It's important to show the performance gain is from the algorithm instead the memorization of the training set. To this end, I suggest authors to construct a non-redundant test set (such as sequence identity of 50%) for the infilling tasks.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper presents the IgBlend method, a pre-trained model for antibody modeling. The model can encode both sequence and structure information, and predict the masked sequence. The model can also be used in other tasks like inverse folding or CDR design.

### Strengths
-	The authors studied an important research topic.
-	The authors conducted experiments on many downstream tasks.

### Weaknesses
-	I have concerns about the novelty because jointly modeling sequence and structure have already been widely studied in many scientific tasks, such as organic small molecules [1,2] and protein [3].

-	In Table 2, for the sequence only settings, the IgBlend performance seems to be worse than baselines.

-	Minor: the caption of Fig.2 should below the figure.

[1] Dual-view Molecular Pre-training, KDD 2023

[2] One Transformer Can Understand Both 2D & 3D Molecular Data, ICLR 2023

[3] Protein sequence and structure co-design with equivariant translation, ICLR 2023

### Questions
-	Does authors have any plan to release the pretraining code, data, and checkpoints for reproducibility
-	As there are three coordinates for each position (C-alpha, N, and C), how are they processed by GVP-GNN? How are they combined?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors present IgBlend, a model that combines antibody sequences and to enhance antibody engineering. The model shows improved performance in several benchmarks and shows strong correlation with binding affinities in experimental validation.

### Strengths
1. The authors presented an effective pre-training framework, unifying antibody sturctures and sequences.

### Weaknesses
1. The authors did not compare IgBlend with another similar method, LM-Design, despite mentioning it in the related work section.
2. The whole structure information ("Struct Guided") was used in sequence recovery and CDR editing tasks, which may introduce potential data leakage.
3. The sequence recovery and CDR editing tasks are quite similar, and the observed improvements appear marginal.
4. This paper is not clearly written, and there is a typo in the title (IgBleng -> IgBlend).

### Questions
Why did the authors choose GVP as the structure encoder instead of other models, such as MPNN?

### Soundness
2

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
This paper introduces IgBlend, a new method which adds structural information to an 
antibody language model using a Geometric Vector Perceptron. The authors describe a 
pre-training approach which combines multiple objectives for learning sequence and 
structure-based tasks. 
Currently, I would recommend to reject this paper for the following reasons: (1) The novelty 
of the approach is limited. There have already been several models proposed which 
integrate structure into protein language models, including LMDesign which is cited by the 
authors, and the model is very similar to ESM-IF which has already been adapted to 
antibodies (2) The comparison to other methods is not fair. In particular, other methods 
were not designed to be run on single chains and/or Nanobodies. (3) The apparent 
inclusion of modelled structures in the test set means that the results cannot be trusted 
since IgBlend may simply be learning bias in IgFold, especially for Nanobodies.

### Strengths
• The paper is well-written and it is easy to understand the differences between the 
experimental settings. 
• The comparison to sequence-only settings throughout demonstrates that this 
model operates effectively as a sequence model in the absence of structure. 
• The analysis of model-guided HER2 design was interesting and motivated the 
inclusion of structure. 
• The ablation of the different model inputs is helpful and informative.

### Weaknesses
• The AntiFold results are much worse than those quoted in the AntiFold paper. This 
is likely because AntiFold was trained using both chains whereas this paper tests it 
on individual chains (including Nanobodies) which is not a fair comparison. 
• The test set seems to include a large number of IgFold-generated structures. If this 
is the only method trained on IgFold structures, then it is very likely to perform 
better by learning bias in IgFold. Previous approaches test only on high-quality 
experimental structures. This point holds especially when comparing RMSDs to 
IgFold-predicted structures. 
• “Notably, IgBlend is the first inverse folding model to achieve results on nanobodies 
comparable to heavy chains” - this claim is not well supported since it appears that 
all tested Nanobody structures were modelled. 
• The authors note the similarity to LM-Design in the introduction but do not provide a 
comparison in the experiments. 
• The biggest advantage of this work compared to the extensive work on antibody 
language modelling and inverse folding is the “Seq + Struct Guided” setting, 
however the benefit of this is not demonstrated in any experiments except 
pretraining. Could this be used to generate better embeddings or in a realistic 
antibody design task?

### Questions
• I assume the title IgBleng is a typo? 
• Is there some inherent benefit to language models which justifies the exclusion of 
tools such as AbMPNN?

### Soundness
2

### Presentation
2

### Contribution
2
