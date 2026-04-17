# Learning Representations Through Contrastive Neural Model Checking

- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Model checking is a key technique for verifying safety-critical systems against formal specifications, where recent applications of deep learning have shown promise. However, while ubiquitous for vision and language domains, representation learning remains underexplored in formal verification. We introduce Contrastive Neural Model Checking (CNML), a novel method that leverages the model checking task as a guiding signal for learning aligned representations. CNML jointly embeds logical specifications and systems into a shared latent space through a self-supervised contrastive objective. On industry-inspired retrieval tasks, CNML considerably outperforms both algorithmic and neural baselines in cross-modal and intra-modal settings. We further show that the learned representations effectively transfer to downstream tasks and generalize to more complex formulas.  These findings demonstrate that model checking can serve as an objective for learning  representations for formal languages.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a technique to improve the ability to associate circuit designs with their respective specifications by training models to learn a joint representation over the two domains. The authors demonstrate the effectiveness of this technique by evaluating trained models on retrieval tasks, classification tasks, and generalizability.

### Strengths
I really like the problem this paper is tackling: it is unique yet important, and seemingly understudied especially in the age of large models. The hypothesis is sound and builds on well-established findings, and the results reflect the advantages of the proposed technique. They also test the generalizability of the trained models by splitting the formulas.

### Weaknesses
I see some weaknesses in the paper:
1. Need for a contrastive learning approach: The paper has not really justified the need of using contrastive learning as such. While it would not harm the performance, I would assume given a dataset of ~300K pairs, the models would be trained in a rather straightforward manner.
2. Choice of encoder: I also am not sure why CodeBert specifically was chosen -- while it is trained on code, I don't expect the code in pretraining being considerably in distribution with respect to LTL formulas. I wonder what would have happened if we just used a standard Bert model or RoBerta instead of something specialized like CodeBert.
3. Generalizability experiments: I would be more convinced by the generalizability if you had shown model performance on composed specifications (if that is even possible), and more importantly, a held-out dataset consisting of specifications that were designed by professionals/that exist in model-checking textbooks or references. I also don't get a sense of how aligned your generated specs are with industry-standard specifications, even if it is just 100 samples.

### Questions
Refer to weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a representation-learning view of model checking via Contrastive Neural Model Checking (CNML), which jointly embeds LTL specifications and AIGER circuits using two CodeBERT encoders trained with a CLIP-style contrastive objective. A large synthetic dataset (cnml-base) is built, and a single-guarantee variant (cnml-split) is derived by formula splitting for a generalization study. The method is evaluated on intra-modal retrieval with N=100 and N=1000, against algorithmic (e.g., Bag-of-Keywords, WL kernel) and neural baselines, an cross-modal retrieval (neural baselines only). Finally, they fine-tune for model checking and show generalization from single- to multi-guarantee formulas.

### Strengths
* A novel contrastive approach to neural model checking (joint embeddings for LTL and circuits).
* A synthetic dataset created by first sampling LTL formulas and then synthesizing matching circuits.
* Broad retrieval evaluation (cross-modal and intra-modal).
* Mini-batch construction that avoids duplicates and reduces off-diagonal false negatives.
* Clear, well-organized writing with intuitive embedding analyses (e.g., cosine-similarity distributions and a heatmap).

### Weaknesses
* Mini-batch false negative analysis is limited 
* Cross-modal retrieval lacks non-ML baselines (e.g. edit distance between paired LTL/AIGER string forms).
* Runtime benchmarks versus non-ML baselines are missing scalability and variance are not characterized.
* No qualitative results overall (true positives/false positives/near misses, error taxonomy, or interpretability visuals).
* Sequence pooling is under-specified (CLS/mean/max/attention-pool not clarified), and its impact is not quantified.
* Practical metric such as “recall after model checking (top-k)” is not reported alongside raw recall.
* Potential data overlap/near-duplicates in synthetic sets are not analyzed; de-duplication is unclear.

### Questions
Please address the weaknesses listed above.

Please explain the sequence pooling layer in more detail.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper uses contrastive learning to learn joint embeddings of linear temporal logic (LTL) formulas representing specifications, and and-inverter graphs formulas representing systems to be checked. Both kinds of formulas are in raw ascii text, and pre-trained CodeBERT models are used as encoders, which are further fine-tuned on synthetic tasks. Experimental evaluations show that, compared to vanilla CodeBERT and  Sentence-BERT, contrastive learning with CodeBERT outperform both on two retrieval tasks (i.e., cross-modal retrieval and intra-modal retrieval). Furthermore, CodeBERT after contrastive learning outperforms the original CodeBERT for downstream fine-tuning task, binary classification on circuit-specification pairs.

### Strengths
- the targeted research problem, model checking, is of great importance in hardware verification
- decent background such as LTL, And-Inverter Graphs, model checking, and contrastive learning, are provided

### Weaknesses
- contrastive learning has been widely explored in similar tasks like code analysis and theorem proving; applying contrastive learning for LTL specifications is fairly incremental, especially given this work simply applies this standard idea to a synthetic dataset generated with existing tools. 
- evaluation tasks like cross-modal retrieval and intra-modal retrieval are artificial, and there is no clear indication how and to what extent, these retrieval tasks really help to tackle the model checking challenge (e.g., state exploration issue explicitly highlighted in the introduction). 
- the findings is somewhat well-expected, CodeBERT with some fine-tuning on the synthetic dataset shall outperform the original CodeBERT.
- the assume-guarantee format is essential for this work, however, there is no discussion (including appendix) about the specific syntax for assume and guarantee sub-formulas. One concern is that they may be biased in a limited category.

### Questions
It is surprising that the authors believe LTL is short for "Linear-Time Temporal Logic" (see background section), especially given that LTL is the focus of this work. What is the complete syntax for assumption and guarantee sub-formulas? Are they limited in some category, for instance, certain operators like "implies" is not allowed. 

Why are remaining N^2-N pairs (implicitly) considered negative? Shouldn't validation checking be performed?

How do cross-modal retrieval and intra-modal retrieval help model checking? Are they purely hypothetic or used in any model checkers? To what extent, do these retrieval affect the performance of model checking algorithms?

Certain discussions of the introduction (line 58 - 65) do not make much sense; if LLMs are used for the paper writing, the authors shall explicitly acknowledge that.

### Soundness
1

### Presentation
1

### Contribution
1
