# LM$^2$otifs: An Explainable Framework for Machine-Generated Texts Detection

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
The impressive ability of large language models to generate natural text across various tasks has led to critical challenges in authorship authentication. Although numerous detection methods have been developed to differentiate between machine-generated texts (MGT) and human-generated texts (HGT), the explainability of these methods remains a significant gap. Traditional explainability techniques often fall short in capturing the complex word relationships that distinguish HGT from MGT. To address this limitation, we present LM$^2$otifs, a novel explainable framework for MGT detection. Inspired by probabilistic graphical models, we provide a theoretical rationale for the effectiveness. LM$^2$otifs utilizes eXplainable Graph Neural Networks to achieve both accurate detection and interpretability. The LM$^2$otifs pipeline operates in three key stages: first, it transforms text into graphs based on word co-occurrence to represent lexical dependencies; second, graph neural networks are used for prediction; and third, a post-hoc explainability method extracts interpretable motifs, offering multi-level explanations from individual words to sentence structures. Extensive experiments on multiple benchmark datasets demonstrate the comparable performance of LM$^2$otifs. The empirical evaluation of the extracted explainable motifs confirms their effectiveness in differentiating HGT and MGT. Furthermore, qualitative analysis reveals \textit{distinct and visible linguistic fingerprints} characteristic of MGT.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes an explainable framework that specifically focuses on machine-generated text detection, which brings together two existing approaches: TextGCN and GNNExplainer. Specifically, the framework first builds a token-document graph, then trains a GNN detector for machine-generated texts; GNNExplainer is positioned as a post-hoc process that additionally explains the framework’s decisions at inference time by highlighting motifs relevant to those decisions. The authors conducted extensive experiments across various test settings and compared the performance of the proposed method with a large set of SOTA MGT detectors.

### Strengths
- This paper is properly organized, with a clear task statement and an illustration of the proposed pipeline, which facilitates easier understanding.
- The scale of the experiments is large. The experiments bring together different detectors and test them under various settings, reflecting the SOTA in MGT detection.

### Weaknesses
- The novelty raises a significant concern. This paper presents a typical A+B work, which reassembles existing approaches for a specific task.
    - The framework directly adopts TextGCN as the core for detection. Apart from the simplification of the graph by binarizing the edges, equations (2) and (3) appear largely overlapping with equations (3) and (8) in the TextGCN paper.
    - The title and the claimed contribution emphasize “explainable” as a main contribution of this work, yet the framework directly adopts an existing explanation tool (GNNExplainer) with a minor modification of the regularization term. This modification appears trivial, as it does not affect or appear in the subsequent discussion, and the use of $\lambda$ is overwritten by a different concept in Section 4.
- Some details about the GNN detector are either unexplained or confusing.
    - The detailed functionalities and implementations of AGG and COMBINE are not clear.
    - The relationship between $\boldsymbol{H}$ and $\boldsymbol{Z}$ is confusing. They are supposed to have different dimensionalities and therefore cannot be connected directly through a softmax.
    - See questions for further details.
- Explainability, as a central claim, is underrepresented. The authors leverage GNNExplainer, which appears rather isolated from the detection part. The missing connection between the two components harms the consistency of the paper as a whole.
- Some experimental settings are not clarified (see more details in questions). The claimed results that compare to other explanation baselines are missing (Figure 3 only shows the random baseline).

### Questions
- What is the novelty or contribution beyond the knowledge of TextGCN’s capability to perform text classification?
- What are the AGG and COMBINE functions? How do they alter/update node representations?
- What is the relationship between $\boldsymbol{H}$ and $\boldsymbol{Z}$? Let $d$ be the dimensionality of the node embedding. It is confusing how a softmax function can transform $\boldsymbol{H}\in \mathbb{R}^d$ to the prediction probability vector $\boldsymbol{Z}\in\mathbb{R}^2$ in a binary classification task.
- The dataset information presented in the appendix does not match the sources. Could the authors clarify the reason for downsampling the entries and how that is implemented?
- Could the authors explain the mismatch of the performance by DeTeCtive on M4? The original paper reports an accuracy of ~98% (can be estimated from the reported recall and F1).
- What are the training settings for the competitors? Are they fine-tuned for each test case? If the answer is positive, what does the “*” symbol indicate in the table?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a new method for authorship attribution for distinguishing between human and machine generated text. The task is treated as  a binary classification task. Given a set of human and machine generated texts, first the text is tokenized. A graph is created including all tokens and the documents/texts (either human or machine generated) as different types of nodes. Given an unseen text, a GNN has been trained to predict the node type (human or machine) of the text by first adding any new tokens and connecting the unknown text with all its token nodes. Empirical results on standard benchmarks show that the proposed method is more effective compared to other classification methods.

### Strengths
- The proposed method seems to work well and it has interpretable properties.

### Weaknesses
- The paper propose a well known technique applied on this specific llm vs human generated classification task. 

- The performance of the proposed method but also other methods looks saturated (e.g. achieving 100% accuracy in many cases). Either the task is solved or the datasets are too easy. This is not a criticism of the method proposed in this paper but for the whole area of llm/human authorship attribution.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes an explainable machine-generated text detection framework based on Probabilistic Graphical Models (PGM) and eXplainable Graph Neural Networks (XGNN). It constructs word co-occurrence graphs and employs GNNs for detection, while extracting “explainable motifs” that reveal linguistic structures distinguishing human-written and machine-generated texts.

### Strengths
S1: The introduction of a PGM perspective into machine-generated text detection (MGT detection) provides a theoretical justification (Theorem 4.1) and represents a meaningful level of innovation.

S2: The method is comprehensively validated across six mainstream datasets (HC3, M4, RAID, Yelp, Essay, Creative) and multiple LLMs (GPT-4, Claude3, Gemini, etc.), showing wide coverage and robustness.

S3: The model achieves strong performance and high inference efficiency (testing time only 0.005–0.009s/sample). It outperforms major baselines such as DetectGPT, Binoculars, Fast-DetectGPT, and DeTeCtive on both ACC and AUC metrics.

### Weaknesses
W1: Although the paper claims theoretical novelty from the PGM perspective, the actual modeling is limited to co-occurrence graphs plus GCNs, making it conceptually similar to TextGCN (Yao et al., 2019) and GNNExplainer (Ying et al., 2019). The theoretical analysis lacks new mechanisms or proof-level innovations.

W2: Several methodological details are unclear. For example, in the graph construction stage, the choice of PMI threshold, sliding window size, and hyperparameter λ is not specified. In motif extraction (Eq. 4), the selection of λ and the reproducibility of the extracted subgraphs are insufficiently discussed.

W3: The model’s cross-domain performance drops significantly (e.g., only 0.59 in the Reddit domain), indicating dependence on dataset-specific statistics and limited generalization capability.

W4: Typos and formatting issues: In Section 4, several mathematical symbols are densely presented without clear definitions. Minor typos exist (e.g., in Eq. (3), “Ydℓ is the ground-truth label,.” has an extra comma). Some equations are broken across lines, and figure–table references are inconsistent (e.g., Figure 3 and Table 2). Table captions sometimes lack consistent expansion of abbreviations (e.g., “DaV.”, “Dol.” should be fully spelled out on first mention).

### Questions
In Eq. (4), λ controls the balance between fidelity and complexity in subgraph extraction. How stable are the resulting motifs across multiple runs or random seeds? Do the motifs remain semantically consistent, or do they vary significantly due to stochastic optimization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces LM^2OTIFS, an explainable MGT detection framework, inspired by probabilistic graphical models (PGM).
The framework uses the word cooccurrence to capture the lexical dependence; then it applies XGNNs to learn to predict and generate motifs for explanation.
The framework outperforms baselines in terms of interpretability, with competitive performance.

### Strengths
1. The idea of linking the explanability in the graph neuron network to explain machine-generated text detection is impressive.
2.  The experiment covers adequate baseline methods and datasets.

### Weaknesses
1. The explanation results have space to be expanded and deepened. 1) Unclear how to explain the difference between human and machine text. For the example in Figure 4, I believe "sleep" and "patients" are also very common words in human-written medical-related documents. It is still unclear why these words are a strong clue. 2) There are also highlighted non-common words, e.g., "in", "with". They may indicate more generalizable patterns, but the paper fails to discuss.  3) My understanding of the current explanation is limited to the cooccurrence of words, which might be highly conditioned on the training corpus. I sense why "obstructive sleep" is a matify for GPT-4 might just be because it has appeared as a training example (which I can be wrong). But what if we remove those "obstructive sleep" examples as a small perturbation of the dataset collection? Can the detector still be robust on the test example?

2. I am concerned about the computation cost. It seems each time we predict a new document, we need to compute its tokens' occurrence with all training dataset (I can be wrong), which could be a large cost when we scale up the training set and test set. But for other baselines, the estimated distribution is mostly parametrically stored, which only requires a pass on the to-be-predicted document.

3. There are spaces to improve writing clarity. Sections 3.1 and 4 seem to have large overlap content on the graph construction/definition parts.

4. (A minor point) The explanation of detection is limited to the lexical level. As token nodes are initialized with one-hot features and sequence nodes with all-zero features, I would suggest that no semantic information is introduced. As some papers point out that semantics is important for some MGT detection (like some pos types of tokens may contribute more), it would be cool to see if we include semantic features, for example, initializing the graph node with embeddings from some LLM, the motifs would be the same or not.

### Questions
1. Section 2 MGT Detection: Should T_h and T_m be an input to the function f for the tasks? It seems to include the few-shot detectors. I assume you are mainly considering your methods, but what about all your other baselines? Do you mean they can all also be expressed as this function?

### Soundness
3

### Presentation
2

### Contribution
2
