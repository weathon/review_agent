# BrainFCIR: Functional Context Informed Representation Learning for Intracranial Neural Signals

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Intracranial neural recordings (e.g., stereo-ElectroEncephaloGraphy (sEEG)) have offered a unique window to measure neural signals across multiple brain regions simultaneously. Recent works have focused on developing neurofoundation models that learn generalizable representations across both subjects and tasks from such recordings. These models achieve exciting advances, yet overlook the modular functional organization of the brain, where neurons from multiple adjacent anatomical regions collectively support specific cognitive functions (e.g., the Wernicke area for speech perception). A key challenge remains how to effectively incorporate this functional contextual information into representation learning to improve both interpretability and decoding performance. To tackle this challenge, we propose a novel pre-training framework, BrainFCIR, that explicitly integrates functional context into model design via spatial-context-guided representation learning. We evaluate BrainFCIR on the publicly available sEEG speech-perception dataset. Extensive experiments show that BrainFCIR, as a unified representation learning framework for intracranial sEEG signals, significantly outperforms previous decoding methods. Overall, our work underscores the significance of functional context in developing more biologically plausible and high-performing neural decoding models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In contrast with foundation models that learn temporal or spatial representations of sEEG signals, the proposed model BrainFCIR represents the spatio-temporal relationships. This is done by learning at the same time the tokenization of the signal and a representation based on a spatio-temporal transformer approach. In addition to providing an unified representation, the model also outperforms current decoding methods.

### Strengths
the model is original and addresses an important topic, both in terms of representation learning and in terms of performance.

### Weaknesses
The topic of building representations integrating contextual constraints is central in this work. Here the context is related to responses of the neighborhood, but it is not clear (and not discussed) if other kind of constraints are absent in the current SOTA and should be also considered here.

### Questions
-the need to have a mapping that also integrates contextual information is also present in many other domains than sEEG. Do you think your work could be applied to other signals ?
- the database corresponds to recording of 10 different people. Do you think that it is enough or that better performance could be observed with more people ?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces BrainFCIR, a neurofoundation model for intracranial sEEG recordings that aims to model functional context (e.g. inter-channel relationships reflecting coordinated brain activity). The model is pretrained with constrastive learning and combines temporal and spatial transformers from which they estimate inter-channel connectivity. Then, they cluster channels into functional groups that are finally used for decoding and interpretability. The results they provide on the Brain Treebank human sEEG dataset show improvements over Brant and PopT on four speech related decoding tasks. The proposed architecture and pretraining objective are technically sound and the evaluation protocol follows established practice. However the experiments use a limited number of seeds, comparison are made with only two baselines, and quantitative gains are moderate. While the method appears valid, stronger statistical validation and more comparative results would increase confidence in the claims.

### Strengths
BrainFCIR successfully unifies spatial and temporal modeling within a contrastive pre-training framework to identify functional connectivity patterns in the brain using transformer attention matrices. The model achieves modest but consistent improvements over the compared baselines on sEEG decoding tasks. The paper is well structured, conceptually coherent, and relies on publicly available data, which supports reproducibility and transparency.

### Weaknesses
The results, while promising, remain preliminary. The experimental validation is limited to four tasks and only two baselines; additional benchmarks would help position the contribution. The qualitative interpretability analysis lacks depth and justification (e.g., why only one hemisphere, how clusters map to known regions). Figures are difficult to read (notably text and legend are very small), captions are minimal, and the code is not yet released. Finally, while computational gains from channel selection are mentioned, no quantitative comparison supports the claim.

### Questions
1. What is the rationale for showing only one hemisphere in the cortical visualizations ?
2. Could you provide quantitative results on inference speed and efficiency after channel selection ?
3. Why are there only two baselines for comparison? Please either clarify the reasons (e.g., lack of other available models) or include more baselines in the evaluation.
4. Why is the code not available with the submission? This raises concerns about the reproducibility of the work.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BrainFCIR, a spatiotemporal transformer for sEEG that is pre-trained using a “functional-context replacement” task and later used for channel clustering and selection. The idea is to encourage the model to capture cross-channel dependencies, and then exploit the resulting attention structure to identify informative channels. On several speech-related classification tasks, BrainFCIR reports modest gains over prior baselines and reduced inference cost after channel selection.

### Strengths
- The concept of treating functional context as a self-supervised signal is creative and potentially useful for building population-level representations from intracranial data.
- Leveraging the learned attention maps for channel selection is a nice practical touch — if validated properly, it could be valuable for faster or more portable brain–computer interfaces.

### Weaknesses
- Ambiguity in the pretraining task. It’s not clear exactly how the replaced segments are sampled (from the same session? same channel? matched by power spectrum?). The teacher/student inputs and the label definitions in Eq. (4–5) are hard to follow, and the objective could be solved by simple statistical shortcuts rather than genuine contextual reasoning.

- Questionable interpretation of attention as connectivity. Averaging spatial-attention weights is not equivalent to functional connectivity, especially since anatomical coordinate embeddings are also injected. The resulting maps could mostly reflect geometric proximity. There’s no comparison to standard signal-level FC measures (e.g., coherence or partial correlation) or distance-controlled analyses.

- Unclear clustering visualizations. In Fig. 3, the spatial distribution is hard to interpret (why only one hemisphere? why uneven density?). The matrix view in panel (b) lacks meaningful axes or clear correspondence to panel (a). The number of clusters k=10 is fixed without sensitivity analysis.

- Experimental results:  Weak cross-subject generalization. Under the hold-one-subject-out setting, BrainFCIR drops more sharply than PopT (≈ 0.10 vs 0.05 gap). Multi-subject training barely improves performance, suggesting poor subject scalability.

- Possible data leakage in channel selection. If the “functional connectivity” used for clustering is computed on the entire dataset (even unlabeled), this would constitute double-dipping and inflate test scores.

- Baseline alignment. Preprocessing (sampling rate, referencing) and hyperparameter changes make comparisons with PopT and Brant hard to interpret. Also, model selection is by accuracy while reporting uses AUC — these should match.

- Parameter inconsistency. The reported 3.32 M parameters don’t match the configuration in Table 4; just the transformer MLPs would already exceed that count.

- Limited task diversity. Only binary classification is shown. Including a regression or retrieval task (e.g., continuous speech features or EEG image reconstruction [1]) would better demonstrate representational value.

### Questions
- how are replacement segments chosen and normalized?

- What input does the teacher see?

- Is channel selection computed strictly on training data per fold?

- How does attention-derived connectivity relate to Euclidean distance or to coherence-based FC?

- Can you reconcile the parameter count with a detailed component breakdown?

### Soundness
3

### Presentation
3

### Contribution
3
