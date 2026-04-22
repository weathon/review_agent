# Quantum Autoencoder: An Efficient Representation Learner for Quantum Features

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Quantum machine learning methods often rely on fixed, hand-crafted quantum encodings that may not capture optimal features for downstream tasks. In this work, we study the power of quantum autoencoders in learning data-driven quantum representations. We first theoretically demonstrate that the quantum autoencoder method is efficient in terms of sample complexity throughout the entire training process. Then we numerically train the quantum autoencoder on 3 million peptide sequences, and evaluate their effectiveness across multiple peptide classification problems, including antihypertensive activity, blood–brain barrier-penetration, and cytotoxicity. The learned representations were compared against Hamiltonian-evolved baselines using a quantum kernel with support vector machines. Results show that quantum autoencoder learned representations achieve accuracy improvements ranging from 0.4\% to 8.1\% over Hamiltonian baselines across seven datasets, demonstrating effective generalization across biologically distinct datasets, with pre-training enabling effective transfer learning without task-specific fine-tuning. This work establishes that quantum autoencoder architectures can effectively learn from large-scale datasets (3 million samples) with compact parameterizations ($\sim$900 parameters), demonstrating their viability for practical quantum applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to use a quantum autoencoder to preprocess Hamiltonian-encoded classical-data-representing quantum states for the downstream task of classification with a support vector machine. The main contribution is the proposal of this application of quantum autoencoder, and also experiments / benchmarking on some peptide datasets.

The claims of the paper are mostly accompanied by empirical results. However, a more careful design of the experiments and processing of the results is needed to convincingly support the claimed benefits of the quantum autoencoder for downstream tasks.

The paper lacks novelty. The quantum autoencoder used is not new and is almost identical to the original model presented. Usage of the quantum autoencoder is not motivated by sufficient theoretical analysis. The empirical performance comparison is not entirely fair to encourage the broader community to try further in that direction.

### Strengths
The paper is clearly written, and the results are prettily plotted and discussed. Good presentation and interesting application to peptide datasets.

### Weaknesses
Not enough theoretical analysis to explain why a quantum autoencoder should help with the downstream support vector machine. Not entirely fair numerical experiments.

### Questions
1.	It is not very clear to me, from a theoretical perspective, why the compressed quantum states should perform better in the downstream classification task, especially when the quantum autoencoder is trained in an unsupervised way, meaning it does not know beforehand the downstream task. Can you make a theoretical analysis to explain why?
2.	The comparison between the baseline data (fixed Hamiltonian encoding) and the quantum autoencoder-processed data is not fair for a few reasons. First, the kernels used for the two cases are different, making it unclear whether the claimed benefits come from the different choice of kernel or really from the compression. Can you either make some ablation studies to clarify this, or try to use the same kernel?
3.	Second, I worry that the benefits appear only because they are cherry-picked from abundant runs over many different hyperparameter settings. This is not fair because the baseline is fixed. Can you present the average performance, not just the best performance, and show the confidence intervals?
4.	It is also an important question to understand whether the benefit of the quantum autoencoder, if there’s any, comes from discarding excessive qubits, or from the extra non-linearity due to partial trace. Can you either theoretically or experimentally clarify this point?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a quantum autoencoder (QAE) framework for data-driven quantum representation learning, focusing on peptide sequence modelling and classification. The authors argue that most quantum machine learning (QML) pipelines rely on static encodings (e.g., Hamiltonian or angle encodings) that fail to adapt to data structure. The proposed QAE learns compact quantum representations from data in an unsupervised manner, similar to pre-training in classical deep learning.

The paper provides (i) theoretical results on the depth and sample complexity of the variational circuit; (ii) large-scale training of QAEs on 3 million peptide sequences using Hamiltonian-evolved quantum states; and (iii) downstream evaluations on seven peptide classification datasets, comparing learned QAE representations against fixed Hamiltonian baselines.

The results show accuracy improvements ranging from 0.4% to 8.1% over Hamiltonian encodings across tasks, with the learned representations showing transferability without fine-tuning. The study positions quantum autoencoders as an analogue of classical representation learning (e.g., PLMs) within quantum settings .

### Strengths
* The paper takes an important step beyond static quantum encodings by treating representation learning as a core problem in quantum pipelines. The analogy to unsupervised pre-training (as in PLMs) is both conceptually interesting and timely for ICLR’s machine learning audience.
* The pre-training on 3 million peptide sequences represents one of the largest QML simulations to date, demonstrating the feasibility of scaling quantum-inspired learning beyond toy datasets.
* Using seven biologically diverse peptide datasets (antihypertensive, antioxidant, BBB-penetrating, cytotoxic, hemolytic, and neurotoxic peptides) provides convincing evidence that the learned quantum representations generalise across domains.

### Weaknesses
* The reported improvements (0.4–8.1%) over Hamiltonian baselines are relatively modest compared to PLM gains (1–16%), suggesting limited immediate practical impact. Discussion of why the performance gap is smaller would strengthen the narrative.
* Downstream testing is limited to SVMs with quantum kernels, rather than using QAE features directly for diverse classifiers. While justified by density-matrix representation issues, this restricts insight into how QAEs could integrate into hybrid pipelines.
* The study benchmarks only against Hamiltonian evolution encoding. Other fixed encodings (e.g., amplitude or angle encoding) are not tested, leaving open whether improvements generalise.
* It would be valuable to show which aspects of the learned latent space (e.g., entanglement structure or qubit utilisation) correlate with downstream performance.
* The paper’s comparisons are limited to Hamiltonian-encoded quantum baselines and brief contextual references to PLMs. There are no quantitative benchmarks against classical representation-learning methods such as autoencoders, variational autoencoders (VAEs), or simple self-supervised embeddings trained on the same peptide data. Without such comparisons, it remains unclear whether the observed 0.4–8.1% gains could be replicated by small classical models with similar parameter counts? I'm not wanting to see quantum advantage (which I believe should not be the sole goal of quantum computing), but demonstrating clear empirical advantage to more classical baselines would be almost be essential instead of just comparing to quantum baselines.
* Although the datasets are biologically motivated, the paper does not analyse whether the learned representations correspond to meaningful biochemical properties (e.g., hydrophobicity, charge patterns), which could make the work more relevant to bioinformatics audiences.

### Questions
* Do the learned latent representations exhibit measurable structure or interpretability (e.g., clustering by peptide function)?
* How sensitive are the results to the number of qubits

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
5

### Summary
This paper studies QAEs as learned quantum representations for classical sequence data (peptides). The authors (i) give two theoretical statements: a polylogarithmic depth upper bound for the QAE encoder under local Hamiltonian data encoding, and a uniform-in-training sample complexity bound to estimate the SWAP-test–style loss; (ii) pre-train QAEs on 3M unlabeled peptide sequences encoded via Hamiltonian evolution; and (iii) evaluate the resulting compressed states as features for SVMs with a trace-distance kernel on seven downstream peptide classification tasks. They report +0.4%–8.1% test-accuracy gains over a fixed Hamiltonian encoding baseline.

### Strengths
The paper explicitly targets representation learning within QML, comparing learned QAE features against fixed Hamiltonian encodings while holding the downstream classifier constant, which isolates the contribution of representation learning on the quantum side. 

In my view, unsupervised training on 3M peptide sequences with small circuits (8–10 qubits, 10–30 layers, and 1–3 traced-out qubits) is a useful empirical data point. 

QAEs uniformly outperform the Hamiltonian baseline across seven datasets in radar plots. 

The paper candidly notes simulation-only results, kernel-method dependence, and the non-optimality of the Hamiltonian encoding for long sequences.

### Weaknesses
The only quantum baseline is the fixed Hamiltonian encoding. While this isolates representation learning, it is a relatively weak comparator. Missing are (a) other learned quantum encoders (e.g., variational feature maps without autoencoding; data-reuploading encoders) and (b) end-to-end quantum classifiers (keeping the same resource envelope) to contextualize the benefits of pretraining. The “PLM vs one-hot” comparison is explicitly not directly comparable and thus does not strengthen the quantum claims (Sec. 4.2, Fig. 4). 

The radar plots summarize relative improvements but omit tables of absolute accuracies, standard deviations, and statistical significance across splits. A results table per dataset (mean ± std over 5 folds) would allow readers to judge the practical effect sizes; currently, the gains (0.4%–8.1%) may be within the noise for some datasets.

The path from high-dimensional one-hot peptide vectors to 8–10 qubits via Hamiltonian evolution (Sec. 3.1) compresses massively before autoencoding. It is not obvious how information-preserving or task-aligned this mapping is for long sequences. More ablations are needed: (i) different qubit counts; (ii) alternative encodings (angle/amplitude; learned/Trotterized Hamiltonians or something else); (iii) sensitivity to sequence length thresholds. 

All experiments are noiseless simulations; yet the kernel requires estimating the trace distance between reduced density matrices.

Ablations on pretraining value. The claim of “pretrained without fine-tuning” would be stronger if compared against (a) training the same QAE from scratch on each downstream dataset; (b) pretraining on fewer sequences (data-scaling curve); (c) varying the number of traced-out qubits m and depth with fixed compute budget.

### Questions
See Weaknesses.

In addition, why fix Hamiltonian evolution as the input encoding before QAE? Can you report results with angle or amplitude encodings (same qubit budget) to show the robustness of the pretraining story?

For a realistic n=10, m∈{1,2,3}, what is the measurement complexity to approximate the trace-distance kernel on current devices? Any preliminary noise simulations (depolarizing/readout) or classical-shadow-based surrogates? Strengthening ablations and reporting are needed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies quantum autoencoders (QAEs) as a method for unsupervised quantum representation learning for classical peptide sequences. The authors pre-train QAEs on 3 million peptide samples encoded via Hamiltonian evolution, producing compressed quantum states used as representations. The learned representations are evaluated on seven downstream peptide classification datasets using quantum kernels with SVM classifiers. Compared against fixed Hamiltonian-evolution encodings, the QAE-derived representations improve test accuracy by 0.4–8.1%. The authors also provide theoretical results claiming (1) polylogarithmic circuit depth is sufficient for implementing QAEs under local Hamiltonians, and (2) efficient sample complexity for estimating the QAE loss throughout training. The work emphasizes that the goal is not quantum advantage but establishing QAEs as effective, scalable quantum representation learners on large classical datasets.

### Strengths
- The pre-training on 3 million peptide sequences is a significant step up from the toy problems (e.g., MNIST, Iris) often seen in QML literature. This demonstrates an attempt to scale these methods to a problem of practical and real-world size, which is conceptually aligned with classical deep representation learning.

- The comparative framework (in Figure 1) is well-designed. By having both the QAE pipeline and the baseline pipeline start with the same Hamiltonian encoding, the experiment effectively isolates the contribution of the QAE's learned transformation.

- The paper provides two theoretical results: polylogarithmic depth for implementing QAEs under specific Hamiltonian assumptions, and a sample-complexity bound via local scrambling and classical shadows. Although high-level, these results offer theoretical grounding for scalability.

### Weaknesses
- The primary empirical evidence for the QAE's utility is an accuracy gain of 0.4% to 8.1%. On several datasets, the gains are marginal (e.g., 0.4%, 0.6%, 0.8%, 1.4%). These small improvements make it unclear about the practical significance of the approach, especially when considering the overhead of pre-training the QAE. Also the experiments are conducted entirely in a noiseless simulation. Since variational quantum algorithms are sensitive to hardware noise, a gain of 0.4% or 0.6% would almost be erased by any realistic noise model. Without at least a preliminary analysis of noise robustness, the paper's claim of "viability for practical quantum applications" feels a bit unsubstantiated.

- The use of constant-weight Pauli Hamiltonians, locality conditions, and scrambling assumptions for two-qubit gates are strong constraints. It is unclear how these results translate to realistic QAEs used in the experiments. The connection between the theoretical analysis and the specific circuits used is not fully convincing, partiuclarly in real world applications.

- Only Hamiltonian evolution is used as baseline. Other natural baselines—angle encoding, amplitude encoding, shallow variational encoders, or random features—are omitted.

### Questions
- How sensitive are the reported gains to the choice of kernel? Would fidelity kernels or projected kernels produce similar improvements?

- Why is the baseline restricted to Hamiltonian evolution? Could the authors include comparisons against at least angle encoding or random variational encodings?

- Given the small performance margins, how do you expect these gains to hold up under realistic noise? Even a simple depolarizing noise simulation would add significant weight to your claims of practical viability.

### Soundness
3

### Presentation
3

### Contribution
2
