# Lightweight Transformer for EEG Classification via Balanced Signed Graph Algorithm Unrolling

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Samples of brain signals collected by EEG sensors have inherent anti-correlations that are well modeled by negative edges in a finite graph. 
To differentiate epilepsy patients from healthy subjects using collected EEG signals, we build lightweight and interpretable  transformer-like neural nets by unrolling a spectral denoising algorithm for signals on a balanced signed graph---graph with no cycles of odd number of negative edges.
A balanced signed graph has well-defined frequencies that map to a corresponding positive graph via similarity transform of the graph Laplacian matrices.
We implement an ideal low-pass filter efficiently on the mapped positive graph via Lanczos approximation, where the optimal cutoff frequency is learned from data.
Given that two balanced signed graph denoisers learn posterior probabilities of two different signal classes during training, we evaluate their reconstruction errors for binary classification of EEG signals.
Experiments show that our method achieves classification performance comparable to representative deep learning schemes, while employing dramatically fewer parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed an EEG classification method based on unrolling a sequence of graph learning and low-pass graph filtering operations. At each step, the authors first learn a signed graph Laplacian, then cut off the high-frequency components for denoising. The authors interpret the final output as the posterior mean of the clean signal, and derive a classifier based on its distance to the noisy input. Experiments show that their method outperforms existing graph and non-graph EEG classification models in normal and LOSO settings.

### Strengths
1. The paper is well-written and easy to follow. More details are provided in the appendix.
2. The authors interpret the graph learning and denoising as a form of transformers. The parameter space of the proposed model is substantially smaller than many existing methods, while the performance is stronger.

### Weaknesses
1. The graph construction is heuristic. Some design choices are not well-justified, e.g. the use of the Manhattan distance.
2. The evaluation on the real dataset is on the weaker side. The authors only evaluate on two datasets, with one relegated to the appendix. More informative metrics such as PR-AUC should also be considered.
3. The paper would benefit from some in-depth discussion about why the proposed model outperforms the competing ones.

### Questions
1. What is the justification for using the signed Laplacian? The total positivity of the combinatorial graph Laplacian is shown to be a useful bias for many problems. If both positive and negative correlations are allowed, one can just use eigenvectors of the empirical covariance matrix for denoising, essentially PCA. This is somewhat similar to the authors' choice, given their way of selecting the signs in the graph, but much simpler. Is there any ablation study on this?
2. How did the authors select the cutoff value $\omega$ for denoising?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the problem of EEG classification. The authors propose a discriminant
analysis model that predicts classes based on the smallest reconstruction error to a 
class-wise autoencoder. The autoencoder represents the multivariate EEG time series
as a graph where each node represents a segment of a univariate channel. 
Consecutive segments in the same channel are connected by edges, and concurrent
channel segments are connected by edges weighted by the correlation between their 
segments. The autoencoder denoises by using a graph low-pass filter that uses only 
the k eigen dimensions of the graph Laplacian with highest eigenvalue. The authors 
address specifically the problem that some of the edge weights / correlations can 
be negative, and then the graph Laplacian is no longer guaranteed to be positive
semidefinite. For this case they approximate the graph with a balanced signed 
graph, flipping the signs of some edges; then a pos.semidef. Laplacian is guaranteed
again. In experiments on a large EEG dataset they show that their method outperforms
several baselines, while reaching 98.4% of the performance of a way less
parsimonious model with 1000 times more parameters.

### Strengths
- s1. denoising using the channel segment correlation graph is an interesting approach.
- s2. finding a good balanced signed graph approximation is a novel idea in this context.
- s3. very good results with a model with very few parameters.
- s4. the method is well explained.

### Weaknesses
- w1. several strong baseline papers for EEG classification have been missed.
- w2. an ablation study for low pass filtering is missing.
- w3. experiments are hard to reproduce as no source code is provided and the method
  is somewhat complex.

more details:

w1. several strong baseline papers for EEG classification have been missed.
- for example, MAtt [Pan et. al., NeurIPS 2022] is often compared to recently.
- also the EEG foundation models such as EEG2Rep [Baeza-Yates et al., KDD 2024]
  might be interesting models to compare to. They also report on TUAB
  (like your appendix G), but way lower scores. Is the experimental protocol
  the same?

w2. an ablation study for low pass filtering is missing.
- would the model deteriorate if you do not low-pass filter at all?
- then your model is basically a graph attention neural network.
- what is not clear to me: to softmax-like normalization of the 
  attention weights in eq. 8 should denoise already by pushing
  the smaller weights close to zero. Low-pass filtering now sets them
  to exactly zero. Why does this help?
- you argue that your model does not use dense and large key,
  value and query matrices K, V and Q, but the Mahalanobis kernel M 
  in eq. 6 you could interprete as say the key weights matrix Q.
  Then you just choose the identity for K and V. 

w3. experiments are hard to reproduce as no source code is provided and the method
is somewhat complex.
- re-implementing all the different steps of your method will put a high burden on
  researchers trying to work with your paper.

smaller points:
- p1. the model's performance crucially depends on the contrastive loss (eq. 23), 
  but you do not mention this in the main paper. In ablation study F1, w/o 
  the constrastive loss your model performs worse than all your baselines 
  in tab. 1.
  

references:
- Pan, Yue-Ting; Chou, Jing-Lun; Wei, Chun-Shu (NeurIPS 2022):
  "MAtt: a manifold attention network for EEG decoding."
  Advances in Neural Information Processing Systems.

- Baeza-Yates, Ricardo; Bonchi, Francesco; Nguyen, Nam; Foumani, Navid
  Mohammadi; Salehi, Mahsa; Mackellar, Geoffrey; Ghane, Soheila; Irtza,
  Saad (KDD 2024): "EEG2Rep: Enhancing Self-supervised EEG
  Representation Through Informative Masked Inputs."  Proceedings of the
  30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD
  2024, Barcelona, Spain, August 25-29, 2024.

### Questions
- q1. Can you compare the performance of your model with recent strong models
  such as MAtt and EEG2Rep?
- q2. As additional ablation study: how does the model perform w/o low-pass filtering?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a lightweight, interpretable “transformer-like” network for EEG seizure classification by unrolling a balanced signed-graph denoising algorithm. Key ideas: (i) learn a balanced signed graph of EEG channels, then map it via a similarity transform to an equivalent positive graph with well-defined frequencies; (ii) implement an ideal low-pass graph filter (cutoff learned from data) and interleave it with a balanced-graph learning (BGL) module; (iii) train two class-conditioned denoisers and classify by comparing reconstruction errors. On the Turkish Epilepsy EEG dataset the method attains 97.57% accuracy with only ~14.8k parameters, and shows strong leave-one-subject-out (LOSO) generalization.

### Strengths
S1. Principled modeling. The use of balanced signed graphs gives a rigorous frequency notion via a Laplacian similarity transform to a positive graph, enabling classical spectral filtering while still modeling anti-correlations that are common in EEG. This is technically neat and well-motivated.

S2. Clear algorithm-unrolling design with interpretability. The network cleanly alternates LP filtering and BGL. Figure 2 makes the pipeline easy to follow。

S3. Good parameter efficiency and competitive accuracy.

### Weaknesses
> W1. “Attention equivalence” is overstated.  

The normalized weights use
    $$\bar w_{ij}=\beta_i\beta_j\frac{e^{-d_{ij}}}{\sqrt{\sum_\ell e^{-d_{i\ell}}\sum_k e^{-d_{kj}}}},$$
    which (a) can be negative via $\beta_i\beta_j$, (b) are *symmetric* rather than row-stochastic, and (c) need not satisfy $\sum_j \bar w_{ij}=1$. This differs materially from softmax attention
    $$a_{ij}=\frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}\ge 0,$$
    so calling $\bar w_{ij}$ “essentially attention weights” risks confusion.

> W2. Possible leakage in polarity initialization.  

Initializing polarities via an empirical covariance $\bar C$ computed from “collected EEG data” is ambiguous: if $\bar C$ uses the *entire* dataset, the train/test split is compromised. Clarify that initialization uses train-only statistics.

> W3. Temporal modeling is inconsistent across sections. 

One place “assumes a single chunk” and works on $G_B$, elsewhere the implementation uses a product graph with 6 windows (nodes $N=210=6\times35$). It is unclear whether the experiments use the single-chunk simplification or the 6-slice product graph with temporal edges, and how this choice affects results.

> W4. Feature metric is under-specified.  

Distances use $d_{ij}=(f_i-f_j)^\top M (f_i-f_j),\quad M\succeq0,$  but Appendix text suggests $M=Q_iQ_i^\top$ selected from a candidate set. Is $M$ global, per-node, or per-block? How is the candidate set regularized to avoid degenerate metrics? What prevents collapse(e.g., rank-1 $M$) beyond the implicit CNN bottleneck?

### Questions
**I found that the main text exceeds the 9-page limit, which seems to violate the strict page requirements for submissions.**

For rebuttal, I hope the authors to answer the following questions:

Q1: Please clarify whether the cutoff is implemented via explicit spectral truncation or via a smooth transfer function on eigenvalues (e.g., sigmoid). If both appear in the paper, which one produced the reported numbers, and why was it preferred?

Q2: You ensure PSD by setting $L_B=\bar L_B+\delta I$. What is the empirical distribution of $\delta$ across subjects/blocks? How does varying $\delta$ shift the effective spectrum and the LP response (e.g., energy scaling)? 

Q3: How imbalanced are the learned signed graphs before enforcing balance (e.g., frustration index, fraction of odd negative cycles)? A short diagnostic could justify that hard balance enforcement does not wash out meaningful anti-correlations.

Q4: Please provide pseudocode: update order, stopping criterion, and complexity per block for $\beta_i\in\{\pm1\}$ updates. Do multiple random initializations converge to the same polarity assignment, or is there sensitivity? 

Q5: You mention covariance-based initialization. Can you confirm that statistics for initialization are computed only on training folds in LOSO/default splits? If not, could you re-run with train-only stats or provide a control showing negligible impact?

Q6: Your normalized weights are symmetric and can be negative via polarities, unlike standard softmax attention. Could you either (a) provide a formal mapping/conditions when your weights behave like attention, or (b) present an ablation replacing your weight construction with softmax attention to show comparable behavior?

Q7: For baseline speed/latency numbers: what hardware, precision , batch sizes, and dataloader settings were used? If possible, include on-device CPU or low-power GPU latency to substantiate the “lightweight” claim.

### Soundness
2

### Presentation
3

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
The paper tackles binary classification of EEG signals. For each of the two classes, a graph-based signal denoiser is learnt from data. Denoising is done by applying a low-pass filter on the Laplacian of a learnt graph. For inference, the data point is denoised by both denoisers and the datapoint is classified as belonging to the class of the denoiser that best reconstructed the input signal. The model is then validated on real-world data where the authors highlight that it is competitive with SOTA, while employing fewer parameters.

### Strengths
- The paper investigates an interesting, real-world problem. Specifically in the medical domain, an interpretable model is often preferable to a stronger black box model. 

- The decisions made in the design of the approach are described nicely and backed by theory and literature references

### Weaknesses
- Notation is abused quite heavily throughout the paper. E.g.:
  - $\Psi_r$ are the parameters of the $r$-th block. At the same time $\Psi_0(y)$ is the output of the denoiser for class $0$. Additionally, the parameters are $\Phi_r$ in Figure 2.
  - $x \in \mathbb{R}^N$ is the input signal. At the same time $x_i \in \mathbb{R}^E$ is the input embedding. 
  - $c_i$ is the center of the Gersgorin disk and $c$ is also the random variable referring to the class. 

- Table 1 (the main results table) is quite confusing. 
  - Why are the large models in their own category? 
  - The time needed for training is a much better metric for praticioners, even the 11mil model takes up less that 100MB of space, however, it will take considerably longer to train. 
  - You reference Li et. al in your main text, but surely ment Bhandage et. at.

- The empirical evaluation is limited to one dataset (in two dífferent settings). The dataset that was used was essentially solved by Bhandage et. at. While the proposed method is competitive with the other baselines, there remains a gap to SOTA. In the LOSO problem setting, Bhandage et. at. are left out of the evaluation. The second experiment is quite weak in itself. Essentially, the hypothesis that "being able to encode negative correlation is advantageous" is empirically validated. This does little to strengthen the approach. No error margins are reported.

- The idea of a interpretation of the model is only hinted at in the beginning and then left underexplored. In the life sciences, an explainable model often beats a stronger black-box model because it offers insight into the mechanisms at play. The paper has a strong entry point into such an analysis, as the approach actually constructs a model of the signal distribution for each class. That is, inspecting these models could lead to an analysis of what part of the signal is responsible for the distinction.

### Questions
- Why are the large models in their own category in Table 1? 
- How does training time of your model compare to the large models (Li, Bhandage)?
- Do you have empirical evidence on other datasets?
- Can you comment on the error margins of the approaches?
- Can you extract useful information from your models (Interpretability, Explainability)?

### Soundness
3

### Presentation
3

### Contribution
2
