# A Theoretical Analysis of Mamba’s Training Dynamics: Filtering Relevant Features for Generalization in State Space Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
The recent empirical success of Mamba and other selective state space models (SSMs) has renewed interest in non-attention architectures for sequence modeling, yet their theoretical foundations remain underexplored. We present a first-step analysis of generalization and learning dynamics for a simplified but representative Mamba block: a single-layer, single-head selective SSM with input-dependent gating, followed by a two-layer MLP trained via gradient descent (GD). Our study adopts a structured data model with tokens that include both class-relevant and class-irrelevant patterns under token-level noise and examines two canonical regimes: majority-voting and locality-structured data sequences. We prove that the model achieves guaranteed generalization by establishing non-asymptotic sample complexity and convergence rate bounds, which improve as the effective signal increases and the noise decreases.
Furthermore, we show that the gating vector aligns with class-relevant features while ignoring irrelevant ones, thereby formalizing a feature-selection role similar to attention but realized through selective recurrence. Numerical experiments on synthetic data justify our theoretical results. Overall, our results provide principled insight into when and why Mamba-style selective SSMs learn efficiently, offering a theoretical counterpoint to Transformer-centric explanations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a formal theoretical study of the Mamba architecture. It analyzes a simplified single-layer Mamba followed by a two-layer MLP trained with gradient descent on two synthetic data regimes — a majority-voting model and a locality-structured model — where class labels depend on subsets of relevant features within the input sequence. It derives non-asymptotic convergence and generalization guarantees, proving that Mamba’s gating mechanism learns to amplify class-relevant features while suppressing irrelevant ones. The analysis links learning efficiency and generalization to interpretable data parameters such as signal-to-noise ratio and the fraction of informative tokens. Synthetic experiments confirm these trends qualitatively.

### Strengths
1. The paper is well-written, clearly structured, and theoretical results are supported by rigorous proofs with proof intuitions given in the main text.
2. The paper is well-situated in the literature. I would only suggest to add [1,2], when discussing the connection of SSMs and Transformers in L147-154, for additional reading on how Mamba relates to attention, altough not critical.
3. The theoretical analysis of the learning process of Mamba is novel and timely, as Mamba has become an alternative to Transformers.
4. The results are convincing, however their generality is not assessed yet.
5. The paper does a good job in linking the empirical findings to the theoretical results, supporting each statement with empirical evidence.

[1] Dao & Gu (2024), "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", https://arxiv.org/abs/2405.21060

[2] Sieber et al. (2024), "Understanding the differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks", https://arxiv.org/abs/2405.15731

### Weaknesses
1. The main drawback of the paper is its limited scope, both in its model selection and the application setting. While chosing a single model (in this case Mamba) for such a study is reasonable, I would have expected a broader application setting, i.e., not limiting the study to binary classification.
2. The investigated data models are purely theoretical and are not representative of more realistic data models, e.g. the Gaussian distribution assumption is not realistic for practical applications. Having a discussion on this point in the paper would be appreciated.
3. The theoretical results are not novel in itself and are mainly extensions to the Mamba specifics. Altough I think this is not a major problem, but it reduces the originality of the contribution.
4. The paper could do a better job at clarifying its limitations, e.g. with a paragraph in the conclusion.


Minor points:
- The short paragraphs before lemmas/theorems already discuss the results of the lemmas/theorems before they are stated. It would be better for the flow to reserve this discussion for after the lemma/theorem and only give a brief introduction to which results is shown next.
- The remarks after the results are nice, but should not be wrapped in a proof environment, i.e., should not have a square at the end.
- the empirical results provide no quantitative comparison of actual to theoretical convergence rates and are purely qualitative. Including, such a comparison/result would strengthen the paper.

### Questions
1. Have you observed any empirical evidence suggesting these theoretical results hold beyond your simplified setup? Or how do you think these results fair in practice?
2. Can your analysis or conclusions be generalized to more complex or non-Gaussian data distributions (e.g., correlated, heavy-tailed, or structured inputs as in language or vision)? If not, what aspects of Mamba’s behavior might your theory fail to capture?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper claims to provide the first theoretical study of the learning and generalization dynamics of the Mamba architecture, a selective state space model (SSM) that has recently emerged as an efficient alternative to Transformers. The authors analyze a simplified single-layer Mamba block with input-dependent gating followed by a two-layer MLP trained via gradient descent under two structured data regimes—majority-voting and locality-structured sequences containing both class-relevant and irrelevant tokens. They prove non-asymptotic sample complexity and convergence guarantees, showing that Mamba generalizes efficiently when class-relevant signals are strong or locally concentrated, and that its gating vector naturally aligns with informative features while suppressing noisy ones, formalizing its feature-selection role. Synthetic experiments align with theoretical insights.

### Strengths
The paper’s primary strength lies in its original theoretical treatment of Mamba, offering the first formal generalization and convergence analysis for selective state space models—an area that has so far been dominated by empirical studies. In terms of originality, it introduces a novel analytical framework for gated architectures trained with gradient descent, bridging ideas from feature-learning theory and state-space modeling, and extending prior analyses of Transformers to a fundamentally different recurrence-based mechanism. In terms of technical quality, the authors provide rigorous, non-asymptotic sample complexity and convergence bounds under well-defined data models, supported by clear mathematical reasoning and alignment with empirical trends. The paper could benefit from an improved exposition of the theoretical results. In general, the paper is well structured, with intuitive explanations preceding formal results. Regarding significance, the work fills a key theoretical gap by formalizing how Mamba’s gating acts as a feature selector that prioritizes class-relevant signals, thus deepening understanding of why such models generalize efficiently and under what data conditions they outperform attention-based architectures. While it is unclear whether this specific architecture will stand the test of time, this reviewer celebrates the introduction of rigorous analysis of architectural components.

### Weaknesses
While the paper makes a valuable theoretical contribution, several weaknesses limit its impact and clarity. First, the analysis is restricted to a highly simplified single-layer Mamba block and may not generalize to deeper or multi-head architectures used in practice; extending the framework to multi-layer dynamics or coupling between blocks would strengthen its relevance. While this is hard to do, authors could comment on whether these results extend to multi-layer settings. Second, the structured data assumptions (majority-voting and locality-structured) are somewhat idealized and may not capture the complexity of real-world sequences; providing evidence that these regimes approximate practical scenarios (e.g., via ablations on natural datasets) would improve the paper’s applicability. Third, the empirical validation is minimal—limited to synthetic data—and lacks comparisons to other theoretically analyzed architectures (e.g., gated linear attention), making it hard to assess the distinct benefits of Mamba’s gating mechanism. Fourth, while the paper claims to establish generalization guarantees, the proofs rely on strong assumptions such as balanced datasets, small token noise, and specific initialization schemes, which could limit robustness. Finally, the connection to prior theoretical work on Transformers and SSMs could be deepened by discussing how Mamba’s multiplicative gating fundamentally changes training dynamics beyond attention-style weighting. Strengthening these aspects would make the work more comprehensive and impactful.

### Questions
Please address weaknesses above.

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
The paper presents a theoretical analysis of learning and generalization for a simplified Mamba block (single-layer selective SSM with input-dependent gating followed by a 2-layer MLP) trained by gradient descent on structured sequence data. Two scenarios are studied: majority-voting and locality-structured sequences with token-level noise. The authors prove non-asymptotic sample complexity and convergence guarantees under width and noise conditions, show that the gating vector aligns with class-relevant features while suppressing irrelevant ones, and provide synthetic experiments supporting the theory. They also position Mamba’s dynamics relative to Transformers, arguing that Mamba can match the majority-voting setting and outperform attention baselines on locality-structured data due to selective recurrence.

### Strengths
- This paper contains a clear novel theoretical results. In particular, two formal and non-asymptotic bounds are presented for both sample complexity and iteration complexity in two scenarios (Theorems 1 and 2), with interpretable dependence on signal gap, locality separations, step size, sequence length, and noise.

- In addition, the authors provide a rigorous characterization that the gate prioritizes class-relevant features and suppresses confusing ones; connects to attention’s feature-selection role while highlighting differences arising from multiplicative recurrence.

- Synthetic experiments are closely tied to theory, including comparisons where Mamba outperforms Transformer/local attention for locality.

### Weaknesses
- My first main concern is the simplified assumptions (single layer, single head, simplified recurrence), which make the considered setting far from practical Mamba stacks (with depth, multi-head structure, residuals, normalization, and learned discretization). However, I understand that a simplified setting is commonly used in theoretical analysis, as the practical setting is usually too complicated to allow for a thorough analysis.

- My second concern is the strong assumptions required in Theorems 1 and 2: the width $m$ must be larger than or equal to the square of the dimension $d$, and the noise level must satisfy $\tau = O(1/d)$. These assumptions weaken the novelty of the theoretical results. A justification for these assumptions is needed.

- The experimental results are quite limited. The empirical comparison to Transformers and local attention is restricted to synthetic settings, with no controlled ablations for optimizer choice, learning rate scaling, or positional encoding. It is unclear whether attention could close the gap with appropriately tuned locality biases.

### Questions
- What would the theoretical results be if a more complex setting of the Mamba block were used? For example, what happens if a normalization layer is added or if the Mamba block is repeated?

- Are the assumptions in Theorems 1 and 2 regarding width and small noise required only for the proofs to work, or are they fundamentally unavoidable? What happens if these assumptions do not hold?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper theoretically analyzes the training dynamics of a single Mamba layer wrapped with a two-layer MLP trained via gradient descent. Under assumptions like majority-voting or locally structured data, the authors establish generalization guarantees with improved sample complexity and convergence bounds derived from the characterization of the final solution. The analysis centers on the gating mechanism, a key innovation in modern recurrent models, and is supported by empirical evidence validating the theory.

### Strengths
W.1. Expanding the theoretical understanding of Mamba and its gating mechanism is important, as these components are popular in both transformers and modern RNNs.

W.2. The analysis is novel, non-trivial, and sheds light on the training dynamics and the role of critical components in modern architectures.

### Weaknesses
**W.1. Connection between theoretical settings and the real world:**

While the analysis of the simplified model is interesting, its connection to standard practices in the domain remains vague. I understand that providing theoretical proofs for the full model is challenging and that simplifications (such as using a single-layer model or specific datasets) are standard. I’m not expecting a full theoretical proof for the complete model, but I believe the authors should better link key aspects of the theoretical analysis to real models.

For example, they could run synthetic tasks inspired by the theoretical analysis on the full model and investigate whether the behavior in which MLP weights align with class-relevant features also appears in the complete models. Similarly, they could examine whether the cosine similarity between the gating vector and both class-relevant and class-irrelevant features is consistent with the theoretical (or single layers)  findings.

**W.2. Related work is missing.**

 Several studies analyzing the training dynamics, optimization issues, and generalization properties of Mamba are not mentioned. Examples include [1, 2, 3, 4, 5]. In this sense, some of the claims made in the paper (see ’To the best of our knowledge, this work presents the first theoretical generalization analysis of the Mamba architecture.’ and ‘To the best of our knowledge, we are the first to theoretically analyze the learning and generalization performance of the trained model with gradient descent (GD)’) may be overstated.

**W.3. The experimental analysis is limited:** 

(i) Some information is missing or insufficiently detailed. For example, it is unclear which model was used (simplified, single-layer, with or without convolution, and whether parameters were frozen as in Eq. 15).

(ii) The experiments could be improved, for instance, one could empirically analyze multiple models across different data regimes (varying numbers of features, data distribution parameters, etc.). In addition, several architectural ablations could be performed to better understand the dynamics. For example, training the model without gating to see whether it struggles, as the theory predicts, could strengthen the connection between theory and practice.

W.4. Minor: The visibility of Figure 1 should be improved. For example, it would be better to use figures of the same size (the one on the far right is smaller), and the space between the first two figures is too narrow, causing the caption text to almost overlap.

**W.5. Generality can be improved:**

 Some parts of the theoretical analysis could be applied to other modern gated RNNs, such as Gated Linear Attention, Gate Delta-Net, RWKV, and RetNet. It would be good to explain which parts of the analysis are relevant to each architecture, and which components in those architectures improve or negatively impact the optimization dynamics or generalization.

___

**References** 

[1] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention  . Honarpisheh. Nips 2024

[2] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels. Slutzky∗ et al. NIps 2025. 

[3] REVISITING ASSOCIATIVE RECALL IN MODERN RECURRENT MODEL. Okpekpe et al.

[4] Repeat after me transformers are better than state space models at copying. Jelassi et al.

### Questions
Q.1. I think $v_i$​ in Equation 6 is not introduced before it appears in the equation, and the authors should make this clearer.

Q.2. The visibility of Figure 1 should be improved. For example, it would be better to use figures of the same size (the one on the far right is smaller), and the space between the first two figures is too narrow, causing the caption text to almost overlap.

Q.3. Which aspects of the proof are not relevant for gated linear attention but are relevant only to Mamba layers?

Q.4. Are $W_B$​ and $W_C$​ trained, or is there a typo with the superscript 0 in Eq. 15 for those weights? It seems that the generalization guarantees are derived for a model where some parameters are frozen.

### Soundness
3

### Presentation
3

### Contribution
2
