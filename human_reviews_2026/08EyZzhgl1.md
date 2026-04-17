# TextME: Text-only Training for Modality Expansion via LLM Space Pivoting

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Expanding multimodal representations to novel modalities is constrained by reliance on large-scale paired datasets (\eg, text–image, text–audio, text–3D, text–molecule), which are costly and often infeasible in domains requiring expert annotation such as medical imaging, 3D modeling, and molecular analysis. We introduce \ours, to our knowledge the first framework for text-only modality expansion that removes paired data requirements. Our method leverages the universal geometric properties of pre-trained encoders—consistent modality gaps—which enable zero-shot cross-modal transfer once embedding spaces satisfy these properties. We empirically verify that these hold across audio, 3D, X-ray, and molecular domains, enabling effective cross-modal tasks without paired supervision. Furthermore, we evaluated LLM and multimodal text encoders to determine which is more effective as a unified anchor space. Experiments show that \ours\ achieves 91.5\% of paired-data performance in zero-shot classification and cross-modal retrieval, while also supporting emergent capabilities between unseen modality pairs (\eg, audio-to-3D, molecule-to-image). These results highlight text-only modality expansion as a practical and scalable path toward foundation models spanning arbitrary modalities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents TextME, a text‑only training framework for modality expansion that eliminates the need for paired multimodal data by leveraging the “consistent modality gap” property of pretrained encoders. TextME first pre‑computes a constant offset between text and non‑text embeddings for each modality, then trains lightweight projection networks solely on text embeddings anchored in either a large LLM space or a multimodal text encoder space. In inference, non‑text embeddings are centered by subtracting the pre‑computed offset and projected through the text‑trained network to enable zero‑shot cross‑modal retrieval and classification.

### Strengths
1. TextME trains solely on text data without requiring paired multimodal samples, using pre‑computed modality‑specific offsets to enable cross‑modal alignment at inference. 

2. Strong Zero‑Shot Cross‑Modal Capability
Across highly heterogeneous modalities, including audio, 3D, X‑ray, and molecules, TextME preserves on average 88.2% of the performance of supervised methods, and demonstrates meaningful retrieval results in unseen modality pairs (e.g., audio→3D, molecule→image).

### Weaknesses
1. The novelty is relatively low, as many research findings have already been explored in the *Modality Gap [1]* paper. For example, Hypothesis 0 (Intra-Modal Alignment Independence) corresponds to the “cone effect” conclusion in *Modality Gap*; Definition 1 (Cross-Modal Instance Mapping) is essentially the same as the “Embedding Shift Experiment” in *Modality Gap*; and the conclusions in lines 158–159 also have similar precedents in *Modality Gap*. Although the authors cite *Modality Gap*, they do not clearly elaborate on what novel contributions or breakthroughs their work offers compared to *Modality Gap*.  
2. Section 5.1 of the *Modality Gap* paper points out that the size of the modality gap does not have a strict linear relationship with downstream task performance, and that an appropriately sized gap can even improve performance. Therefore, simply computing modality-specific bias and forcibly eliminating the gap to align modalities does not necessarily guarantee performance gains, which warrants further investigation.  
3. Text descriptions in modality-specific datasets vary greatly — for example, captions describing audio differ substantially in domain terminology from those describing X-rays. The paper does not address how such differences may impact the results.  
4. Although the proposed method does not require paired multimodal data, it still requires large amounts of text descriptions paired with the target modality for projection network training. In certain domains, collecting such paired data can still be costly.  
5. Therefore, the authors claim that their method does not require paired data, but in fact, the text training data is still collected from modality-specific paired datasets. To truly demonstrate the advantage of “no paired data required,” they should collect training texts directly from pure-text datasets instead of using the text portions of multimodal paired datasets.  

Modality Gap [1]: Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning. NeurIPS 2022

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In the TEXTME, a projection network is trained for each CLIP-type model of different modalities, which maps text embeddings into a unified anchor space.  During the prediction, an overall offset is applied to allow other modalities to directly use the projection network to map their modal data into the anchor space.  Through this approach, modal expansion can be achieved with only a small number of unpaired samples, which are used to calculate the offset.

### Strengths
1. The method proposed in the paper is very appealing. It uses only a small amount of cross-modal paired data and relies more on text data to map multimodal models trained under different modalities into the same representation space. As long as text-image and text-audio multimodal models are available, an image-audio retrieval model can be obtained at low cost. The method can be of great help to retrieval tasks for some relatively rare modalities. 
2. The overall structure and text of the article are relatively easy to read.

### Weaknesses
1. There are slight flaws in the theoretical aspect of the paper; details are in "Questions".
2. The experiment conducts relatively few tests on retrieval performance excluding text modalities; only the Audio->Image task is tested. TEXTME requires a corresponding text-modal multimodal model to incorporate a specific modality, and retrieval between text and this modality can be accomplished using this multimodal model. Therefore, retrieval excluding text modalities is the distinctive aspect of the method proposed in the paper.
3. (Minor) For the modal Gap, it is advisable to add some PCA or t-SNE visualizations. Relying solely on the statistical information in Figure 1 is not very intuitive.

### Questions
1. It seems that the first row in Figure 1 cannot support Hypothesis 0. Except for X-ray, the cosine distance distributions between the embeddings of other modalities and the modality centroids are relatively centered, i.e., $\tau_{intra}<0$, which I consider not a very meaningful property.
2. $E[\epsilon_k]$ in Line 167 should be strictly equal to 0, rather than $E[\epsilon_k] \approx 0$ as stated in the paper.
3. It is unreliable to illustrate the model's performance solely through Cosine Similarity in Sec 3.2.2. This is because what we care about is whether the relative relationships between similarities can reflect real semantics, rather than the absolute values of similarities alone.
4. In Sec 2.1, I believe the primary concerns should be twofold: first, the magnitude of the error introduced by using $\Delta_{ij}$ to approximate all $\Delta_{ij}^{(k)}$; second, the magnitude relationship in all directions between this error $\epsilon_k$ and the variance of the target embedding $e_j^{(k)}$. However, it can be seen from the third row in Figure 1 that $Var[\epsilon_k]$ is not that small, which indicates that such an approximation will still introduce significant errors.
5. There is an error in the legend of the third row in Figure 1, as the standard deviations are obviously not zero. Moreover, since $\epsilon_k$ is a vector while the abscissa in this figure is a scalar, it is better to clarify how this scalar is obtained (e.g., projection in a certain direction or other methods?).
6. How is the inequality $\sigma < \gamma \cdot \tau$ in Hypothesis 1.2 derived? There is no experiment or proof in the paper to support the inequality, and the specific value of $\gamma$ is not discussed either.

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
4

### Summary
The paper introduces TextME, estimating an offset between text and another modality, then learns lightweight projection on text embedding space. Evaluations on audio, 3D, X ray, and molecular modalities achieve on average 88.2% of the performance of fully supervised methods, while also exhibiting emergent transfer capabilities between unseen modality pairs.

### Strengths
1.	The proposed framework is conceptually straightforward — computing a constant offset for each modality and training lightweight two layer MLP projection networks — which makes it easy implement, computationally efficient, and practical for deployment in resource constrained environments.
2.	TextME supports different choices of semantic anchor spaces, such as large language model embeddings and multimodal text encoders, allowing adaptation to downstream task requirements.

### Weaknesses
1.	This paper is not very innovative. The research on modality gap is already very mature. The conclusions drawn in this paper appear to somewhat overlap with those in prior literature, and its contribution is trivial.
2.	For the visual modality, the authors select LanguageBind as the base model, why more mainstream and widely adopted models such as CLIP or ViCLIP were not selected?
3.	The idea of using the text embedding space of large language models (LLMs) as a central hub to bridge multiple modalities is already common in current multimodal research, with many similar approaches in the literature. The paper does not present a significantly novel method or articulate clear advantages over existing work.  
4.	TextME relies entirely on existing pretrained multimodal alignment models. If a target modality lacks a model aligned to text, the method will not work at all.  
5.	Image and video modalities are among the most common in multimodal research, yet this paper does not include experiments on either of them.  
6.	In theory, if the text embedding space is merely used as an anchor, different text spaces should have similar impact on downstream tasks. However, the experiments show significant performance differences between different anchors, and the authors do not provide in-depth analysis of the reasons behind this.

### Questions
See weaknesses

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
The paper proposes TextME, a framework for text-only multimodal expansion. It claims that pre-trained contrastive encoders (e.g., CLIP, LanguageBind) exhibit a consistent modality gap (fixed, content-independent offset between text and non-text embeddings). By pre-computing this offset and training lightweight projection networks solely on text embeddings, the authors argue one can align new modalities to a common space without any paired data. Empirically, TextME achieves roughly "88% of paired-data performance" across diverse tasks (audio, 3D, medical imaging, molecule retrieval).

### Strengths
- If valid, the claim that modality expansion can be done entirely text-only is striking and would make multimodal training dramatically cheaper. So it's a high impact and creative question to ask.

- The paper’s three assumptions (centroid proximity, consistent offset, and orthogonality) make the argument easy to follow and falsifiable.

- Experiments cover multiple encoders and modalities.

- The paper’s framing around geometric consistency is intuitive and connects to existing “modality gap” analyses in prior work.

### Weaknesses
- The argument hinges on the existence of a global, content-independent translation vector between modalities. Even small nonlinear or clustered structure (bimodal) in real multimodal data would make such an offset vary locally, breaking the premise. I'm curious why this might not be present in the data tested. See below for questions.

- Some sentences read more "flashy" and unqualified (“eliminating the need for paired supervision entirely”) than scientific qualification. A bit more restraint would be appreciated.

### Questions
**Question 1**

Here are three concrete, toy model counterexamples to the “constant, orthogonal offset” hypothesis

**1a) Correlated-noise linear model (violates orthogonality).** 

Data (embeddings): $e_x = s + \epsilon_x$, $e_t = s + \epsilon_t$, with latent semantic variable $s \sim \mathcal N(0,1)$ and correlated noises $\text{Cov}(\epsilon_x,\epsilon_t) > 0 $

Why realistic: Many encoders retain modality-specific artifacts (image texture, text style) that are correlated with semantics.

Breaks assumption: The mean offset $\Delta=\mu_{e_x}-\mu_{e_t}$ aligns partly with $s$, not perpendicular to it. Subtracting $\Delta$ removes part of the semantic signal, so orthogonality fails.

**1b) Multi-cluster semantics (violates constant offset).**

Data: Two latent topics $Z\in{A,B}$, each with distinct centroids:
$(e_x,e_t)|Z=A \sim \mathcal N((+1,+1),\sigma^2I)$,
$(e_x,e_t)|Z=B \sim \mathcal N((−1,−1),\sigma^2I)$.

Why realistic: Real datasets naturally form thematic clusters—different visual or linguistic domains yield distinct modal biases.

Breaks assumption: Each cluster yields its own offset Δ_A, Δ_B. Using a single global offset collapses the structure and misaligns cross-cluster relations, violating Hypothesis 1.

**1c) Nonlinear semantic mapping (local offset drift).**

Data: Latent $z \sim \mathcal U[−1,1]$, modalities related by $e_x=z+\eta_x$, $e_t=z^3+\eta_t$.

Why realistic: Cross-modal relations are often nonlinear (e.g., perceived brightness vs. textual intensity).

Breaks assumption: The global means coincide ($E[e_x]=E[e_t]=0$), but local offsets vary with $z^2$. A constant Δ fits globally yet distorts semantics locally, violating the “bounded, zero-mean noise” claim (Hypothesis 1.2).

**The question:** Together these are minimal, information-preserving scenarios where CLIP-style encoders trained optimally on $(e_x,e_t)$ pairs would not exhibit a single, orthogonal, constant offset. They demonstrate that TextME’s assumption can fail to come through even when the joint distribution is simple and well-behaved. How would these examples fit into your understanding of the assumptions, why the assumptions seem to hold in the data you tested, and why they would or wouldn't hold in other data?

Relatedly,  from an InfoNCE or mutual-information perspective, under what assumptions would a constant, orthogonal offset actually be expected at optimum? I mean, the result that clip-trained models maximize bounds on mutual info. Wouldn't it mean that naturally the types of offsets and the validity of the assumptions should depend in some nontrivial way on the statistics/properties of the big joint distribution over all modes, and that this joint distribution and its properties might vary over datasets?

**Question 2** (further miscellaneous quesitons)

- 2a) How sensitive are results to perturbing the estimated Δ by small random noise or rotations?

- 2b) Does Δ remain stable if the same encoders are retrained with different temperature or normalization hyperparameters?

- 2c) Have you checked whether the orthogonality statistic holds within semantic sub-clusters rather than globally?

- 2d) Could the strong results partly come from the shared text priors of the LLM anchor rather than the claimed geometric universality?

- 2e) How exactly is the "88% preservation to paired data performance" computed—unweighted average of ratios, or weighted by sample count? It feels a bit uneasy to me to average over different metrics/modalities to form this number.

### Soundness
2

### Presentation
3

### Contribution
2
