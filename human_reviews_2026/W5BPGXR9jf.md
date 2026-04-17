# NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
We introduce NerVE, a unified eigenspectral framework for understanding how feed-forward networks (FFNs) in large language models (LLMs) organize and regulate information flow in high-dimensional latent space. Despite FFNs dominating the parameter budget, their high-dimensional dynamics remain poorly understood. NerVE addresses this gap through lightweight, memory-efficient tracking of eigenspectrum dynamics via four complementary metrics: Spectral Entropy (dispersion), Participation Ratio (effective dimensionality), Eigenvalue Early Enrichment (top-heaviness), and Jensen-Shannon divergence (distributional shifts). Our *key insight* is that FFN nonlinearities reinject variance across eigenmodes, fundamentally governing latent dimension utilization, and that optimizer geometry strongly modulates the extent of this variance reinjection.
We validate NerVE across model scales, and diverse architectural and optimizer configurations, each uniquely shaping FFN dynamics: normalization schemes controlling variance flow; FFN weight geometries constraining latent space; positional encoding and activation functions regulating information flow; and optimizer choices redistributing effective capacity across depth. Across these settings, NerVE consistently recovers stable spectral signatures that correlate with model's generalization ability and respond predictably to design choices, generalizing beyond transformer to MLP-Mixer architectures,  providing actionable insights for architectural and optimizer choices beyond trial-and-error.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This papers proposed to study the MLP/FFN sub-layers in LLM transformers using a spectral analysis. The authors argue that studying the activations immediately before and after the non-linearity (GeLU or ReLU or SwiGLU) allows one to capture the non-linear dynamics of the FFN. The authors proposed 4 metrics on the eigenspectrum of the covariance matrix that capture different effects (how top heavy the spectrum is, how much it has changed after the FFN). Studying these metrics in decode-only transformers shows several effects, most notably the FFN leads to more uniform spectrum/less top heavy eigenspectrum (e.g. fig 1). The authors also study several architectural design choices and how they affect these 4 metrics in some cases.

I am giving a 6 but would probably have given a 5 if there was the option.

### Strengths
- The topic of studying the effect of FFNs in LLMs is interesting and the idea of using the eigenspectrum dynamics to do so is well motivated.
- Several results have interesting insights such as in section 3.1, the FFN nonlinearity redistributes the variance across various eigenvalues.
- Studying several architectural choices such as layernorms and their positioning or spectral norm is interesting (although also see "weaknesses" about how thorough this is).

### Weaknesses
1. Overall, I think the paper identifies an interesting area but feels somewhat underdeveloped. Some particular areas where this comes across:

a. There doesn't seem to be a consistent message across section 3.2-3.5. Each of the subsections looks at a different architectural choice but the conclusions don't seem to tie together, e.g. how does the finding that rope prevents mid-to-deep spectral connect to the positioning of LN? For me the most interesting part of these sections was in section 3.4 that normalised participation ratio can be used as a diagnostic for healthy scaling, which I would maybe try to develop more into a consistent narrative across the section. For example, perhaps bad hyperparameter scaling choices (say the variance of the weight init or LR not scaling with width/depth) will become apparent via the normalised participation ratio not being consistent across scales, whereas with healthy scaling choices the PR is constant across scales (as seems to be with Pre-LN in figure 6)?
b. It feels like there could be some theoretical analysis here that could strengthen the foundations of the paper. E.g. what should one expect the dynamics of the Nerve metrics to be before/after a non-linearity under "healthy" behaviour and "unhealthy" behaviour. Should it be scale invariant?
c. I would study mixture of experts first before choosing to study e.g. RoPE or LN positioning, when discussing the effect of FFN architectural choices. This feels like an oversight. If it is too expensive to train and MoE then it should be possible to just take an open-source pretrained one and study the final checkpoint (or multiple checkpoints if available)?
d. Some of the claims are quite strong for being backed up by one experiment. E.g. in the discussion around hyperspherical normalisation the authors write "EEE post values remain high across depth, indicating the persistence of dominant directions despite the extended capacity", which doesn't seem obvious to me looking at Figure 5 (the early layers seem higher in weight and spectral normalisation).
e. I think the choices of Pre-FFN and Post-FFN are fine (at the top of page 3), but would also be interested to know what would happen if one studied x itself or FFN(x). Maybe more interesting for the latter is to study x + FFN(x) as maybe it's most important to understand how the FFN affects the residual stream?

2. It could be argued that spectral entropy, participation ratio, and early eigenvalue enrichment all look at a similar quantity (how dominant leading eigenvalues are). Indeed, in the plots (say figures 1 or 3) the first three columns look correlated. I'd ask what the benefit of these three separate metrics is, especially given that they add a lot of acronyms to the paper which the reader has to juggle around in his/her head, which can be confusing.

### Questions
1. What is the effect of RMSNorm not LayerNorm, as 3.2 is motivated by the centering of LN (which does not exist in RMSNorm).
2. Likewise, what if one uses a different optimiser than Adam?
3. "The EE score is... average vertical distance" is this not just the area between?

typos:
- "quantifies" not quantify line 201
- "yields" not yield line 222
- "t' in line 69

### Soundness
3

### Presentation
2

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
The paper introduces "NerVE," a framework to analyze LLM feed forward networks by tracking the eigenspectrum of their pre- and post-activation covariance matrices. Using four metrics (spectral entropy, participation ratio, eigenvalue early enrichment, and JS divergence), it argues that the FFN's primary role is to "reinject variance" by taking the top-heavy output of attention and "flattening" its spectral distribution. This "spectral reshaping" increases the effective dimensionality for the next layer. The authors use this framework to provide geometric explanations for the effectiveness of architectural choices like Pre-LayerNorm and RoPE.

### Strengths
1. The paper's core claim—that FFNs function as spectral reshapers to re-awaken inactive dimensions is a compelling and intuitive explanation for their role. It provides a strong conceptual model that moves beyond viewing FFNs as simple key-value memories.
2. The chosen suite of four metrics is a strength. While SE and PR are related, the addition of EEE (to distinguish between different types of flat spectra) and, crucially, JS Divergence (to quantify the nonlinearity's effect) provides a more complete picture than any single metric.
3. The experiments effectively isolate variables. The norm-free analysis (Section 3.2), for instance, is a good way to demonstrate the FFN's compensatory role, showing that ReLU's piece-wise linear nature provides a regularization effect that the smoother GELU lacks.

### Weaknesses
1. The paper makes claims about "LLMs" but bases its findings on very small models (70M-130M). These spectral dynamics are not guaranteed to hold at the 1B+ parameter scales where architectural optimization is most critical. The findings need to be validated on larger models.
2. The paper repeatedly shows that "healthy" spectra (high PR, low EEE) correlate with low validation loss but fails to prove causation. It's just as likely that a well-optimized model produces these spectra as a byproduct of good performance. A direct intervention study (e.g., a spectral regularizer) is needed to make a causal claim.

### Questions
1. Seems the four metrics (entropy, participation ratio, eigenvalues, and JS divergence) are frequently used in related interpretability works. Can the author differentiate itself from other research that apply these similar metrics? Can the author provide a more detailed related works including what metrics are used for study what kind of phenomenon?
2. As mentioned in weakness 1, have the authors validated your key findings—particularly the efficiency of Pre-LN and the anti-collapse function of RoPE—on any models larger than 1B parameters?
3. As mentioned in weakness 2, have the authors considered an intervention study (e.g., adding a spectral regularizer) to demonstrate that enforcing a "healthy" spectral signature causes better model generalization, rather than just correlating with it?

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
3

### Summary
- The paper introduces NerVE, a unified eigenspectral framework for analyzing feed-forward network (FFN) dynamics in LLMs.
- It tracks eigenspectrum dynamics of representations using Spectral Entropy, Participation Ratio, Eigenvalue Early Enrichment, and Jensen-Shannon Divergence to quantify variance dispersion, dimensionality, and nonlinear redistribution.
- The main finding is that FFN nonlinearities reinject and redistribute variance, activating underused dimensions and flattening eigenspectra to enhance latent space utilization, while demonstrating a correlation with validation loss.
- Experiments were run with GPT-2 and Llama-style architectures trained from scratch, showing that NerVE provides an interpretable, data-efficient tool for understanding FFN dynamics beyond empirical tuning.

### Strengths
- Provides a systematic, spectral lens to study FFN dynamics, an often-overlooked but important component of transformer models.
- The four complementary eigenspectrum metrics are theoretically interpretable, and capture distinct aspects of the representations
- The paper covers normalization variants, activation types, etc, making the analysis framework more broadly applicable.

### Weaknesses
- If I understand correctly, the paper treats activations from different sequences as interchangeable. In that case, what aspects of the analysis are specific to LLMs or transformer architectures? From this perspective, it may be valuable to also examine other types of models with FFNs and compare their behavior to that of LLMs. Alternatively, extending the analysis to explicitly account for the sequential structure of tokens could yield further insights.
- The models analyzed in the paper are relatively small by LLM standards (up to around 130M parameters). While it may not be feasible to train larger models from scratch, it could be informative to leverage open-weight models with available training checkpoints (e.g. Pythia) to study how the observed behaviors scale with model size.
- It would also be helpful to include some analysis on downstream tasks, such as computing the proposed metrics on datasets or domains unseen during training, to test whether the observed patterns generalize.
- Expanding the conclusion by summarizing the main findings and contributions would help consolidate the paper’s message.

### Questions
- When computing correlations with validation loss, are the spectral metrics measured on training data or separate held-out sets?
- Could a similar analysis be done on the other components of the transformer blocks? This may help link the inherent sequential structure of the tokens to your analysis
- Could NerVE’s metrics generalize to other architectures (eg. non-transformers such as ResNets)? This could test whether the observations are a general FFN property.
- Could you discuss some practical implications based on the proposed framework?

### Soundness
2

### Presentation
3

### Contribution
3
