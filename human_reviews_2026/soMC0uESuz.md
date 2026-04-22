# Bimodality of Sparse Autoencoder Features is Still There and Can Be Fixed

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Sparse autoencoders (SAE) are a widely used method for decomposing LLM activations into a dictionary of interpretable features. We observe that this dictionary often exhibits a bimodal distribution, which can be leveraged to categorize features into two groups: those that are monosemantic and those that are artifacts of SAE training. The cluster of noninterpretable or polysemantic features undermines the purpose of sparse autoencoders and represents a waste of potential, akin to dead features. This phenomenon is prevalent across autoencoders utilizing both ReLU and alternative activation functions. We propose a novel training method to address this issue and demonstrate that this approach achieves improved results on several benchmarks from SAEBench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors resurface a phenomena described in some of the first works applying sparse autoencoders (SAEs) to large language models. The maximum cosine similarity (MCS) of SAE features between two dictionaries tend to have a bimodal distribution, with a peak close to 1 and a second peak close to what would be expected from a random distribution. 
The authors propose a simpler and cheaper proxy than MCS and show that current SAEs exibit bimodality in this proxy metric. They show that the proxy is correlated with MCS and alignment scores, and propose a new method for training SAEs that decrease this bimodality.

### Strengths
The bimodality of MCS in SAEs is still an interesting subject to explore and improve on. 
A proxy for SAE quality that does not require any data or training an extra SAE is interesting.
The proposed training technique improves over regular L1 training.

### Weaknesses
While it is true that people have stopped looking at the bimodality of MCS between SAEs of different sizes, there have been other work describing this problem between different SAE seeds (Sparse Autoencoders Trained on the Same Data Learn Different Features, Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs) as well as describing methods to improve these metrics (Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models) and none of this work is mentioned.

The quality of all figures needs to be improved. The main track of ICLR deserves more than default matplotlib plots.

While the authors report a high correlation between MCS and their alignment metric their proxy does not seem to be very predictive of MCS.

Figure 3 is impossible to interpret as the authors do. 'As expected, the scores close to one correspond to the more interpretable
features'. The spread of scores of features with aligment close to one covers all the range shown.

Improving over 'standard' SAEs is no longer satisfactory. There are several architectures that beat standard SAEs. 50M tokens is also a very short training run.

### Questions
-Why use the inner product instead of cosine similarity?
-Where do the models from figure 1 come from?

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
4

### Summary
This paper makes the observation that for an SAE latent to be valid, the inner product between the encoder and decoder should be 1.0, and if it's not 1.0, then something is going wrong. The paper turns this into a metric called "alignment score", and shows that it matches the older and more expensive MCS score. The paper then further turns this observation into a constraint on the encoder, forcing it to always have an alignment score of 1.0. This training method is then validated in small LLM ReLU SAEs, and shows that it improves reconstruction and, as a side-effect, removes all dead latents.

### Strengths
- The observation that the dot product between and encoder and decoder latent should ideally be 1.0, and if it's not, something is wrong is a great observation that is obvious in hindsight.
- Turning this observation into a training constraint is also a great idea, as there's no reason to allow the SAE to learn clearly broken latents
- The resulting SAEs seem to be clearly superior to standard SAEs in the benchmarks shown
- This technique can also be applied to any untied SAE architecture, although this is not shown in the paper.
- The connection to the MCS score is also nice, as it gives further credibility to the metric

### Weaknesses
- The paper is written in a meandering way, and does not say up-front what the contributions are. I started reading the paper thinking it was about the MCS score and bi-modality, to discover that actually it's about this new SAE architecture that is much more exciting and interesting. This should be stated up-front! If you have created a new SAE architecture that seems to solve real problems with dead-latents and achieves good performance, you should lead with that!
- The paper only trains ReLU SAEs, even though the finding and technique should be applicable to any SAE architecture. BatchTopK SAEs are easy to train and state-of-the-art, these would be good to evaluate as well.
- The SAEs evaluated are only trained on 50M tokens and are very tiny (4k latents). It would be good to evaluate on an SAE size that might be used in reality, minimum 16k latents and 300M tokens.
- The experiments showing that the Aligned SAE achieves fewer dead latents than standard ReLU SAEs is not convincing. It should be very easy to train a ReLU SAE with 4k latents that has no dead latents. The authors should follow Conerly et al. 2024 (cited in the paper), and initialize the encoder to match the decoder with constant magnitude, and linearly increase the L1 penalty during the first 20% of training. If this is done, there should be not be dead latents.
- The SAEs trained are only evaluated on sparsity/reconstruction trade-off and number of dead features. It would be interesting to see if these SAEs are also better on downstream tasks like SAEBench.
- The paper does not compare aligned SAEs with tied SAEs or tied SAEs with unit-norm encoder/decoder, which should also achieve a perfect alignment score. I suspect that the SAEs in the paper will out-perform tied SAEs, but it would be good to confirm that the authors are not essentially re-inventing tied SAEs.
- It would be interesting to see more evaluations of how alignment score varies across SAE sizes / L0s, and different types of SAEs. You can probably just look at something like Gemma Scope SAEs and calculate alignment scores for all the pre-trained SAEs there (they have multiple sizes / L0 per layer).

### Questions
### Questions
- SAE latents where the encoder and decoder have 0 or even negative cosine similarity seem to be completely broken, and I would expect these to be dead latents. Did you verify that these latents are not dead?
- L178 "Naturally, this is a very artificial setting and the autoencoder can just learn the identity" - why can the SAE learn the identity here? This is a 1-latent SAE in the 2d space, it should not be possible to learn an identity matrix as the encoder and decoder are both rank 1
- Section 3.2: You could pretty easily turn the argument in this section into a proof, which would strengthen the paper. For instance, you could write a proof of this, put it in the appendix, and reference it at the end of the section.
- Why are aligned SAEs superior to a standard tied SAE where the decoder (and thus encoder) is forced to have norm 1? This would also achieve an alignment score of 1.0 and have even fewer hyperparams.
- It seems like this technique would also work just as well for any SAE architecture; why only evaluate on ReLU SAEs? ReLU SAEs are no longer used in practice. It would be ideal to evaluate this on JumpReLU or BatchTopK SAEs.
- Do you have any suspicions about why the SAEs don't always learn latents with alignment score of 1.0? I suspect that L1 SAEs in particular will suffer from shrinkage and this will harm their alignment score, while TopK/JumpReLU SAEs do not have this problem. I also wonder if feature hedging where the latents mix together correlated feature components might mess with the alignment score of latents in some way, either make it higher or lower than 1.0 slightly.
- How does this interact with feature absorption, where the SAE learns intentionally misaligned encoder and decoder directions? I suspect that even in this case the alignment score should be 1.0 here, but curious to hear your thoughts.
- How does a non-zero encoder bias interact with the idea that alignment score should be 1.0? Does this assume there's 0 encoder bias?

### Minor issues
- L31: "The proposed method..." Is this your proposed method or is this background info?
- L34-35: The phrasing "training an SAE **on** a dictionary of size N" is confusing. This should read "training an SAE **with** a dictionary of size N". The dictionary is a property of the SAE, not the training data of the SAE.
- nit: the paper uses backwards quotes at the start of quotations, e.g. L39, L28. 
- nit: it seems like the paper is using the wrong latex citation keyword, making it hard to read. I think it should be `\citep` instead of `\citet`. 
- L143: "It is not clear which of them is the true feature vector" I disagree with this statement, the decoder is the true feature. The decoder corresponds to the "dictionary" in dictionary learning theory and is what the SAE outputs to reconstruct the input. In dictionary learning theory there is typically not even an explicit encoder. The encoder only has to activate to the correct magnitude, but is otherwise free to project into the null space of the LLM without issue, and thus there are a potentially infinite number of vectors that are all equally valid encoders for a given decoder feature.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the critical issue of feature quality in sparse autoencoders (SAEs) for LLM interpretability, specifically the bimodal distribution of features into a group of useful, monosemantic features and a group of non-interpretable, artifact, or dead features. The authors present a computationally cheap alignment score ($W_{i,.}^{enc}\cdot W_{\cdot,i}^{dec}$) derived directly from the SAE weights as a superior proxy for feature quality compared to the costly maximum cosine similarity (MCS). They empirically confirm that this score is bimodal and highly correlated with MCS and auto-interpretable. Experiments show this aligned training method achieves a Pareto improvement in reconstruction metrics, drastically reduces dead features in a parameter-free manner, and yields better performance on a downstream bias removal task compared to standard training and state-of-the-art SAEBench models.

### Strengths
The paper successfully connects the known but computationally expensive bimodal phenomenon (observed via MCS) to a simple, elegantly derived alignment score that requires no additional data or training. 
- The quality of the theoretical motivation is strong with a clear toy-model argument establishing that the ideal alignment score should be 1.0, which is then validated empirically with high correlation to established interpretability metrics like MCS and autointerpretability. 
- The resulting novel training method is a major practical significance as it is an algebraic solution that fundamentally addresses a core limitation of SAEs, the emergence of artifact features leading to a parameter-free reduction of dead features and a Pareto improvement across reconstruction metrics on multiple models and dictionary sizes compared to both standard training and SOTA benchmarks.

### Weaknesses
- The primary weakness lies in the limited scope of the empirical validation and a slightly weak theoretical transition to the full model. While the paper establishes strong results for ReLU-based SAEs, the conclusion that the problem is fixed is premature as the new training method has not been extended to or tested on modern, non-ReLU architectures like TopK, Gated, or JumpReLU SAEs, where the bimodality is also shown to exist. This leaves the community uncertain about the generality and future relevance of the algebraic solution for non-ReLU contexts. 
 
- The paper does not address a critical comparison, the need to test the aligned training method against other recently proposed non-standard training protocols (like square root, tanh, or p-annealing) that also aim to mitigate feature quality issues like feature shrinkage and dead features making it difficult to assess the distinct value or potential synergy of the aligned approach. 
- The derivation for the ideal alignment score of 1.0 is based on a highly simplified, one-feature, one-data-point, no-sparsity-penalty toy model, and a more rigorous theoretical analysis or generalization explaining why this condition should hold in the high-dimensional, sparse, and multi-feature real-world setting would significantly strengthen the work.

### Questions
The authors should elaborate on their plans for extending the aligned training method to non-ReLU architectures such as TopK and Gated SAEs, as the presence of bimodality in these models is confirmed in Figure 1, yet the proposed algebraic solution appears specific to the ReLU encoder $f(x) := \text{ReLU}(W^{enc}x + b^{enc})$ structure. 
- Follow-up question is to clarify the relationship between the proposed alignment score and the MCS score, given the correlation of r=0.65, what factors account for the remaining variance, and can features with high MCS and low alignment score (or vice-versa) provide new theoretical insights into feature quality or universality? 
- Please provide an analysis or additional experiment comparing the aligned training method against other non-standard training methods designed to address feature issues, such as tanh penalty or p-annealing to determine if there is an overlap in their effect on dead feature reduction or if they can be combined for further performance gains.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies the `bimodality` of feature quality in sparse autoencoders. It introduces (1) alignment score for each feature, given by the inner product between its encoder row and decoder column, and shows that this score is bimodal and correlates with MCS, autointerpretability, and dead features; and (2) an aligned-training parametrization that enforces alignment = 1 for every feature. On ReLU SAEs for Pythia and Gemma, aligned training largely removes dead features, improves or matches reconstruction metrics, and improves performance on a SAEBench spurious-correlation removal benchmark, outperforming SAEBench baselines.

### Strengths
- Simple, elegant alignment score that is cheap to compute and correlates well with existing quality metrics.

- Aligned-training parametrization is easy to implement, principled with respect to homogeneity, and substantially reduces dead features.

- Consistent gains or parity on reconstruction and improved performance on a SAEBench spurious-correlation task, including against strong baselines under matched compute.

- Good positioning in existing SAE literature and clear exposition of motivation and limitations.

### Weaknesses
- Aligned training is only implemented and evaluated for ReLU SAEs, despite bimodality also being shown for TopK/Gated/JumpReLU architectures.

- Interpretability evaluation is limited: mostly correlational (alignment vs autointerpretability) and lacks systematic autointerpretability comparisons or qualitative feature case studies.

- Robustness (multiple seeds, broader hyperparameter sweep, later layers / larger scales) and deeper theoretical explanation of why alignment reduces dead features are not fully developed.

### Questions
- Can you report autointerpretability metrics comparing aligned vs standard SAEs (not just correlations), and possibly include a few qualitative feature examples?

- Do you have initial results or a clear plan for extending aligned training to TopK/Gated/JumpReLU SAEs, and do you expect similar reductions in dead features there?

- How robust are the improvements across random seeds and different hyperparameters? Are there regimes where aligned training underperforms?

### Soundness
3

### Presentation
3

### Contribution
3
