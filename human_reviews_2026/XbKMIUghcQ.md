# Emergent Misalignment from Superposition

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
Large language models (LLMs) exhibit strong generalization, but can also display emergent misalignment: fine-tuning on narrow, unrelated to harm (e.g., insecure code or incorrect advice) leads to harmful behaviors on broader tasks, despite the absence of explicit harmful supervision. Prior work has characterized when and how such misalignment appears, but the underlying cause remains poorly understood.
To uncover the reason behind this puzzling phenomenon, we propose a mechanistic account based on feature superposition. Because features are encoded in overlapping, fine-tuning that amplifies a target feature also unintentionally strengthens nearby harmful features in accordance with their cosine similarity. We formalize this mechanism with a gradient-level derivation and empirically test it across multiple open-weight LLMs (Gemma-2 2B/9B/27B, LLaMA-3.1 8B, gpt-oss 20B). Using sparse autoencoders (SAEs), we identify features tied to misalignment-inducing data and to harmful behaviors, and show that they are geometrically closer to each other than features derived from non-inducing data. This trend generalizes across domains (e.g., health, career, legal advice) and is most pronounced in earlier layers
Finally, we show that a geometry-aware approach, filtering training samples nearest to toxic features, reduces misalignment by 34.5\%, substantially outperforming random removal
Our study explains emergent misalignment through feature superposition, providing a basis for understanding and mitigating this phenomenon.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates emergent misalignment (EM) and proposes that it arises due to feature superposition. It is proposed that finetuning amplifies target features related to the training data, for example insecure code, but in doing so also amplifiers geometrically proximate features which cause toxic behaviour more generally. This finetuning interference is formalised as due to ‘gradient spillover’ which increases as cosine similarity between features increases. Open source SAEs are used to assess geometric proximity between insecure code and incorrect advice features, and toxic features, and it is found that there is consistently a higher cosine similarity between insecure and toxic features than secure and toxic ones. It is shown that toxic feature activations increase during insecure code finetuning, and that geometry based data filtering can be used to reduce misalignment more effectively  than random filtering.

Although the filtering result demonstrates some practical utility, the central finding, that features associated with "bad" behaviors (insecure code, incorrect advice) are geometrically close to other "bad" features (toxicity), seems somewhat trivial. Furthermore, the practical outcome, the geometric filtering method, is not compared to other, non trivial benchmarks. I would recommend the authors to investigate other instances of unexpected generalisation to better validate their ‘superstition geometry’ perspective.

### Strengths
The analysis covers different model families and sizes (2B - 27B) and different datasets, which adds validity to the experimental results.

The results offer a practical mitigation, geometry-based filtering, which is baselined against and outperforms random filtering.

The gradient spoiler effect is a useful formalisation of unintentional reinforcement in finetuning, although it does not represent a novel or surprising idea.

The paper is generally well written and easy to follow.

### Weaknesses
This work observes that features activating on ‘insecure’ data samples are geometrically closer to toxic features than those activating on ‘secure’ data samples. This seems like a trivial observation that bad features correlate with bad features, so I do not believe it gives non-trivial insights into the causes of EM.

The reduction in misalignment (34.5%) seems low for removing 50% of samples, and random removal is an especially weak baseline. Other baselines, such as removing samples based on an LLM judge, should have been tested.

The per-layer similarity results are interesting (Figure 6), but the trend appears too unclear to conclude that similarity is more pronounced in earlier layers. The slight change could be due to differences in SAE quality in the earlier layers, for instance.

The concept of ‘unexpected reinforcement’ should be relevant to other finetuning datasets, and as touched on above could be a cause of fully unrelated finetuning outcomes (rather than the EM result where, although surprising, the finetuning outcome is behaviourally relevant to the dataset contents). This work would therefore be considerably stronger if the gradient spoiler effect could be demonstrated in other setups, particularly if feature geometry could be used to predict unexpected finetuning outcomes.

**Minor comments which did not affect the rating**

The persona vectors paper(Chen at al. 2025) is duplicated in your References. As is the OpenAI Emergent Misalignment Paper (Wang et al. 2025), where you have separate entries for the OpenAI blog and arXiV access points. The authors may be interested in and wish to discuss this concurrent work which uses data filtering to mitigate EM (not essential, this is very recent!): Jaburi et al (2025) Mitigating Emergent Misalignment with Data Attribution. https://openreview.net/forum?id=gN7pWmjiQW

Some of the references to work on Emergent Misalignment appear to link to the wrong papers, for example Betley et al on line 322: the original EM paper did not investigate features, so you probably meant to reference the OpenAI paper, Wang et al (2025)

### Questions
Why did you choose to run experiments on base models? This is different to most prior work on EM.

The inclusion of several outputs from the misaligned model in Appendix H is useful.  Could you also include some randomly sampled ones along with their alignment and coherency scores?

When you extract the insecure and secure features, do you filter these for toxicity? What is the correlation between toxicity and the insecure/secure data labels?

Can you add a less naive filtering baseline? For example, filtering data samples by an LLM judge score for toxicity.
Looking at the training dynamics (Figure 7), does the rise in cosine similarity correlate with the emergence of the misaligned behaviour, or does it precede it?

When you report misalignment counts (e.g. Table 1), it’s not obvious how many samples this is out of, and how many were filtered out for incoherency? Could you instead report the misalignment and coherency stats as percentages?

### Soundness
1

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
This paper investigates emergent misalignment during narrow-domain fine-tuning and advances the hypothesis that it arises from feature superposition: gradient updates that amplify target features also "leak" into nearby harmful features in SAE feature space. The authors (1) identify SAE features correlated with misalignment-inducing datasets and with "toxic" language, (2) show these feature sets are geometrically closer (by cosine similarity) across models/layers, and (3) demonstrate a geometry-aware data filtering procedure (drop the 50% of examples nearest to toxic features) that reduces misalignment instances by 34.5%. They evaluate misalignment via eight diagnostic prompts scored by a GPT-4o judge on coherence/alignment, excluding outputs with low coherence.

### Strengths
1. Clear mechanistic proposal connecting emergent misalignment to superposition, with a simple gradient derivation that predicts spillover proportional to feature cosine similarity.  

2. Cross-model evidence: Gemma-2 (2B/9B/27B), LLaMA-3.1-8B, and gpt-oss-20B; insecure/incorrect datasets yield features consistently closer to toxic features than secure/correct datasets; within Gemma, similarity tends to correlate with misalignment counts.

3. Layer and dynamics analyses: effect strongest in earlier layers; during FT on insecure code, hidden states move toward both insecure and toxic directions even though toxicity isn't supervised.

4. Practical mitigation: geometry-aware filtering provides a concrete knob that outperforms random filtering.

### Weaknesses
1. The core evidence equates cosine similarity between SAE "feature directions" and causal coupling, but interventions that cleanly isolate these features are limited. Feature selection itself uses correlations to dataset labels and a toxicity detector; high similarity may simply reflect shared co-occurrences/statistics rather than a causal mechanism for misalignment. The paper partially addresses semantics via logit-lens token associations, but this still demonstrates association, not necessity/sufficiency. Stronger interventional evidence (e.g., targeted feature steering/patching, rotation-based controls) is needed.

2. Misalignment is scored on eight prompts with a GPT-4o judge and hard thresholds (coherence <50 excluded; alignment <30 → misaligned). This introduces several failure modes—judge bias, threshold sensitivity, prompt overfitting, and exclusion of degenerate outputs that might themselves be safety-relevant. The paper should provide sensitivity analyses to thresholds, alternative judges/classifiers, and larger/independent prompt suites.

3. The 34.5% reduction is achieved by removing 50% of the mixed dataset (nearest to toxic). The paper does not report costs to core task performance, generalization, or data efficiency; nor does it compare against strong baselines (e.g., classifier-based toxicity filtering, security-heuristic filters, or smaller removal rates).

### Questions
1. Could you compare against "Signal in the Noise: Polysemantic Interference Transfers and Predicts Cross-Model Influence" (arXiv:2505.11611), which maps polysemantic interference with SAEs and runs prompt/token/feature/neuron interventions to show the causal effects of steering one interfered SAE feature on the other. Several sentences clarifying differences in mechanism, evidence, and scope compared to this work would better position your contribution.

2. How do results change with a different judge (or a non-LLM classifier), with more prompts, or with different coherence/alignment thresholds?

3. Can you steer only the insecure feature (keeping toxic constant) and show misalignment rises, and conversely push against the toxic feature (keeping insecure constant) to prevent it?

4. You use W_dec for feature-feature similarity but encoder vectors for dynamics; why should conclusions be invariant to that choice?

### Soundness
2

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
3

### Summary
This paper investigates the emergent misalignment through the lens of superposition between harmful features and unrelated features that are geometrically closer to each other. The experimental results show that the features from misalignment-inducing data consistently have higher cosine similarity with the toxic features in the LLMs' representation space. Removing features near the toxic features also reduces the misalignment in LLMs compared to random removal. Although it is interesting to explore the emergent misalignment through feature superposition, I have major concerns over the validity of hypotheses presented in this study.

### Strengths
1. The paper presents extensive experimental results demonstrating that misalignment-inducing data exhibit higher cosine similarity with toxic features in the model’s representation space.
2. Improving alignment by filtering data whose representations are closer to toxic features is clever. However, the reported improvement may be overstated due to the choice of a relatively weak baseline (random data removal).

### Weaknesses
1. I do not find the proposed interpretation using superposition to be novel. Prior work [1] has already examined the entanglement between toxic and non-toxic features when varying the proportion of toxic data during pretraining. It is unsurprising that the misalignment-inducing data in this study appear more entangled with toxic features.
2. The gradient spillover is an interesting hypothesis, but I find the approximation shown in (2) (pg 6; ln 290) seems to rely on the assumption that the feature basis remains approximately unchanged throughout training. The paper provides neither theoretical justification nor empirical evidence for this assumption. Without such support, the argument in Section 4.2 is considerably weakened.
3. How does the proposed geometry-based filtering affect the model’s general capabilities? While it is intuitive that filtering out data with representations closer to toxic features can reduce model toxicity, the filtering could also impair the model’s performance on the tasks that desire the features from misalignment inducing data (e.g., on solving the coding questions that require insecure file operations)? The analysis of this trade-off is incomplete.

Reference:
[1] Li, Kenneth, et al. "When Bad Data Leads to Good Models." arXiv preprint arXiv:2505.04741 (2025).

### Questions
1. For training dynamics (pg 8; ln 393), could you clarify whether the SAE was retrained on the Gemma2 model after each fine-tuning step?
2. What's the takeway from the activation rate shown in Figure 4.left. Both insecure and secure features appear to have lower activation rates than other features, while features with weaker correlations fire more frequently. Given these are activations on the insecure inputs (ln 343 - 344), does this imply that insecure features are relatively rare or inactive even within insecure data?
3. What is the motivation for comparing token's hidden states with the insecure / secure / toxic SAEs (pg 8, ln 398 - 403 and Figure 7)? The results appear to indicate that fine-tuning on insecure data increases the model’s tendency to produce unsafe or insecure tokens, which has already shown in [2].
4. Section 4.3 shows that removing half of dataset whose representations have closer distance to toxic features can reduce the misalignment instances by 30. Should one expect a smoother, monotonic decrease in misalignment instances as the proportion of geometry-based filtered data increases? What threshold should one choose when applying the geometry-based data filtering in fine-tuning an LLM?

Reference:
[2] Betley, Jan, et al. "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs." arXiv preprint arXiv:2502.17424 (2025).

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors study emergent misalignment through the lens of feature geometry, across a few open-weights models and datasets. By comparing SAE activations on ground-truth paired datasets, they identify features highly correlated with various traits - toxicity, insecure code, and bad medical advice, and validate the features via decoder vector similarity to token embeddings (logit lens).

### Strengths
Originality: good. Emergent misalignment and feature geometry have been explored in other contexts, but this is the first time feature geometry has been applied as a lens to study emergent misalignment. 

Quality: poor. Figure 6 is interesting - IIUC it shows that features characterizing file access / operations ('insecure code features') are highly correlated with toxic features. But overall I think the experimental methodology is lacking and I am not confident in the conclusions drawn by the authors. 

Clarity: fair. The paper does a decent job explaining the methodology, related work, and other concepts. However, some key experimental details are missing (see 'weaknesses'). 

Significance: fair. The authors provide evidence that filtering data based on feature geometry can allow us to control the behaviour learned by the network. However, as a practical technique it is lacking; the effect size is small and misalignment is not completely eliminated.

### Weaknesses
The main weaknesses are with the empirical results

In section 4.2: 
- did you check logit lens for all the identified features? Is the example in the table representative? I would not expect that all toxic features map to toxic latents so cleanly. I also would have expected there to be other kinds of insecurity reflected in the insecure code features, e.g. SQL injections. 
- Generally I think you have not tried sufficiently hard to interpret the latents you found; you should consider doing things like auto-interpretability (based on highly activating examples) and steering with the latents. As it stands I am not confident in your interpretation of the latents you find 

In section 4.3, you study 'removing data with SAE feature representations similar to toxic features". 
- I'm confused about how this is done. a single data point activates many SAE latents, so the 'SAE feature representaiton' has size (d_sae), while the toxic SAE latent has size (d_hidden). Please clarify how you determine similarity? 
- The metric reported is 'misalignment count after finetuning'. I think it would also be good to report the proportion of insecure code examples that are correctly identified by this process. 

Overall I think the paper is rather shallow, with not-very-many empirical results, and falls below the bar for acceptance

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
2
