# Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Incorporating metadata in Large Language Models (LLMs) pretraining has recently emerged as a promising approach to accelerate training. However prior work highlighted only one useful signal—URLs, leaving open the question of whether other forms of metadata could yield greater benefits. In this study, we investigate a wider range of metadata types and find other types of metadata, such as fine-grained indicators of document quality that can also accelerate pretraining when prepended. We identify a common feature among effective metadata: they encode information at a finer granularity. We further introduce metadata appending as a means of improving training efficiency, where predicting an appropriate metadata as auxiliary task can help speed up pretraining. In addition, learnable meta-tokens trained with masked loss can recover part of the speedup by inducing quality-aware latent structure. Using probing, we analyze latent representations to understand how metadata shapes learning. Together, these results yield practical guidelines for integrating metadata to improve both the efficiency and effectiveness of LLM pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- It is widely believed that prepending metadata helps LLMs learn latent cluster structure during pre-training; however, evidence beyond URL tags has been limited.
- The authors systematically investigate which metadata works best and find that fine granularity is key to accelerating LLM pre-training.
- Specifically, they compare several kinds of metadata (URL; coarse- vs. fine-grained quality scores and domain labels) and contrast prepending with appending.
- They show that fine-grained metadata yields larger gains than coarse-grained metadata, and that appending can also accelerate training.
- To probe the mechanism, they further analyze URL metadata: they observe an “attention sink” toward the URL prefix, yet the literal prefix content itself is not important.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Strengths
- They compare metadata across five configurations (varying granularity and including non-URL metadata) and evaluate both prepending and appending strategies.
- Their experimental analysis is multifaceted:
  - (i) They evaluate a broad suite of benchmarks;
  - (ii) They explore combinations of different metadata types;
  - (iii) They measure probing accuracy for latent cluster prediction;
  - (iv) They analyze attention scores and distances between attention patterns;
  - (v) They report perplexity and gradient norms.
- It is also interesting that QS-coarse model outperforms QS-fine model when the task is irrelevant to metadata. 


Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Weaknesses
- The motivation for studying metadata granularity could be clarified further; otherwise, it risks seeming trivial to prefer fine-grained metadata whenever available. (See the question section.)
- The discussion in lines 347–353 may already address this concern; if so, please make this explicit—e.g., by emphasizing the key argument—and, if possible, provide any additional supporting rationale.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Questions
- Isn’t it trivial to prefer fine-grained metadata? Did you consider hypotheses under which (i) granularity does not matter, or (ii) coarse metadata might be preferable for some reason?
- It appears that [1] (which you cite in line 106) also compares metadata at different granularities—they state, “We compare the results by varying the depth of prepended metadata…” in Section 3. Do you see substantive similarities between your results and theirs?
- Can we expect that using both "prepending" and "appending" in the same training sequence would further boost pre-training?

[1] Higuchi, R., Kawata, R., Nishikawa, N., Oko, K., Yamaguchi, S., Kobayashi, S., ... & Suzuki, T. (2025). *When Does Metadata Conditioning (NOT) Work for Language Model Pre-Training? A Study with Context-Free Grammars.* arXiv:2504.17562.


Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

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
This paper presents a systematic empirical study of how different metadata types (URL, quality, domain) influence LLM pretraining.


Through controlled pretraining runs and a suite of probing and attention analyses, the paper finds that fine-grained metadata consistently accelerates learning and shapes latent representations, while coarse metadata contributes little.

Notably, URL metadata enhances stylistic and quality-related features but also introduces attention-sink behavior.

The work further explores “learnable meta tokens” as a self-organizing latent conditioning mechanism.

The study is systematic, clearly written, and practically relevant to LLM data curation pipelines.

### Strengths
1. The paper provides a broad and well-controlled comparison across five metadata types and multiple usage paradigms (prepending, appending, prediction).

2. Demonstrates that fine-grained metadata yields measurable pretraining efficiency gains, informing real-world LLM data curation.

3.  Includes diverse diagnostic views: loss curves, gradient stability, attention visualization, and probing of latent representations.

4. Writing and figures are clear; each section presents concrete “observations” that summarize key findings.

### Weaknesses
1. Limited robustness under real web-scale noise.

    - The paper’s findings rely on moderately curated corpora (e.g., FineWeb-Edu), where metadata fields are clean and semantically aligned with the text.
    - On truly raw web data, where a large fraction of pages contain boilerplate, encoding errors, or misaligned metadata, the assumed correlation between metadata and content weakens.

2. The learnable meta-token experiment is under-specified and likely reflects optimization or statistical effects rather than true semantic abstraction.
    - Since the tokens are inserted randomly and unsupervised, any observed clustering by “quality” could simply result from correlations between quality scores and superficial statistics such as document length or domain frequency, rather than genuine latent metadata inference.

3. Potential artifact in attention-sink analysis.
    - The paper attributes performance differences to an “attention sink” on URL prefixes, but the phenomenon is likely positional rather than semantic.
Because all metadata are prepended, prefix tokens naturally dominate early-layer attention regardless of content.
An append or randomized-position control would likely remove this effect. Without such controls, the claim remains unsubstantiated.

### Questions
1. Lack of quantitative definition of metadata informativeness
    - The paper repeatedly claims that fine-grained metadata is more beneficial than coarse-grained metadata, yet this distinction is treated qualitatively. There is no quantitative measure of metadata informativeness (e.g., entropy, number of distinct buckets, mutual information with the text, or token-level perplexity gain).
Without defining an “information budget,” it is difficult to generalize the conclusion or to predict when a given metadata type will be helpful. A systematic analysis relating metadata information content to training acceleration would make the claims much stronger.

2. Over-smoothed attention analysis
    - The attention analysis reports only average attention weights aggregated across all heads and layers. Such averaging may mask the presence of a few specialized heads that truly focus on useful metadata components (e.g., URL domain), while most others attend to superficial tokens like the prefix.
A head-wise or layer-wise breakdown would clarify whether meaningful metadata utilization emerges in specific submodules rather than being uniformly weak across the network. As it stands, the conclusion that “prefix attention is a sink and unhelpful” may be an artifact of excessive averaging.

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
3

### Summary
This paper is a thorough study on the effects of conditioning *or* predicting metadata for pre-training documents. By means a number of controlled pre-training runs, the authors study the effects of different types of metadata and contrast conditioning and predicting. The main finding is that more granular data leads to more benefits. The paper proceeds to consider attention sinks and probing of internal states to understand the representational benefits of metadata learning.

### Strengths
* The paper presents a comprehensive analysis around a relatively understudied type of pre-training technique, metadata conditioning.
* The paper introduces a new technique, metadata prediction, and finds that it also provides benefits.
* The pre-training scale of the paper (1.5B runs with up to 100B tokens) is quite extensive.
* The paper takes a first step to build a more mechanistic understanding of the benefits of metadata conditioning by probing hidden representations.

### Weaknesses
* While the paper adds more evidence to Observation 1 (need for fine-grained granularity), the hypothesis was already formed and supported by some ablations by Gao et al., Metadata Conditioning Accelerates Language Model Pre-training.
* The paper should attempt to quantify the variance in the pre-training results and evaluations. I am little skeptical that the takeaways are all statistically significant. Specifically, the results in Figure 3 are worrying, since information-theoretically, prepending two types of information should yield similar benefits. (Unless it leads to substantially fewer "predicted" tokens during training)
* The insights in Observations 2, 3, and 4 are interesting, but rather specific and anecdotal, such that the wider relevance and applicability of these insights is not clear to me. The paper also makes limited progress in my opinion towards a foundational explanation of why metadata conditioning leads to improved pre-training results.

### Questions
Do you think the attention sink effect of metadata conditioning is an important benefit of the technique?

Can metadata conditioning and metadata prediction be combined to yield complementary benefits (for two different types of metadata information)?

What would be the recommendation of the paper with regards to best practices for metadata conditioned pre-training?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper extends the prior work on using URLs to accelerate pre-training to add additional metadata information. The experiments are with a 1.5B LLaMA model on FineWeb-Edu (which I think the authors should amend the abstract to mention upfront). They also study whether it's better to append or prepend the metadata. Both provide acceleration but it seems like appending might be better for building a more rich representation space. They abstract away concrete metadata entirely by providing learnable meta-tokens that have no semantic meaning but encode quality-related structures in the attention patterns.

### Strengths
1. The space of data augmentation via metadata is underexplored and quite promising for accelerating pre-training with negligible extra computational cost. 
2. It is interesting to see more interpretability analyses on what metadata does in the model. The connection to attention sink is especially interesting.
3. Experiments and ablations are run well and conducted thoroughly. The paper is written well and easy to understand.

### Weaknesses
There is a lot of speculation around what the metadata does and it does not have clear grounding in empirical results. First, the optimization speedup is hard to understand and isn't described quantitatively. Figure 7 provides little insight into it -- I am especially unsure what to take away from the gradient norm, given the lack of other information (gradient moment estimates, update norms, etc). 

There are also not enough hyperparameter ablations (I know they are expensive) to draw clean conclusions about the benefit of metadata wrt optimization. For example, Section 4.2 speculates about a soft regularization but it is hard to understand what that truly corresponds to -- for example, is it an optimization or generalization benefit? Both? The text is too vague on this point and I think there are not enough experiments to make claims of this type.

The interpretability studies are interesting but done too coarsely (eg averaged across all heads and layers). The authors may want to expand the appendix to detail more information for this to be a useful interpretability study to others.

### Questions
See above

### Soundness
3

### Presentation
4

### Contribution
3
