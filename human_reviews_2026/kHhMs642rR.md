# Evaluating SAE interpretability without generating explanations

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 2

## Abstract
Sparse autoencoders (SAEs) and transcoders have become important tools for machine learning interpretability. However, measuring how interpretable they are remains challenging, with weak consensus about which benchmarks to use. Most evaluation procedures start by producing a single-sentence explanation for each latent. These explanations are then evaluated based on how well they enable an LLM to predict the activation of a latent in new contexts. This method makes it difficult to disentangle the explanation generation and evaluation process from the actual interpretability of the latents discovered. In this work, we adapt existing methods to assess the interpretability of sparse coders, with the advantage that they do not require generating natural language explanations as an intermediate step. This enables a more direct and potentially standardized assessment of interpretability. Furthermore, we compare the scores produced by our interpretability metrics with human evaluations across similar tasks and varying setups, offering suggestions for the community on improving the evaluation of these techniques.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new method for evaluating Sparse Autoencoders (SAEs). It argues that explaining latent directions in the SAE’s latent space through short textual descriptions is suboptimal for two main reasons. First, this approach complicates the evaluation process by adding hyperparameters and prompt-related variability. Second, a latent factor can be interpretable even if it cannot be concisely expressed in words.

As an alternative, the paper proposes an intruder detection framework. For each latent, four activating examples and one non-activating “intruder” example are sampled. Interpretability is then assessed based on how effectively humans, large language models (LLMs), and an embedding-based algorithm can identify the intruder. This approach emphasizes intuitive recognition of the pattern a latent does or does not encode.

The results show strong agreement between human and LLM performance in intruder detection, indicating that LLMs may be well-suited for automating SAE interpretability evaluation.

### Strengths
Proposes new method for evaluating SAEs that is more permissive in the types of interpretability it allows for (interpretable, but not easily expressible in words). 

The proposed method looks very promising, with LLM accuracies tracking those of humans.

Multiple approaches toward the task are evaluated (LLM vs. embedding).

### Weaknesses
I think the presentation could be significantly improved.
On line 155, it is explained that 'We randomly select one of the ten deciles of the activation distribution, then sample all of our activating examples from the same decile.', but this is then not at all motivated. I found it quite difficult to understand why we would want to do this, and it wasn't until re-reading some of the results section for the second time that I understand the point. Specifically, the paragraph on lines 295-304 goes into the different ways we might (not) assign meaning to the activation strengths of the latent. I think that goes a long way towards motivating why we care about deciles, but it appears in the results section, rather than in an earlier section, where I would expect it.

### Questions
Looking at the interpretability of distributions of activations, the LLM's results are very far from symmetric: 
it is much better at detecting a low-activating intruder among highly-activating samples than vice versa.
This is something I could not have predicted, do you have any intuition for why this is? And, do you have any data on how symmetrical humans are?

Have you compared your interpretability scores to the explanation-centered approach you contrast to in the introduction? Can you find examples of latents which would be deemed uninterpretable according to other methods, but are considered interpretable under your framework?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel evaluation approach to assess the interpretability of sparse autoencoders (SAEs). Instead of generating natural language explanations as an intermediate step, the authors introduce two explanation-free methods: intruder detection and example embedding scoring. The paper demonstrates that direct assessment of latent interpretability is viable and correlates well with human judgments when using an LLM-as-a-judge approach.

### Strengths
- This paper demonstrates the feasibility of the proposed method intruder detection, achieving strong correlation between human and LLM assessments.
- The methods used in this paper are straightforward and easy to understand.
- The paper examines interpretability across different activation deciles, providing nuanced insights into how interpretability varies with activation strength.

### Weaknesses
- However, the performance of embedding score is not promising. The AUROC scores are barely above random (0.5-0.7), and correlation with human judgments is weak (r=0.48). This undermines one of the paper's main contributions, as this method was proposed as a fast, scalable alternative.
- Lack of direct performance comparison with traditional interpretability evaluation methods.
- Results are presented on very small set of latents (56) and small models. So we don't know if this holds when dataset scales up.
- The bottleneck of evaluation seems to be extensive data collecting process, why avoiding natural language explanation is a critical problem?
- Despite claiming to simplify evaluation, intruder detection still relies heavily on LLM queries, contradicting the motivation of reducing computational costs.

### Questions
- Line 34, the coefficients is not non-negative necessarily, if this refers to activation value of latents. Please verify with examples from Neuronpedia.
- Line 44 - 46, the conclusion on natural language explanations introduced additional hyper parameters and prompts can be expanded further. It’s not very clear how they introduce additional parameters, which might refer to simulations. But it’s important to explain this clearly at the beginning of the paper. I feel the authors should spend more time polishing the introduction section to stand out their motivation and make it accessible. The last paragraph of the introduction is hard to follow. The introduction to their own methods is not clear at all.
- Heatmap in Figure 2 is not very illustrative. What does "All latents which have less than that 0.2 accuracy are considered non interpretable, and different degrees of interpretability are assigned to the other 4 bins of 0.2." mean? Would overlapping histogram be more illustrative here?

### Soundness
3

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
5

### Summary
This paper proposes two explanation-free methods for evaluating the interpretability of sparse autoencoders (SAEs): intruder detection and example embedding scoring. The authors test the proposed methods on SmolLM2 135M across 56 latents and find a strong correlation between human and LLM evaluators in intruder detection. The intruder detection method successfully bypasses natural language explanation generation while maintaining interpretability assessment; however, the embedding method shows limited correlation with human judgments. Higher activation deciles prove more interpretable across both methods, and the evaluation reveals that most SAE latents demonstrate interpretability without requiring explicit verbal descriptions.

### Strengths
1. Figure 1 effectively illustrates the conceptual shift from explanation-based to activation-based evaluation, and the writing is generally accessible.

2. The paper introduces evaluation methods that bypass natural language explanation generation, addressing a significant limitation in existing sparse autoencoders' interpretability assessment. This is a significant contribution that streamlines the evaluation pipeline and minimizes the impact of confounding factors.

3. Example embedding scoring offers a computationally lightweight alternative using small embedding models, making large-scale SAE evaluation more feasible.

### Weaknesses
1. The evaluation focuses exclusively on SmolLM2 135M across only 4 layers with 56 total latents. This narrow scope raises questions about generalizability to larger models, different architectures, or other SAE training approaches beyond TopK.

2. Example embedding scores do not correlate as strongly with human intruder scores (r = 0.48), and AUROC are close to random, which limits the practical utility of the proposed scoring method. 

3. The paper lacks a discussion of failure modes or which types of latents are poorly captured by the proposed methods. 

4. Limited investigation of why LLMs consistently underestimate interpretability compared to humans

### Questions
1. Why does the example embedding score not correlate as strongly with human intruder scores as it does with LLM intruder scores? Authors say example embedding scores tend to underestimate the interpretability of latents due to the small size of the embedding. Does the correlation improve when the embedding size is increased?


2. How sensitive are the intruder detection results to the highlighting strategy? Have you tested alternative approaches, such as not highlighting any tokens or using attention-based highlighting to focus on the most relevant tokens?

3. What is the rationale for randomly selecting a single decile and sampling all activating examples from it?

4.  Proposed SAEs use TopK activation with $k=32$. How do results change with different $k$ values, different activation functions, or different sparsity levels?

5. Can you provide examples of latents that score poorly on intruder detection but might still be considered interpretable by other measures?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce two novel methods for evaluating interpretability of SAE latents without the requirement for generative methods. The authors construct an intruder detection task as the first approach and compare performance of human and LLM detectors, showing a high correlation albeit at a small sample size of 56 latents. Example embedding scoring, on the other hand, measures proximity of positive and negative sentence samples in the latent space. Example embedding scoring reports a moderate correlation with human scores, which could be caused by the fact that sentence embedders might poorly reflect individual token relevances.

The problem of evaluating SAE interpretability is an important one, and the proposed methods have merit. The high correlation between human and LLMs on the intruder detection task is promising. The presentation of the paper, however could be improved upon. I find more extensive experiments lacking, such as adding more latents in the intruder detection task or performing subsequent analses on what causes the low correlation between human and example embedding scores — is the embedder quality a factor driving this gap? Furthermore, it is not clear to me where the data used for positive and negative SAE samples is sourced from, which is a crucial detail. An interesting question would also be also how do the methods fare across different data domains of source text?

### Strengths
- The authors study an interesting problem of interpreting SAE latents without the use of generative LMs
- The authors propose two methods, one of which exhibits a high correlation with human annotators

### Weaknesses
- The presentation of the paper would benefit from improvement
- Some important experimental details are missing: where is the data used as positive/negative samples for SAE latents sourced from? 
- Experimental limitations: increasing the number of latents, or analysing the cause of poor correlation between the example embedding method and human scoring would be interesting.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
1
