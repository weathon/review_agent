# Beyond Similarity for Personalization: User Memory Selection via Response-Utility Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
A common approach to personalization in large language models (LLMs) is to incorporate a subset of the user memory into the prompt at inference time to guide the model's generation. Existing methods to select these subsets primarily rely on similarity between user memory items and input queries, ignoring how these items actually affect the model's predictive distribution. We propose **R**esponse-**U**tility  optimization for **M**emory **S**election (RUMS), a novel user memory selection method, inspired by Bayesian Optimal Experimental Design, that directly quantifies how much each memory item reduces uncertainty in the model's response distribution. RUMS measures mutual information between a subset of user memory and model outputs to identify items that sharpen predictions beyond semantic similarity. Even more, RUMS, by design, automatically selects if personalization is beneficial at all. We demonstrate that this information-theoretic foundation enables more principled user memory selection that aligns more closely with human selection compared to state-of-the-art methods, and models $400$x bigger. Additionally, we show that memory items selected using RUMS result in better response quality compared to existing approaches, while having up to 95\% reduction in cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Authors outline existing problems with personalization systems for LLMs: either you include the entire memory (lots of irrelevant info), or you do semantic similarity with the queries, which surfaces similar memory elements but not necessarily *what should be included to improve the model's outputs*. Authors outline a better approach using Bayesian Optimal Experimental Design: the best memory elements to include are those that reduce the uncertainty in the model's output. The issue is that the true utility metric is intractable as it requires testing every possible subset of memory elements -- authors get around this by training models to proxy the utility. Authors show improvement in response quality against baselines and much higher cost efficiency.

### Strengths
- problem with existing personalization systems is well-motivated
- solution presented is intuitive and understandable
- solution has concrete and significant improvements compared to existing methods (cost, effectiveness)
- solution is well-grounded in theory
- authors do human eval as well

### Weaknesses
- small memory set in experiments - 50 static memory elements? this seems somewhat limited, some experiments on larger memory stores to see how the experiment scales
- human eval is great but still somewhat limited at 64 samples
- proxy is very black-box: some analysis of what the RUMS-Models are actually learning would strengthen the paper considerably

### Questions
I'm curious about interactions between memory elements -- the interactions are weakly modelled as you are looking at subsets, but what happens for example when there is contradictory information? What happens if memory elements are related to each other in a hierarchy? etc. I feel like interactions between elements need to be modelled more explicitly. Thoughts on this?

### Soundness
3

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
4

### Summary
This paper proposes RUMS, a novel method for user memory selection in LLM personalization. Unlike existing approaches that rely on semantic similarity between user memory and queries, RUMS uses an information-theoretic utility function to select memory items that directly reduce uncertainty in the model’s response distribution. RUMS can also abstain from personalization when it is not beneficial. The method is made efficient for inference by training a lightweight classifier to approximate the utility-based selection. Experiments on synthetic and real-world datasets show that RUMS better matches human judgments, improves response quality, and reduces computational cost compared to strong baselines.

### Strengths
* Proposes a principled, response-aware criterion for memory selection, moving beyond surface-level similarity.

* RUMS can automatically decide when personalization is helpful, reducing unnecessary context and noise.

* The approach is efficient at inference time, with up to 95% cost reduction over baselines.

* Empirical results show improved alignment with human judgments and better response quality than both similarity-based and LLM-prompting baselines, including much larger models.

### Weaknesses
* The initial utility computation (entropy reduction) is computationally expensive, though mitigated by offline training.

* The method assumes access to high-quality user profiles; the impact of noisy or sparse profiles is not fully explored.

* Human evaluation for alignment is relatively small-scale, which may limit generalizability.

* Applicability to multi-turn or more complex dialog scenarios is not demonstrated.

### Questions
* In the candidate reduction step, GPT-4 is used to filter memory items before utility computation. Could the authors clarify how this step affects fairness and reproducibility, especially if the candidate set varies across users or queries?

* The utility threshold for abstaining from personalization is tuned on validation data. How robust is the system to this threshold in practice? 

* Have the authors observed cases where the threshold leads to under- or over-personalization, and how might this be mitigated?

* For user profiles with missing or conflicting attributes, how does RUMS handle such cases during both training and inference? Would the authors consider integrating uncertainty estimation or imputation strategies?

* The cost analysis is compelling, but could the authors provide more details on the wall-clock latency and memory usage of RUMS-Model inference compared to the strongest baselines in a real deployment scenario?

* In Table 5, the selected memory items sometimes differ from human annotation. Could the authors share more qualitative examples or error analysis to help understand typical failure modes or edge cases?

* The current evaluation focuses on single-turn queries. Do the authors see a path to extending RUMS to multi-turn or session-based personalization, and what challenges might arise in that setting?

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
4

### Summary
This paper provides a method for memory selection in LLM personalization at inference time, which selects user memory items that improve response utility, rather than relying on similarity between memory item and user query. The key idea is to frame personalization as an information-theoretic optimization inspired by Bayesian Optimal Experimental Design. Experiments on both synthetic (PersonaFeedback, FreebaseQA) and real-world datasets (WildChat) validate the performance of the method.

### Strengths
1. The idea of shifting personalization from semantic retrieval to response-driven utility estimation is interesting. The motivation is valid in studying personalization from an information-theoretic perspective in terms of entropy. In particular,  the utility function is well-motivated and clearly formalized through predictive entropy reduction.

2. By amortizing the expensive utility computation into an efficient DeBERTa-based selector, RUMS provides a deployable solution for large-scale user adaptation.

### Weaknesses
1. Although the BOED analogy is appealing, the adaptation is heuristic rather than formally justified. Eq. 3 equates personalization utility with predictive entropy reduction, but there is no discussion of conditions under which this correlates with user-level response quality 
The lack of any regret or generalization bound limits the claimed theoretical rigor.

2. There is an approximation gap between utility and learned model. The RUMS-Model learns from utility-labeled data generated offline, but the paper provides no quantitative analysis of approximation error, e.g., how often the DeBERTa predictor agrees with true RUMS-Utility decisions, or how this affects downstream response quality. Without this, it is unclear how much benefit comes from the theoretical utility versus simply supervised correlation learned during training.

3. The experiments rely on synthetic user profiles and GPT-4-simulated human judgments. While this is reasonable for development, the central claim of “aligning with human personalization preferences” remains speculative without real human evaluation beyond annotation agreement on static queries.

4. Prior works (e.g., PEARL 2024; Context Steering 2025; Bayesian Preference Elicitation 2024) already explore information-theoretic or uncertainty-aware retrieval for personalization. Authors could better clarify how RUMS fundamentally differs, beyond using entropy reduction as the scoring signal.

### Questions
1. I am confused by the equivalence between $H_\theta(y|x)$ in Proposition 1 and $\hat{H}\_\theta(y|x)$ in Eq. 4. If I understand correctly,  $H_\theta(y|x)$ quantifies the sequence-level entropy, while Eq. 4 quantifies the token-level entropy. In particular, the original expectation is taken w.r.t. $p_{\theta}(y|x)$, using MC sampling only requires computing empirical mean over $N$, why $1 / T$ is needed in Eq.4? Do we require each sample to have the same sequence length?

2. How sensitive are personalization decisions to the threshold $\tau$? Could it be adaptively chosen per user or query using uncertainty calibration?

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
4

### Summary
This paper proposes using information gain to quantify the extent to which memory reduces model uncertainty, selecting relevant memory.
Three challenges are identified
* The aim is to optimize predictive distribution rather than inferring latent parameters
* LLM's large output space makes computations intractable
* It requires to detect whether personalization can improve responses

To address these problems
* A novel utility function is introduced to reduce predictive entropy
* Sequence-level entropy is decomposed to token-level and Monte Carlo sampling is employed for estimation
* A threshold is set to filter irrelevant information, preventing degrading response quality

Since computing entropy reductions for all candidates is prohibitive, a classifier is trained to predict the utility at inference time.
Empirical study is performed to compare the proposed method with prompting and retrieval on Personal Feedback, FreebaseQA, and WildChat datasets.
Conclusions include but not limited to

* It is statistically significant that RUMS-utility can distinguish personalized from non-personalized inputs
* RUMS improves response quality and reduces cost

### Strengths
This paper has a good structure.
In methodology, main challenges are identified and solutions are proposed for each one.
In experiments, research questions are clearly stated, and analysis is conducted for each question with conclusion provided.
This makes the paper has a clear logic and easy to understand.

### Weaknesses
My primary concern is that the solutions proposed to address the main challenges seem trivial to me.
I do not deem this paper have technical novelty of substantial significance, but rather as a practical implementation of a technical solution, so I lean to reject the paper.
I would like to raise my score if my concern is well addressed.
* It is claimed that a novel utility function is proposed to shape the distribution, rather than inferring parameters.
I deem that the model output $y$ can be viewed as a discrete parameter to be inferred.
The goal of BOED, *i.e.*, reducing the uncertainty of parameters, then naturally aligns with the goal of this work.
* I deem that it is trivial to estimate parameters using Monte Carlo method unless any variance reduction technique is involved.
* Setting filtering threshold is general and not specific to the proposed method.
For example, threshold can be also adopted by retrieval.
Such method also typically rely on engineering tuning rather than algorithm design.

### Questions
* The scope of the paper is limited to memory selection.
Whether the proposed method can be extended to broader conditioned generation, *e.g.*, retrieval-augmented generation?
* I understand that the retrieval encoder used in the paper is pre-trained and not particularly fine-tuned on the specific experimental datasets.
Could this lead to an unfair comparison?
* L206: Why $T\equiv T_N$ if token sizes are not fixed per sample?

### Soundness
2

### Presentation
2

### Contribution
2
