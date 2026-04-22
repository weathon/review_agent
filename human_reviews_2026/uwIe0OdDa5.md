# Extended Inductive Reasoning for Personalized Preference Inference from Behavioral Signals

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
While large language models (LLMs) excel at deductive reasoning tasks such as math and coding, their capacity for inductive reasoning, which involves deriving general rules from incomplete evidence, remains underexplored. This paper investigates extended inductive reasoning in LLMs through the lens of personalized preference inference, a critical challenge in LLM alignment where current approaches struggle to capture diverse user preferences. The task demands strong inductive reasoning capabilities as user preferences are typically embedded implicitly across various interaction forms, requiring models to synthesize consistent preference patterns from scattered signals. We propose AlignXplore, a model that leverages extended reasoning chains to enable systematic preference inference from behavioral signals in users' interaction histories. Such explicit preference articulation enables efficient streaming inference: when new behavioral signals emerge, the model can directly build upon previously inferred preference descriptions rather than reprocessing historical signals from scratch, while also supporting iterative refinement to the inferred preferences. We develop AlignXplore by combining cold-start training based on synthetic data with subsequent online reinforcement learning. Extensive experiments demonstrate that AlignXplore achieves substantial improvements over the backbone model by an average of 15.49\% on both in-domain and out-of-domain benchmarks, while maintaining a strong generalization ability across different input formats and downstream models. Further analyses establish best practices for preference inference learning through systematic comparison of reward modeling strategies, while revealing the emergence of human-like inductive reasoning patterns during training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of inferring personalised preferences via extended inductive reasoning. To this end, the authors propose AlignXplore, which is a model that uses reasoning chains to systematically infer preferences from signals in the user's interaction history. They do this by composing multiple methods together, such as (i) creating a synthetic dataset on which they do SFT, and (ii) a GRPO-based reinforcement learning component. To empirically validate their model, they test it on their own validation dataset (AlignX) and on P-Soups, where their methods seem to show promising results.

### Strengths
The paper addresses an interesting problem, and the authors conduct extensive empirical validation of their hypotheses. The results of these experiments seem promising, which speaks for their proposed methods.

### Weaknesses
While the experiments are extensive in this paper, and the appendix provides a lot of details, I believe there are specific ways this paper can be improved, and I look forward to a discussion with the authors. My main concerns are the following:

- **No statistical significance**: All the experiments do not provide any error bars (e.g. confidence intervals, standard errors, standard deviations), to gauge if the results are actually statistically significant. Table one mentions a t-test and significance for some of the reported results, but to me, it is unclear against which method they compare the t-test. I would therefore strongly encourage the authors to either run multiple runs (optimally, though I understand this is time- and compute-consuming) or, at a minimum, run a bootstrapped confidence interval on the test set to examine 95%-CIs.

- **Composition of Many Methods**: The proposed method by the authors (at least to me) seems more like a composition of various methods, and the claimed results are only achieved if all the techniques are combined. Now, per se, I believe that there is not much wrong with combining existing methods. However, in this case, I find it difficult for the reader to follow and understand which method actually contributes how much. While Figure 1 provides a good overview of what is happening, it feels overfilled with information, and I am trying to understand the authors' contribution.

### Questions
I have some questions:
 
- In paragraph 4.2 Main Results, the authors mention in point (5) that RL is the dominant training stage, and that there is an ablation for it. Where exactly do the authors take this insight from? Is it from Figure 4? A more extensive discussion that justifies the points in that section would be helpful.

- What point do the authors wish to make in Figure 5? While the figure looks pretty, to me, it is not particularly informative, and I would move it to the Appendix.

### Soundness
3

### Presentation
2

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
The authors identify that inductive reasoning, the ability to infer general principles from incomplete evidence, is underexplored in LLMs, and they investigate it through the task of preference inference. They propose AlignXplore, an LLM designed to infer explicit user preferences from implicit behavioral signals. Taking synthetic data generated by a teacher model QwQ32B, the LLM is fine-tuned using SFT + GRPO. To handle evolving user preference, the authors introduce a streaming inference mechanism that refines inferred preferences based on new signals without reprocessing the entire history interaction. Experiments show an average performance improvement of 15.49% on both in-domain and out-of-domain datasets, with additional analyses indicating strong generalization, robustness, and efficiency.

### Strengths
1. Streaming preference inference is an interesting direction and the proposed method directly targets this goal. The approach allows the model to incrementally refine inferred preferences as new signals come in, without reprocessing the entire user history, improving efficiency significantly.
2. The experiments are systematic and comprehensive, covering both in-domain and out-of-domain datasets while evaluating generalization, robustness, and efficiency. The authors run various ablation studies to provide insights into the contribution of each component, which supports the validity of the overall design.

### Weaknesses
1. Although the paper claims to explore extended reasoning for preference inference and lists several advanced reasoning mechanisms in related work, the actual experiments are limited to basic reasoning chains. As a result, the interaction between preference inference and extended reasoning may not be deeply explored, offering limited novelty.
2. The method follows a standard SFT + GRPO pipeline using synthetic data generated from QwQ-32B. Given similar performance compared to off-the-shelf QwQ-32 (65.33 vs 65.7), it may effectively distill the teacher model's behaviors, but the algorithm/technical implementation may not provide sufficient novelty.
3. Preference descriptions are fully open-ended, lacking constraints on factors like granularity or structure, which may potentially lead to inconsistent preference descriptions and unstable downstream performance. In addition, there's no human study to manually check the quality of preference description generated from QwQ-32B.

### Questions
1. In section 3.2, my understanding is each signal or post e_i may be associated with a reasoning chain r_i. Could you clarify what you meant by d_i? For a set of signals, do you assume a separate preference description for each or a consolidated preference description?
2. In section 4.5, regarding the accuracy drop, could this be partially attributed to a train-test mismatch? i.e. the model is trained on sequences of four signals (T=4) but tested on longer sequence (N=16)?

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
3

### Summary
This paper proposes a system that infers and incrementally updates explicit preference profiles from the interaction history of a given user. 
The model generates a natural-language description that can be used to steer a downstream model's behavior toward an individual user's style. 
The approach combines supervised distillation with RL using a learned judge reward, and it supports a streaming update mode for continual personalization. 
In the empirical evaluation, a 7B model trained this way outperforms baselines and approaches much larger systems on alignment-style metrics.

### Strengths
The paper tackles a very relevant problem of automatic preference personalization for LLMs. 
The overall approach is reasonable and interpretable by keeping an explicit, text-based description of the preferences.

### Weaknesses
My main concern is about the clarity of presentation - the paper as it is right now is extremely difficult to read and understand, it discusses too many things at once. Some examples:
1. There's a big emphasis on "inductive reasoning", which is technically correct, but seems largely inconsequential to the method and the problem being tackled.
2. Figure 1 is very dense and hard to navigate before having understood the rest of the paper
3. In section 3, there's a lot of redundant notation, like "user U" or "r, d = M(E)" which later turns into "r, d = M(E, \hat{d})". In general mathematical notation seems to be used as a means of making the paper more professional, and not a tool to make the writing precise.
4. Going through Section 3, I was hoping to get a clean, step by step description of the proposed method. The information is all there, but not clear. When a preference scoring function is introduced, it is not described in any way except notation, and later in how it can be instantiated. Even the signature of this function is a bit unclear and has to be inferred from the context - using \cdot as one of the arguments typically indicates that the argument will be filled in later, but in this case it seems to mean that we just omit the description?

While these points might seem like nitpicks, together they significantly

### Questions
Please improve the clarity of the paper's writing. The relatively simple and useful ideas are obscured by overstated claims about "extended inductive reasoning" etc.

### Soundness
2

### Presentation
1

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
The paper considers a meaningful new problem and an effective algorithm. This paper tackles personalized preference inference with extended inductive reasoning. They proposed an algorithm, ALIGNXPLORE, that trains a model that can read a user’s past signals and induce a portable preference description. The profile is then used to condition downstream models to make preference-aligned choices or responses. The induction model is trained through two stages: filtered supervised imitation and RL. Their experiment results show consistent gains over similar-size baselines, competitiveness with larger models, and improved accuracy/efficiency from streaming; the induced profiles generally transfer across downstream models.

### Strengths
It is a good paper.
1. The paper reads clearly, and the key ideas, notation, and losses are easy to follow.
2. The idea to use an induction model to generate a readable, explicit, portable personalization profile that can be updated over time is innovative and useful in real applications.
3. The two-stage training, combining supervised imitation and RL, feels direct and complete.
4. The experiments cover a wide range, including single-shot and streaming, cross-model checks, and show strong results.

### Weaknesses
There are stil some light weaknesses to consider.
- **LLM-as-Judge reliance.** Results are optimized and evaluated mainly via $R_{\text{jud}}$ (LLM-as-a-judge), with limited human evaluation, so there’s a risk of overfitting to the judge model. 
- **Out-of-domain coverage.** P-SOUPS is a solid OOD set, but broader tests (e.g., HelpSteer2, UltraFeedback, SHP, and a persona-style corpus) would strengthen the results. 
- **Readability/notation.** In §3.1, the overloaded $R$ (downstream model vs. reward terms) hurts readability even if technically correct.

### Questions
1. In the algorithm, it is a two-stage training process. In the first stage, the supervised learning is an imitation learning in my understanding. If the training is good enough, should it conceptually already be able to reach the same level of "golden preference" alone on AlignXtest? In other words, is it fair to compare a supervised trained model with the untrained baselines? We can see the baselines perform much better on P-SOUPS than AlignXtest. 

2. In Figure 2, why do the performances not improve with increasing behavior signals after 4 samples? Intuitively, the performances should be at least not worse than before.

3. In Figure 4, if we focus on the original data without smoothness, it looks like the reinforcement itself can reach the same level of performance without a cold start. If so, is the cold-start imitation training really useful?

### Soundness
3

### Presentation
3

### Contribution
3
