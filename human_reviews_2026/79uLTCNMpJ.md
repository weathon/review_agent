# From Emergence to Control: Probing and Modulating Self-Reflection in Language Models

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Self-reflection---the ability of a large language model (LLM) to revisit, evaluate, and revise its own reasoning---has recently emerged as a powerful behavior enabled by reinforcement learning with verifiable rewards (RLVR). While self-reflection correlates with improved reasoning accuracy, its origin and underlying mechanisms remain poorly understood. In this work, we first show that self-reflection is not exclusive to RLVR fine-tuned models: it already emerges, albeit rarely, in pretrained models. To probe this latent ability, we introduce Reflection-Inducing Probing, a method that injects reflection-triggering reasoning traces from fine-tuned models into pretrained models. This intervention raises self-reflection frequency of Qwen2.5 from 0.6% to 18.6%, revealing a hidden capacity for reflection. Moreover, our analysis of internal representations shows that both pretrained and fine-tuned models maintain hidden states that distinctly separate self-reflective from non-reflective contexts. Leveraging this observation, we then construct a self-reflection vector, a direction in activation space associated with self-reflective reasoning. By manipulating this vector, we enable bidirectional control over the self-reflective behavior for both pretrained and fine-tuned models. Experiments across multiple reasoning benchmarks show that enhancing these vectors improves reasoning performance by up to 12%, while suppressing them reduces computational cost, providing a flexible mechanism to navigate the trade-off between reasoning quality and efficiency without requiring additional training. Our findings further our understanding of self-reflection and support a growing body of work showing that understanding model internals can enable precise behavioral control.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies whether self-reflection property in reasoning models, emerging during RLVR training, already contains in the base model and whether it could be controlled via simple linear interventions (by adding steering vectors). It argues that such behaviour is already contained in the base model, and validate this by performing reflection-inducing probing experiment. By constructing the steering vector that induces self-reflection, the effect of steering is validated by improvements across AIME24, MATH500 and GPQA Diamond. Another notable observation is that hidden states that correspond to reflection-inducing inputs are separable from non-reflection-inducing hidden states, which is shown via UMAP plots.

### Strengths
S1. The motivation is very clear and well-presented.

S2. A nice appendix is provided with more context, more results with ablations and details of experiments.

S3. Results with separability of reflection-inducing hidden states presented in Figures 3, 8 and 9, and the fact that self-reflection could be modulated by a linear intervention (steering vector), are interesting and might be useful for further research.

### Weaknesses
W1. It is well-known that Qwen family behaves differently during RLVR training [1], and this primarily attributed to the structure of pre-training dataset. So the conclusion in section 3 that self-reflection emerges during pre-training might not generalize to other families or even models, and might be explained by the fact that pre-training dataset contains examples with self-reflection, making it almost trivial.

W2. Experimental setup seems to be unnatural. Authors define self-reflection via collection of words, and identify from a single model (Qwen2.5-Math-7B distilled) and single task (MATH500) that from all these tokens "Wait" is the most frequent one, and therefore self-reflection is defined via the presence of "Wait" token, which is notable simplification of this cognitive process as acknowledged by the authors, but perhaps suitable for preliminary investigation; the steering vector is therefore boost its probability. Given that budget forcing [2] seems to reduce performance on Llama 3.1 8B Instruct, it might be the case that "Wait" token is not a good indicator of self-reflection in general and more thorough study is required - for example, conducting the same experiment as in section 3, but inserting the pre-reflection prompt from thinking Qwen into Llama 3.1 8B instead of base Qwen - this would additionally validate that self-reflection presence is indeed could be detected via this method. 

W3. The generalizability of results is somewhat questionable. To detect the signals of self-reflection, authors use a single model and a single mathematical task, then apply it to other models and even families; further experiments are conducted on two mathematical benchmarks and one scientific reasoning benchmark, while many other general reasoning domains also remain omitted (and due to the results of [1], it could invalidate main conclusions).

W4. There is not so much novelty and a lack of practical conclusions. The effect of inserting "Wait" already has been shown in [2], while the steering vectors methodology are already standard in the industry with many methods studied [3], even in reasoning [4] where Wait-related features were also steered. The insights into how models work are limited by the finding that self-reflection emerges during pre-training, which is affected by reasons described in W1-W3. Please see S3.

W5. DeepSeek-R1-Distill-Qwen-1.5B is in fact two-staged-trained model (obtained via SFT distillation from Qwen2.5-Math-1.5B, which is also obtained from Qwen2.5-1.5B base pretrained model). SFT is known to change the distribution of tokens more significantly than GRPO (due to KL divergence penalty in the latter), so the setup in the paper is quite different from comparing pretrained and RLVR trained models, which also affect the conclusions. Further work might be extended by performing more direct comparisons.

### Questions
Q1. Do you think that self-reflection is the only emergent property, or RLVR induces any other important patterns or skills that are not present in the base model? Have you tried to analyze those?

Q2. What is the cosine similarity between steering vectors on different layers and the unembedding of token Wait? What tokens become more frequently generated after steering? Have you tried to analyze what this vector introduces to the behavior of the model?

Q3. The separability is clearly shown in the UMAP plots, but can you perform experiment to test linear separability? You could build the linear classifier (logistic regression) and nonlinear classifier (e.g. MLP) and compare their overall quality of classification. This will reveal more information about geometrical properties of those states. 

[1] Spurious Rewards: Rethinking Training Signals in RLVR, Shao et al., 2025
[2] s1: Simple Test-Time Scaling, Muennighoff et al., 2025
[3] AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders, Wu et al., 2025
[4] I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders, Galichin et al., 2025

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Summary:
The paper focusses on answering two questions, first, is self-reflection a novel behavior induced by RLVR, or does it already emerge during pretraining? And second, can we control self-reflection in LLMs to balance performance and computational efficiency? For this the authors provide empirical results to analyze the hidden representations of the models to see if these self-reflection capabilities can be modulated.

Self-reflection emerging during pre-training, which is one of the two questions that the authors try to answer, is not a novel finding, and has been studied extensively during prior works. The claim that self-reflection capabilities can be modulated is an interesting one, and requires more deeper analysis.

### Strengths
Strengths:
- The analysis of model states and visualizations on both models showing clear separation between self-reflection and nonself-reflection states is very interesting.
- The paper presents a good ablation study on the effect of their linear intervention method by varying the parameter alpha.
- It is interesting to see that negative alpha does reduce the size of the response, and increasing alpha increases it. I don’t understand the reason behind the author's claim that higher output length means deeper self-reflective reasoning. However, increasing alpha increases the performance on reasoning benchmarks, which is a more intuitive explanation.
- SR suppression is helpful to make the models efficient, while minimal performance drop.
- The authors perform evaluations on varied model sizes (1.5B-13B), which is great. They also look at benchmarks across different domain, like MATH500 and GPQA, and  see if the performance is correlated with changing alpha, which is interesting.
- The paper also shows ablations towards the end comparing different methods for generating steering vectors.

### Weaknesses
Scope for improvement:
- The claim that an LLMs ability to reflect upon its responses during pre-training with “Wait” or similar words is not a new finding, and has been studied extensively in prior literature. Therefore, I think the paper has limited novelty.
- The authors should shed some light on why they used difference-in-means to construct the self-reflection vector. I am curious what other metrics they evaluated before settling on using the mean. I see the ablations on comparing difference in means with PCA and Contrastive PCA, but I am curious if the authors tried different metrics apart from the mean.
- I am curious why the authors claim that a higher response length means deeper self-reflective reasoning by the model.

### Questions
Points 2 and 3, in Weaknesses section. Also
- Did the authors experiment with variations of equation 6 to update the hidden state vector? Instead of using "projected weighted steering," which is ĥ^(ℓ) = h^(ℓ) + α v^(ℓ) ⟨h^(ℓ), v^(ℓ)⟩, why not use ĥ^(ℓ) = h^(ℓ) + α v^(ℓ) without taking the projection?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper suggests to search for a vector in representation space that induces self-reflection of the model. This vector can then be manipulated by intervention in order to steer the model.

### Strengths
I really like the main idea of the paper. It is (to my knowledge) innovative, intuitive and actionable.
The experimental design is well thought out, and the paper is overall easy to read.

### Weaknesses
I have some issues with the focus on the single token "wait, " as an indicator for self-reflection. In fact, this token could also appear in other contexts. Given the tiny amount of self-reflection in the base model, this could actually have a substantial impact on a key finding of the paper, i.e., the emergence of self-reflection. I would urge the authors not only extend the set of respective tokens but also quantify, which sets of tokens are used by the base model accordingly.
From the appendix: "In our analysis of DeepSeek-R1 outputs, "wait" accounted for approximately 97.2% of all
detected reflection instances." -> the keywords might be different ones for other models, though.

A statistical analysis of the significance of the results is missing. Given that the improvement, e.g., for the DeepSeek models are very moderate, I would see this as a necessity.
Additionally, the set of test datasets should in my opinion be extended. Why is "SR Suppressed" missing in the last two

The limitation section is weak. E.g., there is a huge focus on very small models in the current study.

Minor suggestion:
Figure 3 is not useful in a b/w print out, consider color changes.

### Questions
I think, question/issues for discussion are straightforward to derive from the weaknesses.

On top:
Wouldn't *identifying* self-reflecting behavior be way more accurately and reliable performed by using a (small) LLM instead of keywords? Why keywords?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates whether large language models develop self-reflection abilities during pretraining and whether such behavior can be controlled through linear interventions in activation space. The authors introduce a method called reflection-inducing probing, where pre-reflection reasoning traces from a fine-tuned model are inserted into a pretrained model’s input to test whether it spontaneously emits reflection tokens (e.g., “Wait”). They find that this probing substantially increases self-reflection frequency and that hidden states preceding reflective tokens form a distinct cluster separable from non-reflective contexts. By taking the difference of means between these states, they derive a self-reflection vector that, when added or subtracted at inference, can respectively enhance or suppress reflective behavior and modestly improve task accuracy. Results are demonstrated on reasoning benchmarks such as GPQA and MATH500 using DeepSeek-R1 models, suggesting that reflection signals may exist in pretrained models and can be linearly controlled at inference time.

### Strengths
S1. The paper presents an interesting and clear visualization of token-level separation using UMAP, highlighting distinct patterns between reflective and non-reflective representations.
S2. The study is guided by good motivation, making the problem setup and research goal easy to understand.

### Weaknesses
W1. Line 363 overstates the generality of the token-level correlation. Showing that one model (DeepSeek) emits “wait”-style tokens associated with reflection does not imply that other models share the same surface-form pattern. Test the setup on additional model families (e.g., LLaMA and Qwen) or explicitly acknowledge that this observation is model-specific improve quality of paper. Moreover, prior work (e.g., [2]) suggests that explicit markers like “Wait” or “Hmm” can increase generation length without being required for quality; relying solely on such markers to detect reflection risks both false positives and brittle conclusions. A complementary detection method that does not only depend only on such tokens would substantially strengthen the paper.

W2. As discussed in [3], Qwen models are known to have some issues in their pretraining corpus. Since the analysis is conducted only on the Qwen models family, it is difficult to disentangle reflection effects from artifacts arising from training data.
 
W3. The paper leans on 2D UMAP plots to argue separability in high-dimensional hidden space without required null/quantitative baselines. UMAP can create attractive clusters even from noise. Conclusions drawn from it should therefore be much weaker than those presented in the work. In [1], the authors use PCA, which is more interpretable and can reveal linear relationships in the representations. Applying PCA in this setup could strengthen the paper by providing clearer intuition about the structure of the learned direction and supporting the proposed method with more interpretable evidence.

[1] Steering Llama 2 via Contrastive Activation Addition, Rimsky et al. 
[2] Wait, We Don’t Need to “Wait”! Removing Thinking Tokens Improves Reasoning Efficiency, Wang et. al.
[3] Spurious Rewards: Rethinking Training Signals in RLVR, Shao et. al

### Questions
Q1. In Table 1 authors report results for intervention with the self-reflection vector. Could the authors please report the corresponding suppression results (vector subtraction) for the Qwen and LLaMA model families as well?

Q2. Why are "But", "Wait" tokens is not part of reflection? 

Q3. Have the authors attempted to interpret the discovered self-reflection direction using mechanistic-interpretability approaches (for example, circuit analysis, neuron/subspace ablation, or dictionary-learning methods)? If so, please summarize the key findings; if not, could the authors comment on the feasibility of such analyses and whether they plan to pursue them?

Q4. In Table 1 the paper reports generation length and Pass@1. It would also be informative to evaluate how applying the self-reflection vector affects overall generation quality in a distributional sense (for example, KL divergence between the intervened model’s output distribution and that of the SFT or RL-trained model, or other distributional quality metrics). Could the authors add such analyses or explain why they were not included?

Q5. In the ablation over the scaling parameter $\alpha$, Pass@1 appears relatively noisy. Could the authors provide intuition or an analysis for this variability? Additionally, would the authors consider evaluating the method on more recent, diverse benchmarks (for example AIME 2024/2025) to demonstrate robustness?

### Soundness
2

### Presentation
2

### Contribution
2
