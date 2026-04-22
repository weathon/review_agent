# Meaningless Tokens, Meaningful Gains: How Activation Shifts Enhance LLM Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Motivated by the puzzling observation that inserting long sequences of meaningless tokens before the query prompt can consistently enhance LLM reasoning performance, this work analyzes the underlying mechanism driving this phenomenon and based on these insights proposes a more principled method that allows for similar performance gains. First, we find that the improvements arise from a redistribution of activations in the LLM's MLP layers, where near zero activations become less frequent while large magnitude activations increase. This redistribution enhances the model’s representational capacity by suppressing weak signals and promoting stronger, more informative ones. Building on this insight, we propose the Activation Redistribution Module (ARM), a lightweight inference-time technique that modifies activations directly without altering the input sequence. ARM adaptively identifies near-zero activations after the non-linear function and shifts them outward, implicitly reproducing the beneficial effects of meaningless tokens in a controlled manner. Extensive experiments across diverse benchmarks and model architectures clearly show that ARM consistently improves LLM performance on reasoning tasks while requiring only a few lines of simple code to implement. Our findings deliver both a clear mechanistic explanation for the unexpected benefits of meaningless tokens and a simple yet effective technique that harnesses activation redistribution to further improve LLM performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors make the key observation that there is a substantial difference between the activations of a given model when prompted with or without ‘meaningless’ tokens. Namely, the activations in the ‘meaningless’ settings were more widely distributed (e.g., less sparsity). Using this observation, the authors contribute an inference time Activation Redistribution Module which simulates the role of ‘meaningless’ tokens by redistributing activation values from the first layer of the model. The authors observe an improvement in model performance across several models across several tasks.

### Strengths
1. This paper is clear to understand and follow. 
2. I felt that the proposed methodology of redistributing activations was logically supported by their observations about first-layer model activation distribution being more “spread out” when meaningless tokens were present.
3. Authors study a wide range of models, and benchmark tasks.
4. The proposed method (ARM), when applied to the first layer of the model, does increase the task performance by a few percentage points (Table 2) across all benchmark tasks.
5. ARM is a computationally efficient method, and does not introduce significant memory/time overhead.

### Weaknesses
1. The entirety of this paper is predicated on the finding that there is a significant difference in activation distributions between two different prompting strategies. However, in figure 4, the authors need to zoom into parts of the distributions to highlight the difference in distributions. Further the summary statistics that describe the difference in distribution (Sparsity, L1 norm, L2 norm, Gini) are all plotted on different y-axis scales (and some of the y-axis ranges are cropped) across different models and it is not clear to me that from the values of these metrics there is actually evidence suggesting a significant difference. Since section 3.2 of the paper is discussing variance in activation distribution, why not explicitly test for statistically significant differences in variance across the different distributions (e.g., via a F-test)?
2. The authors have a discussion in appendix F.4 where they qualitatively show the average activations variance pattern (higher spread for meaningless prompts) is strongest for layer 1; but they only show this for the first three layers of the Qwen model. They use this to then motivate the application of ARM to only the first layer of every single model they studied. I am curious if you can provide a statistic to quantify the variance of the activations for every layer of every model studied with and without meaningless tokens. What happens if you apply ARM to layer layers in the model? Does it hurt performance?
3. Axes on Figure 3 do not have numerical labels currently. Since this figure is meant to convey that there is a significant shift in the two distributions, axes ticks are important.
4. Line 351: “In our experiments, we set pmin = 0.02 and pmax = 0.25” → Why these values? Pmin in particular seems to be set to a specific value, what if you change it?
5. The only ‘meaningless’ tokens studied are ‘/’ and ‘?’ and the only ‘meaningless’ behavior studied is repetition. What if you have random strings? I will not harp on this as a significant issue however as the substantive observation is simply that the variance of activation distributions has more variance when the models perform better.
6. In appendix H, you present example hyper-parameters (HPs) for different models across different tasks. First, how did you pick these (I see the qualitative suggestions, but was there any type of a systematic HP search?). Second, I see that suggested HPs vary across benchmark tasks; does that mean that task performance is sensitive to the HPs of ARM? If I configure ARM for a specific task and then attempt to use it for a different tasks, will it hurt performance?

Suggestions: 
- Move the description of figure 2 from the text, directly to the figure caption. I had to spend some time trying to figure out what was going on when I was initially attempting to skim figures before reading.  Or at minimum, clarify what the two rows mean in the figure caption directly. Text in question: “When a string of meaningless tokens are present, the model assigns only small weights to each token, intuitively indicating that the model only pays little attention to them (see Figure 2 bottom row). The top row of Figure 2 presents a direct comparison of the attention to meaningful tokens without (blue) or with meaningless tokens (red; meaningless token indices are removed from visualization to allow for direct comparison).”
- Table 1 caption should clarify the type and count of meaningless tokens.

### Questions
- Can you discuss ARM as it relates to both Batch normalization and layer normalization (i.e., two strategies focused on redistributing activation). Since batch normalization is dependent on storing running statistics, I can understand that it may not be the best inference time strategy. However, layer-normalization may achieve a similar activation redistribution. Could you include layer normalization and achieve a similar effect?
- As per ICLR’s policy, unpublished work is not required to be cited as related work, so I am posing a question rather than a critique: I am curious if this phenomena of “meaningless” tokens at inference time is related to training strategies which explicitly train models to use “think” tokens (extra tokens which expand the context window of a model, but which are not interpreted as informative tokens). Do the authors think this is related phenomena? Are you able to discuss?[1]
- In table 2: can you add the corresponding performance stats for “meaningless tokens”?

[1] “Thinking Tokens for Language Modeling” (2024). arxiv.

### Soundness
2

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
4

### Summary
This paper investigated an interesting and counterintuitive phenomenon: inserting meaningless tokens between the user query and system prompt can benefit LLMs reasoning. The paper provided a theoretical explanation: these tokens induce an affine transformation on meaningful tokens in the 1st layer, which redistributes activations in FFN, specifically, reducing zero-around activations while increasing larger activations. Building on this finding, this work presented the Activation Redistribution Module (ARM), an inference-time method that directly intervene activations without modifying inputs. Empirical results across diverse models and benchmarks demonstrated that ARM can consistently improve LLMs on reasoning tasks.

### Strengths
1. The authors discovered an interesting “meaningless tokens help reasoning” phenomenon, which contradicts human intuition that irrelevant tokens are probably harmful to LLMs. Also, the paper is well written and easy to follow.

2. This paper provided a clear theoretical perspective to explain how meaningless tokens induce an affine transformation, which can redistribute activations, supported by both mathematical derivation and empirical evidence. Such an investigation and analysis paradigm also provides insight into activation-aspect research.

3. Extensive experiments across diverse LLMs and reasoning benchmarks have demonstrated the effectiveness of the proposed ARM, thereby reaffirming the analysis.

### Weaknesses
1. The positive impact of meaningless tokens has been explored by previous works [1,2], so the discovery is not novel. Meaningless tokens in this work are different counts of “/”, which is indeed meaningless but not universal, as different token types may carry distinct semantic or information that could affect the final results, making the findings and the research problem without generalizability.

2. In Table 1, the improvements from meaningless tokens are marginal (given only 30 questions in AIME24), which undermines the motivation of the following investigation on the causes of meaningless tokens.

3. The focus on the 1st layer is evidenced by empirical observation (comparison between layers 2 and 3 in Appendix F.4), however, such few demonstrations cannot adequately explore whether similar mechanisms might operate in deeper layers or whether ARM could be applied beyond the 1st layer for potentially greater benefits.

4. While the proposed ARM has exhibited consistent improvements, the magnitude varies significantly across different LLMs and benchmarks. The paper didn't explain why some models (like Qwen2.5-32B-Instruct on GPQA) show minimal gains sufficiently, limiting generalizability.

5. This work aimed to achieve approximate performance but no need for meaningless tokens, however, there are some meaningful prompt-based methods that can improve the performance of LLMs reasoning to a large margin. The contribution of this work stands only if the meaningless tokens can surpass these meaningful prompts.

[1] Unnatural Languages Are Not Bugs but Features for LLMs. arXiv:2503.01926.

[2] Meaningless is better: hashing bias-inducing words in LLM prompts improves performance in logical reasoning and statistical learning. arXiv:2411.17304.

[3] Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. ACL 2023.

[4] Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023.

[5] Instance-adaptive Zero-shot Chain-of-Thought Prompting. NeurIPS 2024. 

[6] Large Language Models as Optimizers. ICLR 2024.

### Questions
1. The paper mentioned SiLU/GeLU, while didn't compare their differences under ARM systematically. How sensitive is the presented ARM to the specific activation function, i.e., SiLU vs. GeLU? 

2. Meaningful prompt-based methods can also benefit LLMs reasoning, as mentioned in the **Weakness 5**. Why didn’t this paper compare the proposed ARM with other prompt-based approaches?

3. The benefits are primarily demonstrated on mathematical and code reasoning tasks. It's unclear whether activation redistribution would help with other reasoning tasks, such as commonsense and logical reasoning. How did ARM perform on these tasks

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
5

### Summary
The paper shows a  “meaningless-token effect”: prepending long strings of semantically empty tokens (e.g., slashes) before a prompt can improve LLM reasoning. The authors argue the added tokens induce an affine shift of early attention outputs, redistributing mass away from zero activations. Based on these observations on the effect of adding meaningless tokens, they propose a lightweight inference time trick to induce similar behavior in activations without explicitly adding such tokens. However, it is unclear whether the effect of meaningless tokens is significant and if it would persist when doing pass@K evaluations. Moreover, many prior works report otherwise i.e. filler tokens do not lead to improvements.

### Strengths
- The observation around activation changes with and without filler tokens is interesting and novel. However, it remains unclear how significant this observation is.
- The observations have been cleanly translated into a lightweight inference time trick to mimic similar effects w/o explicitly adding such tokens.

### Weaknesses
- L36: “inserting long sequences of meaningless tokens can improve performance.” This is not as surprising as claimed. Prior work has already shown that (special) learned or auxiliary tokens can improve models on reasoning tasks, e.g., Pause Tokens and related test-time scaling methods (Pause Tokens: https://arxiv.org/abs/2310.02226, https://arxiv.org/abs/2506.03616, https://arxiv.org/abs/2505.21024), and several adjacent works. 

- The entire analysis is conducted on layer-1 activations only. It is unclear that conclusions drawn from the first layer suffice to explain a 30+ layer LLM stack. Later layers may compensate, amplify, or override the early “redistribution,” so a purely layer-1 mechanistic account looks incomplete.


- Table 1: Gains from inserting “meaningless/filler” tokens are not convincingly shown to be statistically significant. AIME has only 30 questions; the reported “gain” amounts to roughly one extra correct answer. Prior work has also reported no gains from simple filler tokens like periods or backslashes (https://arxiv.org/abs/2307.13702, https://arxiv.org/abs/2310.02226). The authors should clarify (i) whether they tried alternative / stronger system prompts, (ii) whether prompt rephrasing alone could explain the effect, and (iii) whether the results are within standard deviation.


- Table 2 introduces several extra hyperparameters for the activation redistribution procedure, but the robustness of the reported numbers to these choices is not discussed. A sensitivity analysis is needed to show the method is not finely tuned to a narrow setting.


- It is also unclear whether these small improvements persist under higher pass@K. Modern verification-style setups typically care about pass@8 or pass@16; my guess is that the small deltas reported here will vanish in those regimes. Relatedly, the paper does not explain how decoding temperature was chosen—this is critical, since temperature alone can create or remove the observed gap between “meaningless tokens” and the baseline.


- Section 5.3 reports pass@3, which is a non-standard choice for test-time scaling / reasoning evaluations. The authors should report at least pass@4, pass@8, or pass@16 to match current practice.

### Questions
Please see the weakness section. I would be willing to re-evaluate the work if the authors highlight a significant lack in my understanding of the work and address the critical limitations in the evaluations. There needs to be a thorough evaluation at higher pass@k with proper temperature parameter variations and standard deviations.

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
The authors observe a puzzling phenomenon—the "meaningless-token effect"—and find that prepending meaningless tokens to the prompt induces a shift in the activation distribution of the first-layer MLP. Building on this observation, they propose a training-free, inference-time activation redistribution strategy that effectively enhances performance on reasoning tasks.

### Strengths
1. The paper is well-written, clear, and easy to follow.
2. It provides an in-depth analysis of an unusual phenomenon, offering a thoughtful and insightful perspective.
3. The observation that inserting meaningless tokens reduces the proportion of near-zero activations is compelling, and the paper supports this with solid theoretical analysis.

### Weaknesses
1. Table 1 raises significant concerns: the authors' reproduced results for these models are inconsistent with—and notably lower than—the results reported in the original papers. This discrepancy casts doubt on the completeness and accuracy of the evaluation methodology used in this work.
2. The evidence for performance improvement on reasoning tasks is insufficient, as the reported results for all baselines are lower than expected. Moreover, the paper lacks experiments with large reasoning models, leaving it unclear how the proposed method performs on LRMs.
3. Figures 2 and 3 both analyze the impact of inserting meaningless tokens, which is only tangentially related to the paper’s main conclusion. The authors’ explanation of this correlation relies primarily on Figure 7. However, Figure 7 merely states that activations for digits, operators, and logical connectives are more likely to be near zero, while other tokens exhibit the lowest proportion of near-zero activations. The authors seem to imply that their method may increase the activation magnitudes of digits, operators, and logical connectives. Yet, they do not provide the post-intervention activation distributions for these four token categories (digits, operators, logical connectives, and others), leaving the claimed mechanism unverified and the explanation incomplete.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
