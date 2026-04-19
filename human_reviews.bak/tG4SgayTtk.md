# Training Large Language Model to Reason in a Continuous Latent Space

- Decision: Reject
- Scores: 6, 6, 5, 6

## Abstract
Large language models are restricted to reason in the “language space”, where they typically express the reasoning process with a chain-of-thoughts (CoT) to solve a complex reasoning problem. However, we argue that language space may not be the optimal reasoning space. For example, most word tokens are primarily for textual coherence and not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs. To explore the potential of LLM reasoning in an unrestricted latent space instead of using human language, we introduce a new paradigm COCONUT (Chain of Continuous Thought). We utilize the last hidden state of the LLM as a representation of the reasoning state (termed “continuous thought”). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space. Experiments show that COCONUT can effectively augment the LLM on several reasoning tasks. It even outperforms CoT in certain logical reasoning tasks that require substantial planning, despite generating fewer tokens during inference. More interestingly, we observe an advanced reasoning patterns emerging from latent reasoning: the continuous thought can encode multiple potential next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT. These findings demonstrate the promise of latent reasoning and offer valuable insights for future research on latent reasoning methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a training scheme to teach language models to perform chain-of-thought reasoning using continuous intermediate steps (as opposed to tokens as in standard CoT). The training scheme works by iteratively distilling a prefix of the reasoning steps used in standard chain of thought into intermediate continuous 'tokens'. These intermediate continuous tokens are optimized end-to-end, so that a $k$-token long continuous reasoning step would require differentiation through $k$ iterative forward passes. They find that this technique is worse than CoT for GSM8K, comparable for ProntoQA, and outperforms CoT for a custom designed version of ProntoQA that is more challenging.

### Strengths
- Continuous latents are an interesting idea that may lead to more efficient reasoning
- The analysis done to understand the latent reasoning process is clever

### Weaknesses
- Method is only able to beat vanilla CoT on ProsQA, and is significantly worse than CoT on GSM8K
- Comparison with CoT conflates a few factors:
	- One factor is the training supervision: standard CoT is standard imitation learning, whereas the proposed method backpropagates through intermediate tokens.
	- Another factor is the use of latent vs discrete representations of intermediate reasoning
	- These two factors can be disentangled by either modifying proposed method to not backpropgate through various layers, or to somehow enable CoT to be backpropagated through
- Sec. 5 did not make it clear enough that the model being interpreted is not the same as the models used in the results for section. It is possible that the modified training procedure in sec. 5 (i.e. mixing in different stages) could lead to a qualitatively different model, both in terms of performance and reasoning mechanisms. I suggest highlighting this difference more clearly at the start of sec. 5
- The body of section 5.3 is rather poorly written. I am largely confused about what your hypotheses are, and experiments exactly you are conducting. This is a shame --- I think it likely contains interesting findings, but I cannot understand it.

### Questions
- Can you elaborate more on section 5.3? What is the overall claim, what are the experiments done to support that, etc.
- The interpretation that latent thoughts let you search in parallel is interesting. Do you have a sense of how parallel this can be, or how you can quantify this?

### Soundness
3

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
This paper introduces a new LLM training/inference paradigm called Coconut, which includes a reasoning step in the continuous space. By training the model to predict hidden representations instead of actual tokens, the proposed solution could "think" before giving the final outputs without generating the CoT tokens. This paper also conducted experiments on three datasets: GSM8K, ProntoQA, and ProsQA to demonstrate that the proposed method is a more effective CoT parading than baselines such as iCoT and Pause Token.

### Strengths
1. The paper is clearly written and easy to follow.
2. The proposed method is simple and yet effective a well-defined tasks.
3. The experiment is solid and comprehensive.

### Weaknesses
The proposed method's main weakness is that it requires training on specific datasets. This makes it a task-specific solution rather than a general-purpose LLM solution, which diverges from the current research trend. Specifically, through the proposed training, I agree that the model could learn to compress the "thinking steps" into hidden representations. However, such representations could not be generalized to the general domain, which is an advantage of the CoT method. For example, I suspect that simply changing the prompt of asking the questions might significantly influence the final performance.

### Questions
1. Why do you only evaluate the efficiency in terms of tokens? Clock-time might be a better metric since you use the same foundation model as the baselines.
2. The analysis of the impact of thoughts per step in Figure 3 is good. How about more thoughts on the GSM benchmark?
3. The performance conclusion on different benchmarks seems to diverge. Can you summarize what is the best task scenario of applying the proposed method?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new reasoning paradigm for language models, called COCONUT (Chain of Continuous Thought), which allows reasoning in a continuous latent space rather than traditional language-based chains of thought (CoT). By using the last hidden state as an input embedding for subsequent reasoning steps, COCONUT avoids language constraints, potentially enhancing reasoning efficiency.  The authors use a multi-stage curriculum training scheme that gradually shifts from language-based reasoning to continuous thoughts, helping models learn effective latent reasoning representations. Experiment results with pre-trained GPT2 on synthetical datasets show that COCONUT is most effective in planning intensive tasks.

### Strengths
1. The paper proposed a novel approach to study the potential of making all verbal CoT steps latent.

2. The experiment results show the potential of COCONUT to be useful in terms of improved generation efficiency.

### Weaknesses
1. The proposed method requires training with verbal CoT steps but the performance is significantly lower on GSM than vanilla CoT training. While the generated tokens are significantly fewer, it is unclear if the efficiency benefit overweight the performance loss. 

2. A generation efficiency analysis is needed to show the potential efficiency benefit of the proposed method.

3. More number of thoughts per step should be shown in Figure 3 to get a clearer picture of the effect of the thought length.

3. The experiment setting is in general very synthetic, with all three datasets synthetically generated. Some real-world reasoning datasets are needed to show the practical effectiveness of the proposed method.

4. Only one small LLM is used in the experiment, GPT2. More recent LLMs should be used to validate the effectiveness of the method.

5. The latent tree search analysis concludes that latent CoT can do BSF does not make sense to me. It seems that you can do a similar analysis to verbal CoT too.

6. The proposed method directly feeds the last layer output representation as the input embeddings might introduce extra difficulties in learning as the mismatch in input and output latent space, even the input and output embedding matrices are tied in GPT2. I'd suggest adding a trainable linear projection before feed the output representation back to the LLM.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces COCONUT (Chain of Continuous Thought), a new reasoning framework for large language models (LLMs). The authors argue that traditional Chain-of-Thought (CoT) reasoning is inefficient and aims at coherence rather than reasoning. Thus, the motivation is to allow reasoning in a latent and continuous space. The method involves feeding the model's last hidden state directly back as the next input embedding, bypassing the need to generate intermediate reasoning steps into language tokens. COCONUT requires staged training to progressively introduce continuous thoughts in place of explicit language reasoning steps. Evaluations on logical reasoning show that COCONUT allows for greater efficiency by reducing token generation while improving accuracy. On the math reasoning task, COCONUT's performance is close to explicit CoT and outperforms prior methods on latent CoT. Through case studies, the paper shows how COCONUT represents multiple potential reasoning paths and progressively narrows down choices, avoiding premature commitments.

### Strengths
1. The proposed latent CoT framework is a novel contribution to reducing computing for more efficient CoT reasoning. The framework is well-motivated and theoretically sound.
2. The proposed framework outperforms prior implicit CoT methods. On logical reasoning tasks, it improves over explicit CoT with more efficient inference.
3. An interesting analysis of interpretability shows that latent CoT allows the model to progressively eliminate incorrect options in subsequent steps to achieve higher accuracy. This can potentially contribute to the ongoing research about scaling test-time compute.

### Weaknesses
1. The proposed latent CoT framework requires multi-stage training to learn the internalization process from explicit to latent, which requires a large number of epochs over the full model parameters. This training effort hinders the method's general usability, especially with longer reasoning chains and larger parameter sizes. It would be more informative to see a trade-off analysis with larger models (ideally above or within the 70B scale) to compare the required compute and efficiency for COCONUT (internalization training + inference) and explicit CoT (inference).
2. CoT is a pretty general-purpose inference strategy, so I don't think evaluating GSM8K and the two logical reasoning benchmarks is sufficient. The paper should at least include some tasks outside of the training distribution to show that COCONUT matches or is close to the generalizability of traditional CoT, which works out-of-box on many tasks for powerful foundation models. For example, common benchmarks like MMLU (regular & pro), TheoremQA, Human-Eval, ARC, and MedQA. Or hard challenges like "Game of 24", mini crosswords, and ChessBench.
3. GPT-2 does not seem sufficient for experiments involving a general tool like CoT. I would like to see more results with larger and more advanced models like the LLaMA and Qwen families.

### Questions
1. It looks like most of the CoT reasoning chains are under 100 tokens. I wonder if the internalization training will still work if the CoT chains are very long (some medical and pharmaceutical problems can have around 512-1024 tokens of the reasoning process). Will increasing the number of training stages handle long-context CoT chains?

### Soundness
2

### Presentation
3

### Contribution
2
