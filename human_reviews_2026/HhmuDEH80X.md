# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Synergizing-Oriented Multi-Turn Reinforcement Learning

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores reasoning compression for large language models (LLMs) by analyzing the importance of individual thoughts within long Chain-of-Thought (CoT) traces. The authors propose **Long⊗Short**, a two-model framework where a "long-thought" LLM generates critical reasoning steps and a "short-thought" LLM handles less important segments. The method combines cold-start supervised fine-tuning and multi-turn reinforcement learning (RL) between the two models. Experiments on multiple reasoning benchmarks (MATH500, AIME24/25, AMC23, GPQA) show moderate accuracy gains and over 80% reduction in token length compared to long-CoT baselines.

### Strengths
- The paper addresses an important and active topic: improving reasoning efficiency of LLMs through CoT compression.  
- The experimental section is comprehensive, including diverse datasets and ablation studies.  
- The asynchronous multi-turn RL design is well-documented and empirically stable.

### Weaknesses
- **Limited novelty**: The core contribution is mainly a pipeline combining existing techniques (CoT chunking, two-model SFT, RL fine-tuning), without a clear new theoretical or algorithmic insight.  
- **Two-model framework**: Requires training and maintaining two separate LLMs, which increases complexity and weakens industrial practicality.  
- **Unfair comparison**: The baselines (e.g., UPFT, DAST, C3oT) use single-model setups, while this approach leverages two coordinated models.  
- **Marginal improvements**: Performance gains are small (≈1–2 points) and become saturated for larger models (>8B), suggesting that scaling benefits are not fully realized.  
- **Weak ablation**: The paper does not sufficiently isolate the contribution of the “Long⊗Short synergy” beyond generic SFT and RL benefits.

### Questions
1. How sensitive are the results to the specific choice of long vs. short thought partitioning rules?  
2. Could the same compression effect be achieved with a single model using controlled CoT length or token-level regularization?  
3. How does the method perform under real inference-time constraints (e.g., limited decoding budget)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method to improve the reasoning process of large language models by distinguishing between “genuine” and “non-genuine” thoughts within the chain-of-thought framework. The authors introduce a filtering mechanism designed to identify productive intermediate reasoning steps and suppress unhelpful or misleading ones, with the goal of enhancing both reasoning quality and interpretability. Experiments are conducted on a range of reasoning benchmarks to evaluate the method’s effectiveness.

### Strengths
1. The idea of treating reasoning traces as heterogeneous and selectively weighting or pruning them is interesting to me.
2. The experimental section is well-organized, with comparisons across standard benchmarks such as GSM8K and StrategyQA. The paper is generally well written.

### Weaknesses
1. The paper’s central notion of “genuine thought” is not well-defined in operational or mathematical terms. The method appears to rely on subjective or post-hoc labeling of reasoning steps, rather than any verifiable criterion. This makes the approach unscientific and difficult to reproduce.
2. The proposed filtering algorithm seems to be an ad hoc combination of existing techniques such as CoT pruning or confidence-based re-ranking. There is no theoretical justification or ablation showing how each component contributes to performance.
3. The evaluation is not taht unconvincing. The experiments are limited to small-scale reasoning benchmarks with unclear experimental settings. It is still questionable that if the method can generalize beyond the chosen datasets.
4. The manuscript contains exaggerated statements about “redefining thought” or “unifying reasoning and cognition,” which are not supported by any empirical or theoretical results. Such philosophical or speculative claims weaken the credibility of the technical content.

### Questions
1. What is the formal definition or measurable criterion for “genuine thoughts” in operational or mathematical formulation?
2. Is there any ablation studies that report performance with different filtering thresholds or criteria?

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
This paper explores how to compress long CoTs in LLMs to improve reasoning efficiency.
It uses a three-step approach.
The authors analyze the importance of different thoughts using automatic CoT chunking and Monte Carlo rollouts.
They introduce a metric that jointly measures thought effectiveness and efficiency.
Based on this, they propose Long&&Short, a collaborative reasoning framework involving two LLMs: one focusing on key long thoughts, the other on concise short thoughts.
Both models are fine-tuned with cold-start data and further optimized via a multi-turn reinforcement learning approach encouraging synergy.

### Strengths
1.  The efficiency of reasoning CoT is indeed one of the key challenges for large-scale LLM applications.
2.  The formulas and figures in this paper are presented clearly, and the authors carefully highlight key points with different colors.
3.  The paper first explores how different parts of the CoT affect the results, then proposes a comprehensive metric, and finally uses these insights for training — this overall approach is very well-grounded and makes perfect sense.

### Weaknesses
There is still much to explore regarding the experiments; please refer to the questions section below.

### Questions
1.  The authors use LLMs to chunk the CoT, but since different problems may have CoTs of varying lengths and formats, have the authors considered the potential influence of CoT length and structure?
2.  Table 1 is not presented clearly enough—at first glance, it’s hard to intuitively understand how the two LLMs collaborate and what benefits this brings, which makes the paper more difficult to follow.
3.  Could the authors further elaborate on the conclusions drawn from Figure 2(b)?
4.  Why did the authors choose to compare their method with distilled model versions in the experiments?
5.  The paper seems to lack more analytical experiments, such as testing different training settings or algorithms to directly quantify the improvement in reasoning efficiency. If I understand correctly, the two LLMs are used to generate training data (for long-thought and short-thought reasoning), and this dataset is then used to train a single LLM. Could the authors provide evidence that the trained LLM has indeed acquired this native reasoning style?

### Soundness
2

### Presentation
4

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
This paper proposes "Long & Short," a novel framework for efficient CoT reasoning that bypasses the inefficiency of uniform compression by leveraging two specialized LLMs. The method first quantifies the importance of individual thoughts via Monte Carlo rollouts, and then trains a long-thought and a short-thought LLM to collaboratively solve problems using synergizing-oriented multi-turn reinforcement learning, achieving over 80% token reduction with comparable accuracy to full CoT models.

### Strengths
1. The Monte-Carlo rollout study shows front thoughts contribute the most to accuracy while often inflating length, motivating selective preservation.
2. The cold-start SFT builds long and short datasets directly from scored thoughts.
3. The paper achieves a token length reduction of over 80% while maintaining performance on challenging, multi-step benchmarks.

### Weaknesses
1. In Figure 2, the general trend should show accuracy improving as more thinking steps are added, followed by a slight decrease. However, the highest accuracy for Math500 occurs at 0 thought chunks, indicating that the best performance is achieved without any reasoning process. The curve declines too early. Could the authors clarify why this happens?
2. In Lines 200–201, the phrase “assign higher scores to thoughts with shorter context length” seems inaccurate. It would be more precise to state “assign higher scores to earlier thoughts.”
3. Figure 3 seems to have a similar pattern to Figure 2, where the score decreases as the number of thoughts increases. Does this imply that responses without thoughts achieve the best performance?
4. In Lines 214–215, the authors mention that high scores are predominantly associated with front thoughts. Given this, why not employ a training-free approach such as early stopping instead? What is the benefit of combining SFT and RL under this setting?
5. It is better to include an ablation study on the decay penalty term $\delta (y_i)$.

### Questions
1. The paper is not well-written, and several parts require further clarification.
2. In line 157, the authors mention performing rollouts at each thought. Does this mean that the model is prompted with the question and the accumulated thoughts as the instruction, and then asked to generate the final answer directly without additional reasoning?
3. The use of $N_i^{sum}$ and $N_i^{right}$ is confusing. Since each rollout contains multiple thoughts, and $N_i^{sum}$ represents the total number of rollouts, it should not be specific to a particular thought $i$. Could the authors clarify this notation?
4. Could the authors elaborate on how the SFT training data is constructed? If my understanding is correct, the process starts from long-thought data, and for less important thoughts, the model switches from long to short thoughts. Then, how are $D_{long}$ and $D_{short}$ created? How are tags such as <rethink> generated or assigned during this process?

### Soundness
2

### Presentation
2

### Contribution
2
