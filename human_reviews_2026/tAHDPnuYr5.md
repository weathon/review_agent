# Done Is Better than Perfect: Unlocking Efficient Reasoning by Structured Multi-Turn Decomposition

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large Reasoning Models (LRMs) have gained increasing attention over the past few months. Despite being effective, LRMs are criticized for the excessively lengthy Chain-of-Thought (CoT) to derive the final answer, suffering from high first-token and overall latency. Typically, the CoT of LRMs mixes multiple thinking units, some of which are split by markers like "aha", "wait", or "alternatively"; each unit attempts to produce a candidate answer to the original query. Hence, a natural idea to improve efficiency is to reduce the unit number. Yet, the fact that the thinking units in vanilla CoT cannot be explicitly managed renders doing so challenging. This paper introduces Multi-Turn Decomposition (MinD) to decode conventional CoT into a sequence of explicit, structured, and turn-wise interactions to bridge the gap. In MinD, the model provides a multi-turn response to the query, where each turn embraces a thinking unit and yields a corresponding answer. The subsequent turns can reflect, verify, revise, or explore alternative approaches to both the thinking and answer parts of earlier ones. This not only makes the answer delivered more swiftly, but also enables explicit controls over the iterative reasoning process (i.e., users may halt or continue at any turn). We follow a supervised fine-tuning (SFT) then reinforcement learning (RL) paradigm to realize MinD. We first rephrase the outputs of an LRM into multi-turn formats by prompting another LLM, and then tune the LRM with such data. Observing that the tuned model tends to consume even more tokens than the original one (probably due to that the multi-turn formats introduce additional answer tokens), we advocate leveraging RL algorithms like GRPO to prioritize correct outputs with fewer turns. Trained on the MATH dataset using R1-Distill models, MinD can achieve up to ~70% reduction in both output token usage and time to first token (TTFT), while maintaining competitive performance on benchmarks such as MATH-500, AIME24, AMC23, GPQA-Diamond, and LiveCodeBench.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The excessively lengthy Chain-of-Thought (CoT) of Large Reasoning Models (LRMs) substantially increases computational costs and latency. To mitigate this, the paper introduces Multi-Turn Decomposition (MinD), a method designed to improve the reasoning efficiency of LRMs by restructuring the traditional "think-then-answer" CoT process into a structured multi-turn format. Following a supervised fine-tuning (SFT) then reinforcement learning (RL) paradigm, LRMs are trained to provide an answer after every thinking turn and keep every thinking turn brief.

**Key Contributions:**

1. A structured multi-turn reasoning paradigm, where long CoT sequences are decomposed into explicit, turn-based interactions.

2. Experiments: significantly reduces the number of reasoning turns, token usage and time-to-first-token (TTFT), while maintaining competitive accuracy.

### Strengths
1. The idea of decomposing long CoT into multiple intermediate thinking turns and providing intermediate answers is novel.
2. The organization of this paper is logical, and the background work and methodology are easy to understand.
3. This paper focuses on reducing the output length of reasoning models, which is an important issue in the field.

### Weaknesses
1. In Section 3.3, the proposed method is not introduced clearly.

   - The "correct final answer" in Figure 3 and the conventional accuracy reward $\mathcal{R}_{\text{accuracy}}$. Section 3.2  suggests that SFT data is filtered by the "correct final answer," implying the use of complete reasoning chains. Given this, how does the model develop the ability described in Section 3.3 to actively "choose to continue or abort" its reasoning? The observation in 3.3 that the model "tends to generate even more output tokens" further highlights this issue. 

   - Unit Compactness Reward $\mathcal{R}_{\text{unit}}$. The criterion for determining "**contains multiple exploratory trajectories**" requires clarification. In Section 3.2, trajectory splitting is performed by an LLM. Is a similar approach used in RL training process? A more detailed explanation would help the community benefit more from this work. Furthermore, if an LLM is invoked for this judgment *during* training, the associated  API costs would be substantial. I recommend that these potential costs be explicitly stated to underscore the practical considerations of the proposed method.

2. Experiments. The extremely small sizes of the AIME24 (N=30) and AMC23 (N=40) datasets raise concerns regarding the statistical significance of the results reported in Tables 3 and 4. Conducting only a single run makes it difficult to draw reliable conclusions. Notably, the baseline methods DEER [0] and ThinkPrune [1] in Table 3 have already set a precedent by reporting averaged results over multiple runs.
3. About time-to-first-token (TTFT). Since the method does not appear to modify the model’s architecture or its initial processing stage, the connection between the proposed approach and the improvement in Time-To-First-Token (TTFT) is unclear to me.

References:

[0] Yang, Chenxu, et al. "Dynamic Early Exit in Reasoning Models." *arXiv preprint arXiv:2504.15895* (2025).

[1] Hou, Bairu, et al. "Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning." *arXiv preprint arXiv:2504.01296* (2025).

### Questions
Please see weaknesses for the questions. Besides, given that the SFT model already performs explicit reasoning while its outputs are not shorter, the observed length reduction appears to stem from $\mathcal{R}_{\text{unit}}$.

However, the result in Table 5—where length decreases even with a zero $\mathcal{R}_{\text{unit}}$ weight—is highly unexpected and seems to contradict this intuition. The empirical conclusions in Section 4.3 and discussion on "explicit reward term regarding the number of turns" could be strengthened by a more in-depth discussion of this issue. For example, presenting the learning curves (metrics vs. RL steps) for the experiments in Table 5 might help illustrate how the model’s behavior evolves and offer clearer insights into the underlying factors driving length compression.

### Soundness
2

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
4

### Summary
The authors propose the Multi-Turn Decomposition framework to address the issues of excessively lengthy outputs, high time to first token, and the inability to explicitly manage thinking units in the Chain-of-Thought reasoning of Large Reasoning Models. This framework reconstructs Chain-of-Thought through a two-stage process of Supervised Fine-Tuning and Reinforcement Learning: in the Supervised Fine-Tuning stage, another Large Language Model is used to convert single-segment Chain-of-Thought into an explicit multi-turn format of "thinking units—intermediate answers"; in the Reinforcement Learning stage, a tripartite reward signal based on GRPO is employed to suppress redundancy, solving the problem of increased token consumption after Supervised Fine-Tuning. Trained on the MATH dataset, Multi-Turn Decomposition reduces both output token usage and time to first token by approximately 70% while maintaining competitive performance across multiple benchmark tests, filling the gap in the regulation of traditional Chain-of-Thought and providing a new path for efficient reasoning in Large Reasoning Models.

### Strengths
The authors propose a concise method to compress the reasoning process of Large Reasoning Models, which effectively reduces the number of tokens consumed during inference. Additionally, the paper is clearly written, allowing readers to easily follow the content and grasp the key ideas.

### Weaknesses
1. Insufficient reproducibility: The paper fails to provide detailed experimental settings for both the proposed method and baselines, such as specific hyperparameter configurations and whether repeated experiments were performed. This lack of key information hinders other researchers from replicating the study’s results and verifying its conclusions.
2. Limited diversity of baselines: Most baselines in the experiments fall into the category of methods that control early stopping via token budget constraints. There is no comparison with other representative approaches for token optimization, including those based on length penalty rewards, reasoning path pruning, and reasoning token simplification. This narrow baseline scope restricts a comprehensive assessment of the proposed method’s competitiveness in reducing inference tokens.

### Questions
1. The paper would benefit from comparing the proposed method with a wider range of reasoning compression approaches, alongside a detailed discussion of the advantages and drawbacks of each category. Such a comparison would more prominently highlight the unique strengths of MinD in token reduction and inference efficiency.
2. If the method were applied to other model families (e.g., QwQ), would similar or different findings emerge?
3. What impact would providing positive $R_{unit}$ values during Compliance have on the model’s performance?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a methodology to reduce overthinking in the reasoning trace of Chain-of-Thought (CoT)-based ‘reason-then-answer’ methodology employed in off-the-shelf large Reasoning Models (LRM). This methodology adapts the LRM from ‘reason-then-answer’ to ‘Mult-Turn-Decomposition’ (MinD) via Supervised Finetuning (SFT), followed by Reinforcement Learning based GRPO, enabling the generated response as a set of independent reasoning traces, each containing a final answer. Hence, this output format enables early exit and reduces latency and token utilization. Through experiments on two mathematical question answering datasets, they empirically verify the reduction in token usage during the generation of reasoning traces to arrive at the final answer with reduced output latency.

### Strengths
- The research problem under study, i.e., ‘overthinking’ in reasoning generation to arrive at the final answer, is very prominent in LRMs, resulting in frequent wrong generation and latency overload compared to non-LRMs. This work introduces a competitive methodology for addressing this issue. 
- The proposed methodology reformulates the CoT mechanism in LRMs into multi-turn decomposition, where each turn provides independent reasoning with a candidate final answer, allowing early exit and reduced token utilization (e.g., a 68% reduction on MATH-500 for 1.5B) and redundancy in the reasoning chain (Fig. 2, left). 
- The proposed methodology also enables reduced total latency as the computation time between processing the input prompt and final answer generation (Fig. 4).

### Weaknesses
- **Generalization of empirical results beyond the choice of LRM**: The entire experiments are conducted considering only one reasoning model (DeepSeek-R1-Distill-Qwen) with two different model sizes (1.5B and 7B). Though the empirical results regarding reduced token utilization and latency while maintaining model accuracy are clearly demonstrated for the Qwen model, the absence of incorporating other reasoning-based models (eg, DeepSeek-R1-Distill-Llama-8B) limits its generalization and reliability across diverse mode families of LRMs. 
- **Generalization across diverse knowledge-intensive tasks**: The experiments are conducted on a mathematical question answering dataset. However, it would be interesting to analyze the reduction in reasoning trace using MinD on non-mathematical datasets such as commonsense reasoning Q&A datasets with varying reasoning depths (eg, TruthfulQA, OpenBookQA). 
- **Proper definition of thinking units for segmentation of CoT reasoning**: The segmentation of CoT reasoning into a set of independent reasoning units is done using GPT-4o. The quality of this segmentation is crucial for SFT and GRPO-based training. However, there is no error analysis on CoT segmentation quality (through annotation) and the proposed MinD performance.
- **Lack of proper description of the mechanism of early exit**: The proposed mechanism converts the CoT-based ‘reason-then-answer’ to a sequence of independent thinking uint delimited by <think></think> (lines 254-255), each containing a candidate final answer. However, there is no clear discussion on the early exit mechanism.

### Questions
Please see the weaknesses above.

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
This paper aims to address the issue of excessive token usage of LRMs. The authors propose the MinD (Multi-Turn Decomposition) method. Specifically, they first conduct SFT to segment and restructure CoT data into a structured, multi-turn format, where each turn is a "reasoning unit". Then, they leverage the GRPO algorithm's "implicit bias" for short sequences to incentivize the model to use fewer reasoning turns to reach the correct final answer. The method achieves good accuracy and reduces the number of tokens in both in-domain and OOD data.

### Strengths
1. The paper conducts an exploration of reasoning unit redundancy, which better demonstrates the motivation rather than just directly claiming LRM tokens are redundant.

2. A novel method design that utilizes GRPO's implicit ability for shorter turns rather than incorporating a length penalty directly into the reward function.

### Weaknesses
1. Regarding the reasoning units, it is necessary to measure the quality of the obtained unit splits and whether they clearly reflect the LRM's reasoning process.

2. The authors explored too few LRMs with only two models distilled from DeepSeek-R1. This makes it difficult to prove whether the unit segmentation is overly dependent on DeepSeek-R1's text style, and it also needs to be verified if the multi-turn training method is applicable to other LRMs.

3. More baselines could be considered, such as [1][2]. 

[1] Yi, Jingyang, Jiazheng Wang, and Sida Li. "Shorterbetter: Guiding reasoning models to find optimal inference length for efficient reasoning.". 

[2] Zhang, Jiajie, et al. "Adaptthink: Reasoning models can learn when to think, 2025".

### Questions
1. Why does MinD achieve much higher accuracy than the Original LRM on AMC23 but not on other datasets?

2. In Table 4, why does SFT-Only lead to a decrease in accuracy? It seems that it merely changes the format and does not require a reduction in reasoning steps.

### Soundness
3

### Presentation
2

### Contribution
2
