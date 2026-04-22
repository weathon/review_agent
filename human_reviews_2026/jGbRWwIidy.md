# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Recent advancements in long chain-of-thought (CoT) reasoning, particularly through the Group Relative Policy Optimization algorithm used by DeepSeek-R1, have led to significant interest in the potential of Reinforcement Learning with Verifiable Rewards (RLVR) for Large Language Models (LLMs). While RLVR promises to improve reasoning by allowing models to learn from free exploration, there remains debate over whether it truly enhances reasoning abilities or simply boosts sampling efficiency. This paper demonstrates the profound impact that RLVR has on the reasoning capabilities of LLMs. We revisit Pass@K experiments and show that RLVR can extend the reasoning boundary for both mathematical and coding tasks. This is supported by our introduction of a novel evaluation metric, CoT-Pass@K, which captures reasoning success by accounting for both the final answer and intermediate reasoning steps. Furthermore, we present a theoretical framework explaining RLVR's incentive mechanism, demonstrating how it can encourage correct reasoning even when rewards are based solely on answer correctness. Our analysis of RLVR's training dynamics reveals that it incentivizes correct reasoning early in the process, with substantial improvements in reasoning quality confirmed through extensive evaluations. These findings provide strong evidence of RLVR's potential to enhance LLM reasoning, offering valuable insights into its mechanisms and performance improvements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript investigates a fundamental open question in reinforcement learning with verifiable rewards (RLVR): does RLVR merely improve sampling efficiency, or does it endow the base model with fundamentally new reasoning capabilities? The authors first revisit the Pass@K metric, which the RL community has often considered a signal of exploration, and identify its potential pitfalls. Specifically, base LLMs are capable of producing incorrect Chains of Thought (CoTs) yet coincidentally arriving at the ground truth. To address this, they introduce a new metric, CoT-Pass@K, which simultaneously considers the correctness of both the reasoning trace and the final answer. By re-examining the model's reasoning performance with CoT-Pass@K, the authors observe that RLVR successfully extends the reasoning boundary of the base model in both code and math tasks. The manuscript also presents primary theorems on GRPO's advantage estimation to interpret these empirical observations.

### Strengths
- The exact role of RLVR in enhancing the reasoning capabilities of base models is an open and fundamental problem for the RL community. Thus, this manuscript begins with a very strong motivation and is, in my view, highly timely. 

- Although the proposed Theorem 1 is relatively simple and straightforward, this preliminary explanation is helpful for the community to understand the underlying reasons why RLVR works. 

- The authors base their conclusions on extensive experiments. I particularly appreciate that they not only experimented with the Qwen+Math setup but also conducted additional experiments on coding tasks.

### Weaknesses
- The correctness of a CoT is determined by a R1-distilled-Qwen3-8B model, which may not be a sufficiently strong verifier. Since reasoning models may frequently exhibit rethinking or backtracking behavior, I have concerns about whether an 8B model is reliable enough to verify complex reasoning traces that include self-correction, reflection, and backtracking. For example, what if a model initially thinks incorrectly but then corrects its answer on its own? Is the distilled 8B model strong enough to verify such a CoT? The authors acknowledge this limitation in line 468, but I still feel it is noteworthy. Stronger frontier close-source LLMs maybe a better choice, e.g., Gemini-2.5-pro, if the cost of API calls is affordable.

- The empirical observations from the post-SFT experiments (lines 429-431) are difficult for me to follow. It is counterintuitive to me that: 1) tuning on erroneous CoTs can also improve Pass@1 accuracy, ultimately leading to nearly the same performance, and 2) SFT on all CoTs results in worse accuracy than SFT on only incorrect CoTs. Does this observation indicate that the "incorrect" CoTs identified by Distilled-Qwen3-8B only contain erroneous steps at the beginning (which the model self-corrects later) rather than the entire reasoning trace being wrong? And thus SFT on these partially incorrect reasoning traces can also benefits pass@1 accuracy. If this is true, I have further concerns about the reliability of the CoT verifier and the definition of a "correct" CoT.

In summary, I am positive about this manuscript, and I find its findings inspiring and important. However, due to the two concerns mentioned above, I do not have enough confidence to give it a higher rating.

### Questions
1. How would the authors define the correctness of a CoT? That is, if a model makes mistakes initially but then corrects them in subsequent tokens through self-reflection, should such a CoT be considered incorrect?

2. It would be good to see the influence of SFT on both Pass@K and CoT-Pass@K. For example, DAPO-Qwen-32B and DeepSeek-R1-Distilled-Qwen-32B have similar avg@32 on AIME24. It would be interesting to see which model performs better when using CoT-Pass@K as a metric.

### Soundness
2

### Presentation
3

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
* **Starting Point**: The paper aims to argue that **RLVR truly enhances the model's capabilities**
* **New Metric**: The paper proposes the **CoT-Pass@K** metric, which is better suited than the commonly used Pass@K to demonstrate the **model's capability improvement after RLVR**
* **Experimental Validation**: The experiments verify that for multiple math tasks, **CoT-Pass@K better highlights the performance gap** in model capabilities compared to Pass@K

### Strengths
1. The paper reads smoothly and presents its ideas in a clear, easy-to-follow style.
2. The experimental design is comprehensive and well thought-out.
3. RLVR provides only marginal gains at Pass@1024 or Pass@2048, but demonstrably boosts Pass@1—a widely recognized pattern. This supports the prevailing view that RLVR achieves its effect by learning correct reasoning, a claim for which your experiments offer some evidence.

### Weaknesses
Please refer to the "Questions" section for details.

### Questions
1. Why opt for a smaller model to judge the reasoning paths? Doesn't the accuracy of the judgment become constrained by the capacity of the referee model? Should a more powerful referee model be employed instead?

2. For tasks where a direct reward signal is unavailable in standard RL training, the common approach is often to use a rubric combined with a powerful evaluation model to assign rewards. Given this, could your CoT-Pass@K metric be integrated into the RLVR training process to yield beneficial results?

3. Based on the experimental results, we observe that the CoT-Pass@K score for 'no-thinking' models is generally suppressed compared to their Pass@K score. However, this method does not appear to provide a significantly better evaluation metric for RLVR models (as the difference between Pass@K and CoT-Pass@K for RLVR seems minor). Therefore, what are the practical applications or use cases for your proposed method?

### Soundness
2

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
3

### Summary
This paper investigates whether Reinforcement Learning with Verifiable Rewards (RLVR) genuinely improves the reasoning capability of large language models, rather than merely increasing sampling efficiency. It provides both theoretical analysis and empirical evaluation to show that RLVR implicitly incentivizes correct reasoning chains. The authors introduce a new evaluation metric, CoT-Pass@K, which verifies both intermediate reasoning correctness and final answers, and demonstrate that GRPO-based RLVR improves reasoning quality across multiple math and code benchmarks. Results suggest that RLVR not only reinforces correctness but also enhances reasoning consistency and generalization.

### Strengths
1. The paper provides a well-structured theoretical analysis explaining how GRPO gradients implicitly favor logically consistent reasoning traces, offering insights on why RLVR often improves reasoning without explicit step-wise rewards.

2. Experiments on multiple reasoning domains (math and code) and models (Qwen, Nemotron) show consistent gains under both Pass@K and the proposed CoT-Pass@K metrics. The analyses of training dynamics and reasoning correctness lend strong empirical support to the theoretical claims.

3. The new CoT-Pass@K metric could be a useful tool for the community, bridging the gap between evaluating correctness of answers and correctness of reasoning.

### Weaknesses
1. The reliability of CoT-Pass@K hinges on judgments from a single verifier (DeepSeek-R1-Qwen3-8B). While multi-verification is discussed, evaluating under different verifiers would improve confidence in these results. How robust are CoT-Pass@K results when using different verifiers? -- we can fix the verifier used in training and just use different verifiers for evaluation.

2. Potentially questionable assumptions. The theoretical analysis assumes that reasoning traces with correct answers have systematically higher probabilities, which may not always hold in domains like commonsense or multi-hop reasoning. The practical validity of this assumption could be discussed more thoroughly. Do we have any empirical examples?

3. While the paper shows improved correctness, it does not analyze whether RLVR reduces reasoning diversity or leads to overconfident, deterministic reasoning traces. Add some quantitative analysis of reasoning diversity would further strengthen the paper.

### Questions
- Could RLVR’s implicit reasoning incentive be combined with explicit step-wise rewards to further boost reasoning reliability?
- It'd be great to see more details on training cost/efficiency and scaling behavior

### Soundness
3

### Presentation
3

### Contribution
3
