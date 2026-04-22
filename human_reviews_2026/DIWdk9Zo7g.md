# Interleaved Reasoning for Large Language Models via Reinforcement Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Long chain-of-thought (CoT) significantly enhances the reasoning capabilities of large language models (LLMs). However, extensive reasoning traces lead to inefficiencies and increased time-to-first-token (TTFT). We propose a novel training paradigm that uses reinforcement learning (RL) to guide reasoning LLMs to interleave thinking and answering for multi-hop questions. We observe that models inherently possess the ability to perform interleaved reasoning, which can be further enhanced through RL.
We introduce a simple yet effective reward scheme to incentivize correct intermediate steps, guiding the policy model toward correct reasoning paths by leveraging intermediate signals generated during interleaved reasoning. Extensive experiments across five diverse datasets and three RL algorithms (PPO, GRPO, and REINFORCE++) demonstrate consistent improvements over traditional think-answer reasoning, without requiring external tools. Our method improves final task accuracy and overall efficiency by enabling more effective credit assignment during RL. Specifically, our approach reduces TTFT by over 80\% on average, reduces overall reasoning length by 37\%, and achieves an average 12.5\% improvement in final Pass@1 accuracy. Furthermore, our method, trained solely on question answering and logical reasoning datasets, exhibits strong generalization to complex reasoning datasets such as MATH, GPQA, and MMLU. Additionally, we conduct in-depth analysis to reveal several valuable insights into conditional reward modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Interleaved Reasoning (IR), an RL framework that enables LLMs to alternate between thinking and answering during reasoning. The approach introduces special <think> and <answer> tags and applies a conditional reward scheme to encourage correct intermediate answers while maintaining reasoning structure. Experiments across five reasoning datasets and three RL algorithms show consistent improvements in both reasoning efficiency and task accuracy, with notable reductions in time-to-first-token (TTFT).

### Strengths
- The paper addresses an important issue in reasoning LLMs — the trade-off between long, reflective reasoning chains and real-time efficiency — and provides a clear framework to investigate this trade-off.
- The method is easy to reproduce and compatible with standard RL algorithms, offering practical insights into the impact of intermediate rewards and structured reasoning patterns.
- The experiments covering multiple datasets, RL algorithms, and detailed ablations that help clarify the empirical effects of the proposed framework.

### Weaknesses
- Marginal technical novelty. The core mechanism of interleaved reasoning — introducing <think> and <answer> tags and training models to alternate between them — is relatively incremental. The method essentially reformulates the output format and applies a rule-based reward scheme within a conventional RL training setup.
- Potential conflict with self-reflection. Traditional think-then-answer reasoning allows the model to perform internal reflection and correction before producing a final response. The proposed interleaved approach may hinder such self-correction by encouraging the model to provide partial answers prematurely, promoting short-sighted reasoning and reducing the chance of global reassessment of reasoning paths.
- Limited evaluation of model adaptability. Although the paper compares different RL algorithms, it does not analyze how well the interleaved paradigm adapts across models with different reasoning characteristics. In particular, models like Qwen3, which already support explicit structured thinking, may behave differently, and this dimension is not sufficiently explored.
- Reward validation details. While a “format reward” is introduced to enforce the alternation between <think> and <answer>, the paper does not clearly describe how this formatting is verified or how to ensure the semantic distinction between thinking and answering segments. This raises concerns about the reliability of the reward signal.

### Questions
- The current experiments focus on relatively simple multi-hop reasoning tasks. Would the proposed method remain effective on more challenging benchmarks such as AIME 2024/2025 or similar high-difficulty reasoning datasets?
- Can the proposed interleaved reasoning framework be effectively applied to thinking models such as Qwen3, which already perform explicit reasoning-then-answering? Since Qwen2.5 reportedly “inherently possesses the ability to interleave,” could the observed gains simply reflect properties of the base model rather than the proposed RL scheme?
- In the reward design, the format reward depends on correctly distinguishing between <think> and <answer> segments. How is this verification implemented? How does the system ensure that the content inside <think> truly represents reasoning rather than a disguised partial answer?

### Soundness
2

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
The paper proposes training LLMs to alternate \<think\>\</think\> and \<answer\>\</answer\> with a conditional, rule-based reward for intermediate steps. The authors evaluate the proposed method on both in-domain (K&K, Musique) and out-of-domain (GPQA, MMLU, MATH) datasets.

### Strengths
- Clear motivation.

### Weaknesses
1. The definition and interpretation of the TTFT metric appear conceptually weak. In real applications, users primarily care about the final answer. For example, in Figure 1, the user’s actual need is to know the final “director,” not the model’s intermediate reasoning. If users are interested in the reasoning process, they can view the entire “think” chain, so interleaving does not inherently improve usability. Moreover, TTFT can be easily reward hacking: a model could output any meaningless statement early to artificially boost the metric. This undermines its validity as a measure of responsiveness. 

2. The paper’s presentation of the intermediate reward mechanism is highly unclear. Although Equation (3) introduces the concept, it never specifies how correctness of intermediate steps is evaluated. The brief note that “f(y_answer^(k)) evaluates the answer correctness at step k” is insufficient and leaves readers guessing. Appendix D.2 defines a Correct() function, but again fails to explain the actual criterion for correctness. Determining intermediate correctness without ground truth is a well-known open challenge, and this work appears to sidestep it. The authors only declare that such a function exists, without revealing its logic, heuristics, or data sources. As a result, the core mechanism of this paper cannot be understood or reproduced. 

3. The paper seems to sum all intermediate rewards and treat the result as a single final reward, according to Equations (3) and (5). If this is true, the model receives one identical reward signal across all tokens and steps, which defeats the goal of localized credit assignment. The approach would then be functionally equivalent to the traditional final-reward RL setup, with no genuine per-step feedback. The authors should clarify this point explicitly and explore using true step-level intermediate rewards, where each intermediate step receives its own immediate feedback. 

4. The reported baseline results differ drastically from known benchmarks, casting doubt on the validity of the claimed improvements. In Tables 8 and 9 of the Qwen2.5 technical report, Qwen2.5-7B-Instruct reaches about 75.5 on MATH and 36.4 on GPQA, but this paper reports only 56.1 and 19.2. Similarly, Qwen2.5-1.5B-Instruct achieves 55.2 and 29.8 on those tasks respectively in the official report, whereas the paper reports only 30.8 and 6.6. Such large discrepancies cannot be attributed to random variation. The authors should reconcile these differences.

### Questions
See Weaknesses

### Soundness
2

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
This paper proposes an RL-based paradigm intended to improve LLM reasoning efficiency by alternating between <think> (internal reasoning) and <answer> (public intermediate outputs). The authors introduce a conditional intermediate reward to encourage correct sub-answers, claiming reductions in TTFT and gain in Pass@1 across several reasoning datasets. Experiments are conducted on Qwen 2.5-1.5B/7B using PPO, GRPO, and REINFORCE++.

### Strengths
- The motivation of reducing TTFT while maintaining or improving reasoning quality is timely.
- The interleaved format is simple and compatible; the use of special tags (<think>, <answer>) is pragmatic.

### Weaknesses
- Training relies on K&K and Musique, both containing explicit sub-problem labels. 

- Generalization to domains lacking intermediate supervision is claimed but not empirically tested. 

- No ablation shows sensitivity to ε or to the relative weighting β in Eq. 2. 

- No direct wall-clock latency or token-throughput numbers are reported.

- The experiments are conducted solely on the Qwen series models, without validation on other model families. In addition, the method has not been tested on models larger than 7B parameters.

- The proposed “conditional intermediate reward” essentially re-weights existing PRM signals based on batch accuracy; this appears to be an incremental engineering trick with limited contribution to the community.

### Questions
- Please provide variance across random seeds and confidence intervals for all Pass@1 results.

- In Figure 1, the “long” vs. “short” TTFT examples are described qualitatively. Could you quantify this difference with concrete numbers or timing statistics?

- Can you demonstrate actual wall-clock TTFT improvements under identical inference conditions?

- Does the model truly generalize without intermediate-ground-truth supervision? 

- Ablate each term in Eq. (5): What happens if r_format or r_intermediate is removed?

### Soundness
1

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
3

### Summary
This paper introduces a novel training paradigm utilizing RL to encourage LLMs to adopt an interleaved reasoning strategy, which mixes thought steps and answer steps, for multi-hop question answering. The primary motivation is to mitigate the inefficiency and increased TTFT associated with traditional long reasoning. The authors demonstrate that this RL-based approach can significantly reduce latency while maintaining or improving performance under certain reward schemes.

### Strengths
1. The most interesting result is the significant reduction in the TTFT. This directly addresses a critical and practical limitation of using CoT in real-world applications and demonstrates the potential for faster, more efficient reasoning in LLMs.

2. The paper is generally well-structured and easy to follow. The methodology and experimental setup are clearly explained, which aids in the comprehension of the proposed interleaved reasoning framework.

### Weaknesses
1. A major concern is that the interleaved reasoning appears weak when not supported by dense intermediate rewards. The performance of the model in the setting without strong intermediate reward signals is often suboptimal or even worse than standard baselines, suggesting that the RL framework alone is not sufficient to reliably improve reasoning quality.

2. While the model's performance significantly improves when strong intermediate rewards are introduced, this outcome is largely expected as the model receives a much stronger signal than standard end-to-end RL or supervised methods. The current comparison to baselines in this heavily-rewarded setting is therefore not entirely fair or informative. A more rigorous comparison is needed against models that also leverage similar process-level supervision or intermediate reward signals.

### Questions
1. Why does the proposed interleaved method, without the aid of dense intermediate rewards, struggle to maintain or improve performance compared to baselines? What architectural or training adjustments could be made to improve the robustness of the core interleaving mechanism?

2. Could the authors perform a comparison against representative baselines that also incorporates process rewards or intermediate supervision? This would significantly strengthen the claim that the interleaved structure is the source of the gain, rather than just the dense reward signal.

### Soundness
3

### Presentation
3

### Contribution
2
