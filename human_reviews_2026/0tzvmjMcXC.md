# SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 2, 8

## Abstract
Recent advances have shown success in eliciting strong reasoning abilities in multimodal large language models (MLLMs) through rule-based reinforcement learning (RL) with outcome rewards. However, this paradigm typically lacks supervision over the thinking process leading to the final outcome. As a result, the model may learn sub-optimal reasoning strategies, which can hinder its generalization ability. In light of this, we propose SophiaVL-R1, as an attempt to add reward signals for the thinking process in this paradigm. To achieve this, we first train a thinking reward model that evaluates the quality of the entire thinking process. Given that the thinking reward may be unreliable for certain samples due to reward hacking, we propose the Trust-GRPO method, which assigns a trustworthiness weight to the thinking reward during training. This weight is computed based on the thinking reward comparison of responses leading to correct answers versus incorrect answers, helping to mitigate the impact of potentially unreliable thinking rewards. Moreover, we design an annealing training strategy that gradually reduces the thinking reward over time, allowing the model to rely more on the accurate rule-based outcome reward in later training stages. Experiments show that our SophiaVL-R1 surpasses a series of reasoning MLLMs on various benchmarks (\textit{e.g.}, MathVisita, MMMU), demonstrating strong reasoning and generalization capabilities. Notably, our SophiaVL-R1-7B even outperforms LLaVA-OneVision-72B on most benchmarks, despite the latter having 10 $\times$ more parameters. All code, models, and datasets will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces SophiaVL-R1, a multimodal large language model designed to significantly enhance reasoning capabilities in complex reasoning tasks. Its primary contribution is the novel Trust-GRPO algorithm, which effectively integrates outcome-level and thinking-level rewards to guide model reasoning. This approach addresses a common limitation in R1-style models, where models often produce correct answers despite following flawed reasoning trajectories—a phenomenon referred to as "wrong thinking, correct answer". The authors conduct extensive evaluations on multiple established reasoning benchmarks, demonstrating that SophiaVL-R1-7B achieves strong performance and even surpasses much larger 72B models.
Overall, the paper is technically rigorous and presents a well-motivated and compelling approach.

### Strengths
1. The paper addresses a practical issue that has received considerable attention in the community: models producing correct answers through incorrect reasoning trajectories, a phenomenon referred to as "wrong thinking, correct answer".
2. The motivation for introducing the thinking reward model is well articulated and convincing, and the discussion of how Trust-GRPO balances thinking and outcome rewards is clear.
3. The empirical evaluations are thorough and rigorous, conducted across several established MLLM benchmarks, including mathematical reasoning datasets (MathVista, MathVerse) as well as general reasoning benchmarks (MMMU, MMStar, etc.).
4. SophiaVL-R1-7B demonstrates strong and competitive performance across multiple MLLM benchmarks for both mathematical and general reasoning tasks, surpassing existing models such as LLaVA-OneVision-72B and InternVL2.5-8B-VisualPRM.
5. Two open-source datasets (SophiaVL-R1-130k and SophiaVL-R1-Thinking-156k) are contributed, which will serve as valuable resources for future research in multimodal reasoning and reward model training.
6. The paper is clearly written and easy to understand.

### Weaknesses
1. The evaluated benchmarks are largely similar to the training datasets in terms of task type (e.g., math). It remains unclear how the model would perform on out-of-distribution tasks.
2. The choice of Qwen2.5-VL-3B as the thinking reward model is not fully justified. It is unclear why a stronger model, such as the labeling model Qwen2.5-VL-72B, was not considered, as a more capable model might provide more reliable reward signals.
3. The SophiaVL-R1-Thinking-156k dataset is filtered and balanced by “manually designed rule-based filtering"(Section 3.2), but the paper doesn't specify what these rules are.

### Questions
1. How well does the proposed method generalize beyond the tasks seen during training?
2. Why not use a stronger model as thinking reward model? For example, why not use the labeling model itself as the thinking reward model?
3. What are the specific “manually designed rule-based filtering” criteria mentioned in Section 3.2?

### Soundness
4

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
3

### Summary
The paper proposes SophiaVL-R1 which rewards the thinking processes of multimodal large language models and performs RLVR on top of this.

### Strengths
1. The presentation of the paper is good.
2. The method is simple and easy to follow.
3. The experiments validate the effectiveness of the method.

### Weaknesses
1. The novelty of this paper is limited. The paper proposed to use a process reward model to give step-level feedback for the thinking tokens. There are several other previous works that also do similar things, some of which are already cited in the paper. The only difference lies in the more fine-grained labeling of the reward, i.e., the five dimensions including Logical Soundness, Correct Reasoning, Error Identification, Language Consistency, and Redundancy. However, giving fine-grained rewards is already widely adopted in reward models, such as ArmoRM.
2. The direct comparison with other reward models is missing. I expect to see the comparison of RMs on both the VLRewardBench benchmark and the results after performing RL with the reward model.
3. Prompting the expert model to give fine-grained rewards is not robust in my view. The inaccuracy is even more pronounced when letting the model give a score from many (10) choices.
4. TRUST-GRPO is not theoretically grounded. For example, why do we need this trustworthiness weighting in principle? Is it possible that the CoTs are bad but the final answer happens to be correct, or vice versa?

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose a new RL-based reasoning framework for MLLMs. Specifically, they introduce a novel reward model to enhance the reasoning process and a trustworthiness weight to make the model’s reasoning more reliable. The proposed framework demonstrates effectiveness across multiple V&L benchmarks, including MMMU and MathVista.

### Strengths
1. The paer is well-written and easy to follow.
2. The proposed idea, a thinking reward model, looks interesting and seems to be effective across multiple V&L benchmarks.
3. The dataset curated for the reward model training is potentially valuable to the research community on further improving MLLM's reasoning capabilities.

### Weaknesses
Generalizability of the proposed method. The authors demonstrate the effectiveness of their approach only on the Qwen-based architecture, raising questions about its applicability to other V&L models.

### Questions
- What if larger models (e.g., Qwen2.5-VL-32B-Instruct) are used as both the reward model and the reasoning model? Will the proposed method still be effective?
- Can the proposed method be applied to other V&L architectures, such as LLaVA-based models?
- Beyond the overall accuracy gain, it would be helpful to include a more detailed analysis illustrating the effectiveness of the proposed thinking reward model. For instance, what proportion of examples show reasoning errors in the baseline that are corrected after applying the reward model?

### Soundness
3

### Presentation
3

### Contribution
3
