# Building Math Agents with Multi-Turn Iterative Preference Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6

## Abstract
Recent studies have shown that large language models' (LLMs) mathematical problem-solving capabilities can be enhanced by integrating external tools, such as code interpreters, and employing multi-turn Chain-of-Thought (CoT) reasoning. While current methods focus on synthetic data generation and Supervised Fine-Tuning (SFT), this paper studies the complementary direct preference learning approach to further improve model performance. However, existing direct preference learning algorithms are originally designed for the single-turn chat task, and do not fully address the complexities of multi-turn reasoning and external tool integration required for tool-integrated mathematical reasoning tasks. To fill in this gap, we introduce a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences. This framework includes multi-turn DPO and multi-turn KTO as specific implementations. The effectiveness of our framework is validated through training of various language models using an augmented prompt set from the GSM8K and MATH datasets. Our results demonstrate substantial improvements: a supervised fine-tuned Gemma-1.1-it-7B model's performance increased from 77.5% to 83.9% on GSM8K and from 46.1% to 51.2% on MATH. Similarly, a Gemma-2-it-9B model improved from 84.1% to 86.3% on GSM8K and from 51.0% to 54.5% on MATH.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new method to improve the mathematical reasoning abilities of large language models. Existing direct preference learning algorithms are mainly applied to single-turn dialogue tasks and fail to address the complexities of multi-turn reasoning and external tool integration. To tackle this, the paper introduces a multi-turn direct preference learning framework and specifically implements two algorithms (M-DPO and M-KTO). Experiments were conducted by training the model on the GSM8K and MATH datasets, and the results demonstrate that this new framework significantly enhances the model’s performance on multiple benchmark tasks.

### Strengths
This paper improves the performance of large language models (LLMs) in multi-turn complex reasoning tasks. The study effectively validates the proposed approach using two datasets.

### Weaknesses
Although the proposed method shows outstanding performance, it is relatively focused on specific task types. Currently, the experiments are limited to the GSM8K and MATH datasets, and it would be better to include more datasets in the testing. Additionally, the framework's design is relatively complex, involving multi-turn preference learning and trajectory-level optimization, which may require high computational resources in practical applications. A comparative experiment to demonstrate the time cost and computational consumption compared to previous methods would be helpful here.

### Questions
1.How is the definition and selection of preference signals specifically designed in this approach? Could different preference signals significantly impact the final optimization results?
2.Is it possible to further test the method in other types of reasoning tasks or application domains to verify its generalizability?
3.In terms of external tool integration, is the current method the optimal solution?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a multi-turn direct preference learning framework, tailored for this context, that leverages feedback from code interpreters and optimizes trajectory-level preferences. This framework includes multi-turn DPO and multi-turn KTO as specific implementations. The effectiveness of our framework is validated through training of various language models using an augmented
prompt set from the GSM8K and MATH datasets.

### Strengths
1. The integration of external tool augmented methods and the use of multi-turn preference learning is both interesting and reasonable, effectively addressing the shortcomings of natural language in mathematical reasoning. The exploration of tool-augmented methods in the mathematical domain is well-motivated.

2. The methods presented in the paper are strongly supported by theoretical foundations, which enhances their credibility, and the overall writing is clear and easy to understand.

### Weaknesses
1. WizardMath, being a state-of-the-art math model from a year ago, seems somewhat outdated as a baseline for performance. I hope the authors can include results from recent stronger models in mathematics that incorporate MULTI-TURN ITERATIVE PREFERENCE LEARNING, such as Qwen 2.5 Math and DeepSeek Math.

2. I would indeed like to see results of this method in the context of a process-based reward model (PRM) and combined with MCTS sampling strategies (even preliminary results would be helpful). Although the authors mention this limitation in the limitations section and in line 297, relying solely on ORM for reasoning verification is not sufficiently detailed.

### Questions
Please see the weaknesses section. I also hope to see further details on the ORM, such as the base model and training method. 

Overall, the paper does not have any obvious weaknesses, and I personally find it quite impressive. I believe it meets the acceptance criteria for ICLR. If you address my concerns, I would consider raising the score.

**Missing reference:**

[1] DotaMath: Decomposition of Thought with Code Assistance and Self-correction for Mathematical Reasoning

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the optimization of multi-turn MDP planning under conditions involving interactions with external tool integration. Through rigorous theoretical derivations, it models the optimal conditions for multi-turn MDP planning and demonstrates that when
the external randomness is low, a reward form using DPO can feasibly compute the utility function in multi-turn MDP planning. Furthermore, the authors propose multi-turn DPO and multi-turn KTO algorithms tailored for optimizing multi-turn MDP planning and validate the effectiveness of combining multi-turn MDP planning optimization with an online iterative approach, showing that performance loss relative to the optimal policy is sublinear, thus highlighting the statistical efficiency of the proposed algorithm. Extensive experiments conducted on various model series across GSM8K and MATH datasets affirm the efficacy of these methods in multi-turn MDP planning.

### Strengths
1. This paper proposes a solution to the language model’s multi-turn planning problem involving external environment interaction, offering well-grounded theoretical proofs and a straightforward formulation.

2. The proposed algorithms are empirically validated to be effective in addressing mathematical problem-solving tasks with external tool integration.

### Weaknesses
1. The utility function form in Equation (7) needs a value network and is computable only under the form of DPO rewards and deterministic external conditions, which limits the practical applicability of this theoretical framework. Currently, the theoretic framework cannot utilize methods like PPO to address multi-turn MDP planning problems with external interactions.

2. The experimental section’s baseline explanation for single-turn iterative DPO lacks clarity regarding the KL regularization strategy, especially in terms of whether a fixed reference is employed, which might contribute to the poorer performance observed in single-turn iterative DPO.

### Questions
None.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates enhancing mathematical reasoning in large language models (LLMs) through a multi-turn preference learning approach with tool integration. The authors propose a multi-turn direct preference optimization framework that accommodates complex, multi-turn reasoning and external tool integration, crucial for mathematical tasks. By employing two specific implementations—multi-turn Direct Preference Optimization (DPO) and multi-turn Kernelized Trajectory Optimization (KTO)—the framework improves model accuracy on math benchmarks, GSM8K and MATH. Empirical results with multiple base models show notable gains, particularly for models fine-tuned with iterative multi-turn training, outperforming standard SFT and single-turn baselines. The study also highlights the importance of balancing model exploration and accuracy through KL regularization and mixture sampling to maintain performance in iterative training.

### Strengths
a.	The paper proposed an iterative multi-turn DPO and KTO training framework, which effectively improves the models ability on multiturn reasoning for math problems.
b.	The experiments are solid, and they conduct experiments with multiple base models.

### Weaknesses
a.	Overall, the motivation makes sense, as for some complex math problems that require tool integration, human inputs might be needed. However, only GSM8K and MATH have been used as the benchmarks in this paper, which are relatively easy problems, and may not necessarily need human input as multi-turn conversation. It is more convincing if you can conduct experiments on more complex tasks, like formulating an optimization problem and solving it from a task description. 
b.	The presentation of the results is not clear enough. For example, table 1 is too long, and different models use different settings, e.g. Wizardmath, codellama and Gemma-1.1-it-7B have different settings from other models. It is more readable if the table is splitted, and each sub table has its own conclusion or findings.
c.	Typos, e.g. line 270 DKO, should be KTO.

### Questions
For the example shown in figure 1, the human inputs (o1 and o2) do not provide additional information, why multi-turn conversation is needed in this case?

### Soundness
3

### Presentation
3

### Contribution
3
