# CLPO: Curriculum Learning meets Policy Optimization for LLM Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recently, online Reinforcement Learning with Verifiable Rewards (RLVR) has become a key paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing methods typically treat all training samples uniformly, overlook the vast differences in problem difficulty relative to the model's current capabilities. This uniform training strategy leads to inefficient exploration of problems the model has already mastered, while lacking effective guidance on the problems that are challenging its abilities the most, limiting both learning efficiency and the performance upper-bound. To address this, we propose \textbf{CLPO (Curriculum-guided Learning for Policy Optimization)}, a novel algorithm that creates a dynamic pedagogical feedback loop within the policy optimization process. The core of CLPO is to leverage the model's own rollout performance to conduct real-time difficulty assessment, thereby constructing an \textbf{Online Curriculum}. This curriculum then guides an \textbf{Adaptive Problem Restructuring} mechanism, where the model acts as its own teacher: it diversifies medium-difficulty problems to promote generalization and simplifies hard problems to make them more accessible. Our approach transforms the static training procedure into a dynamic process that co-evolves with the model's capabilities. Experiments show that CLPO achieves \textbf{state-of-the-art (SOTA)} performance across eight challenging mathematical and general reasoning benchmarks, with an average \textbf{pass@1} improvement of \textbf{6.96\%} over ohter methods, demonstrating its potential for more efficiently training more capable reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CLPO, a novel framework for improving LLM reasoning via reinforcement learning. CLPO creates a dynamic curriculum by using the model's own performance to assess problem difficulty in real-time. It then restructures problems, diversifying medium ones and simplifying hard ones, to create an optimal training batch, leading to a more efficient and effective learning process that co-evolves with the model's capabilities.

### Strengths
1.	The core idea of "Guided Self-Evolution" is innovative and directly addresses the limitations of uniform data sampling in RLVR. It creates an adaptive, self-improving loop that is both intuitive and powerful.
2.	The paper provides strong empirical validation through extensive experiments, including detailed ablation studies for each component and a pass@k analysis that demonstrates improved solution diversity.

### Weaknesses
The CLPO framework introduces several steps that seem computationally expensive compared to baselines like GRPO. Specifically, for each batch, CLPO requires (1) generating N rollouts to assess the difficulty of each original problem, and (2) using the LLM itself to rewrite the "medium" and "hard" problems, and then (3) generating another N rollouts for these newly restructured problems to filter them. This overhead is non-trivial and is a critical factor for practical applications. The paper lacks any analysis or discussion of the computational cost (in terms of time or FLOPs) compared to the baseline methods.

### Questions
1.	Which specific Qwen3-8B was used—the base or the post-training?
2.	Did you reproduce all the baseline results in Table 1 yourselves?
3.	The core components like adaptive restructuring and dynamic KL seem similar to prior work [1,2,3]. You seem to have achieved this merely by assembling or tuning parameters. What is the key novelty beyond combining these ideas?
4.	How do you guarantee that "Adaptive Problem Restructuring" strictly preserves the original answer?
5.	How does CLPO compare to DAPO when evaluated under an equal computational budget (e.g., fixed training time)?

[1] Xu C, Sun Q, Zheng K, et al. WizardLM: Empowering large pre-trained language models to follow complex instructions[C]//The Twelfth International Conference on Learning Representations. 2024.

[2] Wang Y, Kordi Y, Mishra S, et al. Self-instruct: Aligning language models with self-generated instructions[J]. arXiv preprint arXiv:2212.10560, 2022.

[3] Yu Q, Zhang Z, Zhu R, et al. Dapo: An open-source llm reinforcement learning system at scale[J]. arXiv preprint arXiv:2503.14476, 2025.

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
CLPO (Curriculum-guided Learning for Policy Optimization) is proposed to create a dynamic pedagogical feedback loop within the policy optimization process.

### Strengths
1.The proposed RL paradigm could run automatically.
2.The experiments show improvements across a variety of datasets.

### Weaknesses
1.It's better to show improvements for some datasets such as AIME25, LCV-V6.
2.The MATH-500 performance is not improved.
3.The improvements is not obvious for some datasets such as AMC23, MMLM-Pro
4.Qwen-3B is in the experiments, there are not other base-model be evaluated. 
5.There is limited novielty in the proposed algorithm.

### Questions
1.Why are improvements demonstrated only on specific datasets like AIME25 and LCV-V6, and not across a broader range of benchmarks?
2.Why has performance on the MATH-500 dataset remained unchanged despite the proposed method?
3.Why are the reported improvements minimal or inconsistent on datasets such as AMC23 and MMLM-Pro?
4.Why is Qwen-3B the only base model evaluated in the experiments, with no comparison to other foundational models?
5.What novel contributions does the proposed algorithm introduce beyond existing methods?

### Soundness
3

### Presentation
3

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
This paper proposes CLPO (Curriculum-guided Learning for Policy Optimization), a novel algorithm to improve training efficiency for Large Language Models (LLMs) in reasoning tasks using Reinforcement Learning with Verifiable Rewards (RLVR). Unlike existing methods that treat all training samples equally, CLPO dynamically assesses problem difficulty based on the model's performance and constructs an Online Curriculum. It restructures problems by simplifying hard ones and diversifying medium-difficulty ones, enabling the model to act as its own teacher. Experiments on eight benchmarks show CLPO achieves state-of-the-art (SOTA) results, improving pass@1 by 6.96%, demonstrating its effectiveness in training reasoning models.

### Strengths
1. The paper addresses an important and timely topic—leveraging reinforcement learning (RL) to enhance the reasoning capabilities of large language models (LLMs). RL plays a crucial role in improving LLMs, and this work contributes meaningfully to advancing RL-based training methods.
2. The experimental evaluation is thorough, covering eight challenging mathematical and reasoning benchmarks. The results clearly demonstrate the effectiveness of the proposed method, with improvements over existing approaches.
3. The authors provide code, which enhances reproducibility and facilitates further research, making the work accessible and impactful for the broader community.

### Weaknesses
1. While the topic is significant, the idea of integrating curriculum learning into reinforcement learning is not entirely new. The paper's innovation is somewhat incremental, and it could benefit from a deeper discussion on how CLPO differs from and improves upon existing similar methods. Highlighting unique contributions more explicitly would strengthen the paper's originality.
2. Although the authors provide code, its usability is limited due to poor readability and the lack of a basic README file or documentation to guide users on how to reproduce results or run experiments. This makes replication challenging and reduces accessibility. Providing well-documented and user-friendly code would greatly enhance the paper's impact and utility for the research community.

### Questions
Please refer to weakness.

### Soundness
2

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
This paper proposes Curriculum-guided Learning for Policy Optimization, a reinforcement learning framework that integrates curriculum learning into RL with verifiable rewards to enhance reasoning in LLMs. Unlike conventional RLVR methods that uniformly sample problems regardless of difficulty, CLPO dynamically constructs an online curriculum by evaluating the model’s rollout performance. This curriculum drives an adaptive problem restructuring mechanism that diversifies medium-difficulty problems to promote generalization and simplifies overly hard ones for accessibility. Furthermore, a difficulty-aware policy optimization strategy introduces dynamic KL regularization to balance exploration and exploitation. Experiments show improvement over several baselines.

### Strengths
- Proposes CLPO, a novel integration of curriculum learning and RLVR, enabling dynamic, self-adaptive policy optimization.

- Introduces an adaptive problem restructuring mechanism, where the model acts as its own teacher by adjusting problem difficulty.

- Employs difficulty-aware policy optimization via dynamic KL regularization to effectively balance exploration and exploitation.

- Demonstrates better results across eight reasoning benchmarks.

### Weaknesses
- The evaluation dataset selection appears curated, focusing on benchmarks (e.g., AIME2024, GPQA Diamond) that may favor mathematical reasoning. It remains unclear why more challenging or recently adopted datasets such as AIME25, GPQA, ACPBench, or HeadQA were not included for broader validation.

- Experiments are conducted only on the Qwen3-8B base model, limiting insights into scalability or robustness across different model sizes and architectures.

- The dynamic KL regularization analysis (varying scaling factors α) lacks clear trends—results appear noisy or unstable, suggesting the method’s sensitivity to hyperparameter tuning.

- The adaptive restructuring mechanism likely incurs non-trivial computational overhead, but the paper does not provide analysis on training efficiency or cost.

- The discussion could be better contextualized with respect to related self-play, auto-curriculum, or competence-based RL literature to highlight conceptual novelty.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3
