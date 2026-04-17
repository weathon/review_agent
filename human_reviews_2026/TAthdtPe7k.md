# Provable and Practical In-Context Policy Optimization for Self-Improvement

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
We study test-time scaling, where a model improves its answer through multi-round self-reflection at inference. We introduce In-Context Policy Optimization (ICPO), in which an agent optimizes its response in context using self-assessed or externally observed rewards without modifying its parameters. 
To explain this ICPO process, we theoretically show that with sufficient pretraining under a novel Fisher-weighted logit-matching objective, a single-layer linear self-attention model can provably imitate policy-optimization algorithm for linear bandits. Building on this theory, we propose Minimum-Entropy ICPO (ME-ICPO), a practical algorithm that iteratively uses its response and self-assessed reward to refine its response in-context at inference time. 
By selecting the responses and their rewards with minimum entropy, ME-ICPO ensures the robustness of the self-assessed rewards via majority voting. 
Across standard mathematical reasoning tasks, ME-ICPO attains competitive, top-tier performance while keeping inference costs affordable compared with other inference-time algorithms. Overall, ICPO provides a principled understanding of self-reflection in LLMs and yields practical benefits for test-time scaling for mathematical reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces the In-Context Policy Optimization (ICPO) framework for understanding the phenomenon of test-time scaling in LLMs, i.e., when models improve their responses through multi-round self-reflection without parameter updates. The main idea is to model this process as an agent that optimizes its response (action $x$), based on a history of in-context action-reward pairs $(x_t, r_t)$, effectively performing policy optimization within the context window. The main theoretical contribution proves that a single-layer Linear Self-Attention (LSA) transformer can imitate a specific policy optimization algorithm: a variant of FTRL for linear bandits. This imitation is achieved when the LSA model is pre-trained using a Fisher-weighted logit-matching objective. The authors present this as a foundational proof of how an attention-based architecture can learn to perform in-context optimization. Building on this theoretical result, the authors propose a practical, gradient-free inference-time algorithm called Minimum-Entropy ICPO (ME-ICPO), which they leverage for complex mathematical reasoning tasks. At each round, ME-ICPO generates multiple candidate solutions, assigns self-assessed rewards using majority voting on the final answers, summarizes the reasoning paths (Chain-of-Thought) to manage context length, and selects the next reasoning step to add to the context based on a minimum-entropy criterion. The experiments show that ME-ICPO achieves competitive, and in some cases state-of-the-art, performance on benchmarks like AIME, AMC, and MATH.

### Strengths
1. To best of my knowledge, the paper introduces a first framework to formally model in-context self-improvement as a policy optimization problem. The authors provide a new and principled perspective on the mechanisms underlying test-time scaling.

2. The main theorems establish population-level equivalence to an FTRL-like algorithm and provide finite-sample guarantees for learning this algorithm from data. They also analyze the stability of the learned policy to reward perturbations. 

3. The proposed ME-ICPO algorithm is an effective method. It demonstrates substantial and consistent performance improvements over specialized base models on multiple mathematical reasoning benchmarks.

4. ME-ICPO is a gradient-free, inference-time-only algorithm. This makes it significantly more computationally efficient (particularly in terms of VRAM) than methods that require test-time backpropagation, such as TTRL. This practicality makes it a more accessible method for improving LLM performance.

### Weaknesses
Please respond to weaknesses, I will consider raising my score from 6 to 8 if all weaknesses are addressed -- this is a good paper!

1. The most significant weakness is the large abstraction gap between the theoretical model and the practical application. The theory is built on a single-layer Linear Self-Attention model solving a linear bandit problem, whereas the experiments are run on deep, multi-layer, non-linear transformers performing complex, structured reasoning. The paper's claim to "explain" the mechanism of self-reflection is an overstatement. The theory provides an elegant proof-of-concept that the attention mechanism can implement a form of optimization, but it does not and cannot prove that this specific linear mechanism is what underlies the sophisticated self-correction abilities observed in models like Qwen2.5-Math. The paper would be stronger if it framed the theory more cautiously as an inspirational, minimal model that demonstrates a core computational capability, rather than a direct explanation of an emergent phenomenon.   

2. The ablation study shows that the minimum-entropy selection criterion is the most critical component of ME-ICPO. However, the paper's justification for this heuristic is purely intuitive, suggesting it avoids "corrupted" responses and encourages "diversified" ones. This justification is somewhat vague and potentially self-contradictory (low entropy implies low diversity). The success of this heuristic may be domain-specific. For mathematical problems with a single correct reasoning path, low entropy (high agreement among future sampled paths) is likely a strong proxy for correctness. However, for more open-ended or creative tasks, the optimal path might be one that leads to a rich and diverse set of possibilities (high entropy). The paper lacks a more formal justification for this algorithmic choice and does not compare it against more standard selection criteria from RL, such as simply selecting the candidate with the highest self-assessed reward.   

3. Modeling a multi-round reasoning process as a sequence of K-armed bandit pulls is a major simplification. This abstraction ignores the stateful and compositional nature of logical deduction. Each step in a mathematical proof is not an independent choice from a fixed set of K options; rather, it generates a new logical state that constrains all subsequent steps. A more faithful, albeit likely intractable, model would involve a contextual bandit or a full Markov Decision Process (MDP). The paper should explicitly acknowledge and discuss the limitations of this memoryless abstraction and how it impacts the interpretation of the theoretical results.

4. (minor) There is clear over-abuse of spacing in the paper in terms of vspaces. While I realize all authors use this, the authors should not abuse it. Please remove these if your paper is accepted.

Typos and grammatical errors:
- ...the model's ability to digest the in-context information to improve their response. $\rightarrow$ ...to improve its response.
- Such an in-context information can be... $\rightarrow$ Such in-context information can be...
- ...without answering why these ability emerge... $\rightarrow$ ...without answering why this ability emerges... (or ...why these abilities emerge...)
- ...learn to optimize it's behavior x by optimizing it's policy... $\rightarrow$ ...learn to optimize its behavior x by optimizing its policy...
- ...how LLM leverage the in-context actions... $\rightarrow$ ...how LLMs leverage the in-context actions...
- ...to improve it's response \$x\_\{t+1\}\$... $\rightarrow$ ...to improve its response \$x\_\{t+1\}\$...
- ...generating it's response \$x\_t\$ and receives... and then improve it's response... $\rightarrow$ ...generating its response \$x\_t\$ and receives... and then improves its response...
- ...into it's policy optimization process and to gradually improves its response. $\rightarrow$ ...into its policy optimization process and to gradually improve its response.
-  ...where the agent generates and improve it's response... $\rightarrow$ ...where the agent generates and improves its response...
- ...denotes its norm \$l\_2\$ For a matrix A. $\rightarrow$ ...denotes its norm \$l\_2\$. For a matrix A.
-...during the test-time can improve... $\rightarrow$ ...during test-time can improve...
- ...including the Monte-Carol Tree Search... $\rightarrow$ ...including the Monte-Carlo Tree Search...
- ...where the LLM evaluate their own response... $\rightarrow$ ...where the LLM evaluates its own response... (or ...LLMs evaluate their own...)
-  ...by directly assume the LLM's ability... $\rightarrow$ ...by directly assuming the LLM's ability...
-  ...trained linear self attention can implement... $\rightarrow$ ...trained linear self-attention can implement...
- ...multi head constructions... $\rightarrow$ ...multi-head constructions...
-  ...in which first layer heads preprocess... $\rightarrow$ ...in which first-layer heads preprocess...
-  ...rare recent literature have covered... $\rightarrow$ ...rare recent literature has covered...
- ...optimize it's policy \$x\_t\$... $\rightarrow$ ...optimize its policy \$x\_t\$...
- ...dataset is generating from the policy... $\rightarrow$ ...dataset is generated from the policy...
- ...similar with the Follow-the-Regularized Leader... $\rightarrow$ ...similar to the Follow-the-Regularized Leader...
- ...defined by \$s\propto log~p\$ In the following... $\rightarrow$ ...defined by \$s\propto log~p\$. In the following...
-  ...prefix of trajectory up to... $\rightarrow$ ...prefix of trajectory \$\tau\$ up to...
-...exploration parametery is wide... $\rightarrow$ ...exploration parameter \$\gamma\$ is wide...
-...and p is a normalization factor... $\rightarrow$ ...and \$\rho\$ is a normalization factor...
-  The LSA model parameterized by starts with... $\rightarrow$ The LSA model parameterized by \$\theta\$ starts with...
- ...the LSA model updates it's policy... $\rightarrow$ ...the LSA model updates its policy...
-  ...corresponding K dimension... $\rightarrow$ ...corresponding \$K\$ dimensions...
- The expected matrix I is inspired by... $\rightarrow$ The expected matrix \$\Gamma\$ is inspired by...
- The Fisher-weighted loss provide new loss... $\rightarrow$ The Fisher-weighted loss provides a new loss...
- ...that common KL loss between... $\rightarrow$ ...that the common KL loss between...
- ...using the KL loss enable the transformers... $\rightarrow$ ...using the KL loss enables the transformers...

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
- The paper introduces ICPO, framing multi‑round self‑reflection at inference as **in‑context policy optimization** that uses self‑assessed or external rewards without parameter updates. The authors prove that, under **Fisher‑weighted logit‑matching** pretraining, a **single‑layer linear self‑attention (LSA)** model can imitate a policy‑optimization algorithm for linear bandits. Building on this, they propose **Minimum‑Entropy ICPO (ME‑ICPO)**, a practical test‑time algorithm that iteratively samples candidates, assigns self‑assessed rewards, and selects low‑entropy, high‑confidence responses; **majority voting** is used to robustify the reward signal.
- Experiments use **Qwen2.5‑Math‑7B** and **Qwen2.5‑Math‑1.5B** across **AIME‑2024, AMC, and MATH L1–L5**, etc., demonstrating the power of ME-ICPO.

### Strengths
1) **Clear mechanistic link:** a theoretically grounded account connecting pretraining under a Fisher‑weighted objective to in‑context policy‑optimization behavior in an LSA. 
2) **Practicality:** ME‑ICPO yields strong math‑reasoning gains with gradient‑free test‑time optimization; **Mean@16 can surpass the base model’s majority‑vote upper bound**, and adding majority vote on ME‑ICPO output brings further gains.

### Weaknesses
- **No variability reported in Table 1.** Table 1 reports only point estimates (Accuracy and Mean@16) with no variability across multiple runs; please add mean±std over, e.g., 5 seeds. 
- **Theory scope.** Guarantees apply to a **single‑layer LSA** and **linear bandits**; practical models may not be LSA, so the theoretical guarantees do not directly cover the standard non-LSA archetictures.

### Questions
1) Although the proofs target single‑layer LSA, **can ME‑ICPO be safely applied to general (non‑LSA, multi‑layer) Transformers in practice**?
2) The paper should clearly articulate the scope and the required setup for ICPO vs ICPO with LSA. In Section 4, it seems ICPO is only defined with LSA, is this correct? If so, is ME-ICPO only defined for LSA models?
3) "that with sufficient pretraining under a novel Fisher-weighted logit-matching objective, a single-layer linear self-attention model can provably
imitate policy-optimization algorithm for linear bandits", does ME-ICPO described in Algorithm 1 **require** such pretraining to work? 
It is not clear whether ME-ICPO can be used as a test-time only method OR it has to be bundled with the specific pretraining procedures.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces ICPO (In-Context Policy Optimization), a framework showing that a one-layer linear self-attention model can imitate a policy-optimization algorithm under a Fisher-weighted training objective. Motivated by insights from the theoretical results, the paper also proposes a practical algorithm, ME-ICPO, which performs multi-round generation, self-assessment with majority voting, chain-of-thought summarization, and minimum-entropy response selection to enable reward-aware prompting and principled feedback selection without further training. The empirical results show improvements on mathematical reasoning benchmarks.

### Strengths
1. It is interesting to formulate ICPO as a bandit-style policy optimization approach. The theoretical grounding for in-context self-refinement is potentially impactful if the claims hold in more realistic settings. 

2. The framework and algorithm diagrams are well-organized, and the writing is mostly easy to follow.

### Weaknesses
1. The theoretical framework in Section 4 uses a linear bandit abstraction and a simplified linear self-attention model, whereas ME-ICPO is demonstrated with models like Qwen2.5-Math-7B. It is not clear how these theoretical assumptions connect to the practical model choices.

2. ICPO requires iterative sampling, which implicitly increases inference compute. The paper only compares with the base model; since this is technically a prompting technique, it is unclear how this improvement differs from test-time scaling methods such as Tree-of-Thoughts, ReAct, and Monte-Carlo Tree Refinement, or from lightweight training methods such as GRPO and TTRL.

3. The ME-ICPO also seems limited. Majority voting requires that (1) the model has sufficient capability to solve the task, (2) reasoning verification is cheap and easier than generation, and (3) the majority answer correlates with correctness. These appear to be strong assumptions that many real tasks may not satisfy.

### Questions
1. How does the method perform on recent long-CoT models, for example Qwen3-4B-Instruct? And since this is a training-free method, how does it perform even on frontier models, such as GPT-5 or Gemini-2.5-Pro?

2. How does ICPO extend to harder tasks—for example HMMT, APEX-shortlist tasks—or tasks without final-answer executability, or where the final answer is not discrete?

### Soundness
3

### Presentation
2

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
This paper introduces In-Context Policy Optimization (ICPO), a theoretical framework that explains how large language models can self-improve during test-time by iteratively refining their responses without parameter updates.The paper formulates multi-round self-refinement in LLMs as a form of in-context policy optimization, where the model treats its previous responses and associated rewards as contextual experience to adjust future outputs. This extends existing in-context learning theory from supervised prediction to policy optimization with bandit feedback. One the theoretical side, the authors prove that a single-layer linear self-attention transformer, when pretrained using a Fisher-weighted logit-matching objective, can provably imitate a policy optimization algorithm for linear bandits, thereby establishing a mechanistic explanation for the emergence of self-reflection in LLMs. Based on the theory, the paper proposes ME-ICPO, a practical inference-time algorithm that performs iterative response refinement using self-assessed rewards and entropy-based selection to ensure robustness to reward noise. Across standard mathematical reasoning benchmarks, ME-ICPO achieves competitive and often state-of-the-art test-time performance while maintaining affordable inference cost, demonstrating that test-time scaling can be improved without parameter fine-tuning.

### Strengths
The authors derive provable guarantees showing that a linear self-attention transformer, when trained under a Fisher-weighted objective, can imitate the behavior of a policy optimization algorithm in a linear bandit setting. This is a novel result from the theoretical perspective. 

The paper proposed Minimum-Entropy ICPO (ME-ICPO) algorithm which demonstrates a practical and implementable version of in-context policy optimization. It integrates entropy-regularized response selection and self-assessed rewards, leading to consistent empirical improvements in mathematical reasoning tasks. The experimental results are strong and align with the theoretical insights. 

By modeling the self-reflection and iterative response refinement as In-Context policy optimization problem, the paper offers a clear mechanistic and mathematically grounded explanation for self-improvement phenomena observed in LLMs.

### Weaknesses
The effectiveness of ME-ICPO depends on choices such as number of refinement rounds, sample count per round, and entropy thresholds. Tuning those hyperparameters are non-trivial and might heavily depend on model sizes and datasets.

### Questions
How can we handle the situation that the model itself cannot score or rank its own responses? How can we handle mis-aligned reward heuristics that the incorrect reasonings are being rewarded or reinforced? How to mitigate such caveats?

### Soundness
3

### Presentation
3

### Contribution
3
