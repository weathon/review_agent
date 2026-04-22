# Proximal Supervised Fine-Tuning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 4, 2

## Abstract
Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on specific tasks. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT), a fine-tuning objective that incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical, human-value, and multimodal domains show that PSFT matches standard SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Proximal Supervised Fine-Tuning (PSFT), which improves the supervised fine-tuning by utilizing a clipped surrogate objective to enforce trust-region-like constraints. The method can stabilize optimization and reduce the loss of generalization during supervised fine-tuning. Experiments demonstrate the effectiveness and statement of the method.

### Strengths
1. The main idea is well-motivated and straightforward.

2. The proposed method is easy to implemented with a simple objective.

3. Extensive experiments demonstrate the statements and efffectiveness of the proposed method.

### Weaknesses
1. Although the proposed method is well-motivated and empirically demonstrated, it seems lack of theoretical support.

2. It is not clear whether the empirical gains obtained by the proposed method or the lower effective lr and gradient clipping. It should be compared to simple baselines like: lower LR, gradient clipping, sample filtering, or reweighting. 

3. The method is only compared with the common SFT and SFT SFT-KL, lack of recent RL-based fine-tuning baselines for comparisons. 

4. It is not clear whether the method also work on larger model sizes and datasets.

### Questions
There are different conclusions in Table 1 and Table 2 for the in-domain performance, where PSFT performs better in Table 1 but not in Table 2. This also happens before and after GRPO in Table 2, where PSFT performs better than SFT after GRPO but not before. Are there any analyses or discussions on that?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Introduces PSFT, an algorithm to replace SFT that achieves comparable performance but archives strong generalization. It maximally preserves the model’s general capabilities (prevents entropy collapse and overfitting), and achieves good performance, especially in out-of-distribution tasks. PSFT also is compatible with RL fine-tuning afterwards, and boots RL performance by initializing with a more general model.

### Strengths
- Very good contribution: the PSFT algorithm is simple, yet shown to be extremely effective. I believe this is a great and timely contribution to the community. It is incredibly important to have algorithms that can fine-tune pre-trained models yet maintain their generalist performance and not overfit.
- Very clear presentation. The motivation is clear, and the preliminary covers the background work sufficiently. Explanation of the method is well done, and the results section makes the take aways crystal clear. The writing is very well done
- Comprehensive experimental evaluation shows that PSFT works very well, and evaluates on multiple base models on extensive benchmarks.

### Weaknesses
- The paper focuses on an algorithm that can finetune a pre-trained model with generalization ability. However, there is not much comparisons with baselines that aim to finetune a model that preserves generalization ability. The only baselines are naively doing SFT and SFT with KL constraints, and it would be nice to see comparisons with more sophisticated methods that aim to keep generalization ability.

### Questions
- How does PSFT compare to doing RL directly from the pre-training model without an SFT-like stage? In other words, can you outperform PSFT if you estimate the actual advantages instead of setting them all to 1.
- nit: the results plots are kind of too small to see, can you make them bigger?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new supervised fine-tuning algorithm for LLMs that addresses the problems of overfitting and entropy collapse by incorporating trust-region updates inspired by TRPO and PPO. First, the paper reformulates the supervised fine-tuning (behavior cloning) objective in terms of the policy gradient objective and proposes an analogous clipped PPO-style objective.
The paper evaluates the proposed algorithm on several math reasoning benchmarks and investigates its effectiveness for RL post-training, human alignment, and vision-language models.

### Strengths
- The paper is clear and easy to follow.
- The proposed method is simple, and the results seem promising.

### Weaknesses
- The analogy drawn between the proposed PSTF objective and TRPO/PPO-style objectives may be oversimplified.

- The paper lacks comparisons with stronger supervised fine-tuning baselines.

- The evaluation is primarily limited to mathematical reasoning benchmarks.

### Questions
- The analogy between PSFT and RPO/PPO-like objectives is not accurate: (1) In the ppo objective in equation 5 the expectation is taken w.r.t the old policy $\pi_old$, while in the PSFT objective, the expectation is w.r.t the offline dataset. (2) The probability ratio in PPO and PSFT loss is $\frac{\pi(a \mid s)}{\pi_{old}(a \mid s)}$, but it should something like  $\frac{\pi(a \mid s)}{\pi_{D}(a \mid s)}$.
Can the authors comment on these differences and how they would explain the performance gains? 
- The method is compared with two baselines, vanilla SFT and SFT with KL. Are there any other SFT baselines that the authors can compare to?
- Performing an SFT warm-up seems to contradict the paper’s stated goal of replacing vanilla SFT with a more stable method that prevents entropy collapse. The paper claims that the warm-up stage helps mitigate the bias resulting from the misalignment between the data distribution and πold\pi_{\text{old}}πold​, but this warm-up phase may also affect some of the model’s capabilities. It is not clear how this aligns with the core motivation of the proposed approach.
- Most of the evaluation is on Math benchmarks; can the authors show some results on coding or general reasoning tasks?

I am willing to increase my score if the authors addressed my concerns.

### Soundness
2

### Presentation
3

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
The paper introduces Proximal Supervised Fine-tuning, a method for supervised fine-tuning that takes inspiration from trust-region policy optimisation (TRPO) and proximal policy optimisation (PPO) from reinforcement learning. The supervised fine tuning setting is matched to the RL context by setting the advantage function to be a constant function and matching the objectives. Then the importance sampling of TRPO and clipping of PPO are applied. Experiments are carried out on a wide number of domains and applications. 

The novelty of this paper is in an interesting composition of existing methods.

### Strengths
- The paper presents a simple yet effective idea.
- Generally well written, except for the points mentioned below.
- Good set of experiments with clear descriptions of findings. 
- Code provided + open-sourced datasets and models.

### Weaknesses
- There are inconsistencies in the description of the mathematical notation:
	- - Section 2: a MDP usually has a reward function as well with respect to which the *best decision* is made.  You have omitted this. Why?
	- Line 78 onwards: What does the $\*$ in $a^\*_t$ signify: the optimal action? If so, state this in the text. What is the distinction between the action in Equation 2 and 3, where Equation 2 contains the $\*$ and Equation 3 does not?
	- Line 83: The authors should define what an advantage is, at least in text. 
- References are missing at places (below are some examples):
	- Line 84: Please add a reference after "policy gradient theorem". 
	- Line 105: "... can be difficult to optimize in practice." — please state why and provide a reference.
	- Line 115: "... preventing entropy collapse..." — please say what entropy collapse is and provide a reference.
	- Line 257: "... DAPO ..." — please add a reference to DAPO. 
		- Can you elaborate on why DAPO was chosen?
- I am wondering if the framing of this paper can be improved / made more accurate by changing it from applying Reinforcement Learning to Supervised Fine Tuning to applying the specific techniques (importance sampling and clipping) to the supervised objective, and observing / realising similar benefits. In particular, because the advantage is set to a constant 1, I am hesitant to classify the presented work as RL. 

### Suggestions
- Line 27: Please expand PPO and GRPO here since this is the first time these abbreviations are used.
- Line 31: "These reasoning models offer an abundant and valuable latent thoughts (Ruan et al., 2025) across the internet" — It is unclear to me what this sentence means. Are the authors saying that foundation models are used to generate the thought data?  
- Line 68: "the state space (partial sequences), ..." — Elaborate on what you mean by "partial sequences", since it is unclear here. E.g., partial sequences of what?
- Line 115: "... while retaining general capabilities..." — can you expand on the specific capabilities you are targeting?

### Questions
- Line 58: "It maximally preserves..." — Do you have a proof to supporting the claim that your method "*maximally* preserves the model's general capabilities"?
- Line 102: "... far from the reference policy, enabling more targeted and ..." — isn't the reference policy $\pi_{\theta_\mathrm{old}}$? If so, is this the policy from the previous update step? If this is the case, is calling it a reference policy valid, since reference policy implies a notion of a ground-truth. Furthermore, if this is the policy from the previous step, how does this method ensure the training to be "more targeted" — what is the target if the target keeps changing at each step? 
	- Alternatively, does $\pi_{\theta_\mathrm{old}}$ refer to the initial policy? If so, couldn't the notation be changed to $\pi_{\theta_0}$, or something else to that effect?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel supervised fine-tuning method called Proximal Supervised Fine-Tuning (PSFT), which aims to address the issues of degraded generalization, entropy collapse, and excessive policy updates commonly found in traditional SFT. Inspired by TRPO and PPO from reinforcement learning, PSFT introduces a clipped probability ratio to constrain the magnitude of policy updates. The authors conduct extensive experiments across various domains, showing that PSFT achieves comparable in-domain performance to SFT while offering advantages in out-of-domain generalization and entropy stability.

### Strengths
- Originality: The paper introduces the trust-region mechanism from reinforcement learning into supervised fine-tuning, proposing PSFT. This is a novel and reasonable perspective that offers fresh insights into improving SFT.
- Quality: The authors validate the effectiveness of the proposed method through extensive experiments and compare it with several existing methods. Results show consistent performance improvements across multiple benchmarks.
- Clarity: The structure of the paper is generally clear, although there are some notational errors that need correction.

### Weaknesses
1. Although PSFT is inspired by TRPO/PPO, it remains largely heuristic, lacking rigorous mathematical analysis on convergence, stability, or generalization error.
2. While PSFT provides a better cold-start for RL, the RL stage still uses standard GRPO/DAPO. The paper does not explore joint optimization of PSFT and RL objectives.
3. All experiments are conducted on 7B–8B scale models. The effectiveness of PSFT on larger models (e.g., 30B+) remains unverified.
4. Current experiments mainly focus on mathematical reasoning and long CoT data. It is unclear whether PSFT is also applicable to dialogue, code, instruction following, or other general SFT tasks.
5. Equation (3) is incorrect — the objective function is not in the right form.
6. The assumption $\hat{A}_t=1$ is too strong. The paper does not justify why this assumption is reasonable.
7. In Equation (7), the meaning of $r_t$ is not clearly explained.
8. The meaning of $π_{old}$ is ambiguous. It is unclear why the same \*π\*old is used throughout training on a fixed dataset, and thus the effectiveness of this approach remains questionable.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
