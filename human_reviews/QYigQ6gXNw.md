# Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
This paper investigates a basic question in reinforcement learning from human feedback (RLHF) from a theoretical perspective: how to efficiently explore in an online manner under preference feedback and general function approximation. We take the initial step towards a theoretical understanding of this problem by proposing a novel algorithm, *Exploratory Preference Optimization* (XPO). This algorithm is elegantly simple---requiring only a one-line modification to (online) Direct Preference Optimization  (DPO; Rafailov et al., 2023)---yet provides the strongest known provable guarantees. XPO augments the DPO objective with a novel and principled *exploration bonus*, enabling the algorithm to strategically explore beyond the support of the initial model and preference feedback data. We prove that XPO is provably sample-efficient and converges to a near-optimal policy under natural exploration conditions, regardless of the initial model's coverage. Our analysis builds on the observation that DPO implicitly performs a form of *Bellman error minimization*. It synthesizes previously disparate techniques from language modeling and theoretical reinforcement learning in a serendipitous fashion through the lens of *KL-regularized Markov decision processes*.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an algorithm called XPO, designed to enable efficient online exploration using preference feedback and general function approximation within KL-regularized MDPs with deterministic transitions. With just a single-line modification to online DPO, XPO is provably sample-efficient and reliably converges to a near-optimal language model policy under standard exploration conditions. Importantly, XPO can achieve optimal policy convergence regardless of the initial model’s coverage.

### Strengths
Overall, the core idea of this paper—introducing a novel approach to promote exploration—is intriguing, even though it is demonstrated in a limited setting, specifically deterministic MDPs. Achieving sufficient exploration with only a single-line modification to the DPO algorithm could have a non-trivial practical impact. 

Additionally, the theoretical result revealing the limitations of Online DPO (Proposition 2.1) is particularly compelling and supports the need for deliberate exploration.

### Weaknesses
1. Although the proposed algorithm is a simple extension of DPO and guarantees a strong sample efficiency, it remains unclear why the algorithm is considered *highly practical* (Line 312). While I understand that this is a theoretical paper, the lack of empirical experiments makes it difficult to be convinced of the algorithm's practicality. Without experiments, one of the authors' main claims—ease of implementation—is not fully persuasive. I am not suggesting that experiments are required; it is simply a limitation of the paper.

2. The optimism parameter $\alpha$ depends on Coverability $C_{cov}(\Pi)$, which is hard to know in practice.

### Questions
1. Line 408: You mentioned that the optimistic exploration objective is differentiable and directly amenable to implementation with language models. Could you explain this in more detail?
2. Are there any advantages to using a token-level MDP (where $s_h$ is simply the initial state and sequence of actions) compared to the contextual (dueling) bandit setting, where each action corresponds to an entire text rather than a single token? In Line 762, you mentioned that the token-level MDP is strictly more general than the contextual bandit formulation. However, since your setting is quite restricted, this claim is not clear enough to me.
3. In Lemma C.1, it appears that the SEC depends on the value of $\beta$. Does the upper bound of the SEC also depend on $\beta$? If so, it would be helpful to explicitly show this dependence on $\beta$.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper gives a theoretical analysis of online RLHF. In particular, they proposed a XPO algorithm that enjoys sample complexity guarantee.

### Strengths
1. The literature review is extensive and helpful for the readers. Due to the fact that RLHF is currently a fast-growing domain, the paper gives a nice comparison between its contribution and concurrent works helps the reader understand how to posit their work.
2. Their theoretical analysis is meaningful. In particular, I like the general algorithm 2 provided in the Appendix C.1.

### Weaknesses
1. Due to that it is a theoretical paper, I do wonder the practical implication of XPO--- I looked up the online DPO work the authors referenced, and their empirical results made me believe in this work (or online approach in general) more.

### Questions
1. The regret bound is scaling down w.r.t. the number of iterations. And for every model update, it just requires one new pair. In practice it sounds very costly. In the simplicity session (line 311), the authors also suggest cutting down the number of iterations and sampling more data during each iteration. I wonder how do they anticipate this way to guarantee a good policy if their analysis is compeletely dependent on a good number of iterations.

### Soundness
3

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
This paper proposes a simple yet provably efficient RLHF algorithm called XPO with general function approximation. The authors focus on a KL-regularized Deterministic Contextual MDP (DCMDP) and decompose the regret into policy expression, thereby formulating it within a direct preference optimization framework. By employing a coverability measure and implicit optimism, XPO provides a provably guaranteed sampling policy of preference pairs, which is a notable theoretical contribution in the RLHF field.

### Strengths
- Despite the limited references due to the field's recent development, the authors have effectively organized concurrent studies and provided ample comparisons with their own contributions.

- The authors clearly address the paper's limitation in focusing on DCMDP, which is valuable information for future researchers working on related papers.

- Particularly in Section 3, the interpretation of theoretical results is well-explained in an intuitive manner, with concepts from conventional regret analysis effectively connected within the DPO perspective.

### Weaknesses
- Although the paper lacks experimental content, this is a minor drawback given the theoretical nature and significance of its contributions. The emphasis on simplicity in the main text raises questions on whether the authors have experience conducting practical experiments with XPO or plan to do so.

- It appears that some implicit assumptions are used in the main text; depending on the authors' response, I am willing to increase the score further. Please refer to the "Questions" section for more details.

### Questions
-	What are the authors' views on the practicality of human "online" feedback in the online learning process?

-	In Line 188 and Line 4 of the pseudocode, $y_t \sim \mathbb{P}(\tau^{(t)} \succ \tilde{\tau}^{(t)})$ suggests that feedback follows the preference model. If labeling is done via human online feedback, it seems quite a strong assumption that this property is satisfied in every iteration. Was this assumption necessary for theoretical development?

-	While I am familiar with regret analysis using eluder dimension, I lack a deep understanding of coverability. Specifically, as stated in Remark 3.3, coverability at the trajectory level appears to increase exponentially with $H$, suggesting a much larger upper bound compared to the eluder dimension. Are there specific advantages of coverability over eluder dimension?

-	Theorem 3.1 requires $|\Pi|$ to be defined, suggesting an implicit assumption that the policy class is finite. Could this be addressed by a covering number?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers the sample complexity of online reinforcement learning with human feedback (RLHF). For contextual MDPs with deterministic transitions, the authors show that an optimism-based variant of Direct Preference Optimization (DPO) is provably efficient with general function approximation.

### Strengths
1. The proposed method is simple to implement; 
2. The paper shows that online DPO can fail if $\pi_\text{ref}$ does not have good coverage (Proposition 2.1);
3. A connection is built between DPO and Bellman error minimization for DCMPDs (Lemma C.3)

### Weaknesses
1. My major concern with the paper is on the assumption of deterministic transitions, which ignores the challenge of learning transitions in RL;
2. The paper does not provide numerical results to show the effectiveness of the proposed method.

### Questions
1. The hardness result in Proposition 2.1 can be bypassed by setting $\pi_\text{ref}$ to be uniform. I think an alternative way is to show that for *any* $\pi_\text{ref}$, there exists a hard DCMDP instance such that the lower bound either 1) scales with the action space (e.g., for uniform policy), which can be prohibitively large for token-level MDPs, or 2) scales with, e.g., $\exp(1/\beta)$.
2. Can we just output the uniform mixture of $\pi^{(1)},\cdots,\pi^{(T+1)}$ in line 7 of Algorithm 1? In this case, we do not need validation data.

### Soundness
2

### Presentation
3

### Contribution
2
