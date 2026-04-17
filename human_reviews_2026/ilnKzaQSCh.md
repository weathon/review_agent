# Automatic Dialectic Jailbreak: A Framework for Generating Effective Jailbreak Strategies

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Large language models (LLMs) can be jailbroken to produce malicious or unethical content with embedded jailbreaking prompts. Unfortunately, current jailbreak attack techniques suffer from adaptability issues due to reliance on the fixed evaluation models and incapability problems of surviving from a wide range of defense mechanisms. In this work, we propose to model the the jailbreak attack problem as a Stackelberg multi-objective game between two LLMs engaged in a Hegelian-Dialectic-style debate enabling the automatic generation of jailbreak strategy (ADJ). In the ADJ, iterative thesis-antithesis-synthesis cycles of Hegelian dialectical reasoning are executed to guarantee that both attacker and defender can maximize their own utility while minimizing that of their opponent. We propose to map the optimization problem from the original parameter space into a Hilbert space via Haar wavelet transformation, for efficiently extracting localized and structurally significant information. In this functional space, we solve a convex multi-objective optimization problem to construct a common descent direction that better aligns with the objectives in the ADJ. In order to ensure sufficient descent for each objective in ADJ, we construct a subset of descent components and directly integrate them into the optimization objective. We theoretically validate the existence of a Pareto–Nash equilibrium achieved by our Automatic Dialectic Jailbreak method and demonstrate that our algorithm is able to converge to this Pareto–Nash equilibrium.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Automatic Dialectic Jailbreak (ADJ), a novel framework that models LLM jailbreaking as a Stackelberg multi-objective game inspired by Hegelian dialectic reasoning. The framework involves two LLMs engaged in iterative thesis-antithesis-synthesis cycles, where an attacker proposes jailbreak strategies and a defender generates countermeasures. The authors employ Haar wavelet transformation to map the optimization problem into Hilbert space and use Armijo backtracking rules for convergence. They provide theoretical guarantees for Pareto-Nash equilibrium existence and convergence, and demonstrate superior performance over existing methods on AdvBench and HarmBench datasets across multiple LLMs.

### Strengths
Originality: The application of Hegelian dialectic and multi-objective game theory to LLM jailbreaking is genuinely novel. The combination of philosophical reasoning frameworks with modern optimization techniques represents a creative interdisciplinary approach that hasn't been explored in this domain.

Quality: The mathematical formulation is rigorous and well-developed. The authors provide formal proofs for equilibrium existence and convergence properties. The experimental evaluation is comprehensive, covering multiple models, datasets, and both white-box and black-box settings.

Clarity: Despite the mathematical complexity, the paper is generally well-written. The motivation is clearly presented, the methodology is systematically explained, and the results are properly contextualized. The Hegelian dialectic analogy is well-motivated and helps readers understand the framework.

Significance: This work advances the state-of-the-art in adversarial ML for LLMs by introducing a principled framework for automatic strategy generation. The theoretical contributions to multi-objective optimization in the context of LLM safety are valuable. The demonstrated robustness against existing defenses highlights important vulnerabilities in current safety measures.

### Weaknesses
1. Theory-Practice Gap
Inconsistent theoretical guarantees: The paper provides rigorous convergence proofs for white-box settings but lacks theoretical foundations for black-box scenarios, which are more practically relevant. This creates a significant disconnect between the theoretical contributions and real-world applicability, as white-box access to commercial LLMs is rarely available.

2. Computational Efficiency Concerns
Prohibitive resource requirements: The framework requires multiple LLMs to interact simultaneously, leading to substantially higher computational costs compared to existing methods. The scalability becomes questionable as model sizes increase, potentially making the approach impractical for resource-constrained environments.

3. Complexity vs. Benefit Trade-off
Over-engineering complexity: The introduction of sophisticated mathematical tools (Haar wavelets, Hilbert space mapping, multi-objective optimization) adds considerable complexity without clear justification that the performance gains warrant such elaborate machinery. Some baseline methods might achieve comparable results with simpler modifications

4. Limited Experimental Validation
Narrow evaluation scope: The evaluation relies primarily on ASR and HS metrics, lacking assessment of content quality, diversity, and semantic coherence. Additionally, only two defense mechanisms are tested, which may not represent the full spectrum of real-world defensive strategies.

5.Little typo: there are double “the”s in the 5th line of Abstract;For formula (1) in 2.1,The subscript n may be replaced with k.

### Questions
Given the substantial computational overhead (29K+ seconds vs. 4K-8K for simpler methods), what is the empirical relationship between resource investment and performance gains in ADJ compared to enhanced baseline methods? Specifically, could simpler approaches achieve comparable ASR/HS improvements through targeted optimizations (e.g., better prompt engineering, improved sampling strategies) without requiring the complex multi-LLM dialectical framework?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an automated jailbreak method designed to adapt to a wider range of attack scenarios and to bypass defenses. It formulates the jailbreak attack as a Stackelberg multi-objective game. This game is implemented as a Hegelian-Dialectic-style debate between two LLMs, cycling through "thesis-antithesis-synthesis" to automatically generate robust jailbreak strategies.

### Strengths
1. The proposed attack (ADJ framework) significantly outperforms baselines in both ASR and HS across various models.
2. The ADJ framework demonstrates robustness against two defense mechanisms. It shows minimal performance degradation against the RAIN defense and perplexity-based defense，which effectively counter other baseline attacks.

### Weaknesses
1. The experiments evaluate only two defense methods, which is not sufficient. The paper should include a broader set of defense strategies—including composite or dynamic defenses—to better support its claims.
2. Some formulas (Eq. (1), $ \pi_{-i}^{*} $, etc) contain unclear or inconsistent notation. Please review and correct the mathematical expressions to ensure clarity.

### Questions
1. Is Equation (1) correct? In Equation (1), the definition of the k-simplex appears to contain an inconsistency. The vector is written as $(x_0, \ldots, x_n)\in \mathbb{R}^{k+1}$, mixing $n$ and $k$. For a k-simplex, the standard definition involves $(k+1)$ coordinates $(x_0, \ldots, x_k)$. It would be clearer and mathematically consistent to use $k$ instead of $n$ in the index range.
2. In the black-box setting, is also the same model for Attacker, Defender, and Target?
3. What is the RAIN defense?
4. What does the x-axis represent in Figure 4?
5. What is the compution cost of ADJ in white-box setting?
6. What is the computation cost of ADJ in the white-box setting?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In  this work,  the authors propose to model the the jailbreak attack problem as a Stackelberg multi-objective game between two LLMs engaged in a Hegelian-Dialectic-style debate enabling the automatic generation of jailbreak strategy (ADJ). 

In the ADJ, iterative thesis-antithesis-synthesis cycles of Hegelian dialectical reasoning are executed to guarantee that both attacker and defender can maximize their own utility while minimizing that of their opponent.

Experimental results demonstrate that the paper's method consistently outperforms prior jailbreak approaches across a wide range of models, while also exhibiting superior robustness.

### Strengths
(1) By simulating the Hegelian-style debate between the attacker and defender, our method enables the attacker to generate diverse jailbreak strategies, thereby mitigating the incapability to any single specific defense method.

 (2) Based on the SMOG, the  algorithm does not rely on a fixed auxiliary model, thereby enhancing the attacker’s adaptability against a wide range of defense mechanisms.

 (3) The proposed method is applicable to both white-box and black-box settings.

### Weaknesses
I only have the following two concerns. 


1. Why using a Stackelberg multi-objective game and  Hegelian-Dialectic-style debate is a good strategy than other schemes, such as reinforcement learning scheme, is not that clear to me. Please explain.

2. Compared with usual improvement, what is the percentage of  the improvement of you attacking effect on average it is? You study an old problem of  jailbreaking prompts. The novel game strategy is still not persuasive enough for me to think this technical is effective and necessary.

### Questions
see above comments

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
5

### Summary
This paper proposes a framework, Automatic Dialectic Jailbreak (ADJ), that automatically generates effective jailbreak strategies against large language models (LLMs). The authors formulate the problem as a Hegelian-dialectic (thesis–antithesis–synthesis) Stackelberg multi-objective game (SMOG). Within this framework, an attacker LLM (the leader) and a defender LLM (the follower) co-evolve through an iterative debating process. To address the non-smooth multi-objective optimization problem arising under the white-box setting (W-ADJ), the authors propose employing the Haar wavelet transform to map parameters into a Hilbert space, thereby identifying common descent directions. They provide theoretical proofs establishing the existence of a Pareto–Nash equilibrium for the game and the convergence properties of their method. Additionally, the paper introduces a black-box variant (B-ADJ) that leverages in-context learning (ICL) to simulate the dialectical process. Experimental results demonstrate that the proposed approach outperforms existing baselines in both attack success rate (ASR) and harmfulness score (HS) across multiple models and datasets.

### Strengths
(1) This paper attempts to frame the LLM jailbreak problem within a dynamic, adversarial game-theoretic framework, representing a novel approach distinct from existing static prompt optimization methods.

(2) The authors evaluate the proposed method on a variety of LLMs, including state-of-the-art closed-source models (such as GPT-4, Gemini 1.5 Pro), and report a high attack success rate.

(3) The paper provides theoretical proofs for the existence of a Pareto–Nash equilibrium (Theorem 1) and the algorithm's convergence (Theorem 2) for the proposed W-ADJ framework.

### Weaknesses
(1) The W-ADJ (white-box) method requires white-box access to both LLMs (the attacker and defender) for joint optimization. This is computationally expensive and far beyond the feasibility of most real-world attack scenarios. Although the B-ADJ (black-box) method is more practical, it replaces theoretically guaranteed gradient optimization with heuristic in-context learning (ICL). To make ICL effective (i.e., to build meaningful $R_A$ and $R_D$ histories), a massive number of API calls and evaluations are likely required, leading to very high API costs.

(2) The core theory of the paper (game theory, wavelet transform, convergence proofs) applies solely to W-ADJ. As a simulation based on ICL, B-ADJ's effectiveness heavily relies on the design of system prompts (e.g., the "Tom and Jerry" setting in Appendix G). It is unclear how much of B-ADJ's success is due to the cleverness of this meta-prompting and how much is attributable to the "dialectical game" process claimed by the paper. There is a lack of argumentation regarding whether B-ADJ truly simulates SMOG or if it is merely an iterative prompt improver.

(3) The paper misuses complex mathematical symbols and terminology, showing a clear tendency toward "mathematical embellishment." Many of the complex derivations (such as the lengthy gradient calculations in the appendix) are not essential for proving the core arguments, but instead obscure the fundamental ideas behind the method. The description of Key Algorithm 1 is confusing, with unclear variable names and a disorganized flow, which severely hampers reproducibility.

(4) The paper constructs a complex theoretical framework (e.g., Hilbert space, wavelet transform, Pareto-Nash equilibrium), but the necessity of applying these advanced mathematical tools is not sufficiently justified. The core driving force behind the method seems to rely more on the "emergent" capabilities of LLMs as debate participants, rather than the intricacy of their mathematical optimization process. The theoretical section appears to be "overkill" and fails to convincingly demonstrate that these complex tools provide a significant improvement over simpler heuristic methods, such as multi-round self-play.

### Questions
(1)Can the authors provide clear evidence through ablation experiments to demonstrate that the complex Haar wavelet transform and Hilbert space mapping yield a statistically significant performance improvement over directly performing multi-objective optimization in the original parameter space? Are these mathematical tools a necessary condition for achieving high performance?

(2)One of the core claims of the paper is the generation of "diverse" strategies. Could the authors provide specific qualitative or quantitative evidence to support that the strategies generated by ADJ indeed exhibit diversity?

(3)To what extent does the effectiveness of B-ADJ rely on the role-playing prompts in Appendix G? If the "Tom and Jerry" game setting were removed and only a single LLM iteratively optimized its historical attack record through ICL (similar to AutoDAN), how much would performance degrade?

### Soundness
2

### Presentation
1

### Contribution
2
