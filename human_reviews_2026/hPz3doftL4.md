# On Discovering Algorithms for Adversarial Imitation Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Adversarial Imitation Learning (AIL) methods, while effective in settings with limited expert demonstrations, are often considered unstable. These approaches typically decompose into two components: Density Ratio (DR) estimation $\frac{\rho_E}{\rho_{\pi}}$, where a discriminator estimates the relative occupancy of state-action pairs under the policy versus the expert; and Reward Assignment (RA), where this ratio is transformed into a reward signal used to train the policy. While significant research has focused on improving density estimation, the role of reward assignment in influencing training dynamics and final policy performance has been largely overlooked. RA functions in AIL are typically derived from divergence minimization objectives, relying heavily on human design and ingenuity. In this work, we take a different approach: we investigate the discovery of data-driven RA functions, i.e, based directly on the performance of the resulting imitation policy. To this end, we leverage an LLM-guided evolutionary framework that efficiently explores the space of RA functions, yielding _Discovered Adversarial Imitation Learning_ (DAIL), the first meta-learnt AIL algorithm. Remarkably, DAIL generalises across unseen environments and policy optimization algorithms, outperforming the current state-of-the-art of _human-designed_  baselines. Finally, we analyse why DAIL leads to more stable training, offering novel insights into the role of RA functions in the stability of AIL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Discovered Adversarial Imitation Learning (DAIL) that stabilizes the training process by using an LLM-guided framework to explore the Reward Assignment (RA) functions.

### Strengths
- This paper presents a search procedure that illustrate how the LLM-guided framework searches for the reward assignment functions
- Empirical experiments on MuJoCo and Minatar tasks have show the effectiveness of DAIL

### Weaknesses
- The paper primarily builds upon existing reward assignment strategies, and thus the level of novelty is limited.
- While experiments support the proposed idea, the paper lacks rigorous theoretical analysis to clarify to what extent DAIL improves stability or under what conditions the improvements are guaranteed.

### Questions
- Some of the text in Figure 2 is too small to read, which affects the clarity of the visualization. Improving figure readability would strengthen the presentation.
- There seems to be a contradiction regarding the reward assignment (RA) function: in Section 5.2, it appears to change dynamically, while the future work section mentions that the RA function in DAIL is static. Could the authors clarify whether the RA function is adaptive during training or fixed throughout?

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
5

### Summary
This paper proposes Discovered Adversarial Imitation Learning (DAIL), a meta-learning framework that aims to automatically discover reward assignment (RA) functions for adversarial imitation learning. 
DAIL employs an LLM-guided evolutionary search to generate and evolve candidate RA functions expressed as executable code, optimizing them based on the Wasserstein distance between expert and policy occupancy measures. 
Experiments on Brax and Minatar benchmarks show that DAIL achieves higher normalized returns and better alignment with expert behavior than existing AIL baselines. 
The paper claims to offer the first meta-learned AIL algorithm and provides empirical analyses linking the discovered reward structure to more stable adversarial training dynamics.

### Strengths
1. *Novelty.* The paper presents a moderately original idea by framing RA discovery in AIL as a meta-learning problem and leveraging LLMs for evolutionary search over reward function space. Its originality stems primarily from the methodological integration of LLM-guided symbolic evolution into the imitation learning pipeline, rather than from a fundamentally new theoretical insight about AIL itself. 

2. *Quality.* The paper is technically sound in its experimental setup and provides comprehensive empirical validation on both Brax and Minatar benchmarks, demonstrating consistent improvements in performance over standard baselines. 

3. *Clarity.* The paper is clear and easy to follow. 

4. *Significance.* The work contributes an interesting demonstration of how LLMs can aid in discovering algorithmic components in AIL. However, its broader impact on the IL/RL community is modest, as the main performance gains stem from empirical search rather than conceptual advances, and the underlying instability of AIL that is driven by the adversarial optimization process, remains unresolved.

### Weaknesses
The main weakness of this paper lies in its limited theoretical novelty and misattribution of the source of instability in AIL. 

1. The idea of studying how RA functions affect learning dynamics is not new. Prior works [1,2,3] have already analyzed and benchmarked multiple RA designs under the f-divergence framework. 

2. The proposed contribution of using an LLM-guided evolutionary search to discover new RA functions only offers a methodological novelty rather than a theoretical or algorithmic one. However, this approach does not address the primary cause of instability in AIL, which stems from the adversarial f-divergence min–max optimization dynamics rather than the form of the f-divergence [4]. As a result, the proposed method may yield empirically smoother rewards but cannot fundamentally stabilize training.

3. The discovered RA functions violate theoretical constraints that ensure valid divergence minimization, which undermines interpretability and convergence guarantees. The paper could be significantly strengthened by incorporating these mathematical constraints into the search process (e.g., by enforcing convexity) or by analyzing the discovered functions’ theoretical properties relative to valid divergences.

4. Empirically, while the experiments on Brax and Minatar demonstrate some improvements, the evaluation is limited to relatively simple continuous-control and grid-world tasks. The generality and robustness of DAIL would be more convincing if tested on high-dimensional or visual IL environments (e.g., Atari) and under different discriminator architectures or regularization strategies. 

5. The paper attributes stability improvements solely to the discovered reward function, but it lacks ablation studies isolating the effect of LLM-guided search versus simpler baselines (e.g., random search, symbolic regression, or gradient-based meta-learning). Including such comparisons would clarify whether the LLM’s involvement meaningfully contributes beyond automated function optimization.

[1] Ke, Liyiming, et al. "Imitation learning as f-divergence minimization." International workshop on the algorithmic foundations of robotics. Cham: Springer International Publishing, 2020.
[2] Ghasemipour, Seyed Kamyar Seyed, Richard Zemel, and Shixiang Gu. "A divergence minimization perspective on imitation learning methods." Conference on robot learning. PMLR, 2020.
[3] Zhang, Xin, et al. "f-gail: Learning f-divergence for generative adversarial imitation learning." Advances in neural information processing systems 33 (2020): 12805-12815.
[4] Arjovsky, Martin, and Léon Bottou. "Towards principled methods for training generative adversarial networks." arXiv preprint arXiv:1701.04862 (2017).

### Questions
1. Why does the method use the Wasserstein distance as the evaluation objective while still training within the f-divergence–based AIL framework? If Wasserstein distance provides smoother and more stable gradients, why not directly optimize it instead of meta-learning new reward assignment functions?

2.How does the discovered reward function r theoretically relate to any valid divergence or distance measure?

3. Does the use of Wasserstein distance for evaluating fitness bias the search toward functions that implicitly approximate it?

4. How does DAIL compare in stability and performance to Wasserstein-based imitation methods [5, 6]?

5. Can the instability in AIL truly be mitigated by changing the RA, or does it fundamentally arise from the adversarial optimization dynamics?

6. Would replacing the adversarial discriminator with a 1-Lipschitz critic (as in WGANs) render the entire reward-assignment discovery unnecessary?

[5] Zhang, Ming, et al. "Wasserstein distance guided adversarial imitation learning with reward shape exploration." 2020 IEEE 9th Data Driven Control and Learning Systems Conference (DDCLS). IEEE, 2020.
[6] Dadashi, Robert, et al. "Primal wasserstein imitation learning." arXiv preprint arXiv:2006.04678 (2020).

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the long-overlooked reward assignment (RA) function design in Adversarial Imitation Learning (AIL) by proposing an LLM-guided evolutionary search framework. The authors formalize RA function discovery as a meta-learning problem optimizing Wasserstein distance, conducting the search on MinAtar SpaceInvaders to discover DAIL's RA function: r(x)=0.5·sigmoid(x)·[tanh(x)+1]. Experiments demonstrate that DAIL outperforms GAIL, AIRL, FAIRL, and other baselines across multiple unseen environments in both Brax and MinAtar suites. The authors further analyze policy entropy and log-ratio distributions to elucidate the mechanisms behind DAIL's enhanced training stability.

### Strengths
1. The AIL literature has predominantly focused on discriminator stabilization (Wang et al. 2024; Luo et al. 2024), while RA function design has been consistently overlooked. This paper is the first to systematically treat RA functions as an independent research subject, filling a significant void. More importantly, the authors shift the RA design paradigm from theory-driven (deriving from f-divergences) to data-driven (directly optimizing performance metrics), offering a fresh methodological perspective for AIL algorithm design. The discovered DAIL function is remarkably concise (one line of code) and easily integrable into existing AIL frameworks, demonstrating high practical value.

2. The choice of Wasserstein distance as the fitness function is natural, given its theoretical robustness in measuring distributional discrepancies. The LLM-guided evolutionary search organically combines the interpretability of code representation with search efficiency, providing more targeted exploration compared to pure random search. The initial population includes classical methods (GAIL, AIRL, FAIRL, GAIL-heuristic), ensuring high-quality starting points—a design choice that reflects the authors' deep domain understanding. Overall, from problem formalization to search strategy selection, the design demonstrates sound intuition.

3.  The training dynamics analysis in Figure 6, examining policy entropy and log-ratio distributions, clearly reveals why DAIL outperforms baselines: DAIL saturates to near-zero in the log-ratio < -1.8 region, effectively filtering noisy signals corresponding to random or low-quality behaviors, thereby achieving more stable training. This mechanistic explanation is not only data-supported (policy entropy converges faster to levels approaching PPO with true rewards) but also holds consistently across multiple environments (though space constraints limit presentation to selected examples), strengthening the credibility of the conclusions. The ablation study in Figure 7 further confirms the necessity of combining sigmoid and tanh, showing that using either function alone cannot match the combined effect.

### Weaknesses
1. The lack of theoretical guarantees is the most serious issue in this work. DAIL's RA function does not correspond to any known f-divergence, and thus lacks convergence guarantees. While the paper acknowledges this limitation (Lines 247-251), it merely justifies the approach with "empirical feedback" and "robust generalization" without providing any alternative theoretical analysis. In contrast, GAIL, AIRL, and FAIRL all have formal proofs of convergence to ρ_π = ρ_E (Ghasemipour et al. 2020). This deficiency means we cannot predict under what conditions DAIL will be effective or when it might fail, nor can it guide future RA function design. For an algorithmic paper, relying entirely on empirical results without theoretical foundation is a critical weakness. Even if rigorous convergence proofs are intractable, the authors should at least provide intuitive analysis under relaxed assumptions (e.g., bounded MDPs, Lipschitz continuous policy classes, approximately optimal discriminators) or explicitly discuss potential failure modes and applicability boundaries.

2. The evidence for generalization is fundamentally insufficient. The paper's core contribution claims to have "discovered a generalizable RA function," but the evidence supporting this claim has serious problems. According to Section 6.1 (Lines 347-353), the authors conduct the search on only one environment—MinAtar SpaceInvaders—and then test on the other 7 environments (5 Brax + 2 other MinAtar). This raises two critical concerns: First, SpaceInvaders is an Atari game with discrete action space, while Brax environments involve continuous control tasks with entirely different dynamics. That an RA function discovered on SpaceInvaders performs well on Brax could be a coincidence (SpaceInvaders happens to be a "good" meta-training environment) rather than evidence of DAIL's true cross-domain generalization capability. Second, different task types (e.g., sparse vs. dense reward environments, or varying state space dimensions) may require RA functions with different characteristics, yet the paper does not explore this at all.

3. The missing critical experiments make the generalization claims unconvincing: We would expect to see searches conducted independently on at least 2-3 different environments (e.g., search on Ant then test on Walker/HalfCheetah; search on Breakout then test on Asterix/SpaceInvaders), followed by comparison of whether the discovered RA functions are consistent. Only if multiple independent searches converge to similar functional forms would that constitute genuine evidence of generalization. The current experimental design at best demonstrates that "SpaceInvaders is a good meta-training task" but cannot establish "DAIL's generalization capability." The performance improvements shown in Figure 4 may merely reflect the peculiarities of SpaceInvaders rather than the method's universality.

4. Several writing and presentation details could be improved. Upon verification, the paper is largely consistent in notation usage (primarily using rfr_f
rf​ and ℓ\ell
ℓ), but the statement "offers less flexibility" on Line 85 is overly vague and requires more specific explanation of which aspects of non-adversarial methods lack flexibility. The title of Appendix A.2 presents potential confusion: since the paper uniformly uses the form Df(ρE∣∣ρπ)D_f(\rho_E || \rho_\pi)
Df​(ρE​∣∣ρπ​), whether FAIRL should be labeled as "Forward KL viewed as reverse" or simply "Forward KL" depends on the reader's interpretive perspective. The authors should clarify this explicitly in the title or main text to avoid misunderstanding.

### Questions
1. If the search were conducted independently on Brax environments such as Ant or Walker, would the resulting RA functions be similar in form to the DAIL function discovered on SpaceInvaders? How large would the performance differences be? This experiment is crucial for validating the method's true generalization capability. Relatedly, are the performance differences among the top-5 functions shown in Table 6 statistically significant? 

2. Although DAIL does not strictly correspond to any f-divergence, could the authors provide intuitive analysis or informal reasoning about convergence under relaxed assumptions (e.g., small learning rates, approximately optimal discriminators, bounded state spaces)? Even without proving convergence, could the authors explicitly discuss the types of environments or conditions where DAIL might fail (e.g., highly non-stationary environments, tasks with extremely sparse rewards)? 

3. If multiple independent searches were conducted (using different random seeds to initialize the LLM or population), how high would the overlap be among the top-K functions? This would reflect the stability of the search process. Additionally, could the authors report the complete computational cost (total GPU hours, rather than the per-evaluation time shown in Table 7)? Given 200 candidate functions, 10 generations of evolution, and each evaluation requiring training 16 agents to convergence, the total cost could be substantial. Clarifying this is important for assessing the method's practicality. 

4. Why does IQ-Learn's performance in Table 8 significantly underperform its original paper's reports (even failing to beat simple BC in multiple environments)? Has the implementation correctness been verified? How were the hyperparameters tuned? This anomalous result weakens the credibility of comparisons with non-adversarial methods. Additionally, since the paper uses Wasserstein distance as both the fitness function and evaluation metric, could the authors specify what cost matrix is used (Euclidean distance? or other?)

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on discovering useful reward functions for adversarial imitation learning, with the goal of improving downstream performance of agents beyond that of hand-designed IL baselines that rely on f-divergence minimization. The paper proposes DAIL, the first meta-learned AIL algorithm, which outperforms standard AIL algorithms on a variety of tasks in continuous control (Brax) and in Atari (MinAtar).

DAIL formulates the meta-learning problem as minimizing the Wasserstein distance between the expert distribution induced by the dataset and the optimal policy with respect to the meta-learned reward function. The meta-learning approach is evolutionary search, which is simulated by an LLM that takes in two "parent" reward functions and aims to produce a "child" reward function that combines desired properties of both parents, making it more suitable for downstream IRL. Once a said reward function is given, standard AIL proceeds, and the Wasserstein distance between the final learned policy and the expert distribution is estimated, which is signal for the meta-learner.

### Strengths
The paper is very cleanly written, and even as a person who doesn't have much of a meta-learning or metagradient RL background, I was easily able to understand the problem and semi-verify that the approach was reasonable. The approach is also fairly novel -- as the paper claims, and as I have verified, there has been no prior work that has attempted to meta-learn a reward function via standard meta-gradient or evolutionary search approaches in AIL.

The results in the paper are also quite strong, improving upon standard baselines such as GAIL and AIRL somewhat considerably based on the probability of improvement plot in Figure 5, meaning that the meta-learning approach is yielding strong reward functions in practice.

### Weaknesses
For the scope of this paper, I don't really know if this should count as a weakness, but I would have liked to see some additional baseline comparison to other AIL algorithms beyond AIRL and GAIL. In particular, there are many possible f-divergences or IPMs that could be minimized to obtain a strong adversarial IL algorithm in practice, including the Wasserstein distance (e.g. see Wasserstein GAN [1]). This is in particular a useful baseline because the meta-learning objective is aiming to minimize the Wasserstein distance between the expert policy and the learned policy, which a Wasserstein AIL objective building on top of the Wasserstein GAN would in theory do. This has also been done in other work, such as the hybrid IRL paper [2].

In addition, it may have been a good idea in my opinion to see how the meta-learned reward function translated to much more complicated tasks than those of Brax and MinAtar. This was done in the Atari suite by Oh et. al. [3] when they discovered RL update rules, so maybe from an experimental standpoint it would be a good idea here. However, this is not a huge weakness in my eyes as Brax and other MinAtar environments are already much different than MinAtar SpaceInvaders, which was used for meta-training. Additionally, maybe some work could've been done focusing on if meta-learning the reward function across more environments would've led to a more robust RL algorithm.

[1]: Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville: Improved Training of Wasserstein GANs (NIPS 2017)
[2] Juntao Ren, Gokul Swamy, Zhiwei Steven Wu, J. Andrew Bagnell, and Sanjiban Choudhury: Hybrid Inverse Reinforcement Learning (ICML 2024)
[3] Junhyuk Oh, Matteo Hessel, Wojciech M. Czarnecki, Zhongwen Xu, Hado van Hasselt, Satinder Singh, and David Silver: Discovering Reinforcement Learning Algorithms (NeurIPS 2020).

### Questions
I don't have any major questions right now -- I think the paper is pretty clear and I think I understood the main points reasonably clearly.

### Soundness
4

### Presentation
3

### Contribution
3
