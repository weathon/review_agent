# Improving Human-AI Coordination through Online Adversarial Training and Generative Models

- Decision: Accept (Poster)
- Scores: 8, 8, 6

## Abstract
Being able to cooperate with diverse humans is an important component of many economically valuable AI tasks, from household robotics to autonomous driving. However, generalizing to novel humans requires training on data that captures the diversity of human behaviors. Adversarial training is a promising method that allows dynamic data generation and ensures that agents are robust. It creates a feedback loop where the agent’s performance influences the generation of new adversarial data, which can be used immediately to train the agent. However, adversarial training is difficult to apply in a cooperative task; how can we train an adversarial cooperator?
We propose a novel strategy that combines a pre-trained generative model to simulate valid cooperative agent policies with adversarial training to maximize regret. We call our method \textbf{GOAT}: \textbf{G}enerative \textbf{O}nline \textbf{A}dversarial \textbf{T}raining. In this framework, the GOAT dynamically searches the latent space of the generative model for coordination strategies where the learning policy---the Cooperator agent---underperforms. GOAT enables better generalization by exposing the Cooperator to various challenging interaction scenarios. We maintain realistic coordination strategies by keeping the generative model frozen, thus avoiding adversarial exploitation. We evaluate GOAT with real human partners, and the results demonstrate state-of-the-art performance on the Overcooked benchmark, highlighting its effectiveness in generalizing to diverse human behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on the problem of training a cooperative agent in a two player cooperative Markov game. It combines a generative procedure for sampling cooperative partner agents with an adversarial procedure for finding agents whose strategies challenge a learning agent to train an agent that is robust to diverse partner strategies while playing cooperatively and maximizing returns.

The generative procedure uses a VAE similar to previous work. The adversarial procedure works by finding embedding values that would lead to partner agents being sampled by the generator that maximize regret. This regret is calculated by calculating the difference between self-play scores and cross-play scores with the learning agent.

Evaluation is done on a toy matrix game, a reaching environment, and two overcooked environments. Evaluation with human participants in overcooked show that the proposed technique, GOAT, outperforms evaluated baselines in terms of dealing with human strategies.

### Strengths
* This paper combines ideas from previous works in a clear and fairly principled manner.
* The idea is explained well. Figure 1 is a good overview of the technique. Section 3 is succinct but clear, and the figures referring to the experiments are also clear and fairly well explained. Some caveats here will be expounded on the weaknesses section
* Human evaluations give the idea a lot of credence. The idea seems to be performing as advertised in the overcooked domain.
* The human evaluation procedure is well designed.
* Appreciate the error bars and their explanation in Figure 4.
* As far as I can tell, all required relevant work seems to be cited.

### Weaknesses
* Figures 5 and 6 are a little hard to follow. After reading the text in the experiments and intuiting the intention, I am able to guess what the authors are trying to show. Figure 6 can be understood well in this manner. But Figure 5 was still a little confusing due to the two different dots for GAMMA and GOAT in addition to the gradient colored dots for the episode numbers. If Figure 5 can be simplified it would get the point across much better.
* Nit about how the value function is defined at the end of the first paragraph of Section 3: Value functions are generally conditioned on the state. Perhaps better to phrase this expression as expected returns?
* Figure 3 was also a little difficult to look at. But I appreciate the summation across methods in the last row. Perhaps this row can be separated or highlighted so a reader can look at the summary first before going through the detailed comparison for each heuristic agent. The description would also benefit from mentioning that the maximum sum would be `11`.


Minor: Line 83, the citation should be a `\citep`

### Questions
* In the paragraph preceding equation 4, the paper claims that self-play is a valid proxy for optimal score without actually stating that this is a proxy. Would self play actually be optimal in all scenarios? I can imagine scenarios where the two agents need very different policies in order to act optimally. E.g. being when agents need to fulfill different roles. Perhaps this paragraph needs to be phrased more carefully.

### Soundness
3

### Presentation
4

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
The authors present GOAT, a method that generates adversarial agents to train a cooperator agent for Human-AI Coordination. The method first utilizes a GAMMA-style VAE to encode an agent population in latent space. Next, the authors propose to train an Adversary to transform the GAMMA latents to Adversarial embeddings, which is then used by the GAMMA decoder to simulate an adversarial agent. The Cooperator Agent and Adversary are jointly trained together in a minimax objective where the adversary tries to maximize the regret (SP to XP gap) between the adversary and Cooperator agent. The authors then evaluate GOAT on 3 cooperative environments, a matrix game, a cooperative reaching game and Overcooked-AI.

### Strengths
- The authors provide a intuitive solution to a long-standing problem in the Human-AI/ZSC community, that of generating viable adversarial cooperative agents without sabotaging effects.
- The paper is very well written and structured.

### Weaknesses
- The authors cites two recent/concurrent related works [1] and [2] but did not compare GOAT with the two methods in the experimental sections. It will be interesting to see how GOAT compares to the two methods especially ROTATE as they also proposes a regret like objective to generate a curriculum of adversarial agents to train a cooperator agent.
- GOAT is evaluated on fairly simple, fully observable grid-world environments which limits its generalizability to more complex environments such as those with partial observable states [3] or continuous actions [4]. 

[1] Wang, C., Rahman, A., Cui, J., Sung, Y., & Stone, P.  ROTATE: Regret-driven Open-ended Training for Ad Hoc Teamwork. arXiv preprint arXiv:2505.23686.

[2] Villin, V., Buening, T. K., & Dimitrakakis, C.  A Minimax Approach to Ad Hoc Teamwork. In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (pp. 2105-2114).

[3] Gessler, T., Dizdarevic, T., Calinescu, A., Ellis, B., Lupu, A., & Foerster, J. N. OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination. In The Thirteenth International Conference on Learning Representations.

[4] Kang, X., Lee, S. W., Liu, H., Wang, Y., & Kuo, Y. L.  Moving Out: Physically-grounded Human-AI Collaboration. arXiv preprint arXiv:2507.18623.

### Questions
- In Figure 5, the authors show that the adversarial latent vectors sampled by GOAT is significantly far away from the cluster of GAMMA latent vectors. As I understand, the GAMMA latent vectors can be interpreted as an interpolation between actual agent trajectories encoded in latent space during the VAE training process. Does this imply that GAMMA could extrapolate new agents out of the standard trained distribution?
- Why is REINFORCE chosen over PPO to train the Adversary?
- The authors state that the main limitation of GOAT is the reliance on a trained VAE model. Could the authors comment on what might the minimally viable population size/amount of data to train a viable VAE model for GOAT?
- Relatedly, does the type of agents (CoMeDi vs MEP) matter when it comes to VAE training?  Does it affect the quality of latent vectors of the VAE?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the problem of learning cooperative agents that are robust to a diverse distribution of human and artificial agents. The models uses a pre-trained generative model to simulate valid cooperative agent policies with adversarial training to maximize regret. The policy itself tries to minimize regret, this induces an adversarial setup, but the challenge is to not the adversary be adversarial _and_ cooperative - two conflicting objectives. If the adversary is not constrained to such policies, then it can become overly combative, resorting to sabotage instead of making the policy robust and foolproof. In contrast to zero-sum games like chess where self-play can improve the performance of the policy by exploiting weaknesses against itself, in cooperative games there is no such incentive for a policy. 
To prevent cross-play performance from self-sabotaging, the model essentially samples adversarial players from a learned distribution of agents using a variational autoencoder. this limits the capability for self-sabotage since the agents are sampled from a predefined distribution. This makes the contribution of the work relevant to the community.

### Strengths
**Strengths**
1. The VAE model and generative policy allows for sampling policies parametrically with a smooth latent space, preventing expensive zeroth-order methods to sample policies. This makes the optimization objective more tractable and inexpensive. This idea is also used in controllable image generation and editing where latent variables are optimized to achieve a test-time objective with a pretrained network. 
2. The formulation is pretty straightforward - given a cooperator policy $\pi_C$, the adversarial policy chooses a partner policy $\pi_P$ to minimize the value function with cross play of the cooperator and partner, while maximize the self-play performance of the partner policy - this is presumably to prevent selecting incompetent policies. This is also interpreted (accurately) as the regret of choosing a cooperative policy over itself in a cross-play scenario. Regret for bad partner policies will be lower (due to low first term in Equation (4)) than a good partner policy that does not work well with the cooperator policy. 
3. The experiment setup is satisfactory - including a cooperative matrix game allows easy visualization of one-step policies and analytically verifying optimal adversarial policies for a given cooperative policy, a 5x5 cooperative reaching game with more complexity but still tractable, and a hard game with a larger action space (Overcooked) with real-time dynamics on which performance is also shown with human cooperative players. 
4. Extensive analysis with related methods - GAMMA and MinMax is shown in RQ4 and RQ5. Figure 5 shows the adversarial objective drifting to new policies that significantly deviate from the normal distribution (in the projection space), showing that the model is indeed aiming to find harder partner policies.

### Weaknesses
1. The robustness of the policy depends on the coverage / support of the autoencoder and its simulated policies. The paper uses fixed agent populations with training methods (Line 178).
2. The paper is not self-sufficient in terms of mentioning how the distribution of partner policies are learned before the adversarial sampling is performed. 
3. Figures 5 and 6 are nice, but they also show that the policies deviate a lot from random normal as training progresses. Since the generative model works off a VAE, are the policies that have a latent so far from the normal distribution meaningful at all? Is a regularization apart from the self-play regret (as mentioned in Line 722) enough to constrain the space of sampled policies?
4. Figures 5 and 6 also show that the coverage of partner policies is very concentrated and lacks diversity - could it lead to a possible oscillation of the cooperative policy if certain partner policies are not compatible with each other? 

Minor nits: 
1. Line 75: "The regret objective proves effective because it challenges the learning agent with a curriculum of increasingly difficult, yet still feasible tasks." - Regret minimization by itself does not motivate a curriculum learning approach. This line could say something like "it challenges the learning agent and expands the frontier / coverage of harder yet feasible state configurations"

### Questions
**Questions**

1. Since the distribution of the partner policies is learned using a VAE, why cant the cooperative agent be trained to minimize regret over the entire distribution of partner policies (by sampling partner policies from the VAE) instead of using an adversarial method to sample partner policies? Is that going to be slower or faster than the proposed training in terms of environment interactions and training time? 
2. Line 376 - "We hypothesize this is because the generative model, trained on cooperative trajectories, encodes a latent space rich in strategic variations." - the paper posits that the generative model encodes a rich space of strategic variations is what leads to fast convergence - this approach is also used in other prior work (some of which are mentioned in the paper). Does GOAT perform better than prior work leveraging generative policies due to its adversarial sampling? 

Minor questions:
1. Line 45/74: Why is Adversary capitalized? 
2. uncommon notation for distribution, i.e. $\Delta$ is used. This could be replaced with a better notation in my opinion
3. Line 363: "PBT methods rely on simulated populations of self-play partners, are often computationally expensive, and have poor coverage of actual human behaviors."  - why is cross play and adversarial training less expensive than self play? I thought the tradeoff was to spend more compute on adversarial training to have better mode coverage of adversarial policies. Figures 4b and 4e show that GOAT is indeed faster but is there any argument or justification as to why that is the case, especially because adversarial training can be slow and brittle.

### Soundness
3

### Presentation
4

### Contribution
3
