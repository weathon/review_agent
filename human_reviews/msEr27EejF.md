# Correlated Proxies: A New Definition and Improved Mitigation for Reward Hacking

- Avg Score: 7.20
- Decision: Accept (Spotlight)
- Scores: 8, 8, 8, 6, 6

## Abstract
Because it is difficult to precisely specify complex objectives, reinforcement learning policies are often optimized using proxy reward functions that only approximate the true goal. However, optimizing proxy rewards frequently leads to reward hacking: the optimized reward function ceases to be a good proxy and the resulting policy performs poorly with respect to the unspecified true reward. Principled solutions to reward hacking have been impeded by the lack of a good definition for the problem. To address this gap, we introduce a definition of reward hacking based on the correlation between proxy and true rewards for states and actions seen by a “reference policy” that breaks down under optimization. We show that this definition captures reward hacking behavior across several realistic settings, including in reinforcement learning from human feedback (RLHF). Using our formulation, we show theoretically that regularization to the reference policy can effectively prevent reward hacking. While the current practice in RLHF applies a KL penalty between action distributions for this purpose, our theory suggests regularizing the χ2 divergence between the policies’ occupancy measures can be more effective. We intuitively show the benefits of this type of regularization and demonstrate that it better mitigates reward hacking in practice across four realistic settings, including RLHF. Our code is available at https://github.com/cassidylaidlaw/orpo.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies reward hacking, including in the context of reinforcement learning from human feedback. It proposes a new definition of what constitutes reward hacking and shows that a specific form of regularization can help with preventing reward hacking. In the context of RLHF, the results suggest a different regularization scheme than a standard KL penalty. The authors further argue that this type of regularization better mitigates reward hacking in practice.

### Strengths
The paper is well-written, enjoyable to read, and relatively easy to follow. It provides an interesting perspective on reward hacking through the lens of regularized RL objectives. The approach is based on a new definition of reward hacking, which assumes access to a base policy-a reference used to determine whether a proxy reward is hackable by a learning agent. This definition is intuitive, although I have some questions and comments about its expressiveness.

I find the connection to the KL-regularized RLHF objective, and the corresponding result (Theorem 3.1), particularly interesting. To the best of my knowledge, this result is novel. The experimental results are extensive and, in my opinion, demonstrate the effectiveness of the proposed approach.

### Weaknesses
Regularization to a base policy in RLHF for language models is somewhat natural, as we aim to preserve the language generation capabilities of our policy. However, as shown in Table 1, it can also lead to considerable performance loss in some environments (e.g., Pandemic Mitigation), where the base policy does not perform well. I wonder how preference-based RL (without regularization) would fare in such environments. It seems that the proposed approach is somewhat constrained, as it requires humans to provide a reliable reference point (base policy), which may be challenging in complex environments.
 
While I am generally positive about this work, I think I didn’t fully understand how it compares to some other well-known approaches to mitigating similar problems (e.g., techniques based on inverse reward design or risk-averse RL). I understand that these approaches do not necessarily address the same problem, but the experimental setup does not seem to include baselines that would illustrate the importance of Definition 4.2. For example, since $\pi_{base}$ is given, why isn’t IRL an adequate baseline?
 
The related work section covers most of the important references, but a more comprehensive literature review on RLHF could be beneficial. For example, the paper mentions:

> Our regularized objective in (5) differs in two main ways from the KL regularization used in RLHF. First, our results suggest optimizing the occupancy measure (OM) divergence between policies, whereas RLHF uses the action distribution (AD) divergence. 

however, it does not explain how this connects to recent results that propose OM-based regularizers. For example, it would be useful if the authors could comment on the MDP formulation of RLHF from

> Nika et al., Reward Model Learning vs. Direct Policy Optimization: A Comparative Analysis of Learning from Human Preferences, ICML 2024.  

which in Section 7.2 introduces the KL-regularized RHF objective based on occupancy measures. 

One potential drawback of the proposed approach is that it is based on estimating occupancy measures, which may be challenging to estimate for high-dimensional state spaces. It would be helpful if the authors could comment on the level of difficulty of the current environments for an RL agent. Furthermore, how accurate are the occupancy measures obtained in the experiments? It seems that it should be possible to estimate this for some of the environments.

Some minor details in the experimental setup were not clear to me. The appendix states:

> We primarily tuned the hyperparameters listed below in order to ensure that the proxy reward would be properly optimized and reward hacking would occur without regularization.

Does this mean that we typically don't observe reward hacking in these environments?

### Questions
Please see my comments and questions *Weaknesses*. I'd appreciate if the authors could respond to these comments and questions.

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
3

### Summary
This paper critically formalizes a notion of exploitability in RL-based
fine-tuning when the reward function is difficult or impossible to
specify exactly, and is instead replaced by a "proxy" reward. It argues
that reward hacking is characterized by the proxy reward admitting
optimal policies that are in fact *not* aligned with our desired
behavior, and their characterization is validated with several intuitive
case studies. Moreover, under this characterization, the authors propose
a RL fine-tuning algorithm that prevents reward hacking by regularizing
the *occupancy measure* of the fine-tuned policy towards a base policy
with respect to $\chi^2$ divergence, under weaker assumptions on the
proxy reward than existing results. Moreover, the paper introduces a
practical implementation using techniques from inverse RL (particularly
GAIL), and demonstrate that their method performs better than baselines
across several fine-tuning tasks.

### Strengths
This paper studies a very hot topic in the field, and in my opinion,
provides some very interesting insight. Most notably, I really enjoyed
the following entangled results in the paper:

1.  Even with a proxy reward that is highly correlated with the ground
    truth reward, reward hacking is possible;
2.  But under such correlated rewards, it is (probably) possible to
    prevent reward hacking by regularizing the fine-tuned policie's
    occupancy measure (wrt $\chi^2$).

It is also nice that their proposed framework fits nicely with that of
IRL, allowing the authors to leverage performant IRL algorithms for RL
fine-tuning.

### Weaknesses
My main concerns with the paper regard lack of clarity in several
respects. I will outline these below, and further examples are given in
"Minor Issues" and "Questions".

1.  I find the narrative of the paper to be a little misguided. The
    paper goes on to try to essentially axiomatize reward hacking,
    leading to many fairly philosophical or imprecise heuristic
    discussions (e.g. "it's not reward hacking if the proxy is just
    bad!"). I think this is, in a sense, actually overcomplicating
    things. Under the hood, I think there is a very clear and
    uncontroversial message here:
    1.  When fine-tuning, we want to learn a policy that is better than
        some base policy at maximizing some ground truth reward (this is
        hard to argue with);
    2.  Proxy rewards that are highly correlated to the ground truth
        reward are still "hackable", in the sense that policy
        optimization with such proxies prohibits improvement over the
        base policy (the authors show this);
    3.  But hackable correlated proxies are not useless—it is possible
        to successfully fine-tune a base policy with such a proxy using
        $\chi^2$ regularization (the authors basically show this, see
        Questions).
2.  While interesting, I think some of the theoretical results are
    actually fairly weak (see Questions). For instance, while the
    theoretical results suggest regularization of the fine-tuned
    policy's occupancy measure, they do not actually suggest that this
    should be an improvement over action-distribution regularization.
    Moreover, it is not actually clear that the lower bound of Theorem
    5.1, which is the main theorem of the paper, can actually be
    positive (in other words, Theorem 5.1 does not explicitly show that
    fine-tuning with occupancy measure regularization can actually lead
    to improved alignment). I suspect it is possible for the authors to
    correct these, and that would strengthen the paper.
3.  Overall, there are several issues with writing, such as discussion
    of related work, redundancy in intro/background sections, and some
    terminology, which I outlined below.

My score is mostly a reflection of weakness 2 here. I trust that the authors can fix weaknesses 1 and 3 fairly easily. Should the authors address weakness 2 (as well as the questions below), I would be happy to increase my score.

## Minor Issues

Stiennon et al. (2020) is definitely not the first example of using KL
regularization to an "offline policy". This is very commonly done in all
of maximum entropy / entropy-regularized reinforcement learning, dating
back at least fifteen years before this reference.

With regard to prior work in offline RL that enforce that the learned
policy does not veer too far away from the data distribution, the work
of \[2\] is the most appropriate reference (see list of references in
Questions section below).

The paper starts off much too slow. The first four pages are fairly
redundant, the isssue of formalizing the notion of reward hacking is
addressed several times, and only very heuristic high-level ideas are
suggested for addressing the issue. I believe this pages can be
condensed a lot, which would make the paper easier to read.

The section about the desiderata for a definition of reward hacking can
be clarified a lot. In particular, it is not clear to me exactly what
the desiderata are, they should be stated very explicitly (especially
since the paper later refers to them in some order, e.g. "the second
desideratum").

At the bottom of page 5, $\sigma_{\tilde{R}}$ and $\sigma_R$ are really
variances, not standard deviations (so they should probably be written
as $\sigma_{\bullet}^2$).

In definition 4.2, there is a missing word — "*reward hacking* occurs
when an optimal policy $\tilde{R}$ **lower** true reward…".

In equation (8), you use $\phi$ on both sides of the equation (you
should use e.g. $\phi'$ as the variable being minimized).

While definition 4.2 is interesting, I don't think "reward hacking" is
the right name for the phenomenon being described in this definition.
Particularly, this definition presents a property on a proxy reward, but
not on the process of fine-tuning a policy with this proxy. Indeed, one
of the contributions presented later in the work is an algorithm that
prevents reward hacking under proxies satisfying definition 4.2! Rather,
this definition presents conditions under which reward hacking *may*
occur. As such, I believe "reward exploitability" is an example of a
more appropriate name: standard policy optimizations *can* exploit/hack
the reward function if you're not careful.

On line 313, "polulation" should be "population".

### Questions
In the introduction, you claim that the formalization of a "good proxy"
has been elusive. Hasn't this issue been addressed by the EPIC distance
\[1\]? What does EPIC leave to be desired?

Is there a mistake in the Table 1 cell for traffic control with state
occupancy KL? Particularly, the standard deviation of $22.6$ is
alarming.

Theorem 5.1 is nice in the sense that it tells us that occupancy measure
regularization (particularly w.r.t $\chi^2$) can prevent reward hacking
even when the proxy is correlated with the true reward. However, there
is no claim about whether or not reward hacking can (at least in
principle) be prevented with action distribution regularization. So,
naturally, I'm curious whether or not a similar result can be achieved
with action distribution regularization. The empirical results suggest
not, but perhaps we just haven't found the right way to do it yet. Do
you think it's impossible?

In the experiments, RLHF is treated as a contextual bandit problem, in
which case there is no difference between occupancy measure
regularization and action distribution regularization (because state
occupancy is independent of actions). However, especially given the
autoregressive nature of many existing LLMs, RLHF can also be viewed as
a sequential problem where the policy sequentially chooses individual
tokens, and a reward is presented after the last token is chosen. How
would OM under this formulation of RLHF compare to AD in the "atomic"
contextual bandit case?

Theorem 5.1 looks nice because intuitovely, it tells us that by
controlling the $\chi^2$ between the learned policy and the base policy,
we can prevent reward hacking. But(!), it is actually not entirely clear
to me if this lower bound admits the possibility that we can actually
*improve* upon the base policy using this regularization. Clearly, as
the $\chi^2$ divergence tends to $0$, the lower bound tends to $0$. But
can the lower bound ever be greater than $0$?

## References

1.  Gleave, Dennis & Legg et al. (2021) Quantifying Differences in
    Reward Functions, arXiv.
2.  Fujimoto, Meger & Precup (2019) Off-Policy Deep Reinforcement
    Learning Without Exploration, ICML.

### Soundness
3

### Presentation
2

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
The authors provide a new definition of reward hacking based on a new definition of correlated policies. Reward hacking is a phenomenon that occurs when optimizing a proxy reward for a true reward function yields worse performance over the true reward. The definition of correlated captures proxy reward functions that match the true reward function in visited state action pairs by a base policy. In addition, the authors propose occupancy measure regularization with respect to a base policy as a method to overcome reward hacking. They show theoretically and empirically that the proposed method mitigates reward hacking.

### Strengths
- The paper is very well written with an astounding flow of information. 
- The authors provide a lot of great intuitive examples for the arguments they make.
- The authors provide a wide range of experiments that show both the soundness and robustness of the claims.

### Weaknesses
- In Theorem 5.1, there is the assumption that the state-occupancy measure of the policy is absolutely continuous with respect to the base policy. However, during training, this is not something that can be guaranteed. I suggest adding an extra discussion on how regularization also pushes toward the learned policy satisfying this assumption.

### Questions
- The definition of correlated rewards allows for the correlated reward to deviate arbitrarily in states with zero occupancy by the base policy.  Doesn't this allow for policies that can be dangerous to still be correlated with the true reward? For example, if the base policy avoids a state with a large negative true reward but the proxy reward function assigns a very high positive reward for that state. Wouldn't this reward incentivize policies that try to visit this state and get a large negative true reward? In addition, wouldn't these kinds of proxy create policies that break the assumption in Theorem 5.1?

### Soundness
4

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
4

### Summary
The paper addresses the issue of reward hacking in RL, where policies optimized with proxy rewards. To address this, the authors propose a novel, formal definition of reward hacking, which is based on the correlation between proxy and true rewards observed by a base policy. The paper also propose a regularization method based on $\chi^{2}$ divergence between the policies' occupancy measures.

### Strengths
The concept presented, while not very complex, demonstrates impressive effectiveness.The authors support their claims with a solid mix of theoretical proofs and comprehensive experiments across various domains. Specific comparison between $\chi^{2}$ vs. KL divergence has been provided. Intuitive examples are also provided. Additionally, intuition about "optimizing the occupancy measure (OM) divergence between policies" might be helpful to the whole community. Overall, this is an excellent paper with good results and insightful contributions.

### Weaknesses
However, my primary concern lies in the assumptions underpinning their framework and the choice of metrics. Please see the specific questions below for further details.

### Questions
## Questions about underlying assumptions
1. Does the paper assume that both the base policy $\pi_{base}$ and the optimized policy $\pi$ are able to acess the same observation states?
2. **Handling of Unobserved Variables**: In the traffic domain, several potentially impactful variables may be unobserved, as noted in [1][2]. For example, factors like the driver’s expertise level [1] or weather conditions [1] are often missing in driving datasets. How does your framework address scenarios where such unobserved variables exist?
3. During the training phase, does the agent have access to the ground truth reward value?
4. What are major assumptions behind your framework?

## Questions about metrics
1. **Unknown True Reward**: In many practical applications, such as driving, it is challenging to properly define a true reward function [1][2]. In RLHF, the system often relies on preference pairs rather than a true reward. When the true reward is unavailable, what alternative metrics would you recommend for evaluation?
2. **Alternative Metrics for RLHF**: For RLHF, additional metrics might be helpful, such as the winning rate (preferred vs. rejected) or a safety score. How could these be considered in your framework?

## Other concerns
1. **Similarity to Prior Work**: This paper is similar to [3], particularly in the theoretical sections and some experiments. It would be beneficial for the authors to clarify the distinctions between their proposed approach and that of [3].
2. **$\chi^{2}$ vs. KL Divergence**: Does $\chi^{2}$ divergence consistently outperform KL divergence in all scenarios? Are there cases where KL divergence yields better results? If so, what trade-offs are involved? If not, could the authors outline situations where KL divergence might be more suitable?

----------
[1] Ruan, Kangrui, and Xuan Di. "Learning human driving behaviors with sequential causal imitation learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 4. 2022.

[2] De Haan, Pim, Dinesh Jayaraman, and Sergey Levine. "Causal confusion in imitation learning." Advances in neural information processing systems 32 (2019).

[3] Laidlaw, Cassidy, Shivam Singhal, and Anca Dragan. "Preventing Reward Hacking with Occupancy Measure Regularization." ICML Workshop on New Frontiers in Learning, Control, and Dynamical Systems.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a new definition for reward hacking, based on the idea that we should only care about hacking of proxy reward functions which are correlated (in terms of state-action occupancy) with the true reward function under some "reasonable" base policy.

They then show theoretically and empirically that regularizing reward using an occupancy measure term as opposed to the standard KL divergence leads to less reward hacking across several simple environments.

### Strengths
The paper is extremely well-written.

The paper builds on past literature in a coherent and helpful way.

The paper shows experiments on interesting environments.

### Weaknesses
While the new definition can be defended as an improvement over past definitions, it still leaves some things to be desired. Specifically:

* it requires a "natural base policy", which might not exist in some real-world relevant settings
* even if a natural base policy does exist, it's not obvious that the state-occupancy distribution induced by it is the one we actually care about: the optimal policy wrt the true reward might be quite different from what we think of as the "natural" base policy.
* the definition only considers hacking to have occurred when looking at the *optimal* policy with respect to the proxy; in practice, we never find the optimal policy, yet hacking occurs nonetheless with intermediate sub-optimal (or locally optimal) policies
* the paper doesn't say how to know whether, and what to do if, the (eg, learned) proxy reward function is not r-correlated with the true one.

### Questions
## Typos
there is an issue with the text in Definition 4.2: I think your $\tilde{\mathcal{R}}$ ate something after "policy" and another thing before "lower".

## Suggestions
* line 24: "superior" feels a bit strong
* line 164: the SEIR and glucose environments diagrams are too small to read/understand. just a comic of the environment would be better. the gridworld is on the edge of too small but ok
* line 217: I would avoid using the word "misaligned", since that has all kinds of connotations. Consider just "different", "not identical to", etc. This is a nit: if you feel strongly about "misaligned", I think it's probably ok.
* line 312: the SEIR model has been around for a long time (a quick search found stuff from 2013 but I think it's been around for significantly longer); please try to find an original reference OR change the sentence to talk specifically about that one from 2020 which you're using.
* 369: RHS = right hand side? some readers might not know this; please define on first use
* 402: "widely used to prevent reward hacking in RLHF" well, I don't think this is quite true. KL divergence, to my understanding, is primarily used to avoid instabilities in the training procedure (important in when using PPO in general, not just RLHF!). Now, it seems likely that there is a side-effect of this regularization which helps prevent hacking, but that's not the main reason (in particular: if you remove KL divergence, you don't get hacking, you just get a failure to train). Please update this sentence! (or please explain if I'm missing something).

## Questions
* copying from above, my biggest concern is this one: even if a natural base policy does exist, it's not obvious that the state-occupancy distribution induced by it is the one we actually care about: the optimal policy wrt the true reward might be quite different from what we think of as the "natural" base policy. The issue I see here is that if the "actually really good" (much better than "reasonable base") policies have a way different state-occupancy distribution than the base policy, then the "check" that the proxy reward is "similar enough" to the true reward over the "reasonable" distribution just doesn't matter: what matters is what happens near optimality. Am I missing something here?
* why do you only consider the *optimal* policy wrt the proxy in Definition 4.2? This seems too restrictive (and you apply the regularization at all points during training, so unless I'm missing something you don't rely on this optimality?)
* how do you expect your technique to scale?

## Closing note
I gave the paper a 5 because I think it's important that some things be changed before it's ready for publication. But I will be happy to increase the score to a 6 if my minor points are met, and possibly more if my greater fears about the utility of the technique in capturing reward hacking in different but relevant parts of state-action occupancy space are allayed. Indeed, it's possible that just acknowledging and writing about this issue would be enough: so long as it is clearly framed what this paper is and is not doing and to what extent it "covers all the bases" wrt reward hacking, I think it's worth publishing. It's really nicely presented and a nice idea with experiments to back it.

### Soundness
4

### Presentation
4

### Contribution
2
