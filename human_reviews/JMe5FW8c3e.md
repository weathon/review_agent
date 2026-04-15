# Stable Offline Value Function Learning with Bisimulation-based Representations

- Decision: Reject
- Scores: 3, 6, 8, 6

## Abstract
In reinforcement learning, offline value function learning is the procedure of using an offline dataset to estimate the expected discounted return from each state when taking actions according to a fixed target policy. The stability of this procedure, i.e., whether it converges to its fixed-point, critically depends on the representations of the state-action pairs. Poorly learned representations can make value function learning unstable, or even divergent. Therefore, it is critical to stabilize value function learning by explicitly shaping the state-action representations. Recently, the class of bisimulation-based algorithms have shown promise in shaping representations for control. However, it is still unclear if this class of methods can \emph{stabilize} value function learning. In this work, we investigate this question and answer it affirmatively. We introduce a bisimulation-based algorithm called kernel representations for offline policy evaluation (\textsc{krope}). \textsc{krope} uses a kernel to shape state-action representations such that state-action pairs that have similar immediate rewards and lead to similar next state-action pairs under the target policy also have similar representations. We show that \textsc{krope}: 1) learns stable representations and 2) leads to lower value error than baselines. Our analysis provides new theoretical insight into the stability properties of bisimulation-based methods and suggests that practitioners can use these methods for stable and accurate evaluation of offline reinforcement learning agents.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a new algorithm for representation learning in offline RL; with the goal to make value function learning more stable and stabilize policy optimization through learning better representations. It builds up on prior works on bisimulation based representations, and proposes a kernel based objective to learn representations for similar state-action pairs. Experimental results are demonstrated on a few tasks to demonstrate why the proposed KROPE algorithm can do stable and accurate policy evaluation.

### Strengths
The paper is well written and proposes a kernel based representation objective through first principles in RL; cares about why policy evaluation is an important measure/metric to look into for analyse stability of value function learning - and thoroughly discusses prior works in this widely studied area of learning good representations for online/offline RL. Furthermore, experiments analyse stability-metrics that are important to look into, when proposing new representation objectives; Figure 4 is a good example of that. It draws insights from prior works and discusses theoretical properties such as the Bellman completeness, that are worth looking into when analysing the strength of a new objective.

### Weaknesses
There are several key weaknesses that I find concerning in this paper : 

1. The proposed objective adds to the line of work around representation learning for RL; in addition to several past works that recently also studied bisimulation based objectives for control and offline RL. While the paper refers to past works, little comparisons are made to those prior works, either theoretically or experimentally - and it seems the authors start with policy evaluartion - discusses metrics related to stability for it - and claims to show expeirmentally why policy evaluation can be done well with the KROPE objective. However, several prior works have done this already, whether implicitly or explicitly showing the policy evaluation measure; and have demonstrated significantly empirically why bisimulation can be a good measure fior learning representations for offline RL. 

2. Following from the above comment - I do not understand the exact benefits the kernel based approach buys for this; whether we do bisimulartion based representation or not (such as looking into inverse or forward dynamics models for learning representations) - the benefits of kernels seem to be not explained too well here. Are there new techniques in deep kernel learning literature that can be exploited here? What happens when we go to large state action spaces, with complex tasks - how well would the kernel based approach scale here? 

3. Experiments are done on simple tasks where kernel based representations can assumably do similar to other objectives. However, as you scale to larger state-action spaces, I do not think the paper demonstrates well how well their algorithm would scale in this context. I would experiment more thorough comparisons with a large body of prior works for comparison, either empirically or theoretically.

4. Recent works (e.g Lamb et al.,) have significantly studied inverse dynamics models and claimed how these objectives can learn the true underlying latent dynamics in an unknown environment. In addition, Zang et al, studied and compared some of these inverse dynamics objectives with the bisimulation based objectives in offline RL - and demonstrated effectiveness of each approach on several empirical benchmarks. How well does the proposed KROPE representation do, in light of all these past works? In all these works, there is also the implicit or explicit demonstration of good enough policy evaluations to then do control. I do not think the discussion of policy evalaution and stability here is enough to show that KROPE can learn good policies for control.

### Questions
Few questions related to the weaknesses : 

1. Can the authors compare theoertically, why bisimulation with the kernel representations can learn the true underlying latent dynamics? How do we know whether inverse or forward dynamics objectives are better/worse compared to KROPE? What does the algorithm mean in context of learning the underlying latent dynamics of the world? 

2. Can authors compare to past works on bisimulation based representations, both theoretically and empirically - and show how well does KROPE perform and scale in complex control tasks in light of these past objectives? 

3. I would expect to see some more experimental results in more challenging benchmarks - I think the current suite of tasks are not enough to show effectiveness of the proposed KROPE objective. 

4. Zang et al for example also studies bisimulation in context of noise (similar to Amy Zhang et al’s past work on bisimulation for control) - can the authors comment on what would happen with KROPE, given reliance on kernels, in presence of noise in the environment?

5. All experiments are done on raw state-action spaces; there are no tasks with pixel based observations? I can imagine kernel representations would fail on more complex pixelized environments. Can authors comment on the trade-offs here?

6. I would like to understand the Bellman completeness claim much better in context of KROPE, compared to past works. Can authors comment on this? Or in short - what is the theoretical guarantee of this algorithm, and how well can we tell that KROPE learns good representations?

7. Is the claim here to learn true underlying latent dynamics, or more on the stability with the KROPE representations? Each claim would have it own questions to answer, depending on how the authors would like to move ahead - what’s the goal here?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores a method for learning representations of state-action pairs that can stabilize the learning of state-action value functions (e.g. for use in reinforcement learning) from offline datasets. The motivating idea is that if state-action pairs with similar values under a policy $pi$ have similar representations, this will make it easier to learn the true value function. First, the authors adapt the kernel-based $pi$-bisimulation representation proposed by Castro et al. (2023) from state to state-action value functions. Then, they present two novel stability results for this representation: first, they prove that using it for value function learning via least-squares policy evaluation will converge to the fixed point. Second, they show that it is "Bellman complete," an alternative notion of stability. Finally, the authors propose an algorithm for learning this representation from data and show that it is useful for stable value function learning in practice both in tabular MDP environments and in DeepMind Control Suite and D4RL tasks.

### Strengths
The paper is well-written, well-organized, and clear about the scope of the contributions and how they relate to prior work. The notation is clearly defined and easy to follow. The experiments compare the proposed representation learning algorithm (KROPE) with several appropriate baseline methods and convincingly show that KROPE supports stable value learning in standard RL benchmark tasks.

### Weaknesses
The scope of the contributions may be a little narrow. I believe the figures could use significant improvement. They are generally hard to read, using only color to differentiate the different methods and thin lines that are difficult to distinguish when the confidence interval is 0. While the learning traces in figure 3 are interesting, I wonder if there might be alternative ways to show these results that are easier to parse, such as summary statistics that capture the degree of instability (similar to figure 4 but focusing on the value error over the course of training in the different environments). For example, in figure 3b the plot is dominated by what looks like a single unstable seed for BCRL-na which makes it hard to tell what "good" vs "bad" seeds look like for each method.

### Questions
* It would be great to have more description for a few of the introduced concepts: why is the notion of generalizability introduced in section 2.3 a notion of generalizability? Why do we think Bellman completeness is a desirable property for stability? Is the condition on the injectivity of the abstract reward function in theorem 2 reasonable in practice?
* I think the first paragraph of the introduction could do a better job of orienting readers who aren't familiar with the idea of value representation learning. What does value function learning look like without special representation learning methods, and how do these methods aim to improve it? I.e. something like $q^{\pi}(s, a)$ --> $q^{\pi}(\phi(s, a))$, and making the role of function approximation here more explicit.
* Why is the higher error of KROPE for WalkerStand unsurprising? Lemma 3 discusses that KROPE may be less accurate when the transition dynamics or policy are stochastic, but I believe all the environments (including Walker) are deterministic and the policy should be stochastic across environments.
* In figure 4b, why does KROPE have a lower condition number than BCRL which specifically optimizes for this? The text at the top of page 10 seems to suggest that BCRL achieves a lower condition number even though the values contradict this.
* The description of FQE in section 4.1 is a little confusing. A cartoon of the $\phi$ network emphasizing the last layers and the representation vs. value function outputs might be helpful.
* I think terms like "one-step similarity" and "future similarity" would be more clear than "short-term" and "long-term" in equation (3) and definition 3.
* Minor issues: mismatched bracket at the end of equation (2); strange spacing around the sampled-from symbols beneath equation (3); FQE acronym was used without being defined yet in section 2.4.

### Soundness
4

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
5

### Summary
The paper studies the stability of bisimulation-based representations in the context of offline RL, specifically as they serve to stabilize the LSPE algorithm and guaranatee Bellman completeness. To this end, they introduce a state-action similarity kernel, called KROPE, based on Castro et al. (2022) and Castro et al. (2023) in the spirit of bisimulation for state similarity. KROPE representations are shown to theoretically guarantee off-policy evaluation (OPE) stability for LSPE and yield Bellman-complete representations. Empirically, KROPE is compared to various alternatives in terms of mean-squared value error and various proxy measures aimed at analyzing stability in more depth. The empirical results verify that KROPE representations indeed provide stability benefits for OPE.

### Strengths
1. I found the paper to be very-well written and a breeze to read.
2. The paper is clear about what the scientific questions being investigated are (L74), what the contributions are and where the key novelties lie. For example, I appreciated the forthcoming note in L215 that the properties discussed in Sec. 3.1 are largely due to prior work, but Sections 3.2, 3.3, and 4 provide novel insights.
3. Given the clear scope of study (stability benefits of the bisimulation-like KROPE kernel in OPE) -- although somewhat narrow -- the theoretical results seem sound and provide clear answers (Theorems 1 and 2).
4. Prior work is discussed clearly and credited appropriately to my knowledge.
5. Experimental setup seems sound, providing additional insights and confirming theoretical results.
6. The empirical results show benefit of the proposed KROPE kernel for OPE, which may be of practical interest for a broader audience.

### Weaknesses
I found the discussion and the empirical study around generalization to be somewhat limited. While stability and realizability are important and seem to be at the center of this paper, bisimulation-based representations have attracted a lot of interest from the community due to their generalization capabilities (e.g., via invariance to distractors in the environment). Pearson correlation between value difference and orthogonality (Fig. 2c) is a useful proxy, but I would have expected to see more direct and thorough experiments in settings where bisimulation-like algorithms shine, e.g., value error over "The distracting control suite" by Stone et al. (2021).

### Questions
A few questions, suggestions & typos:

1. L121: is the number "of" state-actions in the dataset...
2. L156: clarifying what "native features of the MDP" mean may improve readibility
3. Proposition 1: Including $\mathcal{D}$ in the expectations (like in Eq. 1) would help the reader make the connection to how stability of LSPE depends on the distribution shift between $\mathcal{D}$ and $\pi_e$.
4. L373-374: I'm not sure about the higher chance of instability in high dimensions due to the covariance matrix. This seems to require some assumptions on the distribution of $\phi$, which I am not sure if these representations necessarily satisfy. Hence, the sentence sounds speculative to me.
5. L367-374: To what extent is the degradation in stability with increasing dimensionality related to (4) being better satisfied? The difference between LHS and RHS of (4) in practice seems like an important proxy that could be studied in the Appendix.
6. Fig. 3: Legend placement is awkward. 
7. L486:  As "expected"
8. L487: Reads like BCRL-EXP-NA achieves lower condition number than KROPE, which doesn't seem right to me (Fig. 4b).

### Soundness
4

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
The paper introduces Kernel Representations for Offline Policy Evaluation (KROPE), a novel algorithm aimed at stabilizing offline value function learning in reinforcement learning. By utilizing π-bisimulation to shape state-action representations, KROPE ensures that similar state-action pairs are represented similarly, enhancing convergence and reliability. The authors establish theoretical foundations demonstrating KROPE's stability through non-expansiveness and Bellman completeness. Empirical results show that KROPE outperforms non-bisimulation baselines in terms of stability and accuracy.

### Strengths
1. This paper rigorously proves theoretically that the KROPE's representations stabilize least-squares policy evaluation and KROPE's representations are Bellman complete.

2. The authors have shown that the KROPE method can learn offline value functions with both stability and accuracy through a series of experiments.

### Weaknesses
The experimental part of this paper does not explain how $\pi_e$ is obtained, which will result in the inability to complete MSVE calculations.

### Questions
1. How can we obtain $\pi_e$ which is needed for calculation in KROPE?

2. In Garnet MDPs domain and 4 DM Control environments, how can we conduct offline value function learning while we don't have any offline datasets?

3. We are interested in understanding the specific implementation of the representation of the (s, a) pair. Specifically, we are wondering whether states and actions should be concatenated before being inputted into the network for representation, or if they should be inputted into separate networks for representation and then concatenated later. Furthermore, we wonder if the implementation will vary depending on whether the states are image-based or physics-based.

### Soundness
3

### Presentation
2

### Contribution
3
