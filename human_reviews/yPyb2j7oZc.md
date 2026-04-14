# ReZero: Boosting MCTS-based Algorithms by  Backward-view and Entire-buffer Reanalyze

- Decision: Reject
- Scores: 5, 3, 6, 3

## Abstract
Monte Carlo Tree Search (MCTS)-based algorithms, such as MuZero and its derivatives, have achieved widespread success in various decision-making domains.  These algorithms employ the reanalyze process to enhance sample efficiency from stale data, albeit at the expense of significant wall-clock time consumption. To address this issue, we propose a general approach named ReZero to boost tree search operations for MCTS-based algorithms. Specifically, drawing inspiration from the one-armed bandit model, we reanalyze training 
samples through a backward-view reuse technique which uses the value estimation of a certain child node to save the corresponding sub-tree search time. To further adapt to this design, we periodically reanalyze the entire buffer instead of frequently reanalyzing the mini-batch. The synergy of these two designs can significantly reduce the search cost and meanwhile guarantee or even improve performance, simplifying both data collecting and reanalyzing. Experiments conducted on Atari environments, DMControl suites and board games demonstrate that ReZero substantially improves training speed while maintaining high sample efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This research presents an innovative approach to enhance the efficiency of Monte Carlo Tree Search (MCTS) algorithms, like MuZero, by introducing a method called ReZero. By leveraging a backward-view reuse technique and periodically reanalyzing the entire buffer, the authors aim to reduce search costs while maintaining or even improving performance. This work promises to simplify data collection and reanalysis in various decision-making domains.

### Strengths
This work proposes the backward-view reuse technique, which reutilizes previously collected data to boost search efficiency. Additionally, the entire-buffer reanalyze mechanism represents a creative adaptation of traditional reanalyze processes, allowing for more effective data utilization. This originality is further underscored by the application of these techniques across diverse environments.The paper is well-structured and clearly written.

### Weaknesses
The main weakness of the paper lie in the lack of innovation in the method and the scarcity of baseline comparisons in the experiment. The reuse method has some other approaches in previous MCTS works, but this paper fails to conduct a sufficient disscusion. Additionally, the experiments in the article only compare with the MuZero algorithm. Whether the EfficientZero or Sampled MuZero would show improvements when using this method requires further experimental illustration.

### Questions
Main Questions:
1) Previous studies such as "Information capture and reuse strategies in Monte Carlo Tree Search, with applications to games of hidden information" and "Learning policies from self-play with policy gradients and MCTS value estimates" have also proposed methods similar to reuse in MCTS. What are the differences between the methods in this work and those in the previous studies? This requires detailed discussion.
2) The paper presents some theoretical analyses and proofs, but I fail to understand the specific connection between this proof and the method proposed in the article, and a detailed explanation is needed.
3) How the proposed method performs in terms of improvement on Sampled MuZero and EfficientZero requires further experimental illustration.

Some minor issues:
1) It is recommended that the algorithm not use the pseudo-code of Python in Algorithm 1.
2) Line 235: what does $S^{0}s$ refer to?
3) The font size of the text in Figure 3 is too small.
4) Is it understandable that the experimental effect of Figure 6 is not obvious and there is only a small amount of influence?
5) Why is the decision made to periodically reanalyze the entire buffer instead of frequently reanalyzing small batches of data? How does this approach impact search efficiency?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Algorithms like MuZero extend the use of MCTS to environments without known models. However, its extensive tree search incurs a substantial time overhead. This pressing matter has motivated many research papers to propose mechanisms mitigating this wall-clock time problem. The authors in this paper propose a new algorithm of this sort, called ReZero, which is orthogonal to the previous contributions, thus making it readily deployable with many of the previous works. ReZero leverages a backward-view reanalyse technique that prunes the exploration of certain specific nodes by using previously searched root values. ReZero also reanalyses the whole buffer periodically after a fixed number of training iterations. In tandem with this reanalyse technique, ReZero employs a search strategy that is akin to a one-armed bandit algorithm. The authors then analyse ReZero from a theoretical point of view. In Theorem 1, a bound on the expected number of suboptimal action visits is proposed. Moreover, this bound implies that this expected number of suboptimal action visits is sublinear, i.e. $\lim_{n \rightarrow \infty}\frac{\mathbb{E}\left[ T_i(n)\right]}{n} = 0$. In the Appendix, the authors also claim to provide such bound for AlphaZero. Empirical results seem to indicate that ReZero-M, which is ReZero employed with MuZero with SSL, can significantly decrease the wall-clock time compared to MuZero. Similar empirical results are stated in the Appendix comparing ReZero-E with EfficientZero.

### Strengths
- The subject that the paper addresses is very important. Indeed, improving the sample efficiency of MCTS methods such as MuZero, along with the wall-clock time they incur during tree search could dramatically improve the current MCTS methods.
- Explaining MCTS methods can be a difficult task, thus using illustrations such as Figure 2 helps a lot in explaining the approach.
- There is an attempt at providing finite-time theoretical analysis, in the form of a bound on the expected number of visits to suboptimal actions. Such attempts are common within the Multi-Armed Bandits literature but are less frequent in MDP contexts. It is therefore appreciated that the authors made this theoretical effort.
- The experiments, if they are reproducible with a publicly available code, seem to support the authors claims. However, I am not confident enough to assess the experimental setup accurately.
- The contribution is orthogonal to previously proposed solutions. This implies that ReZero is readily applicable with the other proposed methods from the literature.

### Weaknesses
I see two significant weaknesses with this paper. Lack of clarity and mistakes in the theoretical result.

## Lack of Clarity:

- In Algorithm 1 there are many undefined functions: prepare, origin\_MCTS, select\_root\_child, traverse . I did not find these defined in the Appendix either. It is hard for the reader to understand the algorithmic steps from the provided code. I suggest that the authors rather provide an abstract pseudocode or thorough explanations of each step in an iteration, accompanied by some simple illustration. I struggle to deeply understand ReZero, even now I am unsure that I grasp it sufficiently.
- From Figure 2, the considered MDPs seem to have deterministic transition dynamics. This is also implied by Eq. (5). Is ReZero specific to deterministic transition dynamics? If that is the case, then it should be mentioned while introducing the method. Otherwise, Eq. (5) should account for the stochasticity of transitioning to a certain state through a (state, action) pair.
- Section 4.1 is hard to follow. Does ReZero follow the standard MCTS steps: Selection, Expansion, Simulation and Backpropagation? How is the data collected in the replay buffer? In Section 4.3, a policy network will be briefly mentioned, which implies the use of DeepRL. It makes hard for the reader to follow the explanation with scattered information like this. I only realised that there is a DeepRL approach later on, which confused me. What is the network approximating? Is it just the policy or the value function as well? How are the targets constructed in this case? And how are the values updated? There are too many questions here related to the algorithmic machinery itself that I think the authors should spend time explaining. As a suggestion, I think that the authors should delete section 3.1, which I did not find helpful, and spend more space and effort on clearly stating and explaining the algorithmic steps. Figures like Figure 1 and 2 could also be very helpful here, even if put in an illustrative appendix. Figure 3 to me does not seem standalone, it could be useful but only accompanied by this large section just defining the algorithm fully.
- It is unclear how the mini batches come to existence. Do they stem from filling the replay buffer by following the Tree and Simulation policies (as per the standard MCTS terminology)? If so, how are all the trajectories in Figure 2 of the same length? Is this just for illustrative purposes or are they indeed of the same length?
- Could you provide a more thorough explanation of traverse in Line 240? Is it following the Tree and Simulation policies? When does traverse happen compared to reanalyse? I thought it happened at the end of reanalyse but Algorithm 1 suggests otherwise.
- What if during reanalyse, two actions (or more) $a_1, a_2$ have been taken at state $S_l^t$ (in different trajectories)? Would we have $I_l^t\left( a_1\right) = r_1 + \gamma m_1, I_l^t\left( a_2\right) = r_2 + \gamma m_2$? in which case, the policy in (4) and (5) will not taken into consideration the UCB scores of these actions but rather their estimates directly. If many actions of this sort are explored, then how would exploration occur? I think I am missing a crucial detail about the algorithm here, hence why I suggest that the authors restructure the paper in a way that prioritises much more a clear explanation of ReZero.
- In (5), $I_t\left( a\right)$ should be denoted $I_t^l\left( a\right)$ for consistency of the notation with (4).
- This is not a clarity issue per se, nevertheless it should be mentioned. Regarding the non-stationary bandit in (2), the concentration assumption should be around $\mu_{is} = \mathbb{E}\left[ \overline{X_{is}} \right]$ rather than $\mu_i$ for a meaningful analysis. It is true that the text in the paper "Bandit-based Monte Carlo Planning" is unclear about this. In page 5, the authors just "assume the tail inequalities (3), (4)" and we it is unclear whether they mean $\mu_{is}$ or $\mu_i$. However, in a follow-up journal publication by the same authors "Improved Monte Carlo Methods" where they thoroughly rewrite the previous paper and correct some of its theoretical claims, the authors clarify this misunderstanding in Assumption 1. They indeed consider the concentration to be around $\mu_{is}$. If $\delta_{is} = \mu_{is} - \mu_i = \mathcal{o}\left( \sqrt{\frac{1}{s}}\right)$ the Assumption 1 would indeed imply (2) in the reviewed paper for an appropriate constant $C$. However, I believe that no such assumption was made.

To recapitulate this section. I believe that most of the confusion comes from a lack of investment in clearly explaining the algorithm ReZero. I suggest that the authors take this into consideration when revising the paper. Some of the ways to make this improvement is through illustrations, pseudocode and accompanying text. Some sections like 3.1 could be deleted in favour of this restructuring.

## Theoretical mistakes:

I love the fact that the authors committed time to theoretically analyse their algorithm, I encourage such endeavour. I would like to draw the attention of the authors to a number of mistakes in their proofs that unfortunately end up not supporting their claims. At the end of this section, I will provide references that I hope will be helpful to the authors in rethinking their proof approach in the future.
- First, the statement of Theorem 1 is not rigorous. The terms $P_i$ are nowhere defined. I understood from context and the proof that they are the prior terms $P\left( s, a\right)$ introduced in $3$, nevertheless, they should be properly defined in the theorem statement. The same remark applies to other terms like $\widehat{\mu}_{ln}$. Moreover, and even more importantly, What does it mean precisely to choose actions according to $(5)$? A first understanding would lead the reader to believe that once an arm $i$ is played, its index $I_i^t$ will always be equal to its reward, thus removing the UCB exploration bonus, but it seems to me unlikely that this is what is meant by the authors as it would imply full exploitation algorithm. Please provide the precise statement of how actions are chosen.
- General advice that the authors could incorporate straightforwardly. Please start using \left( , \right) , \Bigg\{ , \Bigg\} ... for clearer mathematical notation.

### Lemma 1:
- Minor: Line 706, put a space between "Let" and "$\widehat{\mu}_t$".
- Minor: I think the term $\exp\left( \frac{\sqrt{a}\epsilon}{C^2}\right)$ should rather be $\exp\left( 2\frac{\sqrt{a}\epsilon}{C^2}\right)$ in (13). This is a minor detail as it does not change the result, but could you please verify this?

### Theorem 1:
- $\widehat{\mu_{is}}$ is the average of the first $s$ samples of arm $i$, thus I think that the notation $\widehat{\mu}_{ln}$, in the theorem statement, is inadequate as it implies that arm $l$ has been chosen $n$ times. It should rather be replaced by $\widehat{\mu}_l\left( n\right)$.
- $T_i\left( n\right) = \sum_{t=1}^n \mathbb{I}\\{ A_t = i \\}$, thus $A_t$ denotes the chosen arm at time $t$. This definition should be stated during the proof.
-  I believe there is a mistake in (24). For the passage from (23) to (24), the authors employ $\sum_{t=2}^{n}\exp\left( -a\sqrt{t-1}\right) \le \int_{t=1}^\infty \exp\left( -a\sqrt{t-1}\right)dt$ but they do not make the calculation explicitly. I do it below via change of variable $u = \sqrt{t-1}$:
\begin{align*}
\int_{t=1}^\infty \exp\left( -a\sqrt{t-1}\right)dt &= \int_{u=0}^\infty 2u\exp\left( -au\right)du\\\\
&= \left[ -\frac{2u}{a}\exp\left( -au\right)\right]_{0}^\infty + \int_0^\infty \frac{2}{a}\exp\left( -au\right)du\\\\
&= \frac{2}{a}\left[ -\frac{1}{a}\exp\left( -au\right)\right]_0^\infty = \frac{2}{a^2}\\\\
\implies \sum_2^n\exp\left( -\frac{1}{C^2}\epsilon P_1 \sqrt\{t-1}\right) &\le \frac{2C^4}{\epsilon^2 P_1^2}
\end{align*}
This in turn implies that the term in (24) should be replaced with $1 + \frac{2C^6}{\epsilon^4 P_1^2}$.
- The passage from (26) to (27) is wrong, and this is what will unfortunately go against your claim. I invite the authors to discuss this important specific point with me during the rebuttal. There is an omitted sum here. When upper bounding the term in (26), you should proceed as follows:
$$
\mathbb{E}\left[ \sum_{t=1}^n\sum_{s=1}^{t-1} \mathbb{I}\Bigg\\{ \widehat{\mu}_{is} + P_i\sqrt{\frac{n-1}{\left( 1 + s\right)^2}} \ge \mu_1 - \epsilon\Bigg\\}, T_i\left( t-1\right) = s\right]
$$
Then this term will be further upper bounded by omitting the event $T_i\left( t-1\right) = s$. For a similar illustrative example of this sum, please refer to the paper "Finite-time Analysis of the Multiarmed Bandit Problem", especially the proof of Theorem 1, check (6). Now this double sum will lead to the following:
$$
(25) \le n + \frac{2\left( n-1\right)\sqrt{P_i^2 \left( n-1\right)}}{\Delta_i - \epsilon} + \frac{\left( n-1\right)C^2}{\left( \Delta_i - \epsilon\right)^2}\exp\left( \frac{2\left( \Delta_i -  \epsilon\right)\sqrt{P_i^2 \left( n-1\right)}}{C^2}\right)
$$
Thus leading to:
$$
\mathbb{E}\left[ T_i\left( n\right)\right] \le 1 + \frac{2C^6}{\epsilon^4 P_1^2} + n + \frac{2\left( n-1\right)\sqrt{P_i^2 \left( n-1\right)}}{\Delta_i - \epsilon} + \frac{\left( n-1\right)C^2}{\left( \Delta_i - \epsilon\right)^2}
$$
Now unfortunately the term $n + \frac{2\left( n-1\right)\sqrt{P_i^2 \left( n-1\right)}}{\Delta_i - \epsilon} + \frac{\left( n-1\right)C^2}{\left( \Delta_i - \epsilon\right)^2}$ is not sublinear and as such you can no longer deduce that $\mathbb{E}\left[ T_i\left( n\right)\right]$ is sublinear. In fact that is the reason why UCB, in the Multi-Armed Bandit (MAB) setting, employs a logarithmic term in the index $\widehat{\mu}_i\left( t-1\right) + C\sqrt{\frac{\log t}{1 + T_i\left( t-1\right)}}$ as opposed to a polynomial term. $\log t$ slows down exploration thus leading to the concentration of visits to the optimal arm. With a polynomial term in $t$, this is no longer guaranteed because the index grows quickly leading to too much exploration even for the suboptimal arms. Now the reason AlphaZero employs a polynomial term instead of a logarithmic term is because there is a need for a lot more exploration in MDPs than it is for MABs. You need a lot more samples to estimate the value of a node in an MDP than you need to estimate the value of an arm in a MAB.
- Suggestions to solve this issue. I think there might be a need to change the way you define the index of your UCB. For finite-sample analyses of UCT-like algorithms, please check the following paper "Nonasymptotic Analysis of Monte Carlo Tree Search".

## Experiments:

I do not know how to accurately assess the experimental setup as I feel like I'm not fully grasping the algorithm itself to pinpoint the merits of its contributions in the experiments. Nevertheless, if ReZero-M is just ReZero applied to MuZero, then it does hint at some substancial speedups. My concerne is with variance. From my understanding of your reanalyse strategy, you can stop searching a node prematurely, and with less samples to estimate its values, the variance of this estimate could be important. I think the authors should spend some space discussing this issue. Figure 7, SeaquestNoFrameskip-v4, Gomoku do hint at this phenomenon, while we see that MuZero's variance is stable in these cases.

## Misspelling:

There are a multitude of misspellings in the paper. Although this is a minor issue, the authors should take the time to revise their paper accordingly.
- Line 308: "in in".
- Line 059: "we aims".
- Line 084: "a efficient"
- Line 127: "a last observation sequences"
- Line 137: "we suggests"
- Line 181: "reward $r_A$", shouldn't it be "return $r_A$"?
- Line 201: Remind the reader that the grey box refers to Figure 1.
- Line 212: "Trajectories was"
- Line 744: "$T_i\left( k\right)$ as the times that" should be "$T_i\left( k\right)$ as the number of times that".
- Line 810: $\widehat{\mu}_n = \frac{1}{n}\sum_1^n \widehat{\mu}_t$, what does this mean? Does it mean that you have $n$ samples from the optimal arm?
- Line 311: "we don't need" should be "we do not need".
- Line 349: "compatible to" should be "compatible with".
- Line 399: "an fair" should be "a fair".
- Line 423: "nexe" should be "next".
- Line 527: "we incorporates" should be "we incorporate".
- Line 535: "could broadening".
- Line 539: "for build".


I believe that the subject and the approach of the paper could be very interesting to investigate rigorously. Unfortunately, mainly due to the lack of clarity and the unsoundness of the theoretical results I have decided to reject the paper. Nevertheless, I invite the authors to take a look at my suggestions during the rebuttal period and maybe we can have an instructive discussion about a revised version of the paper.

### Questions
I have stated numerous questions in context in the weaknesses section. I would appreciate if the authors address them during the rebuttal phase.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes an improvement to the MuZero algorithm. Instead of trying the child node selection as a multi-armed bandit problem, the method treats it as one-armed bandit problem, that is a problem of selecting between one stochastic arm or a sure pay-off with a known value. Since this true value is unknown, of course, authors propose a backward search technique that produces an estimate of it.

### Strengths
- The paper is well-written and authors include easy-to-understand figures
- The authors propose a theoretical justification of their framework
- Presented experimental results clearly show the proposed method is faster (in term of wall-clock time) than MuZero

### Weaknesses
- There only baseline authors compare against is the MuZero algorithm. However, in the related work section authors mention many improvements proposed to the MuZero baseline to make it faster.

### Questions
- Why MuZero is the only baseline authors compare with?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
In MuZero, the "reanalyze" mechanism enhances sample efficiency by revisiting and updating past experiences stored in the replay buffer. In this paper, the authors propose a method to use information from future time steps during the reanalyze phase to reduce the search space and accelerate individual MCTS runs.

### Strengths
ReZero demonstrates a reduction in wall-clock time needed to achieve comparable performance levels compared to baseline MCTS-based algorithms.

### Weaknesses
- The writing is not clear enough for the reader to understand exactly what the algorithm is doing. I had a hard time understanding Section 4.1, and Figure 2, so I am not able to evaluate this paper too well.
- To my best guess, the authors are proposing to use previous MCTS runs to approximate the value of nodes in future MCTS runs. This is based on the 
- I don't think Theorem 1, which seems to be based on existing work, is properly grounded in the current setting. This is because MCTS is not a typical bandit: I recommend the authors take a look at the Shah et al. paper "Non-asymptotic analysis of monte carlo tree search".

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1
