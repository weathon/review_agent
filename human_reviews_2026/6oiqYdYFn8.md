# Better Bounds for the Distributed Experts Problem

- Decision: Accept (Poster)
- Scores: 2, 2, 6, 8

## Abstract
In this paper, we study the distributed experts problem, where $n$ experts are distributed across $s$ servers for $T$ timesteps. The loss of each expert at each time $t$ is the $\ell_p$ norm of the vector that consists of the losses of the expert at each of the $s$ servers at time $t$. The goal is to minimize the regret $R$, i.e., the loss of the distributed protocol compared to the loss of the best expert, amortized over the all $T$ times, while using the minimum amount of communication. We give a protocol that achieves regret roughly $R\gtrsim\frac{1}{\sqrt{T}\cdot\text{poly}\log(nsT)}$, using $\mathcal{O}\left(\frac{n}{R^2}+\frac{s}{R^2}\right)\cdot\max(s^{1-2/p},1)\cdot\text{poly}\log(nsT)$ bits of communication, which improves on previous work.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a regret bound for a distributed online convex optimisation task.   The setup is that each expert sees a loss vector of length s, where s is the number of servers, and performance is evaluated by the Lp norm of this vector and so requires communication amongst the servers.  Regret is wrt the best expert.  The motivation primarily comes from hyperparameter optimisation, where each expert corresponds to a choice of hyperparameters and each server to a different dataset on which an ML model is evaluated.  The novelty lies mainly in the extension to Lp norms with p>1.

### Strengths
The paper is well written, and the analysis appears to be sound (I did not carefully check all of the proofs).

### Weaknesses
I found the HPO motivation given for this work unconvincing.  Communication between servers is not really the main bottleneck - sending a few values takes little bandwidth and can be carried out quickly.  Synchronisation seems more of an issue.   It would have been much better to have presented a proper comparison against state of the art hyperparameter optimisation methods in the empirical evaluation section.  That said, the main contribution is probably the theoretical analysis.  I don't see much interesting new maths techniques being introduced, so the contribution is mainly covering Lp loss with p>1.   But online convex optimisation is a mature area these days, with a large literature, and the present work seems like a small extension of previous work and a relatively minor contribution.

### Questions
See comments re weaknesses above.

### Soundness
4

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
4

### Summary
The paper considers learning with experts but in a distributed setting where there are $s$ servers and each of $n$ experts has a 'partial' loss at each server. The total loss of an expert is the $\ell_p$ norm of the $s$-dimensional loss vector over the $s$ servers. Over a sequence of $T$ days, there is a decision maker communicating with the servers which each day has to play an expert with the goal of minimizing the regret compared to the best expert in hindsight. With unlimited communication between the servers, the decision maker can learn the full loss vector and use a standard regret minimizing algorithm, so the interesting question that the paper considers is tradeoffs between regret and communication. Improving upon previous work, they show that (ignoring polylog factors), there is an  algorithm which achieves (normalized) regret $R$ using $O(n/R^2+s/R^2)\max(s^{1-2/p},1)$ bits of communication.

The techniques used in the paper are based on sampling. Instead of all servers communicating the partial loss of an expert, they scale the losses by ($1/p$ powers of) iid exponential random variables and only send the loss of an expert if it exceeds some threshold. The idea is that the maximum of such scaled variables can be turned into an unbiased estimator for the $\ell_p$-norm and additionally, by the thresholding hopefully only a small number of servers will actually have to send an input. 
\paragraph{Strengths}
I found the paper to be quite well motivated, since the problem feels motivated from both a theoretical and practical perspective. I think that the algorithmic ideas in the paper are nice and appear to be a natural approach for the problem.

### Strengths
I found the paper to be quite well motivated, since the problem feels motivated from both a theoretical and practical perspective. I think that the algorithmic ideas in the paper are nice and appear to be a natural approach for the problem.

### Weaknesses
My main concern about the paper is that there are some important steps in the proof where I have doubts about the correctness of the argument. 

(1) This is probably my most important concern and is about the important lemma 3.3. The proof is quite sketchy. It is stated that a conditional expectation (l755) equality holds for all realizations of $p_t$. Realizations of what? I assume of $p(t)$, but what is $p(t)$? I assume the probability vector of playing each expert at time $t$ because then the equation in l755 seems to make sense. However, on the next page, it is shown that $\textbf{E}[C_t \mid p_t]=\textbf{E}[C'_t \mid p_t]$ and then concluded that  $\textbf{E}[C_t]=\textbf{E}[C'_t]$. The conclusion seems wrong, because the $p_t$ are realizations of different random variables in the first equation, namely those obtained from running the algorithm with $\hat s_i(t)$ as opposed to $L_i(t)$. So it seems that the paper has proved something like 
$\textbf{E}[C_t \mid X=p_t]=\textbf{E}[C'_t \mid Y=p_t]$
for two different variables $X,Y$, but I don't think this implies that $\textbf{E}[C_t]=\textbf{E}[C'_t]$

(2) You claim in line 305 that the probability that the max is not send the the paper $1-1/poly(nT)$, but I don't see where you prove this. I can only find the places where you prove an upper bound on the number of values sent to the coordinator.  It's possible that I just missed it, so in that case, you can just point out to me where you prove it.

These concerns are the reason for my low score. It is however possible that I'm missing something, so if the authors can satisfactorily address the above concerns, I would be very happy to update the score.

### Questions
Please address the two concerns under weaknesses. 

A couple of more minor comments:

Abstract: Do you mean $R\lesssim$?

l105: $[1,5]$ is $[a,b]$ in the theorem

Theorem 2.1: The bound should be $\exp(-\mu\delta^2/3)$

Is the max stability property your own result? 

l327: Shared randomness seems like an unstated model assumption. You should probably discuss this earlier.

Algorithm 3: What does it play with probability $1-\rho$ when it continues to the next time?
\newpage

\section{Nearly Space-Optimal Graph and Hypergraph Sparsification in Insertion-Only Data Streams}

### Soundness
2

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
3

### Summary
This paper studies the distributed experts problem in the message-passing model, where the losses for each expert are distributed across multiple servers. The authors introduce a novel distributed protocol that achieves improved communication-regret trade-offs for general $\ell_p$ losses. Specifically, for any $R \geq 1/\sqrt{T}$, they design an algorithm attaining regret $\mathcal{O}(R s^{1/p} \sqrt{\log n})$
using 
$
\mathcal{O}\left( \frac{n+s}{R^2} \cdot \max(s^{1-2/p}, 1) \, \mathrm{polylog}(nsT) \right)
$
bits of communication. This work extends beyond prior results, which were restricted to $\ell_1$ bounded losses.

### Strengths
-  This is the first work to analyze distributed experts in the coordinator model for general $\ell_p$ losses.
    
- The embedding of $\ell_p$ losses into $\ell_\infty$ through exponential random variables, combined with a geometric mean estimator for variance reduction, represents a technically sophisticated and creative approach.
    
- The presentation of three successive algorithms (Algorithms 2--4), a warm-up, a parameterized version that achieves regret-communication tradeoff, and the final algorithm, which does not require the losses to be bounded, offers a clear conceptual progression
    
- The proofs are rigorous, clearly presented, and appear technically correct.

### Weaknesses
- The authors state that it is information-theoretically impossible to achieve regret smaller than $O(1/\sqrt{T})$, but then compare their method to the algorithm of [1] for $R = O(1)$, claiming improved communication in that regime. This causes confusion as $R = O(1)$ is not achievable. Instead, taking $R \ge 1/\sqrt{T}$ seems to reproduce the communication cost of [1], showing no improvement compared to [1]. The claimed improvement in communication complexity should be either removed or clarified.

-  It is unclear why the $\ell_p$ loss is preferable to the additive $\ell_1$ loss, which naturally decomposes across servers and simplifies communication. The paper would benefit from a stronger motivation for why handling these types of losses is relevant.
    
- The paper lacks formal communication lower bounds, making it unclear whether the proposed protocols are optimal or merely improve upon previous works.

[1]  Zhihao Jia, Qi Pang, Trung Tran, David Woodruff, Zhihao Zhang, and Wenting Zheng. Communication bounds for the distributed experts problem. arXiv preprint 2025

### Questions
- The paper states that it is information-theoretically impossible to achieve regret smaller than $O(1/\sqrt{T})$, yet later compares its communication performance to [1] at $R = O(1)$. Could the authors clarify this apparent inconsistency and explain how their claimed improvement holds for the achievable regime $R \ge 1/\sqrt{T}$?

- What are the practical or theoretical motivations for studying $\ell_p$ losses with $p > 1$ in this setting? Since the $\ell_1$ loss decomposes naturally across servers, it would be helpful to better understand scenarios where $\ell_p$ losses provide clear advantages.

 - The paper does not provide matching lower bounds on communication complexity. Can the authors comment on whether such bounds are known, or how close their algorithms might be to optimal in this regard?

[1]  Zhihao Jia, Qi Pang, Trung Tran, David Woodruff, Zhihao Zhang, and Wenting Zheng. Communication bounds for the distributed experts problem. arXiv preprint 2025

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers the distributed best-expert problem where there are n experts and s servers. Losses are a matrix L in R^{n x s}, where each column sits on a server, and the Lp norm of a row corresponds to an expert's loss. The goal is to design a distributed online protocol that minimizes regret with respect to the best fixed expert in hindsight, and experiences bounded communication costs per-round. 

They make use of the following facts. 1) Taking the maximum across a row of L, where each component is divided by an appropriately scaled exponential random variable, allows one to construct an unbiased estimate of the Lp norm of the row. 2) Since this is a maximum, communication costs can be preserved by omitting components that do not a meet a threshold while still exactly computing (1), so long as all components are not thresholded. Moreover, there is a threshold that ensures a polylog number of samples are sent for each expert, and a vanishingly small probability of failure. 3) By repeating this experiment O(1/p) times, one can reduce the variance of this estimate at the cost of a small amount of bias. 

Putting these facts together allows one to execute the multiplicative weights algorithm on a centralized server that receives n*polylog(...) samples per-step via the procedure above, and an addition O(s) communication to check that all servers are done communicating. 

Next, they parameterize this algorithm to achieve a trade-off between regret and communication in a very trivial way. With some probability, the servers can be made to skip the protocol all-together. By sharing randomness, this brings communication complexity down to zero on those rounds, while maximizing regret, and increasing the variance of the loss estimates. For a target regret bound R, they show this protocol requires (n + s)(polylog(...))/R^2 communication complexity. Finally, they employ one more trick, where the threshold selected in step (2) is allowed to be scaled down according to an appropriately chosen distribution to improve the dependency on s even further. 

Empirical evaluations are given on a hyperparameter tuning task, showing improved communication costs  over Jia et al.

### Strengths
Strength #1: The paper is extremely well written, communicating complicated ideas in an incremental fashion. The main ideas of the theory are presented in a manner that allow the proofs to be checked easily. 

Strength #2: The paper improves upon previous work by a) generalizing to arbitrary p-norms, b) attaining the regret-sensitive rate s/R^2 on the dependency on the number of servers, and c) giving an algorithm that improves the dependence on number of servers by a factor max(s^{1-2/p})

Strength #3: I found the approach in the warm-up algorithm to be very elegant, and did a good job of setting up the more technical details of the later algorithms.

### Weaknesses
Weakness #1: The trick to trade-off communication with regret is somewhat artificial, requiring that all servers are silent on some rounds. While the theory works out, I wonder if there is a more natural algorithm that would yield further improvement.

Weakness #2: This is largely a theory paper exploring communication-regret trade-offs in expert learning. It may be a little outside the primary areas of interest for ICLR.

### Questions
Fig 1 seems like it is underselling the improvement for small p. For p = 1, does this work attain communication complexity, (n /(s R^2)) polylog(...) through theorem 1.3? Or am I mistaken? 

The prose in Section 5 claims that losses need not be bounded, and yet the Lemmas assume l_i(j, t) <= 1. Can the authors explain? 

Do the authors think there might exist a more natural protocol that avoids having all servers go silent on some rounds? While the theory works out, this seems artificial. 

"We remark that the choices of the losses being within the range [1, 5] in the statement of Theorem 1.1 are arbitrary" => I think at some point the manuscript was updated to actually state Theorem 1.1 with arbitrary interval [a,b].

There are minor typos in the appendix involving misaligned hat symbols. See bottom of page 16, top of page 17. Please proof-read and fix.

### Soundness
3

### Presentation
4

### Contribution
3
