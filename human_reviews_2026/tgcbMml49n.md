# List Replicable Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 6, 6, 2

## Abstract
Replicability is a fundamental challenge in reinforcement learning (RL), as RL algorithms are empirically observed to be unstable and sensitive to variations in training conditions. To formally address this issue, we study \emph{list replicability} in the Probably Approximately Correct (PAC) RL framework, where an algorithm must return a near-optimal policy that lies in a \emph{small list} of policies across different runs, with high probability. The size of this list defines the \emph{list complexity}. We introduce both weak and strong forms of list replicability: the weak form ensures that the final learned policy belongs to a small list, while the strong form further requires that the entire sequence of executed policies remains constrained. These objectives are challenging, as existing RL algorithms exhibit exponential list complexity due to their instability. Our main theoretical contribution is a provably efficient tabular RL algorithm that guarantees list replicability by ensuring the list complexity remains polynomial in the number of states, actions, and the horizon length. We further extend our techniques to achieve strong list replicability, bounding the number of possible policy execution traces polynomially with high probability. Our theoretical result is made possible by key innovations including (i) a novel planning strategy that selects actions based on lexicographic order among near-optimal choices within a randomly chosen tolerance threshold, and (ii) a mechanism for testing state reachability in stochastic environments while preserving replicability. Finally, we demonstrate that our theoretical investigation sheds light on resolving the \emph{instability} issue of RL algorithms used in practice.  In particular, we show that empirically, our new planning strategy can be incorporated into practical RL frameworks to enhance their stability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper focuses on replicability in the context of RL, with the ultimate goal of designing provably efficient PAC RL algorithms that not only converge to an $\epsilon$-optimal policy with high probability after a polynomial number of samples, but also that are guaranteed to always output a policy belonging to a given list $\Pi(M)$ with polynomial size (weak replicability), or additionally that any policy played by the algorithm during the learning process also belongs to a given list (strong replicability).

After having described the high-level idea and intuition in Section 4, the paper presents a robust planning algorithm, that takes in input an estimated MDP, and computes a near-optimal policy based on some input parameter $r_{\text{action}}$ that trades-off accuracy and replicability. Next, the authors present a strongly replicable RL algorithm. Finally, some numerical simulations are conducted.

### Strengths
- The paper provides both theoretical guarantees and numerical simulations to support their claims.

### Weaknesses
In summary: it is not clear why this notion of replicability is desirable. Moreover, the theoretical analysis is quite poor (too large complexities) and the numerical simulations are too limited (mostly continuous problems) and provide very weird results (the robust planner improves performance?). Lastly, the presentation of the paper is quite poor.

- main contributions: it is not clear why this notion of k-list replicability in RL is desirable. Even though it was desirable, it is not clear why the algorithms presented by the authors are meaningful to achieve it, as they have prohibitive sample complexities (e.g., Algorithm 2 requires $H^{24} S^{11}$ samples !!!) and also quite prohibitive list complexities (e.g., Algorithm 2 requires $S^3$, which limits to very small problems). I recognize the value of these results as "binary" to tell us that polynomial instead of exponential complexity can be achieved, but the rates are too large.
- other limitations: the approach can work only in problems with a finite number of actions (because the main idea is simply to take smallest lexicographically)
- numerical simulations: all simulations (except for mountain car) concern continuous state spaces and not tabular MDPs which is the focus of the paper and the setting where the algorithms work. Moreover, except for cartpole, the claim of the authors at lines 460-461 that their approach makes the RL algorithm more stable is not clear by the graphs. In addition, it is quite strange and surprising that using the robust planner, which by definition is robust and suboptimal (!), actually improves performance for some unclear reason (see Fig. 2).
- presentation: section 4 is too dense and confusing; there is no formal presentation of the results (e.g., the notion of sample complexity in corollary 5.3 is not formalized); some results are reported only in the appendix (e.g., Theorem F.3); the paper redefines some standard symbols (e.g., advantage function in Eq. 1) and re-prove some standard results (e.g., Lemmas E.1,E.2) of RL theory (e.g., see "Reinforcement Learning: Theory and Algorithms"); there are many typos (e.g., bad references like "National Academies of Sciences... 2019" line 033); the presentation of related works through a single giant list of papers is quite useless.

### Questions
Please, address all the weaknesses above and questions below:
- Why should we care about the notion of replicability that you consider in the context of RL?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work studies a PAC version of tabular RL under the notion of list-replicability. It provides results in generative as well as episodic exploration settings using weak-as well as strong list-replicability. The manuscript ends with deep RL experiments using updates updated from the theoretical algorithms.

### Strengths
**Strengths**
**Motivation**
* The problem of studying architectural components that improve deep reinforcement learning is important and well motivated.

**Related Work**
* The related work section on tabular reinforcement learning in both the generative and episodic exploration settings seem largely complete.

**Novelty**
* The paper provides 3 novel results in the area of replicable RL. I am not aware of any results on list-replicable RL.

**Theoretical results**
* While I did not check all the proofs in detail (see in weaknesses) I believe on a high level that the algorithms are correct.

### Weaknesses
**Clarity**
* One key issue that I have with the manuscript is that the structure is very difficult to follow. The text often jumps from algorithm to algorithm rather than finishing one of them completely before moving on. The manuscript follows a traditionally more theoretical standard in which an intuitive explanation of the results is given in the introduction which is then usually later followed by formal statements. In this manuscript however, there is first an intuitive section then a slightly less intuitive section and then (only for some results) a formal part. This leads to repetition and makes it hard to follow which part is following from which. 
    * The formal section should establish all definitions and theorem statements that are the main contributions formally. I believe section 3 would benefit from the usage of definitions rather than paragraphs only. Section 4 only points to the informal statements of the main results. Section 5 then proves things required for the results in section 4. This structure is confusing to me.
    * Section 4 describes the three main results but then only 1 of them gets a formal section in section 6. I think it would be significantly easier to follow the manuscript if each result had its own subsection. One way to do that would be to describe the result intuitively, then give the algorithm and formalize the result. One could also simply talk about the formal and omit the intuitive explanation from section 1. There are a lot of possibilities that would make the text easier to read.
* The notation and description in the example at the beginning of section 4 is confusing. After reading it multiple times I believe I understand it now. I think a simple Figure would help alleviate this confusion since the example in hindsight is not all too complicated.
* I tried to look at the proofs for correctness but the Appendix is largely a conglomerate of Lemmata and Theorem statements without guidance for the reader. This is incredibly difficult to read because I’m not even sure where to start.

**Related Work**
* While the treatment of the RL related work is quite nice, replicability falls a little short in the related work section. There is a large chunk of work in the area and to demonstrate that the presented work is in fact novel, citing relevant replicability work may be useful.

**Empirical Claims and Evidence**
* The experiments are conducted with old algorithms that are known to have issues when it comes to function fitting, raising concerns about spurious effects that are unrelated to the stability from statistical noise.
* The experiment section claims that with the proposed approach “the performance of the algorithm becomes more stable at the cost of worse accuracy.” It is not defined what that means. The work sets out to study list-replicability but at no point is there any evidence that the policies that are being learned are similar. If the metric of stability that is meant is variance, then across the first three experiments the results are effectively statistically insignificant. I cannot make out any difference at all in Acrobot or MountainCar between the different r values.
* The final experiment directly contradicts the theory. The theory argues that list-replicability is more expensive in terms of sample complexity. It does not tell us anything about improved return. It is unclear what this experiment is showing with respect to the stability that the paper is studying.

In summary, I believe that the clarity changes can easily be made and should not hinder acceptance. The big issue is the empirical section. As of right now, the empirical section is disconnected from the theory and the claims made in it are not supported by sufficient evidence. There are two solutions in my mind. First, the empirical section can simply be removed which would get rid of claims that are not substantiated. The theoretical results are likely of interest to the PAC/replicability community regardless. The second option is to strengthen the experiments to validate the claims that are made via additional measurements and by relating the results to the theory.

### Questions
None

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses replicable RL in the MDP setup, where the output policy (and/or the trace) from the algorithm lies within a small list of policies across different runs. This paper proposes two algorithms with weak and strong $(k,\delta)$ replicability, which output a near-optimal policy with polynomial sample complexity. Numerical results on various game environments show that the proposed planning algorithm helps stabilize the learning process.

### Strengths
(+) This paper extends replicable RL to the MDP setup. The analysis and results are technically solid.

### Weaknesses
(-) The numerical results cannot reflect the replicability of the algorithm.

### Questions
Typo: In Line 158, should it be "at least 1-$\rho$" instead?

Q1. Have you thought about providing numerical results to demonstrate the replicability of the proposed algorithm?

Q2. Does Theorem 1.3 imply that there is a gap of $O(HS)$ between Algorithm 3 and the lower bound?

Q3. Are there any trade-offs between the list complexity and the sample complexity? In the work of [1], I can see that a smaller $k$ tends to associate with a higher regret.

[1] Chen et al. Regret-Optimal List Replicable Bandit Learning: Matching Upper and Lower Bounds. ICLR 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the notion of list replicability in the PAC reinforcement learning (RL) framework to formally address instability across training runs. The authors define weak and strong forms of list replicability, requiring that the learned policy—or the entire policy execution trace—lies within a small list of possible outcomes. The main contributions include new tabular RL algorithms, one which is weakly list replicable and one which is strongly list replicable, that achieve polynomial list and sample complexities, in contrast to the exponential instability of existing methods. A complementary lower bound for list complexity is also developed. Empirical results show that incorporating the proposed planner into standard RL frameworks improves stability while maintaining good performance.

### Strengths
- **Quality:** The paper’s main claims are supported by rigorous and well-structured proofs, demonstrating a solid theoretical foundation. Additionally, the empirical results, while limited in scope, align well with the theoretical claims and demonstrate practical relevance.
- **Clarity:** The paper's central contribution—introducing list replicability as a performance criterion in RL alongside efficient algorithms—is both well-motivated and clearly presented. The exposition is supported by intuitive explanations, and informal versions of the main theorems are introduced early in the text to guide reader understanding.
- **Significance:** This work addresses an important issue in RL—replicability—by proposing a formal framework and provably efficient solutions. Notably, the framework extends beyond the theoretical tabular setting, demonstrating practical utility when integrated into empirical RL algorithms.
- **Originality:** The paper introduces list replicability in the context of PAC reinforcement learning, extending ideas from replicable learning and multi-armed bandits to the more complex RL setting. The authors are the first to demonstrate that even strong list replicability—constraining the entire policy execution trace—can be achieved with polynomial complexity in tabular RL.

### Weaknesses
Rather than separating into broad quality, clarity, significance and originality categories, I will outline my main concerns in a more detailed manner below.
- Theorem 1.3 establishes a lower bound on list complexity of $\Omega(SAH)$ for weakly list-replicable RL algorithms, which is notably lower than the upper bounds achieved by Algorithms 2 and 3. This discrepancy raises the question of whether the proposed algorithms admit non-tight bounds that could be further improved. A discussion addressing the gap between upper and lower bounds would strengthen the paper by clarifying whether the current techniques are potentially suboptimal or if the bounds reflect inherent limitations.
- Algorithm 3 exhibits a higher list complexity of $O(S^3 A H^3)$ compared to Algorithm 2's $O(S^2 A H^2)$, despite offering stronger guarantees. A brief discussion clarifying this trade-off—whether it reflects inherent algorithmic requirements or potential room for tightening the analysis—would enhance the paper’s clarity.
- In Figure 1, the authors claim that increased stability comes at the cost of reduced performance. However, this trade-off is not clearly visible in subfigures 1(b) and 1(c), where the degradation in performance is minimal or absent. Conversely, Figure 2 appears to show both improved stability and increased performance when using the robust planner. These observations create ambiguity: it is unclear whether the proposed method generally entails a trade-off between performance and stability, or if it can yield gains on both fronts. A discussion clarifying this point would significantly strengthen the empirical section.
- Although Algorithms 2 and 3 achieve list replicability with polynomial list complexity, their sample complexity scales with $1/\delta$, in contrast to the more favorable $1/\log \delta$ dependence of standard PAC RL algorithms [1]. This increased sample requirement may help explain the reduced performance observed in Figure 1 when the robust planner is applied, suggesting a potential trade-off between replicability guarantees and sample efficiency.
- As the authors note, the theoretical analysis is restricted to the tabular PAC RL setting, which limits direct applicability to large-scale or real-world problems. However, this is not a major concern, given that list replicability is a newly proposed performance criterion. Moreover, the paper demonstrates that key components of the framework can be effectively adapted to practical RL algorithms, supporting its relevance beyond the theoretical setting.
- The authors assume known and deterministic rewards, and briefly mention that their methods can be extended to handle more general cases. However, the paper does not provide further details on how such extensions would be implemented or how they might affect the theoretical guarantees.
- While the paper presents its results within the PAC RL framework and frequently refers to sample complexity, it does not formally define either term. Including precise definitions for PAC RL and sample complexity early in the paper would enhance clarity and make the work more accessible to readers who are less familiar with these concepts.
- Additionally, the paper would benefit from more thorough proofreading. For instance, in line 69 ‘work’ should be replaced with ‘works’. In line 158, $\delta$ should be replaced with $\rho$.  In line 249, ‘the’ should be removed.

[1] Dann, Christoph, Tor Lattimore, and Emma Brunskill. "Unifying PAC and regret: Uniform PAC bounds for episodic reinforcement learning." Advances in Neural Information Processing Systems 30 (2017).

### Questions
1. As noted earlier, Figure 1 suggests a trade-off between accuracy and stability, whereas Figure 2 appears to show simultaneous improvements in both when using a robust planner. Can the authors elaborate on why this occurs, and in what scenarios we should expect each phenomenon?
2. Given the gap between the list complexity upper bounds of Algorithms 2 and 3 and the lower bound established by the hardness result, do the authors believe there is room to tighten their analyses or improve the algorithms further?
3. Additionally, is it possible to establish a hardness result for the strong list replicability setting? If not, what are the main technical obstacles preventing such a result?
4. The proof of Theorem 1.3 is said to rely on a reduction to the multi-armed bandit setting, leveraging known lower bounds. Could the authors clarify what technical challenges arise in this reduction, and what their main contributions are in making it work for the RL setting?

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
4

### Summary
This paper gives a black box reduction from tabular RL algorithms to a list replicable algorithm for the same setting. They give algorithms satisfying two different notions of list replicability: one requiring that only the final policy output by the algorithm lies in some instance-dependent list (weak), and one requiring that all policies used for exploration lie in some instance-dependent list of policy traces (strong). Their weakly list replicable algorithm has list size $k = O(|S|^2|A|H^2)$, and requires $|S|H $ calls to the non-replicable algorithm, while the strongly list replicable algorithm has a modest increase in list size to $k = O(|S|^3|A|H^3)$ and has polynomial sample complexity in all relevant parameters (though unfortunately depends polynomially on $1/\delta$ rather than $\log(1/\delta)$. As an additional contribution, they empirically show that their stable approach to planning can be incorporated into existing RL algorithms to improve stability, giving an approach to variance reduction that may be generally useful.

### Strengths
The paper is very well written and the novel techniques are clearly explained. This is to the best of my knowledge the first work to consider the question of list-replicable RL, and so contributes to our theoretical understanding of the feasibility of stable RL. 

The approach to stable planning is a nice contribution in that it is simple enough to be adapted to a variety of algorithms and empirically improves stability, a common problem for empirical RL.

### Weaknesses
My concerns regarding the results are mostly about comparison to prior results in replicable RL. 

First, this paper omits reference to related work [1] that seems algorithmically similar. [1] improves upon prior work on replicable RL by giving more sample efficient replicable algorithms in the tabular setting. Their results also rely on this idea of stably learning a collection of ignorable states, then doing backward induction with data collected from unignorable states to learn a good policy. 

Second, as this paper acknowledges, replicable algorithms imply list replicable algorithms, though with potentially exponential list size, depending on the amount of randomness required to run the replicable algorithm. To understand the contribution of this work, it seems important to discuss whether existing replicable algorithms have randomness usage that directly implies list-replicable algorithms. 

[1] “From Generative to Episodic: Sample-Efficient Replicable Reinforcement Learning” Hopkins, Liu, Ye, Yoshida

### Questions
Please see weaknesses. How do the list sizes in this paper compare to those implicit in prior work on replicable RL for tabular MDPs? How do the algorithmic techniques in this paper compare to those of [HLYY25]?

### Soundness
4

### Presentation
3

### Contribution
2
