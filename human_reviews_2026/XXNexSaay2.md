# Computing Equilibrium beyond Unilateral Deviation

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Most familiar equilibrium concepts, such as Nash and correlated equilibrium, guarantee only that no single player can improve their utility by deviating unilaterally. They offer no guarantees against profitable coordinated deviations by coalitions. Although the literature proposes notions to address multilateral deviations (\emph{e.g.}, strong Nash and coalition-proof equilibrium), these generally fail to exist. In this paper, we study a solution concept that accommodates multi-player deviations and is guaranteed to exist. We prove a fixed-parameter lower bound on the complexity of computing such an equilibrium and present an algorithm that matches this bound.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies an equilibrium concept describing resistance to multilateral deviations, similar to strong NE and coalition-proof equilibrium. This "Minimal Average-Strong Equilibrium" (MASE) is the strategy $\pi$ which minimizes the utility gain any coalition can gain by any deviation, averaged across the coalition's members. Computing this notion is hard (Theorem 5.1), so a fixed parameter lower bound is provided in terms of strategy space size and the tree width of a tree decomposition of the game's Utility Dependency Graph (Theorem 5.3). The paper then goes on to propose and empirically evaluate an algorithm achieving an $\epsilon$-MASE whose running time matches their lower bound (line 416). The main concepts for this algorithm include a correlator–deviator meta-game with FTPL updates and dynamic programming over a tree decomposition.

### Strengths
This paper makes a meaningful contribution to the learning-in-games literature by providing an algorithm that achieves the tight attainable bound for $\epsilon$-MASE, established in Theorem 5.3. The work is original in its framing of MASE as a zero-sum meta-game between correlator and deviator strategies, and in its use of the tree decomposition of the utility dependency graph to express $\sum_{S \in \mathcal{S}}$. This appears to be a novel and potentially generalizable technique. The theoretical analysis is rigorous, and the experimental results effectively support the claims, demonstrating the algorithm's efficacy in computing the LP-based solution. The paper is generally clear and well-organized, though some technical sections could benefit from additional explanation to improve accessibility.

### Weaknesses
While the paper's theoretical contributions are strong, several aspects of the exposition could be clarified to improve readability and accessibility. In particular, certain definitions and intuitions are introduced later than they are used, and some algorithmic details are not fully explained. For instance:
- After the definition of MASE (line 165), it would be helpful to explicitly mention that minimizing the deviation gap guarantees the existence of MASE, as this may not be immediately obvious to some readers.
Theorem 5.3 (line 216) invokes tree decomposition before defining it (line 316); moving the definition earlier and expanding the discussion of SETH and BPP = P beyond brief footnotes, perhaps in an appendix, would improve clarity.
- The intuition behind the tree decomposition procedure in Figure 1 (line 334) could be expanded, particularly regarding how property 3 ensures no contradictions arise in Equation (6.5).
- The discussion of local assignments of agents to bags (lines 310–312) could be elaborated.
- The FTPL update process (lines 272, 292, 406) could be clarified—are updates alternating, or is there a more complex schedule? Including a concise description of the full procedure for attaining the convergence bound in Theorem 6.4 would be useful.
- In Section 6.2 (lines 301–307), it would be helpful to explain how Equations (6.4) and (6.5) contribute to minimizing $F(\tilde{\pi}, \mu)$
Overall, these issues primarily concern clarity and presentation rather than the soundness of the work, and addressing them would make the paper significantly easier to follow for a broader audience.


Minor:
- Line 230: doesn't this paper only consider deviations of singleton coalitions? (following line 129 and Thm 5.1)
- Line 237: should size be $|A|^n$?
- Line 292: it might be worth saying up front that Section 6.2 is for $\pi$, while the update for $\mu$ is in Appendix F
- Typo line 421 "Social welfare.:"
- Author comments accidentally revealed on lines 712, 760, 
- Accidental red text on lines 900, 971, 987

### Questions
The following points reflect areas where additional clarification could help readers better understand the technical contributions and experimental comparisons:

- (Line 145) How does the complexity of LP (Appendix A) compare to your solution (lines 413--417)? On Line 145 when you say, for non-succinct games, MASE can be computed by LP in $O(\star^N)$, this doesn't clarify the case for succinct games.
- Relatedly, it appears in Figure 2 that MASE converges to LP. For my own understanding, does LP correspond with $\pi^\star$ and $\mu^\star$ in Theorem 6.4? Can you please clarify this connection?
- Is the deviation gap (line 189) for CCE equivalent to $E_{a \sim \pi}[U_i(hat~a_{S}, a_{-S}) - U_i(a)]$ in the definition of MASE (line 167)? For the reader's understanding, it may be worth an additional sentence drawing this connection, somewhere on lines 187--194.
- (Line 191) How does your work differ from Anagnostides et al. (2025)? How does Theorem 5.1 differ from their conclusion about minimizing the average gap of CCE across players?
- Relatedly, how does Anagnostides et al.'s solution compare to your experiments in Figure 2?
- Is your technique for identifying the tree decomposition of the utility dependency graph novel? Or is it inspired by other work?

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
3

### Summary
The paper introduces Minimal Average-Strong Equilibrium (MASE), a solution concept designed to handle coalitional (multilateral) deviations in games. Instead of requiring immunity to every coalition (which often fails to exist), MASE minimizes the maximum average gain any allowed coalition can obtain by deviating from a correlated strategy.

On the theory side, the authors prove NP-hardness even when only single-player deviations are allowed and establish a fixed-parameter lower bound: under SETH, computing (even approximating) MASE must be exponential in the treewidth of a newly defined Utility Dependency Graph.

On the algorithmic side, they show a sparse representation exists for ε-MASE and design a algorithm (up to polynomial factors) that reformulates the problem as a correlator-deviator zero-sum game, solved via no-regret learning (FTPL) combined with dynamic programming over a tree decomposition of the dependency graph. Experiments on classic games (e.g., Prisoner’s Dilemma, Stag Hunt) indicate MASE yields higher robustness to coalition deviations and often better social welfare than standard baselines.

In short, the paper delivers a new coalitional stability notion, a tight complexity characterization tied to treewidth, and a practical algorithm whose complexity matches the theoretical lower bound in its dependence on treewidth.

### Strengths
The paper introduces a clear and well-motivated solution concept for coalitional deviations, MASE, which fills the gap left by nonexistence of strong equilibria. It gives tight complexity lower bounds linked to treewidth and then matches them with an algorithm using FTPL plus tree decomposition. The sparse representation result is elegant and enables practical computation. The Utility Dependency Graph provides a clean structural lens that connects theory and algorithms. Experiments show improved coalition robustness and often better social welfare compared to standard baselines.

### Weaknesses
About model: using the average gain of coalition members as the metric can mask heterogeneity within the coalition. Large gains for some members can offset small losses for others, which may understate the true threat of coordinated deviations. If one instead imposed per-member constraints or a weighted minimization, both the conclusions and the algorithmic complexity could change. The paper lacks a systematic comparison and robustness analysis along these dimensions.
About result: Although the results regarding complexity in the paper are mathematically solid, they are not surprising.
About writing: in this paper, the authors mention that one can construct a Utility Dependency Graph from the game G. What implicit relationship exists between the two, and what aspects of that relationship should be clarified in the main text?

### Questions
First, what is the relationship between your Utility Dependency Graph and the original game? Note that I am not asking how to construct the graph from the game; rather, I want to understand the underlying relationship between the constructed graph and the original game.
Second, why do you rely on a tree decomposition of the Utility Dependency Graph as the core algorithmic vehicle? How do bags correspond to interaction structures or utility terms in the original game?
In a word, I find the description of the tree decomposition and the dynamic programming on the tree insufficiently intuitive, and I would like the authors to clarify it.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Minimal Average Strong Equilibrium (MASE), a concept extending classical equilibria to account for coalition deviations by minimizing each coalition’s average incentive to deviate. Computing even an approximate MASE (ϵ-MASE) is NP-hard. 

Using a Utility Dependency Graph to capture players’ interdependencies, the authors establish a fixed-parameter lower bound: under SETH, computing MASE requires time exponential in the graph’s treewidth. 

They propose a Follow-the-Perturbed-Leader (FTPL) learning algorithm via a correlator–deviator meta-game, achieving $O(\sqrt{T})$ regret for both players, with runtime matching the treewidth lower bound. 

Experiments are conducted on bimatrix games.

### Strengths
1. A novel solution concept, Minimal Average-Strong Equilibrium (MASE), is proposed to addresses coalition deviations.
2. Clear theoretical results of computation complexity.
3. A no-regret learning algorithm is proposed.

### Weaknesses
1. The paper appears to overstate its contributions. In the abstract, the authors claim to propose a tractable equilibrium; however, the theoretical results indicate that computing this equilibrium is NP-hard.
2. Although the theoretical results and learning algorithm are designed for multi-player games, the experiments are limited to simple $2\times 2$ bimatrix games.

### Questions
How dose the learning algorithm perform in more general games?

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
4

### Summary
The paper proposes Minimal Average-Strong Equilibrium (MASE), an always well-defined correlated strategy that minimizes the average incentive to deviate among all coalitions. The idea is to find a correlated distribution over joint actions that minimizes the coalitions' average gain from deviating (differently from CE and CCE, which only prevent unilateral deviations).
The paper proves that computing an approximate MASE is NP-hard even when 1-player coalitions only. It then proposes treewidth-based lower bound via the Utility Dependency Graph (UDG), showing an exponential dependancy on treewidth under the Strong Exponential Time Hypothesis (SETH).
For the approximation result, the reduction assumes BPP = P to derandomize sampling. The method computes MASE by recasting it as a two-player zero-sum meta-game between a correlator and a deviator, then running Follow-The-Perturbed-Leader (FTPL) for both sides while implementing each best-response step with dynamic programming (DP) over a tree decomposition of the UDG. The running time scales polynomially in the horizon and number of coalitions, and exponentially in one plus the treewidth of the UDG (as expected, which matches the lower bound up to constants).
Experiments on standard games such as Prisoner’s Dilemma, Stag Hunt, Chicken, and a Pigou network show lower coalition exploitability and higher social welfare than standard online-learning baselines and a linear-programming oracle designed to compute a MASE for small-sized games, while keeping unilateral exploitability comparable.

### Strengths
The idea of MASE is quite nice and it avoids non-existence issues of strong or coalition-proof equilibria. The link to the treewidth decomposition of the UDG is also rather nice. The analysis is sound and the negative dependancy on the UDG's treewidth is quite interesting. The writing is good and the paper's more technical parts are quite understandable. The whole contribution should be relevant to the game-thereotical/multi-agent literature.

### Weaknesses
Experiments are rather limited and consider rather small games. Only matrix and toy congestion games are shown. No larger and more general families such as polymatrix or congestion game instances are considered. Also, no scaling versus the treewidth of the UDG is shown, which is a pity.
Re: the choice of using an average gain: this is reasonable but also possibly arbitrary and not too strongly motivated in my view.
Some reductions rely on the assumption that BPP = P which is rather strong. Its practical implications could be better discussed.
The appendices need a little polish -- they contain some draft notes.

### Questions
Why average across coalition members? Can you comnpare average versus minimum or weighted objectives on at least a small examples?
  
What happens if the coalitions are restricted by size? Can the algorithm and bounds be parameterized by a size bound?

Did you compute the tree decomposition or assume it to be already there? I'd like to see how compute-intensive it is.
   
Please add polymatrix and larger congestion games, and show the runtime as the number of players and, more importantly, treewidth grows.

### Soundness
3

### Presentation
2

### Contribution
3
