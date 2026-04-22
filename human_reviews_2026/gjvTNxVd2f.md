# Counterfactual Structural Causal Bandits

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Causal reasoning lies at the heart of robust and generalizable decision-making, and the *Pearl Causal Hierarchy* provides a formal language for distinguishing between observational ($\mathcal{L}_1$), interventional ($\mathcal{L}_2$), and counterfactual ($\mathcal{L}_3$) levels of reasoning. Existing bandit algorithms that leverage causal knowledge have primarily operated within the $\mathcal{L}_1$ and $\mathcal{L}_2$ regimes, treating each realizable and physical intervention as a distinct arm. That is, they have largely excluded counterfactual quantities due to their perceived inaccessibility. In this paper, we introduce a *counterfactual structural causal bandit* (ctf-SCB) framework which expands the agent's feasible action space beyond conventional observational and interventional arms to include a class of realizable counterfactual actions. Our framework offers a principled extension of structural causal bandits and paves the way for integrating counterfactual reasoning into sequential decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends the structural causal bandit framework to a counterfactual scenario, where interventions are mixed counterfactuals. In this setup, the action space is defined to satisfy ancestral consistency. Leveraging the existing result on the possibly-optimal minimal intervention set (POMIS), the paper developed a method to search for the POMISs in this counterfactual setup.

### Strengths
- The problem formulation is novel and relevant to the field.
- I skimmed through the theoretical results and found them sound.
- The experimental section effectively demonstrates the merits of the algorithm. In particular, it is helpful to see comparisons when the optimal action lies in $\mathcal{L}_{\leq 2}$.

### Weaknesses
- The challenge of the problem is unclear to me, as the method for finding POMISs is already available.

### Questions
- Could you explain the challenge in algorithm design? By checking Figure 7, Algorithm 1 seems to be an application of Lee and Bareinboim (2018) (algorithm to find POMISs).

- The counterfactual framework in the paper is somewhat confusing. Standard counterfactual inference typically requires observed data to constrain the exogenous variables, and then uses these constraints to reason about what would happen under a hypothetical intervention. Simply replacing the hard interventions in Lee and Bareinboim (2018) with counterfactual distributions does not appear to constitute a substantial contribution. Moreover, the mixed counterfactuals considered here could, in principle, be handled by embedding 
$W$ into a multi-world SCM.

- In Page 2 "(e.g. $X_{1,[w_1]}$)", the  X should not be boldface.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces a variant of the causal bandit setting in which the agent has more power: they can also perform certain "counterfactual" interventions in which a variable $X$ is set to value $x$ as seen by one child, but to $x'$ as seen by another.

### Strengths
The topic is theoretically interesting. I have the impression that the theory is sound.

### Weaknesses
- I am not convinced of the significance of this contribution. My impression is that counterfactual actions as used here are only possible in practice under special circumstances. See also my first question below.
- The paper is very dense in technical material. Additionally, it builds closely on very recent work; familiarity with that work is necessary to build intuition about the present work. This makes the paper hard to review in a reasonable amount of time.

### Questions
- To what extent can counterfactual actions be modelled by defining a new graph which explicitly adds counterfactual mediators, and performing ordinary interventions on it?
- line 78-79 (3rd contribution): what does it mean that suboptimal interventions are "clearly" removed?
- line 100/101: I initially didn't understand what you meant by "when the variables are indexed". Now I think you mean: when the main variable already has a subscript, the counterfactual subscript is put between brackets for visual distinction. Could you confirm?
- In Proposition 1, what does it mean for a counterfactual to "consist of" an action space?

##### Comments
- The limitations section is in the supplement and is not referenced from the main paper.
- line 105: "correlated" should be "dependent" (only the same for Gaussians)

##### Textual
- line 31: "were" -> "was" ("were" is subjunctive mood, but this is factual)
- line 125: "behave**s**"
- Definition 3: "no ~~an~~other"
- line 282: "are cannot be"
- several places: "interventional bo~~a~~rder"

### Soundness
3

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
The paper extends the structural causal bandit framework by introducing **Counterfactual Structural Causal Bandits (CTF-SCB)**, wherein actions corresponds to realizable counterfactual regimes. It defines minimal counterfactual action sets (CTF-MIS), and further refines them by identifying those that are *possibly-optimal* (CTF-POMIS). Building on this, the authors present an enumeration algorithm with complexity $\mathcal{O}(n^2\cdot 2^{|E|})$ that systematically constructs a representative CTF-POMIS set suitable for standard bandit solvers . They prove that restricting exploration to this set preserves optimality and can reduce regret. Across synthetic tasks using Thompson Sampling and KL-UCB, the method consistently achieves lower cumulative regret than baselines that explore either larger counterfactual spaces (CTF-MIS) or purely lower-level action spaces (POMIS). In Markovian graphs, the procedure collapses to intervening on the parents of the outcome node, yielding no additional benefit from counterfactuals.

### Strengths
- Addresses an interesting problem and introduces a novel framework for realizable counterfactual interventions within causal bandits.
- Provides a clear and coherent motivation, positioning the extension of counterfactual reasoning to bandit settings as a natural and meaningful conceptual advance.
- Offers a potentially valuable theoretical foundation for subsequent research that may benefit from richer intervention classes in sequential decision-making.

### Weaknesses
- The exposition presupposes substantial familiarity with the CTF-calculus (Correa & Bareinboim, 2025) and related work (e.g., Correa et al., 2021). Consequently, several statements would benefit from further elaboration.
- The paper is quite dense, and the notation is not intuitive, which makes it hard to read.
- Although the substantial improvement over the super-exponential naive verification, the proposed algorithm remains exponential in the number of edges, raising questions about applicability.
- At present, the manuscript does not address finite-time regret guarantees; a short discussion would be beneficial.

### Questions
1. Could you please clarify the difference between $\boldsymbol{Pa}_V$ and $\boldsymbol{pa}_V$? If you are using the convention: capital letter -> variable and lowercase -> realization, where does the randomness come from in the $Pa$ operator, given a variable $V$?

2. What does $X_{\boldsymbol{w}}$ (line 102-103) refer to? You haven't defined before a rv $X$ with subscript $\boldsymbol{w}$

3. Could you please clarify how $\mathbb{E} N_T(\boldsymbol{X}_*)$ arises in eq. 1? Is an expectation missing?

4. Does $An(Y_x, X_*)$ mean $An(\\{Y_x, X_*\\})=An(Y_x) \cup An(X_*)$? Could you please clarify it in the paper?

5. It would be interesting if the authors can formally discuss (or better derive) finite-time regret guarantees that quantify the benefit of pruning to the representative CTF-POMIS set.


Suggestions:
- move the sentence "We use kinship notation for variable relationships..." (line 105) above mentioning Pa (line 94-95). Also the font used is different.
- end of line 100-101: $\boldsymbol{X}_{1[\boldsymbol{w}_1]}$ should not be bold.
- typo line 282: "... are cannot..".

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Summary
- SCB but also taking into account counterfactuals from L3
- Logical and crucial extension of theoretical base
- well executed

My review is short since I am familiar with the previous SCB work and can see this is an obvious and clear next step of extensions. It is written and present very clear, thorough, yet accessible, with the right amount of empirical experimental evidence.

### Strengths
- Very thorough, clear and precise presentation of
    - CTF MIS, CTF POMIS 
    - Their algorithms  
    - Their theorems

### Weaknesses
The paper is very well written and makes a clear and strong contribution, hence I only have one point on writing style:

- This reads a bit clunky with a rather long subclause” When X⋆∗ lies in L≤2 (i.e., ∆L≤2 = 0)—a special case and does not undermine our theoretical results, since the deployed agent can never be certain prior to interaction whether 471 the optimal arm lies in in L≤2—the smaller action space allows POMIS to converge faster than the 472 others.”
    - Maybe rephrase as: “a special scale that does not undermine”

### Questions
- In Fig 8, Task 3, left: is POMIS about to cross over CTFMIS TS at 100k trials?

### Soundness
4

### Presentation
4

### Contribution
4
