# Fundamental Limits of Game-Theoretic LLM Alignment: Smith Consistency and Preference Matching

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Nash Learning from Human Feedback (NLHF) is a game-theoretic framework for aligning large language models (LLMs) with human preferences by modeling learning as a two-player zero-sum game. When the payoff is defined by the true underlying preference, the framework guarantees desirable alignment properties. However, the ground-truth preference matrix is often unavailable in practice due to limited or noisy data, which substantially constrains the effectiveness of this game-theoretic approach to LLM alignment. In this paper, we systematically study what payoff based on the pairwise human preferences can yield desirable alignment properties. 
We establish necessary and sufficient conditions for Condorcet consistency, diversity through mixed strategies, and Smith consistency. 
These results provide a theoretical foundation for the robustness of game-theoretic LLM alignment.
Further, we show the impossibility of preference matching, i.e., no smooth and learnable mappings of pairwise preferences can guarantee a unique Nash equilibrium that matches a target policy, even under standard assumptions like the Bradley-Terry-Luce model. 
This result highlights a fundamental limitation of game-theoretic LLM alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the theoretical foundations of game-theoretic LLM alignment by studying a generalized NLHF framework where preferences $\mathcal{P}$ are transformed by a mapping $\Psi$. The authors characterize necessary and sufficient conditions on $\Psi$ for three desirable properties:

1. **Condorcet consistency**: Selecting the majority-preferred response when one exists
2. **Smith consistency**: Restricting output to the Smith set (top preference cycle)
3. **Preference matching**: Exactly matching a target distribution reflecting preference diversity

The main contributions are:
- Showing Condorcet and Smith consistency are robust to the exact payoff values (only ordinal properties matter)
- Proving that Smith-consistent methods automatically preserve diversity through mixed strategies
- Establishing an impossibility result for exact preference matching under practical constraints

### Strengths
1. **Theoretical Rigor**: The paper provides complete characterizations (necessary and sufficient conditions) for multiple alignment axioms, going beyond previous work that only verified specific cases (e.g., Liu et al., 2025 for $\Psi(t)=t$).

2. **Robustness Results**: Theorem 3.1 and Corollary 3.2 show that Condorcet consistency is preserved under approximation errors, providing theoretical insights for practical implementations with noisy preference models.

3. **Novel Technical Contributions**: The authors develop techniques for non-symmetric games, extending beyond the symmetric game analysis in Liu et al. (2025).

4. **Impossibility Result**: Theorem 5.1 provides a fundamental limitation result, clarifying that game-theoretic methods cannot achieve preference matching.

### Weaknesses
1. **Limited Practical Motivation**: The transformation $\Psi$, while mathematically tractable, lacks clear practical motivation. Practitioners don't typically apply non-linear transformations to collected preference data. Even the $\Psi$PO framework (Azar et al., 2024) that motivated this work focuses on identity mapping (IPO) in practice.

2. **Lack of Empirical Validation**: The paper is entirely theoretical with no experiments—neither numerical simulations on synthetic games nor actual LLM alignment tasks. This makes it difficult to assess:
   - Which preference models learned from real data satisfy which conditions
   - The practical impact of violations of these conditions
   - Whether the impossibility results matter in practice

3. **Questionable Claims about Existing Models**: The claim (above Corollary 4.2) that "several popular preference models do not satisfy" the skew-symmetry condition on $\mathcal{P}$ is problematic. This can be trivially enforced by setting $\mathcal{P}\_\theta (y \succ y') = \frac{1}{2} (M_\theta (y,y') + 1 - M_\theta (y',y))$ when $M_\theta$ is the imperfect (which may be influenced by the response order) preference model. The cited papers may not explicitly do this, but it's a straightforward fix.

4. **Disconnect Between Title and Content**: Most results are about general two-player zero-sum games and don't fundamentally require the LLM alignment context. The connection to LLMs appears mostly in framing rather than in the mathematical substance.

5. **Section 5 Unclear Connection**: The preference matching section (Section 5) studies payoff matrices $\\{\alpha_{i, j}\\}$ under Assumption 5.3, but the connection to the $\Psi(\mathcal{P})$ framework from earlier sections is unclear. How does composing preferences with $\Psi$ relate to matrices satisfying Assumption 5.3? This disconnect undermines the unified narrative.

6. **Restrictive Assumptions**: Assumption 5.3 prohibits dependence on $n$, but this seems overly restrictive. Allowing $\mathsf{poly}(\log n)$ dependence would be more realistic for LLMs ($\log n \approx$ generation length). It is not clear the influence of such relaxation.

7. **Missing Related Work**: The paper omits a large body of relevant NLHF papers including https://arxiv.org/abs/2401.04056, https://arxiv.org/abs/2410.16714, https://arxiv.org/abs/2503.08942, etc.

### Questions
I don't have other questions than those raised in the Weaknesses part.

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
3

### Summary
The paper studies a generalization of the NLHF objective under a mapping $\Psi$, i.e., the game-theoretic optimization of a preference function $\Psi (\mathcal{P})$ where $\mathcal{P}$ is the true preference model. The authors focus on necessary and sufficient conditions for desirable alignment properites. The authors show that Condorcet and Smith consistency hold under intuitive, general conditions. It is then also shown that similary results do not hold for preference matching.

### Strengths
- The paper is well-written and particularly the first four sections coherent, well-explained and well-contextualized within the existing literature. 
- The topic of Condorcet and Smith consistency under a learned preference model and the robustness of the solution concept under estimation errors of the preference model is an interesting and generally important research direction.

### Weaknesses
- Some of the results are incremental. For example, I personally view Theorem 3.1 as folklore (particularly in the dueling bandit community). As long as we map probabilities $\geq 1/2$ again to probabilities $\geq 1/2$ and similarly for probabilities $< 1/2$, then the ranking among candidate actions does not change. Hence, if there exists a Condorcet winner under $\mathcal{P}$ it is still a Condorcet winner under $\Psi(\mathcal{P})$, and the NE is naturally equal to playing that action. That this is true becomes quite obvious when simply using the notation $y \succ_{\mathcal{P}} y'$ if $\mathcal{P}(y \succ y') > 1/2$ which is typically used when talking about the Condorcet winner and we don't care about the preference strength (the $\Psi$ you define leaves $y \succ y'$ unchanged). 
- While "preference matching" is conceptually interesting, I don't see why it is desirable or a reasonable thing to aim for, especially, in the context of LLMs and NLHF. Firstly, it only exists if the preferences follow a BTL model, so we are in the RLHF framework automatically even though you're primiarly motivated by NLHF. Secondly, you motivated your work with the fact that learned preference models can be biased and suffer from large estimation errors. Aiming for preference matching w.r.t. this estimated preference model that appears exactly the opposite of what we are interested in. We are interested in robustness against the errors in the preference model and do not want to propagate / mirror them. 
- This is theoretical work and I do not believe that theoretical work generally needs experiments. Though, partly due to the lack of empirical validation, the practical relevance of the work is limited in my opinion, and the contributions fairly incremental.

### Questions
1. Why would $\Psi$ ever be such that $\Psi(\mathcal{P})$ is not inducing a symmetric game? In practice, we could always enforce the symmetry, couldn't we? In this case, i.e., when the game is symmetric, isn't Theorem 3.2 trivial when there exists no Condorcet winner? 
2. The notation of Axiom 5.1 (Preference Matching) is unusual. I'd like to suggest to you to not use "$a:b$" but stick to \frac or "/". 


Overall, I consider the paper borderline at this stage. I'd be happy to further discuss with the authors and potentially raise my score after the rebuttal.

### Soundness
4

### Presentation
3

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
This paper provides a theoretical investigation of game-theoretical LLM alignment framework, focusing on how the mapping $\Psi$ of the original preference affects the characteristics of the Nash solution. 

The paper identifies the necessary and sufficient conditions on the mapping to guarantee Condorcet consistency and Smith consistency, where the latter extends the former and considers disjoint subsets of the responses. Conditions on $\Psi$ that ensure the Nash solution is a mixed strategy and preserve diversity are also examined.

While the investigations of consistencies provide a foundation for the robustness of the game-theoretic LLM alignment approaches, the paper also discusses the impossibility of achieving exact preference matching under the general game-theoretic alignment framework where the mapping is assumed to be smooth.

### Strengths
1. The paper discusses properties of game-theoretical LLM alignment methods that are highly relevant to many researchers, as well as extending the scope of analysis from the previous works.
2. The paper is well organized, presenting conceptual extensions gradually so that it becomes easier for the readers to follow up (ex. line 220, 254, 313, 396).

### Weaknesses
While I understand that the paper mainly focuses on theoretical analysis of the algorithms and preference mapping $\Psi$, I think the paper could have benefitted from providing visual examples (ex. mapping functions and the derived properties) and experiments for better understanding.

### Questions
## Questions
1. in line 115-117, the authors say "Smith consistency can be ensured by further maintaining the symmetry of the game", and Theorem 4.2 seems to say the same. However, in line 720-723, it seems to say the opposite: "... preference models must satisfy certain anti-symmetry conditions to ensure Smith consistency". To me there seem to be a contradiction. Am I missing something?
2. In line 380-382, the paper mentions Xiao et al. (2025) which showed that a preference matching policy exists under the BTL model. In line 467-471, however, the paper says the generalized game in line 469-470 cannot achieve preference matching. Could you perhaps explain the differences in the settings between these two that caused the difference?

## Typos and Small Suggestions
1. line 381: “such a policy **exist** -> **exists**”
2. line 1063: “Let us **reformulated** -> **reformulate**”
3. line 1064: there seems to be unnecessary space between “min_pi” and “max_j” operators?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper investigates the theoretical aspects of game-theoretic alignment. The authors discuss from three aspects, revealing game-theoretic alignment's superiority and limitations, as well as common property shared with RLHF.

### Strengths
The results of Smith Consistency shared with both RLHF and game-theoretic alignment is interesting, also intuitive.

### Weaknesses
The Presentation is problematic, making the paper less clear. The results might be interesting but the writings without emphasizing the critical results make me hard to follow.
 
- What is continuity assumption specifically in the paper? The authors didn't define it.

- Axiom 4.1 is not a proper axiom. Smith set is not defined. A proper definition is needed for Smith set. Do not assume the readers know it. And I also tend to not use Axiom in writing as these so called axioms are more like definitions.

- The authors should demonstrate more details and analysis on the meanings of the theoretical results. For example, what can theorem 3.2 imply and how does such theoretical results help understand the game-theoretic alignment field? Does that imply learning from noisy preference data or lack of preference data to learn the general preference model in theory wouldn't harm too much? To make some space, not all the theoretical details are needed in the paper, the authors chouls wrap them into a theorem, leaving the insights and analysis in the main text.

- I think the key concern is that it is unclear the practical meaning of these theoretical results. Some writings are also confusing such as "preserving the diversity in human preference strictly, in the sense of preference matching". Do the authors mean game-theoretic alignment cannot preserve diversity? Or just doesn't satisfy preference matching? What's the practical consequence of not satisfying preference mathcing?

### Questions
Here is a line of works [4-5] doesn't require the full payoff matrix for preference optimization. Will this make the proposed issue of "the ground-truth preference matrix is often unavailable" less concerning?


Add missing related works:

[1] Wang, Mingzhi, et al. "Magnetic preference optimization: Achieving last-iterate convergence for language model alignment." arXiv preprint arXiv:2410.16714 (2024).

[2] Zhang, Yuheng, et al. "Improving LLM general preference alignment via optimistic online mirror descent." arXiv preprint arXiv:2502.16852 (2025).

[3] Tang, Xiaohang, et al. "Game-Theoretic Regularized Self-Play Alignment of Large Language Models." arXiv preprint arXiv:2503.00030 (2025).

[4] Zhang, Yuheng, et al. "Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning." arXiv preprint arXiv:2407.00617 (2024).

[5] Rosset, Corby, et al. "Direct nash optimization: Teaching language models to self-improve with general preferences." arXiv preprint arXiv:2404.03715 (2024).

### Soundness
3

### Presentation
2

### Contribution
2
