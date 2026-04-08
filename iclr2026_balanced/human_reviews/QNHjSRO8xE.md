## Human Reviewer 1

### Summary
This paper studies two new simple regret multi-armed bandit policies, PATSO and CATSO,
and MCTS variants that combines them with power mean backup.

CATSO is a dirichlet-based thompson sampling,
and PATSO is a non-parametric dirichlet-based thompson sampling.
Both stores a form of reward histogram from which a new sample is drawn for arm selection.

The paper provides a theoretical analysis on the regret in a reward distribution
which is non-stationary BUT whose time average converges to some value.

Naive PATSO requires an increasing amount of memory, so the author proposes an implementation that
trades memory with accuracy.

The experiments evaluate them on synthetic stochastic tree environments and Atari environments.
Howeve, the experiments are (1) insufficient, leaving many questions unanswered,
and (2) weak, i.e., the new algorithm does not appear to be particularly strong.

### Strengths
Umm, hard to say, but the method description itself was straightforward and easy to understand. Sorry, I am just being honest.

I see the math proofs in the appendix required a lot of work. Nice work! However, please fix the style, writing, and experiments first.

### Weaknesses
First of all, the overall paper appears work-in-progress and is not polished enough for a review.
Please do not submit such a paper!
Several sections require more in-depth explanations while the space is used quite generously (e.g., plenty of space around Fig1, Fig2, Sec3.1).
Polished paper looks compact; You can tell from its appearance, though this is just a speculation.
Meanwhile, there is definitely some hacks or errors going on in the style file (e.g., section header spacing), which,
as other reviewer also mentioned, should automatically flag this paper for rejection.

This paper has a critical flaw in the experimental design and the method design
by violating a key scientific principle:
Changes must be introduced one piece at a time in order to understand what is happening.

Some key issues in the experiments:

-   p in the power mean is a hyperparameter, but the paper does not explain how to choose it, or which value was used in the experiment.

-   Lack of ablation: PATSO/CATSO with the standard arithmetic mean (monte-carlo backup) and the maximum (bellman backup)

-   Lack of ablation: UCT + power mean

-   The key strength of P/CATSO claimed by the author is the adaptability to the stochastic environment.
    Then the natural question is: how does it behave in the deterministic environment?
    Does it still work as good as UCT etc?
    How does it react to a varying degree of stochasticity?
    To answer this question, synthetic environment experiment requires multiple evaluaton with different stochasticity.

-   I marked the top-3 scores per domain in Table 1.
    I get an impression that CATSO/PATSO look roughly only on par with UCT.
    Even in domains where C/PATSO are in bold, the difference sometimes does not appear statistically significant.
    In other domains where UTC is in bold, the difference sometimes appear significant.
    This shares the key question as the previous one: Why do C/PATSO perform bad in deterministic environment?
    Shouldn't they perform at least on par with UTC in every domain?

line 355-377: section 4.3 needs a complete overhaul with more explanations.
What is this formula in line 363?
Is the near-optimal policy same as $\pi$ in line 363?
Shouldnt it be argmax, not max?
Wp is undefined, and not used.

line 402: "Practical choice of K". I assume this is a paragraph header that is incorrectly turned into italics.

Section 5: fixed-Depth-MCTS, MENTS/TENTS, BTS, DNG: 
you must provide a brief explanation of each algorithm, and explain why these baselines are chosen,
because, otherwise,
since they all performed worse than UCT in Table 1, they are basically straw-man algorithms with bad performances
whose sole purpose to be included in the paper is to artificially inflate the number of figures/table rows
to give an impression that many meaningful comparisons have been done.

One last comment:
Watch a presentation ["How to write a great paper" by Simon Payton Jones (Microsoft Research Cambridge)](https://simon.peytonjones.org/great-research-paper/).
In it, he said: "Many papers contain good ideas, but do not distill what they are".
I suspect the power mean in C/PATSO is a classic case of an unnecessary addition that wastes the space without adding meaningful value to this paper, although I didnt check the proof and don't know how it interacts with the stochasticity.

### Questions
The fact that the time average converges to some value sounds like a pretty big assumption.
I think I saw the same assumption in other non-stationary bandit papers, but still, WHY is it justified? Please explain.

Another important question I have is about whether the definitions of C/PATSO's TS policies have anything specific to the stochastic environments.
Here is my thought:

-   Would power means have anything special about stochasticity and how? I **doubt** that this is happening.
-   Would categorical / non-parametric modeling have anything special about stochasticity? **I doubt this one too**.
-   Both are just different estimators with different parameterizations that converge to the same mean.
    In other words, the first term (exploitation term) of the score in line 244 (ii) probably has nothing to do with the stochasticity.

Then the last diff from UCT is the choice of TS and the different bias term from Shah et al.
This leaves a possible interpretation that may kill this paper:
Isn't the robustness property a direct consequence of the optimism bonus term by Shah et al and
has nothing to do with the new addition introduced in this paper?

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper introduces distributional Monte Carlo tree search, which maintains a distribution of returns for each Q-node instead of a point estimate (inspired by distributional reinforcement learning). The benefit of the proposed approach is avoid the under- or overestimation issues of existing approaches, which use point estimates. To store distributions of returns, the paper proposes to approaches, CATSO and PATSO, which represent the distribution either as a categorical one (with a Dirichlet prior) and as a dynamic set of particles (i.e., keeping track of all observed returns, along with their frequencies), respectively. Both approaches are proven to converge with $\sqrt{n}$ regret. The paper also discusses the memory and time complexity of the algorithms, and it introduces a practical variant of PATSO that limits the number of returns tracked by merging some of them. Finally, the paper compares the proposed approach to baseline MCTS variants on synthetic environment and 12 Atari games.

### Strengths
* The idea of incorporating distributional RL (i.e., maintaining distributions of returns) into MCTS is novel and interesting.
* The proposed approaches are theoretically proven to converge, and have the same regret as some SOTA approaches.
* The paper analyzes memory and time complexity, and introduces a memory-efficient variant of PATSO.
* The numerical results are promising for environments with uncertainty.

### Weaknesses
* The paper seems to violate formatting requirements of ICLR (negative \vspace or similar tricks).
* While the results are promising, especially for environments with stochasticity, they are somewhat mixed compared to baseline approaches.
* For V-nodes, the proposed approach still uses point estimates. The paper acknowledges this limitation; it is not crystal clear why this limitation is necessary.
* Line 472 references the Breakout game, but this game is not included in Table 1.
* What is the point of introducing the probability $p_i$ for CATSO? They do not seem to be used anywhere.

Minor suggestions for improving presentation (beyond removing the negative \vspace):
* Add horizontal space between the two equations on line 112 (or move them to separate lines). It was not obvious that these are two separate equations, which was rather confusing.
* The results in Table 1 are not easy to read. Please (1) align the numbers, so that the $\pm$ symbols are above each other (this way, it will be easier to read the numbers as they will be aligned, instead of being shifted left or right depending on the number of digits) and (2) normalize the results (e.g., report them as fraction of the optimal value) so that the different methods are easier to compare.
* Why not present PATSO with memory management (instead of baseline PATSO) as the main contribution? The convergence still holds, and it is more practical.

### Questions
* What is the point of introducing the probability $p_i$ for CATSO?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 3

### Summary
The paper proposes two  Monte-Carlo Tree Search (MCTS) algorithms for stochastic reinforcement learning environments:
 1. CATSO (Categorical Thompson Sampling with Optimistic Bonus)
 2. PATSO (Particle Thompson Sampling with Optimistic Bonus)
Both algorithms integrate distributional reinforcement learning (RL) with Thompson sampling and an optimistic exploration bonus to improve robustness and exploration under uncertainty.  Non-asymptotic simple regret bounds of order (O(n^{-1/2})) for both algorithms, matching known MCTS convergence rates are presented.
 A connection to Wasserstein distributionally robust optimization (WDRO), showing robustness to model uncertainty and deriving sample complexity bounds is established. Empirical results on synthetic tree environments and Atari benchmarks show that CATSO and PATSO outperform classical and entropy-regularized MCTS methods, especially under high stochasticity.

### Strengths
1. The combination of distributional RL and Thompson sampling with optimism in MCTS is well-motivated and addresses a real gap between learning and planning under uncertainty.

2. The paper establishes convergence results with clear proofs and connects them to non-stationary bandit analysis, which strengthens theoretical grounding.

3. The link to Wasserstein DRO provides a principled robustness interpretation, which may appeal to both theoreticians and practitioners.

4. The experiments cover both synthetic and realistic domains (Atari), showing consistent gains and robustness over baselines.

### Weaknesses
1. The plots in Figure 3 are too small and lack readable axes and legends.
    Spacing between subplots is insufficient, making it difficult to distinguish the different experimental settings.
    The figure captions and layout are not up to ICLR standards-figures should be legible when printed at 100% scale.

2. The Atari experiments use only 12 games, and hyperparameter sensitivity (e.g., number of atoms, exploration constants) is not discussed.
    There is no ablation to isolate the contribution of the optimism bonus versus the distributional representation.

3. While the algorithmic details are extensive, the paper occasionally reads more like a technical report than a conference paper—some derivations could be moved to the appendix. The paper could benefit from a clearer high-level intuition for readers unfamiliar with distributional MCTS.

4. PATSO’s flexibility comes at potential computational cost; empirical runtime comparisons are missing.

5. Results on optimality of the convergence rates are not discussed.

### Questions
1. How sensitive are CATSO and PATSO to the polynomial exploration constant (C)? Is there a principled way to choose it?

2.  For CATSO, how does the number of atoms (N) affect accuracy and runtime? For PATSO, what is the empirical impact of the particle cap (K)?

3. How does your approach compare to recent Bayesian or entropy-based MCTS methods when tuned for stochastic domains (e.g., Boltzmann MCTS or risk-sensitive tree search)?

4. Could these methods be extended to deep MCTS or large-scale POMDPs where state abstractions are necessary?

5.  The WDRO connection is elegant-could you illustrate empirically how the learned policies differ in robustness compared to standard MCTS?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper attempts to explicitly model Q nodes in MCTS with the distributional Bellman operator and adapt the UCT action selection to handle a distribution instead of just a mean estimate.

### Strengths
- I find it an interesting idea to include more explicit modelling of the full Q distribution inside MCTS, and how to adapt the algorithm to this setting.

### Weaknesses
- I found multiple vspace manipulations around section titles. See e.g., section 5.1 lines 425-426, and section 6 lines 475-476. I discussed with the AC for a desk-reject, which we didn't go through with. Still I feel that a warning to the authors is warranted, since giving this paper a review is to a degree disrespectful to the many submissions to ICLR that took meticulous care in formatting their paper to the style guide.

With that said, I still kept an open mind to your ideas and contributions when reviewing this paper and did not let the above comment influence my overall thought of the paper. 

- Section 3.1 to 3.2 is too complicated. In the current state, I'm not sure if I properly understood your method, and I doubt others would from the current presentation. Some small tips:
	- You should guide the reader textually through the ideas of your method. Perhaps even with visualizations of how the Q distribution adapts? 
	- Q-node atom update, instead of algorithmically going through steps (i) and (ii), you can explain what is the "before" state and the "after" state of the algorithm, and defer details to the appendix. 
	- Action Selection, emphasize that we want to approximately draw samples from the estimated Q-posterior. Most importantly, show what the optimism bonus does to this posterior. Again, some visualization helps, but abstract formalism is more useful than the current overly detailed state.

- I have a big issue with distributional approaches to exploration due to the often overlooked fact that distributional methods capture *noise*. When there is still uncertainty about the value functions, it can boost exploration, but this is rather a side-effect and luck over proper design. Typically, we don't want to explore over noise, as this implies risk-seeking behavior.
	- Because of the currently poor presentation of sections 3.1 and 3.2, I am not 100% certain how you deal with this. Could you comment on this? How do you separate irreducible from reducible uncertainty?

- I am not a MCTS theory person, but I found the approach for the analysis a bit strange to jump from MDPs to non-stationary bandits. Maybe this is standard in the literature, then it is fine, but you should include a reference in the opening of section 4.

- **None of the "results" in section 4 are supported**. All proofs and details are deferred to the appendix but the authors did not include an appendix. So I simply assume they do not exist. Because of this, I did not check any of the theory from this point on, I skimmed section 4.3, 4.4, but ignored the details.

- Section 5,
	- Figure 3 is an interesting result, only the color scheme is poor. It is very difficult to see CATSO and PATSO on the plots. The figure is also too small. At least, moving right, it seems that PATSO really reduces value estimation error a lot and consistently. 
	- Unclear how many seeds were used
	- Table 1, most results are not statistically significant, especially since I am unsure of the power behind these measurements. Like, how does Phoenix achieve zero standard deviation with a stochastic method? Is the environment deterministic? If so, why would you include a determinstic environment for testing? Could you perhaps separate/ group environments on their characteristics to make this result easier to parse?

*Minor comments:*
- The introduction quickly dives into a related work section, perhaps it is useful to move the contributions up before thoroughly explaining the context.
- I didn't find any discussion on afterstate-values that were used in stochastic MuZero (Antonoglou I., 2022). This is to my knowledge one of the easier ways to directly use mean-estimates in e.g., puct $Q$ nodes to deal with stochasticity. 

After reading the whole document I think the authors do not really include learning in their method, they only used a pretrained DQN network as feature extraction in the Atari experiments, but I can't call this learning. 
Because "learning" is not really involved, then maybe ICLR is also not the ideal venue for this paper. Maybe, AAAI or UAI (or something else) would be more fitting?

### Questions
- In lines 220-221: How are $T_{s_h}$ and $T_{s, a}$ tracked and updated? 
	- How do you simulate transitions? 
	- Are these somehow cached? Such that you can guarantee that they get revisited?

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
0

### Confidence
2