# Grouping Nodes with known Value Differences: A lossless UCT-based Abstraction Algorithm

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
A core challenge of Monte Carlo Tree Search (MCTS) is its sample efficiency, which can be addressed by building and using state and/or state-action pair abstractions in parallel to the tree search, such that information can be shared among nodes of the same layer. On the Go Abstractions in Upper Confidence bounds applied to Trees (OGA-UCT) is the state-of-the-art MCTS abstraction algorithm for deterministic environments that builds its abstraction using the Abstractions of State-Action Pairs (ASAP) framework, which aims to detect states and state-action pairs with the same value under optimal play by analysing the search graph. ASAP, however, requires two state-action pairs to have the same immediate reward, which is a rigid condition that limits the number of abstractions that can be found and thereby the sample efficiency. In this paper, we break with the paradigm of grouping value-equivalent states or state-action pairs and instead group states and state-action pairs with possibly different values as long as the difference between their values can be inferred. We call this abstraction framework Known Value Difference Abstractions (KVDA), which infers the value differences by analysis of the immediate rewards and modifies OGA-UCT to use this framework instead. The modification is called KVDA-UCT, which detects significantly more abstractions than OGA-UCT, introduces no additional parameter, and outperforms OGA-UCT on a variety of deterministic environments and parameter settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Known Value Difference Abstractions (KVDA), a framework that extends the ASAP (Abstractions of State-Action Pairs) methodology for Monte Carlo Tree Search (MCTS) in deterministic environments. Unlike ASAP which groups only state-action pairs with identical Q-star values, KVDA deliberately groups pairs with different values if their difference can be inferred from immediate rewards. The authors implement KVDA-UCT (a modification of OGA-UCT) that tracks difference-adjusted values during aggregation and a generalized version (epsilon_t-KVDA) for stochastic settings. Experimental evaluation across 19 deterministic domains and their stochastic variants shows KVDA-UCT detects substantially more abstractions than OGA-UCT and typically matches or exceeds parameter-optimized (epsilon_a, 0)-OGA performance without introducing additional parameters.

### Strengths
Elegant and Sound Conceptual Contribution: The core insight relaxing value-equivalence to known value differences is a natural and well-motivated extension of ASAP. Figure 1 effectively demonstrates cases where ASAP fails but KVDA succeeds. The difference-accounted aggregation mechanism is theoretically sound, and the proof correctly establishes the exactness (lossless) guarantee.

Strong Empirical Performance (Deterministic): The method is practical, as KVDA-UCT introduces no new parameters. It empirically detects significantly more abstractions than OGA-UCT. Crucially, it consistently matches or outperforms a parameter-optimized (epsilon_a, 0)-OGA in both raw performance and generalization, all while maintaining negligible runtime overhead.

Comprehensive Methodology and Reproducibility: The evaluation is thorough, covering multiple metrics (abstraction detection, optimized performance, generalization), various iteration budgets, and detailed domain descriptions. The authors also provide reproducible code with compilation details.

### Weaknesses
Poor Stochastic Performance: The method's advantages largely disappear in the approximate (stochastic) setting. epsilon-t-KVDA exhibits mediocre performance, sometimes significantly underperforming. The authors acknowledge this stems from "faulty abstractions" but offer it only as future work, substantially limiting the method's practical applicability.

Experimental Design Lacks Critical Controls: The use of hand-engineered heuristics to create dense rewards for board games may artificially favor KVDA's reward-difference mechanism. Furthermore, some domains show identical performance across all methods and some comparisons have overlapping confidence intervals, yet are still presented as favorable to KVDA.

Significantly Restricted Scope: The paper explicitly states the method does not extend to multi-player games due to the lack of a unique V-star. It also appears dependent on deterministic immediate rewards (not addressing stochastic rewards). These constraints sharply limit the method's impact, as MCTS is highly prevalent in the exact domains that are excluded.

### Questions
Stochastic Failure Modes: The paper attributes epsilon-t-KVDA's poor performance to faulty abstractions. Can you precisely characterize what type of faulty abstractions occur, and why the aggregation/representative-switching mechanism fails to correct for them under transition errors?

Impact of Heuristics: The board game results rely on dense reward transformations via hand-engineered heuristics. How robust are the performance gains to different heuristic choices?

Multi-Player Extension: The paper notes V-star is not unique in multi-player settings. Could the KVDA framework be adapted to work with different value concepts, such as Nash equilibrium values or empirical best-response values from self-play, to broaden its applicability?

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
4

### Summary
Authors propose KVDA, a novel MCTS lossless abstraction method rooted from ASAP. KVDA modifies ASAP’s abstraction criteria, especially removing the criterion of immediate reward equivalence, and leaving the Q differences between the Q nodes as a record when abstracted. For state nodes, KVDA abstracts two state nodes if their all Q nodes are abstracted to each other and Q differences of their all abstractions are same. Authors merge KVDA with UCT as KVDA-UCT, then compare abstraction ratio and general performance of KVDA-UCT with OGA-UCTs on various board games, proving that KVDA-UCT leads more abstractions than OGA-UCT, and that KVDA-UCT outperforms OGA-UCT mainly on deterministic environments.

### Strengths
The idea is distinct and sound, supported with theoretical guarantees. Background description is elaborate and well grounded. Authors adopt adequate counterparts for comparison experiments, which are conducted on appropriate domain and various tasks. They also consider extending KVDA to stochastic environments for generality.

### Weaknesses
#### Major Weaknesses

- KVDA–OGA Performance Ambiguity: The biggest concern is that according to Table 2, different from the authors’ comment, the performance of (ε_a, 0)-OGA with 1,000 iterations seems to generally outperform the KVDA. If it’s the case, authors might claim that since KVDA more generously abstracts than (ε_a, 0)-OGA, KVDA aggressively finds good paths more efficiently under the condition of low iteration budget. Here, at least I can agree that KVDA is better than (∞, 0)-OGA, since (∞, 0)-OGA would yield lots of faulty abstractions while KVDA would not because it’s lossless. This is where KVDA mitigates the potential negative effects of its generosity and turns it to its advantage. Still, if (ε_a, 0)-OGA performs poorly in lower iterations but achieves the best performance at 1,000 iterations, it may imply that the conservative aspect of OGAs contributes to steady but ultimately powerful performance growth. Since 1,000 iterations are not practically high budget, if this trend reproduced in higher iterations, the contribution of KVDA would become more limited, unless OGAs only outperform in practically very high budget. Because KVDA struggled in stochastic envs (and I agree leaving improvement of it for future work), proving the effects of KVDA in deterministic envs would naturally be emphasized to highlight the sufficient significance of the paper.
- Insufficient Method Discussion: Secondly, the lack of discussion on the main method makes authors’ suggestions less convincing. Throughout the paper, it appears relatively limited compared to other parts such as background descriptions and equations. Although introducing the foundations and ASAP (including subsection 3.1) is necessary (But are Sections 2.3 and 2.4 really necessary as standalone subsections?), it would be more convincing and provide denser insight to focus more on the main proposal, especially by providing theoretical or empirical analysis (e.g., a few actual abstraction examples obtained from the experiments) of the expected positive/negative effects due to the KVDA’s lossless but generous abstraction conditions. Explanation of why the results were so good for some tasks and vice versa might also help.
- Poor Structural Cohesion: Lastly, the overall flow of the paper is not very cohesive and somewhat difficult to follow. For instance, it was hard to grasp the key concept of main idea only reading Introduction. The initial introduction of understandable mechanism of the main method is introduced somewhat too late, which made reading before subsection 3.2 a little exhausting. I cautiously suggest adopting a “method-first, explanation-later” structure for the Method section, or providing brief mechanical information about KVDA in the preceding sections. Some sentences were too lengthy or colloquial. Figures and Tables of experimental results were confusing due to their jumpy locations and several formatting errors. Experimental conditions for each result were also confusing (especially the differences between Table 2, 6, 7 and Figure 4. Why do they appear to show different results?). Refining the overall flow with more intuitive and concise writing would be needed.

#### Minor Weaknesses
- Equation 6, it appears that the sign of d was not considered.
- Page 7, unnecessary three “Section 5”?
- Page 13, 14, Table errors
- Page 26, Image error
- Reproducibility link is unavailable.
- Figure 3 (Page 27), numbering error
- Figure 4 (Page 16), legend “(ε_a, εt)-OGA” instead of “(ε_a, 0)-OGA”?

### Questions
All questions and suggestions are included in the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper tackles the problem of low sample efficiency in Monte Carlo Tree Search (MCTS). Existing methods, such as OGA-UCT, improve efficiency by abstracting similar states and state-action pairs, but rely on strict equality of immediate rewards, which limits the scope of abstraction. The authors propose a new framework, Known Value Difference Abstractions (KVDA), which allows grouping states or actions with different but inferable value differences. Incorporating KVDA into OGA-UCT (forming KVDA-UCT) leads to more abstractions, better efficiency, and improved performance across various deterministic environments, all without introducing extra parameters.

### Strengths
- The problem tackled by this work is well motivated. Increasing the number of abstractions helps improve the sample efficiency of MCTS.
- The solution, proposed in this work, sounds. The idea of abstraction using a known difference in values between state(-action) pairs is novel to me.
- I really appreciate Section 2, which is essential to understand the preliminaries and know the related work.
- I believe the results are strong and the proposed abstraction technique does not hurt in terms of performance, except in the stochastic case.
- The limitations are clearly mentioned.

### Weaknesses
- I believe some parts of the main paper need to be adjusted for clarity.
- There are some tables and figures that are placed in the Appendix while being discussed as main results in the main paper. I understand the issue with space, but I believe it is really important for the main paper to be isolated from the appendix. 
- This is especially because the appendix is not well-presented, since there are some figures that are not inserted correctly.
- I believe the baselines benchmarked in this work are fair, but I would be interested in looking into some other methods that were already discussed and mentioned in Section 2.

**Minor Issues:**
- There is an issue with referencing the sections in lines 376-377.

### Questions
- In the deterministic setting, when might the proposed abstraction (based on the known difference in values) harm the performance?

### Soundness
3

### Presentation
2

### Contribution
3
