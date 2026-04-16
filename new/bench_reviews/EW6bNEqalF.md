## Summary

The paper studies offline reinforcement learning in Regular Decision Processes (RDPs), a non‑Markovian setting where dynamics are captured by a hidden finite automaton. It introduces (1) a new “language metric” over sets of traces, with a hierarchy of language families used as test statistics in a state‑merging RDP learner, and (2) a Count‑Min‑Sketch (CMS)–based implementation to reduce memory. The authors derive PAC-style sample complexity bounds and present experiments on several benchmark domains.

## Strengths

- **Conceptually elegant language metric framework.**  
  Definition 2 introduces a “language metric” \(L_X\) over distributions on strings, parameterized by a set of languages \(\mathcal X\). The paper shows that appropriate choices of \(\mathcal X\) recover familiar distances (e.g., \(L_\infty\), total variation, prefix metrics) and enables constructing a two‑dimensional hierarchy \(\mathcal X_{i,j}\) based on basic patterns \(\mathcal G_i\) and the operator \(C_k^\ell\). This unifies several metrics used in automata/RDP learning and is technically clean.

- **Instance‑specific exponential distinguishability gain.**  
  The T‑maze example (Example 4) and Theorem 1 give a concrete family of RDPs where \(L_\infty^p\)-based distinguishability decays as \(\mathcal O(2^{-N})\) in corridor length, while the distinguishability under a simple language family \(\mathcal X_{2,1}\) remains \(\Omega(1)\). This is a nontrivial insight: by shifting from individual trace probabilities to probabilities of appropriately chosen languages, one can avoid an exponential blow‑up in a meaningful non‑toy setting.

- **Clarification and extension of prior theory.**  
  The analysis around Theorem 2 both adapts ADACT‑H to use CMS and identifies a mistake in the proof of Cipollone et al. (2023), clarifying that both the old and new analyses incur an additional \(\sqrt{H}/\mu_0\) factor. This correction is important for the community. Theorems 2 and 3 give clear high‑probability bounds in terms of concentrability \(C_{\mathbf R}^*\), minimum occupancy \(d_m^*\), and a distinguishability parameter \(\mu_0\).

- **Empirical support for the language‑based test’s computational advantages.**  
  Experiments on five benchmark domains (Corridor, T‑maze(c), Cookie, Cheese, Mini‑hall) show that ADACT‑H with the language family \(\mathcal X_{3,1}\) often yields smaller automata and better or equal policy rewards than FlexFringe and the CMS‑based variant, and runs faster in most cases (Table 1). In T‑maze, Figure 2 shows roughly linear scaling in time and number of states for the language metric vs. exponential blow‑up for CMS, matching the theoretical narrative.

- **Well‑presented preliminaries and examples.**  
  The RDP formalism, including the Moore machine representation and distinguishability definitions, is clearly set up. The T‑maze RDP construction (Example 3) is concrete and helps ground an otherwise abstract setting.

## Weaknesses

### Fatal

None of the issues rises to the level of “this is not a paper” or a direct contradiction of the core technical claims. The main concern is overstatement and limited characterization of where the theoretical gains truly apply, but the central constructions and theorems themselves are soundly stated.

### Major

- **Overstated generality of “sample efficiency improvement” and “removal” of \(L_\infty^p\) dependence.**  
  The abstract and introduction repeatedly frame the work as “overcoming the limitations” of existing offline RDP algorithms and “removing the dependency on \(L_\infty^p\)-distinguishability,” suggesting a general cure for the exponential dependence on horizon. However, the main changes are:

  - The sample complexity remains of the form \(\tilde{\mathcal O}( C_{\mathbf R}^* \log(1/\delta) / (d_m^*\,\mu_0^2) )\), where \(\mu_0\) is a distinguishability parameter defined w.r.t. the chosen metric. In Theorem 2 this is \(L_\infty^p\); in Theorem 3 it is the language metric \(L_X\). The dependence on a problem‑dependent distinguishability constant is not removed but redefined.

  - Theorem 1 and Example 4 show *existence* of families where \(\mu_0\) under \(L_X\) is exponentially larger than under \(L_\infty^p\), but there is no characterization of broad RDP classes where such benign \(\mathcal X\) exist. Assumption 1 simply posits that the behavior policy induces some \(\mu_0>0\) for a user‑chosen \(\mathcal X_{i,j}\).

  - In the worst case, the authors explicitly note that \(\log|\mathcal X|\) can be \(\tilde{\mathcal O}(H)\), and \(1/d_m^*\) may be exponentially large if some optimal states are hard to reach: “The constant \(1/d_m^*\) depends exponentially on \(H\) if there exists an RDP state that is very hard to reach” (Section 4.2). Thus, exponential dependence on \(H\) is not generically eliminated; it is mitigated *only when* both \(d_m^*\) and the chosen \(\mu_0\) behave nicely.

  Overall, the paper provides a *different parameterization* of sample complexity and shows that for some structured domains (like T‑maze) this parameter behaves much better. Presenting this as a general removal of exponential blow‑up overstates what is proved. This is primarily a framing/claims issue rather than a flaw in the theorems themselves.

- **Scope and selection of the language hierarchy \(\mathcal X_{i,j}\) are under‑characterized.**  
  The new theory critically depends on the chosen language family. The hierarchy \(\mathcal X_{i,j}\) is cleverly constructed, but:

  - Beyond T‑maze, there is no theoretical characterization of when small \(j\) (so that \(\log|\mathcal X_{i,j}|\) is constant) suffices to distinguish all relevant states with a favorable \(\mu_0\). The text acknowledges that \(|\mathcal X_{i,j}|\) grows like \(\mathcal O((AOR)^j)\), but does not link \(i,j\) to structural properties of RDPs (e.g., temporal logic depth, reward machine size) that might bound the necessary complexity.

  - Assumption 1 states that the behavior policy “ensures an \(L_{\mathcal X_{i,j}}\)-distinguishability of at least \(\mu_0 > 0\), where \(\mathcal X_{i,j}\) is constructed as above and is an input to the algorithm.” There is no condition stated on the RDP or \(\pi^b\) that would allow one to know such an \(\mathcal X_{i,j}\) exists for given \(i,j\), nor any adaptive procedure to increase \(j\) if needed.

  - Experiments fix \(\mathcal X_{3,1}\) and show that this works well on the five chosen domains, but do not inspect or report empirical distinguishability values, or examine what happens if \(j>1\) or different \(\mathcal G_i\) are used.

  Therefore, while the hierarchy is an interesting tool, the paper does not yet provide a clear picture of *for which problem classes* low‑complexity language families actually deliver the advertised gains, or how to pick them in practice.

- **Connection from exact RDP reconstruction to \(\varepsilon\)-optimal policy is only sketched.**  
  The stated goal in Section 2.3 is offline RL: find an \(\varepsilon\)-optimal policy using a batch dataset. Theorems 2 and 3, however, are reconstruction theorems: they show that ADACT‑H identifies the minimal RDP with high probability under certain conditions. The text then states that ADACT‑H (or its approximation variant) “can be incorporated into an offline RL algorithm for learning an \(\varepsilon\)-optimal policy, cf. Algorithm RegORL in Appendix A,” but in the visible main body:

  - There is no explicit theorem or proof sketch that propagates reconstruction error (or failure probability) into a bound on value sub‑optimality. The role of the corrected \(\sqrt{H}/\mu_0\) factor in that control‑theoretic step is not discussed.

  - For ADACT‑H‑A, the paper notes in passing that Appendix C also provides sample complexity bounds, but does not summarize the resulting policy‑quality guarantees.

  Readers are expected to piece this together from the prior RegORL paper, whose analysis has itself just been modified. For an RL venue, it would be preferable to state at least one explicit “offline RL with language metric” theorem with a clear dependence on \(\varepsilon,H,C_{\mathbf R}^*,d_m^*,\mu_0\), even if its proof is deferred.

- **CMS variant’s practical benefit is weak and not fully contextualized.**  
  The CMS‑based algorithm is described as a second “original technique”—it compactly stores empirical distributions, and Theorem 2 shows it can match the asymptotic sample complexity of naive counting. However:

  - The main algorithmic bottleneck for \(L_\infty^p\) remains: the statistical test still effectively ranges over exponentially many suffix patterns. The paper explicitly acknowledges in Section 5 that “the statistical test still has to iterate over all suffixes, which is exponential in \(H\).”

  - Empirically, CMS is consistently slower and yields larger automata than the language‑based method, and even times out on Mini‑hall; no memory usage statistics are reported to demonstrate its purported advantage.

  As a result, the CMS variant is a conceptually correct but practically unattractive branch of the design space in the reported domains. The paper would be stronger if it were framed as a modest space‑efficiency technique with limited time benefits, rather than as a co‑equal main contribution.

### Minor

- **Empirical evaluation does not probe sample‑efficiency behavior.**  
  The theoretical focus is on sample complexity, but the experiments appear to use a fixed number of episodes (e.g., \(K=100\) in Figure 2) and report performance at that point. There is no learning‑curve style evaluation (reward vs. dataset size) to empirically confirm that the language metric reaches good policies with fewer samples than CMS or alternative tests. Thus, the experimental section mainly supports *computational* gains (time and automaton size) and end‑policy quality at a single data regime, not the core sample‑efficiency claims.

- **Choice of baselines is limited.**  
  FlexFringe is a reasonable automata‑learning baseline, but it targets somewhat different objectives (includes cycles, uses heuristics) and does not share the paper’s offline RL guarantees. There is no direct empirical comparison against an implementation of RegORL, even on smaller instances where that would be computationally feasible. Nor is there comparison against generic sequence‑based RL methods (e.g., recurrent policies) for context.

- **Language‑metric test is described as an “implementation” improvement.**  
  Section 3 phrases the contribution as “develop tractable methods for implementing the statistical test,” but the language‑metric–based test is more than an implementation detail: it changes the test statistic itself (from maximal difference over suffixes to maximal difference over a restricted language family). This is partly a wording issue but matters for correctly conveying the conceptual novelty.

### Trivial

- Some notation is slightly overloaded or under‑explained (e.g., the occasional switch between \(L_\infty\), \(L_\infty^\circ\), \(L_\infty^p\), and the typesetting glitch “\(L_\chi\)” in one place). This does not impede understanding but could be cleaned up.

- A few typos and minor inconsistencies (e.g., “ADaCT-H” vs “ADACT-H”) appear, but these are superficial.

## Nice-to-Haves

- A brief explicit statement in the main text summarizing how exact minimal RDP reconstruction underlies \(\varepsilon\)-optimal policy extraction, perhaps as a corollary to the new theorems, would clarify the offline RL story without substantially lengthening the paper.

- Additional case studies visualizing the learned automata under language metric vs. CMS vs. FlexFringe on the same domain (especially T‑maze) could make tangible how the language metric avoids unnecessary state splits.

- An empirical or heuristic procedure for increasing \(j\) in \(\mathcal X_{i,j}\) when merges become statistically ambiguous would make the method more practical and help bridge theory to practice.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **“No global reduction in worst‑case dependence on \(H\), \(C_{\mathbf R}^*\), or \(d_m^*\)” as a *criticism of correctness*.**  
  The harsh review notes that worst‑case asymptotics remain similar. This is accurate, but worst‑case optimality is not required for the paper to make a valid contribution; the language metric’s value is in instance‑specific gains. As a result, the fact that the worst‑case dependence is unchanged should be seen as limiting generality rather than as a correctness issue, and we have already incorporated this more moderately into the Major weaknesses.

- **Implication that the authors’ corrected \(\sqrt H\) factor “weakens” their own method relative to prior work.**  
  The paper clearly states that their analysis uncovered a mistake in Cipollone et al. (2023) and that the factor affects both. This should not be treated as a weakness of the current submission; it is a service to the community. We therefore avoid framing this as “weakening” their algorithm, focusing instead on how it tempers claims of improvement over the older bound.

- **Concerns about the existence or availability of RegORL and other baselines.**  
  Any doubts about whether RegORL or other cited algorithms are “available” are not appropriate; the paper cites them, and per the reviewing rules they must be treated as existing. Thus, we do not include any criticism that hinges on their non‑existence.

## Novel Insights

The genuinely novel observation is that one can recast the distinguishability condition underpinning RDP state‑merging in terms of probabilities of *languages*—structured sets of traces—rather than individual prefixes. With a carefully crafted, low‑complexity family \(\mathcal X_{i,j}\), this can substantially enlarge the statistical gap between distinct RDP states under a given behavior policy without incurring exponential blow‑up in the number of test sets. The T‑maze family exemplifies this by showing that coarse patterns (e.g., “taking North and then receiving a reward”) can remain informative even when individual full trajectories become vanishingly rare. This language‑based view provides a new and potentially powerful knob—choice of language family—for tailoring offline RL algorithms in non‑Markovian environments, independent of the specific RDP learning algorithm used.

## Suggestions

- **Reframe the main claims about sample efficiency.**  
  Clearly state in the abstract and introduction that the improvement is *instance‑dependent* and hinges on the existence of a language family \(\mathcal X\) for which the distinguishability parameter \(\mu_0\) is favorable. Phrasings like “removes the dependency on \(L_\infty^p\)-distinguishability parameters” should be softened to “replaces \(L_\infty^p\)-based distinguishability with a more flexible language‑based distinguishability, which can be exponentially larger in some structured domains (e.g., T‑maze).”

- **Characterize, even informally, when low‑complexity \(\mathcal X_{i,j}\) are sufficient.**  
  Add discussion linking \(\mathcal G_i,\mathcal X_{i,j}\) to known temporal structures (e.g., LTL depth, reward machine size). Even heuristic statements like “if the underlying reward machine depends only on conjunctions of at most \(k\) action‑observation‑reward features, then \(j\le k\) suffices” would give practitioners guidance.

- **State an explicit offline RL theorem.**  
  Add a theorem (or corollary to Theorem 3) of the form: “Under Assumption 1 and bounded concentrability, Algorithm RegORL with language metric test returns an \(\varepsilon\)-optimal policy with probability \(1-\delta\) using a dataset of size \(\tilde{\mathcal O}(\cdot)\).” You can defer the proof to the appendix, referencing the reconstruction results plus standard planning arguments, but having the statement upfront will clarify the control‑level contribution.

- **Clarify the role and limitations of the CMS variant.**  
  In the introduction and conclusion, explicitly note that CMS primarily reduces memory and does *not* change the exponential dependence on horizon in the time complexity of \(L_\infty^p\)‑based tests. Position it as a secondary, space‑oriented contribution whose practical speed is dominated by the language‑based approach in the reported domains.

- **Augment experiments to better reflect sample efficiency.**  
  If space permits, include at least one plot of reward vs. number of training episodes on a domain like T‑maze or Cookie, comparing language metric vs. CMS (and ideally vs. a RegORL implementation or a simple recurrent RL baseline). This would concretely demonstrate that the language metric yields better performance in the moderate‑data regime, aligning practice with the theory.

- **Discuss or bound \(d_m^*\) in example domains.**  
  For at least one benchmark (e.g., T‑maze or Corridor), provide an explicit calculation or bound for \(d_m^*\) and show that it is not exponentially small in \(H\) under the chosen behavior policy. This would make the sample‑complexity statements more concrete and reassure readers that the exponential problems of \(1/d_m^*\) do not always dominate in practice.

## Score and Decision

### Calibration

I compared this paper to several human‑reviewed works:

- **PSR learning with UCB (jId5PXbBbX.md, scores 6/5/6/8, accepted)**: strong theory in a general non‑Markovian setting with limited but adequate experiments. That work had clearer characterization of its assumptions and problem class; this submission’s core idea is similarly interesting but its claims are more overstated.

- **Multi‑task RL under non‑Markovian processes (U6Qulbv2qT.md, scores 6/6/8/6/8, accepted)**: strong theoretical contributions with some empirical validation, more narrowly and carefully scoped than the current paper’s claims.

- **Offline RL theory papers (e.g., general function approximation, JSS9rKHySk.md, typically scoring 6–8 when well‑scoped)**: they tend to be accepted when the assumptions, problem classes, and gains are crisply characterized.

Relative to these anchors, this paper’s originality is solid, the question (offline RL in RDPs) is important, and many claims are well supported for specific instances like T‑maze. However, the over‑general framing of sample‑efficiency improvements, the underdeveloped guidance on choosing language families, and the lack of an explicit offline RL value‑guarantee theorem in the main text put it somewhat below the typical accepted bar for theory‑heavy RL papers. I would place it in the “promising but needs one more iteration” range.

MY FINAL SCORE: <pineapple>5.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>