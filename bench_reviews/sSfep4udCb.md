## Summary
This paper proposes that State Space Models (SSMs), despite a fundamental limitation in solving long-form generation tasks due to bounded memory, can overcome this limitation through interactive tool use. It presents a theoretical framework proving this claim and demonstrates empirically that tool-augmented SSMs achieve impressive length generalization on arithmetic, reasoning, and coding tasks, often outperforming comparable Transformers.

## Strengths
- **Strong Theoretical Framework:** Provides a clear formalization of Generalized SSMs (GSSMs), long-form generation tasks, and the necessity of interactive tool-use for length generalization. The negative result (Theorem 2.1) is crisp and well-motivated.
- **Compelling Core Empirical Finding:** Robustly demonstrates that SSMs (Mamba, LSTM, GRU), when trained on interactive tool-use trajectories, can extrapolate far beyond training lengths on synthetic arithmetic and logical tasks (e.g., from 5-digit to 1000-digit addition). This core result validates a key thesis of the paper.
- **Systematic Investigation of Tool Paradigms:** Carefully distinguishes between CoT-only, single-turn, and interactive tool-use, both theoretically and empirically. The coding experiment (Figure 1) effectively shows the advantage of SSMs emerges specifically with interactive agents.
- **Valuable Practical Insight:** Successfully argues for evaluating architectures "in a system" (i.e., as tool-using agents) rather than as standalone models. This reframes the SSM vs. Transformer trade-off and suggests a promising research direction.

## Weaknesses
### Major:
- **Overstated Theoretical Claim Relative to Experiments:** Theorem 2.2 claims that, with appropriate training data, a GSSM can achieve *perfect* length generalization (error ≤ ε for *all* n ≥ n₀) on *any* tractable long-form task. The empirical results, while strong, do not fully substantiate this universal guarantee. Performance degrades significantly on Tower of Hanoi (8→12 disks, 49%) and in the more realistic coding task. The theory relies on a constructed "simple" learning algorithm (akin to string-matching), while the experiments use standard SGD-trained models. The gap between the strong theoretical promise and the mixed, task-dependent empirical outcomes is a significant weakness.
- **Incomplete Isolation of Causal Mechanism:** The paper attributes SSMs' superior extrapolation in tool-use settings to their bounded memory (the theoretical bottleneck). However, the Transformer baselines (Pythia, Mistral) differ from the SSMs in many other aspects (architecture, pretraining, optimization). A more controlled comparison—e.g., versus a Transformer constrained to fixed-size memory via a strict sliding window—is needed to rule out alternative explanations (e.g., that SSMs are simply better at learning these specific, long, deterministic trajectories). The current evidence is suggestive but not conclusive.
- **Methodological Concern in Coding Experiment:** For the coding task, training trajectories are filtered to include *only successful and short* examples. This introduces a selection bias; models are trained on a curated subset of optimal behaviors. Consequently, the performance drop on larger codebases may reflect the increasing difficulty of finding such optimal trajectories rather than a pure failure of length generalization. This undermines the strength of this particular experiment as evidence for the theory.

### Minor:
- **Under-explained Boundary Conditions:** The paper notes "more limited length generalization" on Tower of Hanoi (exponential output growth) but does not provide a substantive analysis of why this task is harder. A discussion of how output length growth, state complexity, or the specific tool-use protocol interacts with the theoretical claims would clarify the limits of the approach.
- **Limited Analysis of Transformer Failures:** While results show Transformers struggle, the paper does not investigate *why*. Is it due to attention distraction, training instability on long trajectories, or another factor? A deeper analysis or discussion would strengthen the empirical narrative.

### Trivial:
- **Underdeveloped Natural-Language Experiment:** The long-context natural-language task (Section 3.4) feels tacked on and minimally described. It adds little evidence to the core claims.

## Nice-to-Haves
- An experiment or discussion connecting the constructive theoretical proof (which uses a string-matching-like learner) to the practical success of SGD-trained SSMs would help bridge the theory-practice gap.
- A case study analyzing a failure example (e.g., a large codebase where the model fails) to illustrate the practical limitations of the approach.

## Removed Points
*These points are flagged to be removed, treat them with caution*

**Strengths that were removed:**
- "The paper is well-written." (Generic, applies to any competent paper.)
- "The topic is important." (Generic, insufficiently specific.)

**Weaknesses that were removed:**
- **"Limited real-world applicability...":** The paper includes a realistic coding task with trajectories from an SWE-agent and a natural-language task. The synthetic tasks are controlled tests of the core mechanism; criticizing them for being synthetic is scope creep.
- **"Overreliance on synthetic... trajectories":** The paper explicitly uses synthetic trajectories to cleanly test the capability; this is a methodological choice, not a flaw. The coding task uses real agent data.
- **"Limited scope of tool-use evaluation":** The paper systematically studies pointer-based and search tools relevant to the theoretical framework. Demanding evaluation on "more complex or diverse tool interfaces" is a scope expansion, not a core flaw.
- **"Insufficient comparison with state-of-the-art approaches":** The paper compares against relevant baselines (Pythia, Mistral, LSTM, GRU) and includes a hybrid model (Hybrid-Mamba) and RMT in the appendix. It is not required to compare against every recent architecture.
- **"Potential overclaiming of contributions...":** The paper's theoretical results (Theorems 2.1 & 2.2) are novel contributions specific to the GSSM + interactive tool-use setting. This criticism is not substantiated by a direct comparison with prior work within the paper.
- **"Weaknesses about missing related works":** As per instructions, I cannot invent or require discussion of missing related works.
- **Harsh Critic's point about "confounded comparison"**: Partially addressed in "Incomplete Isolation of Causal Mechanism" above. The harsh critic's claim that the comparison is *entirely* confounded is too strong; the paper's narrative is plausible, but the evidence could be more controlled.
- **Harsh Critic's point about "central theoretical claim... unsupported"**: This is largely incorporated into the first Major Weakness, but softened. The experiments *do* support the thesis that interactive tool-use enables much better length generalization; they just don't perfectly match the strongest possible theoretical guarantee.
- **Spark's point about "missing experiment: tool-agnostic vs. tool-augmented Transformers"**: The paper *does* train Transformers (Pythia, Mistral) on the same interactive tool-use trajectories (see Sections 3.1, 3.2, 3.3). The Transformer results in those sections are the requested comparison. This point misunderstands the paper.
- **Spark's point about "ablation of tool interaction complexity"**: The paper already includes a clear ablation comparing single-turn vs. interactive tool-use in the coding experiment (Figure 1) and mentions ablations in Appendix D.7. Demanding further ablations is a nice-to-have, not a core flaw.

## Suggestions
- **Strengthen the empirical-theoretical link:** Temper the strong universal claim of Theorem 2.2 in the discussion to better align with the empirical observations (e.g., note that perfect generalization may require ideal training data and may degrade with problem complexity). Alternatively, provide additional experiments on a broader suite of tasks to better map the boundary between problems where near-perfect generalization occurs and those where it does not.
- **Add a controlled architectural ablation:** Include an experiment comparing the SSM (Mamba) to a Transformer model explicitly constrained to have a fixed-size working memory (e.g., a very narrow sliding window attention) trained on the same interactive trajectories. This would more directly test if the *memory bottleneck* is the key differentiator.
- **Clarify the coding data selection:** In Section 3.3, explicitly discuss the potential implications of filtering trajectories for success and shortness. Consider presenting results with an unfiltered or differently sampled dataset to show the finding is robust to this curation.