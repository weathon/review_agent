# Usefulness-driven Learning of Formal Mathematics

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Creating an AI that can truly "do" mathematics requires more than just solving isolated problems. It must mimic the creative, progressive nature of human mathematicians, who build upon previous work to generate new knowledge. A crucial part of this process is proposing theorems that serve as useful building blocks for proving more advanced theorems. In this paper, we introduce UseForm, a novel framework that formalizes this notion of usefulness, and demonstrates how it can be used to train a usefulness-driven AI mathematician. UseForm determines a theorem's usefulness based on two criteria: its reusability in subsequent proofs and its contribution to increasing proof likelihood. We integrate UseForm into the self-play conjecturing-and-proving setting of Minimo (Poesia et al., 2024). That is, starting from only axioms, we iteratively train conjecturers to propose useful formal statements and provers that explicitly reuse them when generating formal proofs. We experimentally evaluate this usefulness-driven self-play approach across three mathematical domains: arithmetic, propositional logic, and group theory. Our evaluation considers two metrics: intrinsic usefulness, which measures how often our trained provers reuse theorems, and extrinsic usefulness, judged by a state-of-the-art large language model and external provers like SMT solvers. Our results demonstrate that our usefulness-trained model effectively generates a large number of intrinsically and extrinsically useful formal theorems. For instance, our approach outperforms the original Minimo by 2.9 times in extrinsic usefulness for arithmetic. Our work highlights the significant potential of integrating usefulness in AI-driven mathematical discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces UseFor, a usefulness-driven framework for automated discovery of formal theorems. Rather than ranking conjectures only by “difficulty” (low proof likelihood) as in Minimo, UseFor promotes conjectures that (i) are actually used in downstream proofs and (ii) increase the prover’s success likelihood on those targets. The method integrates this metric into a self-play loop (conjecturing & proving), adds triviality filtering and novelty bias, and evaluates across arithmetic, propositional logic, and group theory. Empirically, UseFor yields more reusable lemmas and higher “usefulness” (LLM-as-judge + SMT-proved) than difficulty-only baselines; e.g., in arithmetic they report 2.9× extrinsic usefulness over Minimo.

### Strengths
- **Clear, principled objective**: The dual criterion (usage + likelihood improvement) operationalizes “theorem usefulness” in a tractable way that aligns with theory building, not just local hardness. The formulation and scoring rule are precise.
- **Thoughtful integration into self-play**: UseFor plugs into Minimo’s loop but changes the selection signal, and adds triviality filtering and novelty bias to avoid vacuous lemmas and near-duplicates—practical details that matter for stability.
- **Compelling empirical story**: Multiple domains, explicit intrinsic vs. extrinsic metrics, and a meaningful ablation. The results have demonstrated very meaningful improvements in mathematical discovery.

### Weaknesses
- **Limited Technical Details**: Obviously, this is a good follow-up work for [1]. However, the authors provided little recap of previous technical details, making the paper somewhat difficult to read. The authors should have built the setup more clearly. For example, what kind of LM settings are you using to prevent the contamination issue?
- **Limited Coverage in Mathematics**: The work has good motivations for self-evolving in mathematical discovery, but the axioms and topics are limited as they are also very similar, i.e., heavily relying on the type theory structure.
- **Reporting Clarity**: The paper asserts super-linear growth in usefulness evaluations; please include exact values with error bars per iteration and statistical tests.
- **Scale and generality**: Results are on small models and Peano-style domains; applicability to Lean/Isabelle with larger models remains open, and the paper itself flags contamination risks when scaling with pre-trained LLMs. A small-scale Lean pilot (even with tight libraries) or cross-formalism transfer test would strengthen generality claims.

[1] Poesia, Gabriel, et al. "Learning formal mathematics from intrinsic motivation." Advances in Neural Information Processing Systems 37 (2024): 43032-43057.

### Questions
- Typo: Line 334: domain --> domains

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UseFor, which builds upon prior formal mathematics work Minimo. While Minimo engages in a self-play loop by rewarding hard-but-not-impossible conjecturing, UseFor seeks to measure usefulness of lemmas for a benchmark set $\mathcal B$ of theorems. UseFor also adds library learning and triviality checks. They demonstrate that UseFor is able to increasingly use past theorems and scores higher than Minimo on an LLM-as-judge extrinsic usefulness score. They establish that UseFor beats a baseline that is Minimo but with library learning.

### Strengths
Originality: I think this is a fantastic idea to try and is novel. Intuitively, it makes a great deal of sense that theorem utility in proving other things is critical for deciding what is interesting to include as a theorem. The integration of library learning and triviality checks also make a great deal of sense, and it’s useful to integrate these ideas.

Quality: The treatment of related work and motivation for these ideas is very thorough. The set of RQs for the experiments section is well-motivated and for the most part executed well.

Clarity: For the most part, I found this very easy to follow. Everything was well-motivated.

Significance: the methodological contribution here is I think of considerable interest in this growing field of open-ended learning & formal mathematics. The experiments present good evidence.

### Weaknesses
While this is for the most part very clearly written, I cannot seem to find certain really critical details that would be important for both reproducibility and for interpreting the results. In particular, how is the benchmark set of targets $\mathcal B$ set? If this is something that is created and dynamically updated by the algorithm — if so, how? How is it seeded? How is it updated to reflect the changing capabilities of the model (if indeed it’s supposed to be “hard”)? These seem like important details, not at all trivial. If $\mathcal B$ is something that is not created by the algorithm, but rather is provided, that puts UseFor on a very different footing relative to Minimo, calling all the comparisons into question.

This work also adopts LLM-as-judge for an extrinsic usefulness measure. Minimo has another measure for how useful the play process is, namely, whether it gets better at proving a held out set of theorems provided by people. It would be really helpful to understand what this is and how it relates to the LLM-as-judge metric.

In answering RQ3, the paper claims super-linear growth. I’m not seeing that from the metric, though perhaps I’m misunderstanding something.

It would also be helpful to see the No usefulness training baseline for answering RQ2, as that Minimo didn’t use library learning seems a bit orthogonal to the point you’re making for that plot.

### Questions
How was $\mathcal B$ set up and maintained? I’d be keen to increase my score if this, in particular, is addressed well.

More minor but I think still critical for understanding the method: how is $\mathcal H_i$ (I’m assuming this is a “hard” subset of conjectures) maintained throughout training? Is there a set of “inner loop” successive subsamplings L_i made, in between model updates? If so, how many? If not, how is it that all conjectures eventually receive usefulness credit?

Can we have some error analysis for what’s going on with Group Theory relative to the other domains? The reversal in performance relative to Minimo is striking there.

Does No usefulness training have the triviality check? It’s not 100% clear to me whether a simple hardness criterion doesn’t screen those out.

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
This paper proposes UseFor, a framework for proposing theorems that are useful and can be used when training usefulness-driven automated math. Specifically, the work formalizes the notion of usefulness as a criterion for both usage and improvement. The work is integrated into the Minimo framework, a framework for automatically learn to prove or disprove conjectures. As part of the integration, the authors use usefulness instead of difficulty as a criterion for selecting new conjectures. Moreover, various new techniques are introduced in the method, including stabilization techniques such as filtering and novelty bias. The approach is evaluated on three different mathematical domains.

### Strengths
- This kind of work, incorporating interestingness info into a pipeline for automatic conjecture discovery and theorem proving, is interesting and an important direction in AI math discovery

- The paper shows an interesting and pretty simple way of incorporating a notion of interestingness into an automatic math learning loop

### Weaknesses
- One aspect of the difference between the proposed extension to Minimo and the original Minimo work is that the proposed method is driven by a benchmark set; I.e., a benchmark is used for computing the interestingness score. In contrast, Minimo is not driven by any external benchmark theorems. This is not negative in itself, but a clear difference to the original work, which is important to highlight clearly in the paper. 

- The benchmark setup is not clearly described in the paper. That is, Section 3.2.1 describes how the interesting value is computed using a benchmark set. However, it is unclear in the experimental evaluation where this benchmark set is coming from or how it is integrated into the framework. 

- As already noted by the authors, the experimental setup contains rather small models

### Questions
- Are the three mathematical domains and their definitions identical to the original Minimo paper, or are there any differences?

- Lines 412 to 413 say "The growth of this metric is super-linear, suggesting that UseFor not only maintains but amplifies its ability to generate theorems a mathematician would regard as useful." This is quite a strong statement, since I cannot see how the proposed approach gives any guarantee that the paper's notion of interestingness corresponds to what a mathematician would regard as useful. Please clarify this statement if possible.

- For the Minimo version with "No usefulness training" (line 431), how does it select the proven conjectures? Or is it including all without filtering?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces UseForm, a novel framework to train AI theorem provers to generate theorems that are useful and can serve as building blocks for proving more advanced subsequent theorems. Building on top of Minimo which starting from only axioms, iteratively train conjecturers to propose useful formal statements and provers that explicitly reuse them when generating formal proofs, the authors replaces difficulty-based training with a usefulness-aware self-play loop. The usefulness metric measures how much adding a lemma improves success likelihood and reuse in future proofs. The authors evaluate the proposed approach across three mathematical domains: arithmetic, propositional logic, and group theory, with two metrics: intrinsic usefulness, which measures how often the trained provers reuse theorems, and extrinsic usefulness, judged by a state-of-the-art LLM and external provers like SMT solvers. Experiment results show that UseFor significantly boosts theorem reuse and generates 2.9× more extrinsically useful theorems as judged by GPT-4.1 and Z3 solvers than Minimo.

### Strengths
1. The paper is well-motivated, well-written and easy to follow
2. The concept of usefulness in theorem proving is novel, shifts from proving difficult theorems to discovering useful ones.
3. The proposed usefulness-aware self-play loop is interesting.
4. The authors show consistent gains in both intrinsic (reuse) and extrinsic (human-judged) usefulness across multiple domains.

### Weaknesses
1. The proposed UseForm framework heavily relies on the existing Minimo framework, replacing its difficulty-based objective with a usefulness-based criterion. The improvement seems incremental. 
3. While the authors demonstrate gains in the usefulness of generated proofs, it is unclear whether the UseForm framework improves or hurts proof accuracy.
2. It is unclear whether the UseForm framework can be adapted to other strong theorem-proving systems such as Seed-Prover.
4. The paper lacks comparisons with frontier theorem-proving frameworks such as Seed-Prover [1], Goedel Prover [2], and LLM-based provers such as  GPT-5, Qwen-235B, Claude Sonnet, and Grok.



[1] Chen, Luoxin et al. “Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving.” ArXiv abs/2507.23726 (2025): n. pag.

[2] Lin, Yong et al. “Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction.” ArXiv abs/2508.03613 (2025): n. pag.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2
