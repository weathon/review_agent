# GAR: Generative Adversarial Reinforcement Learning for Formal Theorem Proving

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Solving math problems through verifiable languages such as Lean has significantly impacted both the mathematics and computer science communities. Current state-of-the-art models are often trained with expensive online Reinforcement Learning (RL) or expert iteration. However, these approaches rely on fixed problem sets, which causes inefficient training and limits the model to tackle complex problems. To overcome these limitations, we propose **GAR**: *Generative Adversarial Reinforcement learning*, a comprehensive RL training framework that jointly trains the problem composer and solver in an adversarial loop. **GAR** introduces an implicit curriculum learning mechanism, which aligns task difficulty with the prover's evolving capability. It thereby improves the training efficiency and enables stronger performance of proving advanced theorems. Experiments show that with **GAR** training, Goedel-Prover-V2-8B and DeepSeek-Prover-V2-7B achieve an average relative improvement in pass@32 of **4.20%** on MiniF2F-Test benchmark, while DeepSeek-Prover-V2's pass@32 on ProofNet-Test increases from 22.58% to **25.81%**. Beyond formal proving, **GAR** establishes a general RL paradigm for co-evolution of problem generation and solving under verifiable environments. The training code for this paper is open-sourced in [https://github.com/RickySkywalker/GAR-Official](https://github.com/RickySkywalker/GAR-Official)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Generative Adversarial Reinforcement learning, a framework for adversarial co-training of theorem provers and problem generators in the Lean environment.
GAR jointly optimizes a statement fuser that composes new mathematical statements and a prover that attempts to prove them, forming a self-improving loop. In each iteration, the fuser generates new, harder statements by combining two natural-language problems from a large dataset from Numina-Math and Lean-Workbook, which are then automatically formalized into Lean. The prover produces proofs for these statements, and Lean verification provides reward signals to both models.

Through this interaction, the fuser learns to propose “hard but solvable” problems while the prover learns to solve increasingly challenging ones. The framework yields consistent improvements over strong RL-trained baselines on MiniF2F and ProofNet, and demonstrates that jointly evolving generators and solvers can outperform direct reinforcement learning on static datasets.

### Strengths
* Novel formulation: The paper introduces a clear and well-motivated adversarial training paradigm for formal theorem proving, aligning problem difficulty with prover capability.
* Empirical gains: GAR achieves measurable and consistent improvements on competitive baselines, even when those baselines are already heavily RL-trained.
* Clarity and reproducibility: The training procedure, algorithmic structure, and implementation details are well described, with clear experimental setups and evaluation metrics.
* Significance: The proposed paradigm is general and could influence future work on co-evolving reasoning models in other verifiable domains.

### Weaknesses
1. Lack of a frozen-fuser control:
   The paper always updates both the fuser and the prover in each GAR iteration. A variant that keeps the fuser fixed (for example, using the initial fuser) while training only the prover is missing. This is necessary to determine whether the gains derive from adversarial co-adaptation or simply from exposure to more generated data.

2. Absence of an SFT-only baseline:
   All experiments start from provers that have already undergone heavy RL fine-tuning. It remains unclear whether GAR would still provide improvement if applied directly to a purely SFT-trained prover. This limits understanding of GAR’s generality and its effectiveness from weaker starting points.

3. No comparison with single-input statement generation:
   The current fuser fuses two NL statements into one new problem. There is no experiment showing what happens if the model generates or rewrites new statements from a single input problem (similar to iterative conjecturing). This ablation would clarify whether the two-input fusion design is critical to the observed gains.

4. Ambiguity in the Statement Fuser study:
   The difficulty analysis (Table 2) evaluates only problems produced by the evolving fuser—the same distribution used to train the GAR prover. The observed decline in base-prover accuracy might reflect distributional or stylistic shift rather than genuine difficulty increase. A control using random samples from Dstat would help validate the claimed implicit curriculum effect.

5. Limited exploration of the statement-modification penalty:
   The ablation in Table 3 removes penalties from both the fuser and the prover simultaneously, but does not include one-sided removals or a hard-ban condition. The paper would benefit from a more granular investigation of these choices.

6. Unfair efficiency comparison:
   The efficiency discussion (Appendix B.1) compares GAR, built upon already RL-trained long-CoT provers, with Kimina-Prover, which is trained from SFT. Because the starting points differ significantly, the efficiency claim is not directly justified.

### Questions
The reported statement-modification rate under full GAR (around 30–50%) seems unreasonably high. Since this behavior reflects an instruction-following failure rather than a Lean reasoning limitation, and your prover prompt is identical to the official prompts of DeepSeek-Prover-V2 and Goedel-Prover-V2, there is no prompt-shift factor that could explain it.
   
Please therefore report:
   - the modification rates of DeepSeek-Prover-V2 and Goedel-Prover-V2 under their original prompts (no GAR); and
   - the corresponding rate of the GAR-trained prover under the same prompts.

Such a comparison is important, because a statement-modification rate this high, combined with a relatively small sample size (pass@32), yet still achieving strong benchmark performance, is highly questionable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces a novel RL framework (GAR) that trains not only a solver, but also a fuser designed to generate new problems suitable for the solver's current training stage. To be specific, in the context of automated theorem proving, the prover (solver) is trained to solve problems. Concurrently, the fuser attempts to generate problems that are provable in Lean (by the current provers) but possess a relatively low pass rate, thus making them "difficult but approachable." The fuser is also trained during the RL process, creating an adversarial environment. The experimental results demonstrate the effectiveness of this method.

### Strengths
The overall quality and clarity of this work are good. Its motivation and methodology are described in detail. The proposed method also demonstrates significance, as it is potentially applicable to domains beyond automated theorem proving.

### Weaknesses
Regarding novelty, the idea of generating problems with appropriate difficulty levels has been explored by prior works, such as STP and Goedel-Prover-V2. STP, in particular, also trained a model to generate new problems in an adversarial way. The main difference appears to be that this work uses RL, whereas STP used expert iteration (SFT). This overlap weakens the originality and significance of the current paper.

Additionally, some experimental results in Table 1 are questionable. It appears that the reported results for prior works are all much lower than those originally published in their respective papers.

### Questions
Regarding Table 1, could you please explain why your reported results for previous works (Kimina, Deepseek, Goedel) are all significantly lower than those in their original papers? Did you encounter any issues with your experimental setup, such as a different Lean version, that might account for this mismatch?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a variant of self-play theorem proving (conjecturer-prover adversarial training) where the conjecturer fuses two statements to create a statement with higher difficulty "Composability" is a frequently observed problem in LLMs, and the method directly tackles this issue. With small budgets of additional training, the method improves the initial already highly optimized models by a significant margin.

### Strengths
The method is sensible, the numbers are very good, the paper is very clearly written.
The scope is limited because there are many ways of generating synthetic statements and there was no comparison to other synthetic data methods (STP, Magicoder), but it appears that the authors have found a particularly effective method for this.

### Weaknesses
No comparisons to other synthetic data generation methods.

### Questions
- Can you expand on the issue of <analysis> ?
- Why the statement modification loss and not simply freezing the statement (and punishing any modifications as fully incorrect)? Doesn't this give an optimal policy that is not intended? Why doesn't that happen in practice?

Nits:
eq 7: r_{ijk} needs a correctness term instead of 1?
Alg. 1: typo in line 7

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces GAR (Generative Adversarial Reinforcement Learning) for formal theorem proving in Lean. Unlike traditional expert iteration or RL for theorem proving with a fixed statement set, GAR dynamically generate new statements at the right difficulty level for the prover, forming an implicit curriculum. The method consists of a composer that generates new statements by fusing existing statements and a prover. Both components are trained with RL. The composer is incentivized to generate statements that are challenging but provable for the current prover. Experiments on MiniF2F and ProofNet shows improvements to the prover's capability.

### Strengths
* The paper addresses a critical bottleneck in current neural theorem provers, i.e., the reliance on a fixed and limited set of theorem statement. Having the model conjecturing and proving new statement is a promising direction. The design space here is quite big, and GAR represents one reasonable method in this space.
* Experiments show good improvements on MiniF2F and ProofNet.

### Weaknesses
* GAR is not the first attempt to augment the set of formal statements. For example, the scaffolded data synthesis in Goedel-Prover-V2 tries to generate simpler variants of hard problems and harder variants of simple problems. STP also generates new statements based on existing statements. The fuser in this paper is different from those methods, but it would be great if the authors can make the comparison more explicit. 
* MiniF2F is already saturated. SOTA methods are close to 100%. Experiments on PutnamBench would make this paper much stronger.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2
