# Let's Explore Step by Step: Generating Provable Formal Statements with Deductive Exploration

- Decision: Accept (Poster)
- Scores: 6, 2, 4

## Abstract
Mathematical problem synthesis shows promise in resolving data exhaustion, contamination, and leakage for AI training and evaluation. Despite enormous efforts,
an **expressiveness-validity-complexity trilemma** remains an open question. Existing methods either lack whole-process verifiability, are constrained to a particular domain, or are bounded by external models.
This paper breaks the trilemma by proposing the framework of **DExploration** _(**D**eductive **Exploration**)_, which formulates problem synthesis as a step-by-step exploration process instead of one-shot generation. Agents are equipped with three simple yet powerful atomic actions: _introducing_ variables/hypotheses, _deducing_ new facts, and _submitting_ derived facts. The entire exploration process is formally verified by Lean 4, which encompasses most mathematical domains up to the research level.
Once a conclusion is submitted, the framework outputs a formal statement with guaranteed provability, reducing the need for external models.
To bootstrap training data for DExploration, we propose **Exploratory Transformation** to distill exploration trajectories from existing large-scale theorem-proving data. It rewrites formal proofs into a deductive style, parses dependencies among variables, hypotheses, and proof steps, then reassembles them into exploration trajectories by a topological order.
Experiments validate the effectiveness and efficiency of our methods, achieving an improved success rate ($40.70\\% \mapsto 54.52\\%$), reduced token cost ($52.9\text{K} \mapsto 8.8\text{K}, 83\\%\downarrow$), broader complexity and difficulty distributions, and Pareto optimality.
In $2726$ valid generations, three state-of-the-art provers fail on $60$ (Pass@4) and $8$ (Pass@64).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
I think the authors did a genuinely impressive job with this paper. Let’s Explore Step by Step tackles one of the hardest open challenges in AI4Math: how to synthesize new, provably valid mathematical problems without relying on fragile external provers or autoformalizers. The proposed framework, DExploration, is both elegant and well-motivated. Instead of one-shot problem generation, the model explores mathematics step by step ie introducing variables, deducing facts, and finally submitting a conclusion — all grounded within the Lean 4 proof assistant.

This design ensures whole-process verifiability: every generated problem is backed by a valid proof trace, eliminating the need for an external verifier. I especially like how the authors handle data scarcity through Exploratory Transformation, which rewrites existing proofs into “exploration trajectories” via dependency graphs and topological ordering. It’s a clever and principled way to bootstrap training data from existing Lean corpora.

### Strengths
I think the authors have really nailed something important here with the core idea of DExploration. Instead of trying to generate entire mathematical statements in one shot, they've broken it down into a natural exploration process—introducing variables, deducing facts, and building up to conclusions step-by-step. This feels much more aligned with how mathematicians actually work, and the fact that every single step is verified in Lean 4 is brilliant. It completely sidesteps the validity issues that plague other approaches. The framework doesn't just generate statements; it simultaneously constructs their proofs, which means no need for external provers that might fail on harder problems. That's a genuinely clever solution to what the authors call the "expressiveness-validity-complexity trilemma," and I find it both theoretically elegant and practically useful.

The experimental work is really thorough and convincing. Good job on testing across so many different dimensions; they don't just show their method works, they show it works *better* in multiple ways simultaneously. The 83% reduction in token costs alone is impressive, but combine that with higher success rates (40.70% to 54.52%), better complexity distributions, and harder problems that actually stump current state-of-the-art provers? That's strong evidence this approach is onto something. I particularly appreciate how they've evaluated both formal and informal reasoning, showing the generated problems are harder than AIME24 when translated to natural language. The ablation studies are well-designed too they clearly demonstrate that both the exploration framework and the exploratory transformation pipeline are pulling their weight. The Pareto optimality results across different computational budgets show this isn't just good at one operating point but scales gracefully.

### Weaknesses
My main concern is around how much this approach depends on having really good formal proof data to begin with. The exploratory transformation is clever, but it needs high-quality theorem-proving datasets like NuminaMath-Lean to work. I'm wondering what happens when you want to explore mathematical areas that aren't well-covered in existing formal libraries does the quality drop off? The authors don't really dig into this limitation much. It would've been helpful to see some analysis of how the method performs across different mathematical domains, especially ones with sparser coverage in Mathlib. Also, while they show the method can generate complex problems, there's not much discussion about controlling *what kind* of complexity you get. If you're building educational tools or want problems at specific difficulty levels, you'd need some way to steer the generation, and that's not really addressed here.

Another thing that's bugging me is the question of whether these generated problems are actually *interesting* from a mathematical perspective, not just formally valid. Sure, they're provable and they're complex, but are they elegant? Are they the kind of problems a human mathematician would find worthwhile? The explosion check prevents contradictions, which is great, but it doesn't guarantee the problems aren't trivial or weirdly artificial. The authors mention 97% correctness on a human evaluation of informalized problems, but that's just checking if the translation is accurate not whether the underlying problem is mathematically meaningful or pedagogically useful. I also would've liked to see actual wall-clock time comparisons, not just token counts. The multi-step exploration process involves lots of sequential calls to Lean 4, and I'm curious whether the latency becomes a bottleneck in practice compared to one-shot generation methods. These aren't dealbreakers, but they're gaps that future work should probably address.

### Questions
Some questions that I had,

1. **Domain Coverage and Generalization**: How does DExplorer perform on mathematical domains that have sparse coverage in Mathlib? Have you tested the method on areas like algebraic topology or category theory where formal libraries are less developed? I'm curious if the exploration quality degrades noticeably or if the framework is robust to these gaps.

2. **Controlling Problem Difficulty**: Is there a way to guide the exploration toward specific difficulty levels or complexity targets? For educational applications, it would be really useful to generate, say, "calculus problems suitable for first-year undergraduates" versus "research-level analysis problems." Have you experimented with any conditioning mechanisms beyond just adjusting the step limit?

3. **Mathematical Interestingness**: Beyond formal validity, how do you assess whether generated problems are mathematically interesting or elegant? Have you considered getting feedback from professional mathematicians on whether these problems would be considered "good" problems? I'm thinking about the difference between a problem that's technically correct versus one that reveals some deeper insight or has pedagogical value.

4. **Failure Mode Analysis**: The 3.17% of statements that fail the final typecheck is interesting. What patterns do these failures follow? Are they concentrated in certain mathematical domains or with specific types of tactics? Understanding what goes wrong could really help improve the framework.

5. **Computational Efficiency in Practice**: You show impressive token cost reductions, but what about actual wall-clock time? How does the latency of making sequential Lean 4 calls compare to one-shot generation followed by proving? Is there a sweet spot in terms of step limits where you get the best quality-to-time tradeoff?

6. **Scaling the Training Data**: The exploratory transformation works great on existing proof data, but could you bootstrap the process? For example, could problems generated by DExplorer be fed back as training data after human verification, creating a self-improving cycle?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce DExploration, a step-by-step problem generation framework grounded in Lean 4, designed to scale synthetic mathematical data generation while ensuring provability. This approach offers a principled solution to the trade-off between scalability and formal correctness in mathematical reasoning datasets. Their empirical results show notable improvements in proof success rates, problem difficulty, and efficiency (token cost), though the study’s evaluation setup and comparison baselines raise questions about the robustness and fairness of the reported gains.

### Strengths
- Casting problem synthesis as an exploration process with three atomic actions: Introduce, Deduce, and Submit, is very simple and realistic, which closely mirrors the workflow of mathematicians, as well as a good way to approach the trilemma mentioned.
- Reconstructing a formally checkable Lean 4 statement and proof from the agent’s exploration trajectory ensures validity in a principled, rather than heuristic way.
- The Exploratory Transformation procedure (deductive rewriting → dependency graph → topological ordering) provides a sensible means to convert existing proofs into trajectories for supervised fine-tuning. The depth-prioritised reassembly is intuitively well-founded.
- The paper reports substantial improvements in proof success and validity rates, alongside major reductions in token cost, which could be considered the main contribution of the paper. The generation of novel statements unprovable by SOTA provers further highlights the method’s potential (though if the results are robust).

### Weaknesses
- Lean tactics are tricky and very dependent on context. Common ones like linarith, nlinarith, apply, constructor, and cases can behave unpredictably in different settings. The authors note themselves (around line 1007) that a formal treatment of “deductive tactics” is still future work, and that some tactics can occasionally break the intended constraints. This makes the “guaranteed provability” claim a bit weaker than stated. Without a machine-checkable proof that the tactic set always preserves the invariants in their lemmas, the claim remains conditional. The reported 96.83% success rate is impressive, but it also shows that the guarantee isn’t absolute yet.
- The method relies on heuristic or one-off checks, which are inherently probabilistic; some contradictions could slip through, and some valid but complex cases might get rejected. The paper mentions this, but doesn’t go into detail about how often such cases occur or how sensitive the results are to the choice of prover.
- The manual evaluation covers 100 samples, with 97 marked correct. That’s a good sign, but the sample is quite small compared to the 2,726 valid generations and 39k training proofs. Also, the evaluation only checks for correctness, not for clarity or difficulty of the generated proofs.
- The statement that “DExploration eliminates reliance on external provers and achieves fully provable generation” feels too strong. The system still depends on Lean, Aesop, and LLM-based provers for verification, and about 3% of outputs fail the final check. It would be fairer to say that it largely reduces external dependencies rather than eliminating them.
- Finally, the paper doesn’t include a limitations section. It would help the overall credibility to acknowledge open issues; such as tactic stability, evaluation scope, or remaining failure modes.

### Questions
On tactic reliability:
- How do the authors ensure that all Lean tactics used (e.g., linarith, nlinarith, apply, constructor, cases) preserve the invariants required by their proofs?
- Could they provide a more formal or machine-checkable justification that the selected tactic set does not violate the provability constraint?
- Are there examples where tactic behavior led to incorrect or unverifiable results?

On heuristic verification:
- The approach seems to rely on heuristic or single-sample checks. Have the authors quantified how often these checks fail to catch contradictions or reject valid cases?
- How sensitive are the reported results to the choice of prover or verification strategy?

On evaluation design:
- Why was a sample of 100 items chosen for manual evaluation, given the scale of generated proofs?
- Could the authors provide metrics beyond binary correctness—such as clarity, ambiguity, or problem difficulty?
- How representative is the 100-sample subset of the full set of 2,726 valid generations?

On the claim of “eliminating” external provers:
- Since the system still depends on Lean, Aesop, and LLM-based provers for validation, could the authors clarify what they mean by “eliminates reliance”?
- Would it be more accurate to describe the approach as “reducing” rather than “eliminating” dependency on external provers?

On limitations and open issues:
- Could the authors discuss current limitations, such as tactic reliability, scalability, or remaining failure modes?
- What are the main barriers to achieving a fully formalised notion of “deductive tactics”?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DExploration, a Lean 4–based framework that reformulates mathematical problem synthesis as a step-by-step deductive exploration process.
Rather than directly generating conjectures or proofs, DExploration enables an agent to verifiably explore the mathematical environment through three atomic actions—Introduce, Deduce, and Submit—thereby generating both a formal statement and its proof in one unified loop.

### Strengths
1. The paper makes good contribution by reframing theorem synthesis as verifiable exploration. The Exploratory Transformation pipeline that reuses existing theorem–proof data to construct fine-grained exploration trajectories is a natural and scalable way to create augmented theorem proving dataset.

2. The authors perform a extensive ablation study that clearly supports their design choices, and the observed token efficiency improvement is both large and intuitively meaningful.

### Weaknesses
My main concern is the lack of external evaluation on public benchmarks. The experiments primarily analyze synthetic metrics (success rate, token cost, validity) on self-generated data. It remains unclear whether training on Exploratory Transformation data improves downstream theorem proving performance on public benchmarks such as MiniF2F, ProofNet or PutnamBench. Metrics such as “complexity” and “difficulty” are proxy measures based on proof length and model accuracy; these are helpful but insufficient for assessing real mathematical value.

### Questions
1. Have the authors tried fine-tuning a baseline prover (e.g., Goedel-Prover or DeepSeek-Prover) on the generated trajectories and evaluating it on public Lean benchmarks?
2. Could the authors clarify which sources (Mathlib, NuminaMath, etc.) were used to generate the training data, and how they ensured no overlap with test theorems?

### Soundness
2

### Presentation
3

### Contribution
3
