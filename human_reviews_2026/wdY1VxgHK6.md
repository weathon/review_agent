# IFDECORATOR: Wrapping Instruction Following Reinforcement Learning with Verifiable Rewards

- Decision: Reject
- Scores: 6, 4, 6, 6, 2

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) improves instruction following capabilities of large language models (LLMs), but suffers from over-optimization where LLMs exploit verification shortcuts without aligning to the actual intent of user instructions. We introduce Instruction Following Decorator (IFDecorator), a framework that wraps RLVR training into a robust and sample-efficient pipeline. It consists of three components: (1) a cooperative-adversarial data flywheel that co-evolves instructions and hybrid verifications, generating progressively more challenging instruction-verification pairs; (2) IntentCheck, a bypass module enforcing intent alignment; and (3) trip wires, a diagnostic mechanism that detects reward hacking by injecting trap instructions to trigger and capture shortcut exploitation behaviors. Our Qwen2.5-32B-Instruct-IFDecorator achieves 87.43% accuracy on IFEval, outperforming larger proprietary models such as GPT-4o. Additionally, we demonstrate substantial improvements on FollowBench while preserving general capabilities. Our trip wires show significant reductions in reward hacking rates. We will release models, code, and data for future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes IFDecorator, a framework that augments Reinforcement Learning with Verifiable Rewards (RLVR) for instruction following. RLVR can still be prone to reward hacking, where satisfying instruction intent is bypassed. The authors introduce three synergistic components to mitigate reward hacking:

* Cooperative-Adversarial Data Flywheel – iteratively generates and filters instruction–verification pairs with difficulty control based on solver pass rates rather than constraint counts.
* IntentCheck – a “decorator” module that explicitly extracts the instruction’s core intent and verifies whether model responses fulfill it, mitigating over-optimization.
* Trip Wires – diagnostic trap instructions designed to quantify and analyze reward hacking tendencies.

### Strengths
* Comprehensive and credible evaluation: multiple model families, scales, and benchmarks.
* Addresses a genuine bottleneck—reward hacking in RLVR—more directly than previous “early-stop” fixes.
* Maintains or slightly improves general reasoning and code performance (GA stable).
* Practical utility: a clean wrapper for existing RLVR pipelines.

### Weaknesses
* Conceptual novelty modest: All three components have strong precedents (curriculum flywheels, intent verification, trap-based auditing). 
* Trip Wire coverage narrow: mostly format/placeholder exploits—ignores semantic or contextual gaming (e.g., misleading content that passes human-style checks).
* Evaluation bias: Many metrics depend on LLM-as-a-judge evaluators (potential leakage).
* Limited interpretability: The paper reports lower Hack Hit Rate but does not analyze why certain patterns decline—are models truly more aligned or just penalized for those forms?

### Questions
* How consistent are IntentCheck judgments when re-run with a different seed or model (e.g., Claude vs Qwen judge)?
* Do models trained with IntentCheck transfer to new constraint types not seen in training (e.g., temporal ordering)?
* How do results compare against mixing RLVR and RLHF rewards at equal compute (Pyatkin et al., 2025)?

### Soundness
3

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
3

### Summary
This paper proposes IFDecorator, a framework that augments Reinforcement Learning with Verifiable Rewards (RLVR) for instruction-following LLMs. The method addresses two known issues in RLVR4IF: (1) naive difficulty estimation and (2) reward hacking through verification shortcuts. IFDecorator introduces three synergistic components:

1. Cooperative-Adversarial Data Flywheel for co-evolving instruction-verification pairs with difficulty control.

2. IntentCheck, a bypass verifier that ensures alignment to the instruction’s intent.

3. Trip Wires, diagnostic probes that quantify reward hacking tendencies.
Experiments on IFEval and FollowBench demonstrate improved instruction-following ability and reduced reward hacking, with minimal degradation of general abilities.

### Strengths
1. Novelty and conceptual clarity:
The idea of wrapping RLVR with an intent-aware decorator and diagnostic tripwires is conceptually elegant and practically relevant. Unlike previous RLVR or RLHF hybrids, IFDecorator explicitly disentangles intent alignment from verification correctness, which directly targets reward hacking.

2. Strong motivation and connection to AI safety:
The work clearly situates itself within the literature on Goodhart’s Law and reward hacking, linking practical RLVR challenges to core safety concerns. This contextualization is rare and well justified.

3. Comprehensive framework:
The paper offers a complete system from data generation to evaluation. The cooperative-adversarial data flywheel for instruction evolution is an interesting extension of curriculum or self-play ideas.

### Weaknesses
1. Insufficient analysis of IntentCheck mechanism:
The paper doesn’t deeply analyze how IntentCheck extracts and represents “intent.” The prompt-based approach may risk circularity if the same LLM family is used for both generation and evaluation. A qualitative or error-type analysis of IntentCheck failures would strengthen claims about robustness.

2. Limited novelty in components:
While the integration is well-motivated, each individual part (data flywheel, intent verification, trap-based evaluation) draws on existing paradigms. The paper’s main contribution is more engineering synthesis than a new algorithmic principle. Some reviewers might find the “decorator” framing slightly overstated.

3. Trip Wires evaluation scope:
The diagnostic captures only a few exploit types (placeholders, repetition, formatting). With 37.5% recall, it may underestimate hacking frequency. The paper could discuss scalability of Trip Wires to more nuanced or semantic exploit behaviors.

4. Comparisons and baselines:
Although comparisons to UltraIF and VerIF are included, it’s unclear how hyperparameters, dataset sizes, and judge strengths were normalized. More transparent cost–performance comparisons would help, especially versus recent open-source RLVR variants.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes IFDecorator, a framework that wraps reinforcement learning with verifiable rewards for instruction following (RLVR4IF) to make training more sample-efficient and to mitigate reward hacking. The system combines (1) a cooperative–adversarial data flywheel that evolves instructions by empirical difficulty using measured pass rates, (2) an IntentCheck module that verifies whether model responses satisfy the core intent of instructions, and (3) Trip Wires that expose and quantify reward-hacking behavior. On benchmarks such as IFEval and FollowBench, IFDecorator improves adherence to instruction semantics and reduces exploitative behaviors, particularly in self-alignment settings using large open-source models like Qwen2.5-32B.

### Strengths
- Addresses a concrete and increasingly relevant issue through a simple, modular solution that integrates easily into existing training pipelines.

- Replaces naive constraint-counting with pass-rate–driven adaptive filtering, a principled way to balance challenge and solvability in evolved datasets.

- IntentCheck enforces semantic fidelity beyond rule-level correctness, and Trip Wires provide a tangible diagnostic for detecting reward hacking.

- Consistently improves instruction-following performance while reducing reward hacking, with the self-alignment experiment demonstrating the potential of verifiable self-judging.

### Weaknesses
- Relies on an LLM-driven EXTRACTINTENT step to decompose each instruction into intent, context, input, and constraints, but this process is not validated for accuracy or consistency (e.g., no human agreement or inter-run checks). The paper shows downstream effectiveness (IntentCheck lowers hacking) but does not establish IntentCheck reliability as a decomposition method.

- The cooperative–adversarial flywheel depends on empirical pass rates and fixed thresholds (e.g., $\tau_\text{low}=0.0$, $\tau_\text{high}=0.5$) to classify tasks as too easy, too hard, or acceptable, yet no per-iteration analysis, sensitivity study, or visualization of pass-rate dynamics is provided. The only related ablation disables difficulty control, leaving threshold tuning unexplored.

- IntentCheck and several evaluation components rely on LLM judges—primarily Qwen2.5-32B-Instruct (with a 7B variant) for judging, plus GPT-4o for FollowBench open-ended scoring and for discovering reward-hacking patterns, so cross-judge or non-LLM verifiers are still not demonstrated, and robustness to genuinely different verifiers remains unclear.

- The Trip Wires detector is tuned for high precision (93.5%) but has low recall (37.5%), so it likely undercounts reward hacking. Broader pattern coverage and human correlation would strengthen the claim.

- The pipeline leans heavily on Qwen2.5-32B as the judge/data synthesizer. A 7B variant works but reduces GA, and cheaper or smaller verifiers (e.g., Llama-2-13B) are not analyzed in depth, raising questions about reproducibility and generality.

### Questions
- How accurate/reliable is the LLM-based EXTRACTINTENT decomposition? Do you have any human agreement or consistency checks to support IntentCheck, beyond the effectiveness ablation?

- How do pass rates evolve across flywheel iterations with the chosen thresholds (e.g., $\tau_\text{low}=0.0$, $\tau_\text{high}=0.5$)? Is the method sensitive to these thresholds?

- Since most judging uses Qwen2.5-32B (and a 7B variant) and GPT-4o is used for some evaluation, how robust are the results to alternative, weaker, or non-LLM verifiers?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes IFDecorator, a wrapper around RL with verifiable rewards for instruction following. It has three parts: a cooperative-adversarial data flywheel to curate hard but solvable instruction-verification pairs, IntentCheck to verify core intent beyond surface constraints, and Trip Wires to diagnose reward-hacking via trap instructions and a hack hit rate metric. Experiments show higher IFEval (e.g., 87.43% for Qwen2.5-32B-Instruct-IFD) and better FollowBench, while reducing hacking tendencies, with human study supporting Trip Wires precision.

### Strengths
- The idea of decorating RLVR with intent verification plus independent diagnostics is quite original and practical. IntentCheck directly targets the gap between constraint satisfaction and intent fulfillment, and Trip Wires formalize hack probing with HHR.
- The paper is well-written with clear binary reward formulation and careful hybrid verification. The motivating examples and framework figure make the failure modes and fixes easy to grasp.
- The paper conducts extensive experiments with multiple ablations and a human study.

### Weaknesses
- IntentCheck and soft-criteria rely on a judge model. More cross-judge validation would be stronger. 
- For Trip Wires, human eval shows high precision but only 37.5% recall.
- Although Trip Wires are training-independent, repeated evaluation could still invite Goodhart effects

### Questions
- How does IFDecorator perform if Trip Wires cover new patterns unseen during development?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising approach to enhance instruction following capabilities of large language models (LLMs), but it suffers from over-optimization where LLMs exploit verification shortcuts without aligning to the actual instruction intent. We introduce Instruction Following Decorator (IFDecorator), a framework that wraps RLVR for instruction following into a sample-efficient and robust pipeline. It consists of three components: a cooperative-adversarial data flywheel that co-evolves instruction-verification pairs, generating progressively challenging training samples; IntentCheck, a bypass module that circumvents verifications and directly assesses whether LLM responses align with instruction intent; and Trip Wires, a novel diagnostic tool using strategically designed trap instructions to quantify and capture exploitation behaviors. Extensive experiments validate our approach, with our Qwen2.5-32B-Instruct model achieving 87.43% accuracy on IFEval, outperforming larger models like GPT-40, while human evaluation confirms Trip Wires achieve high precision in detecting genuine hacking. Crucially, Trip Wires show our method significantly reduces reward hacking tendencies and generalizes across different model architectures and scales. We will release models, code, and data for future research.

### Strengths
1. This paper designs a framework called IFDecorator that successfully addresses the long-standing challenge of gauging instruction difficulty by leveraging a cooperative-adversarial flywheel.

2. It proposes the IntentCheck and Trip Wires methods, which effectively mitigate over-optimization and reward hacking in RLVR4IF tasks; I find this direction a particularly interesting angle for RLVR-based instruction-following alignment.

3. Models trained with the approach demonstrate good results across a wide range of parameter scales.

### Weaknesses
1. My central concern is generalization. IFEval and FollowBench consist largely of verifiable instructions, offering limited evidence that the approach will generalize to real-world instructions—especially restrictive role-play prompts that are hard to verify. This raises substantial doubts about the method’s effectiveness on non-verifiable instructions. In addition, several challenging instruction-following benchmarks—such as ComplexBench, Multi-IF, FoFobench and InfoBench—are not covered.

2. A second major concern is the overlap between the paper’s instruction/verification evolution process and prior work like AUTOIF, which weakens the novelty. Moreover, the “instruction evolution” relies heavily on the thresholds τ_low and τ_high; the resulting difficulty seems highly sensitive to these empirical settings, and the paper lacks fine-grained experiments to justify them.

3. Experimentally, the setup mirrors AUTOIF but omits direct comparisons with key baselines (AUTOIF, UltraIF, Conifer, etc.), which is inadequate. Reviewing recent instruction-following papers, I did not see this method establishing a clear performance advantage, which casts doubt on its true contribution. Finally, using only Qwen2.5-32B-IT for the self-alignment setting is insufficient to demonstrate the effectiveness of self-alignment.

### Questions
See Weakness and following questions.

1. The paper should spell out in detail how it differs from closely related work such as AUTOIF and UltraIF, and it should include direct performance comparisons with those baselines.

2. It is unclear how much overlap there is between IntentCheck and the hybrid verification scheme. Is it really necessary to run both checks for every query?

3. The use of Trip Wires in RL training needs clarification. If Trip Wires do not affect the reward, do they actually influence the training process? The current exposition is not sufficiently clear.

### Soundness
2

### Presentation
2

### Contribution
2
