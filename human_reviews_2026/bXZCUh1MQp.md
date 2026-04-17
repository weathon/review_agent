# How Much Backtracking is Enough? Exploring the Interplay of SFT and RL in Enhancing LLM Reasoning

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Recent breakthroughs in large language models (LLMs) have markedly advanced their reasoning progress through two broad post‑training paradigms: supervised fine‑tuning (SFT) and reinforcement learning (RL), particularly on mathematical and logical problems that have verifiable answers. Prior research indicates that RL effectively internalizes search strategies, enabling long chain-of-thought (CoT) reasoning, with backtracking emerging naturally as a learned capability. However, the precise benefits of backtracking—specifically, how significantly it contributes to reasoning improvements and the optimal extent of its use—remain poorly understood. In this work, we systematically investigate the dynamics between SFT and RL on eight reasoning tasks: Countdown, Sudoku, Arc 1D, Geometry, Color Cube Rotation, List Functions, Zebra Puzzles, and Self Reference. Our findings highlight that short CoT sequences used in SFT as a warm-up do have moderate contribution to RL training, compared with RL without any SFT warm-up; however such contribution diminishes when tasks become increasingly difficult. Motivated by these observation, we introduce a backtracking‑centric training recipe. By synthetically varying the number of explicit backtracking steps in the SFT warm‑up, we show that (i) longer CoTs containing backtracks stabilize and amplify RL, and (ii) the optimal backtrack depth scales with task difficulty—zero for Arc 1D, one for Countdown, and five for Sudoku—yielding up to a 28.9\% absolute accuracy boost at the 3B parameter scale. Collectively, our controlled experiments provide concrete guidance for constructing training mixtures that reliably push LLM reasoning beyond current boundaries.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This is an analysis paper investigating the effect of SFT before RL in an RLVR setting, covering 8 tasks using the Qwen2.5-3B-Instruct as the base model. The paper reveals several facts as follows:

1.  Is a warm-up stage necessary for RL?
    * Yes, an SFT warm-up stage is necessary for RL.

2.  What kinds of SFT warm-up matter to RL?
    * Correctness of solutions: This does not matter.
    * Shuffled SFT (SFT on trajectories shuffled with other problems' trajectories): This is important.
    * SFT on synthetic trajectories with explicit backtracking: This is particularly beneficial for more challenging tasks that require deeper backtracking, helping to maximize gains.

### Strengths
1.  Investigating the effect of SFT before RL in an RLVR setting is important.

2.  The tasks in the experiments are diverse, although they are toy tasks.

### Weaknesses
1.  The investigation is based on the Qwen2.5-3B-Instruct model. The limited model family and model scale limit the applicability of the conclusion. Moreover, the authors' reference to Qwen2.5-3B-Instruct as not having an SFT warm-up is a bit misleading, as an instruct model has already undergone the SFT stage. Including the base model is also needed.

2.  The tasks are mainly toy tasks, which limits the application of the conclusion. For example, the authors clearly state the needed optimal backtracking step for various tasks. But for tasks like MATH/AIME, it is unclear how to calculate the backtracking step and whether it will show clear trends.

3.  The conclusions the authors derived have already been included in related work; for example, backtracking [1], the warm-up stage for RL [2], and the model scale and CoT context [3].

[1] Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs, in COLM 2025

[2] Demystifying Long Chain-of-Thought Reasoning in LLMs, in ICLR 2025 Workshop SSI-FM,

[3] LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!, in arxiv 2025

### Questions
* The study's conclusions are drawn exclusively from Qwen2.5-3B-Instruct. How can the authors substantiate that these findings generalize beyond this specific model family and, more importantly, to larger-scale models (e.g., 7B, 70B) which may exhibit different behaviors?


*  The paper's findings, particularly regarding the "optimal backtracking step," are derived from toy tasks where such parameters are well-defined. How would this methodology translate to complex reasoning tasks like MATH or AIME

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how supervised fine-tuning (SFT) and reinforcement learning (RL) interact to enhance LLM reasoning, focusing specifically on the role of backtracking. Using Qwen2.5-3B across eight reasoning tasks (Countdown, Sudoku, Arc 1D, etc.), the authors compare three training approaches: (1) cold-start RL without SFT, (2) RL with short CoT warm-up from self-sampling, and (3) RL with synthetically constructed trajectories containing explicit backtracking steps.

The paper's main claim is that **optimal backtracking depth scales with task difficulty**—zero backtracks for Arc 1D (easy), one for Countdown (medium), and five for Sudoku (hard)—achieving up to 28.9% accuracy improvement. Additional findings include: short CoT warm-ups provide moderate gains over cold-start RL (though diminishing on harder tasks), incorrect trajectories can still benefit RL training, and RL is sensitive to shuffled training data.

**Contributions:** (1) empirical analysis of SFT warm-up effects on RL, (2) a backtracking-centric training recipe with task-difficulty-dependent backtrack depths, and (3) guidance for constructing training mixtures to improve reasoning capabilities.

### Strengths
**Originality:**
The paper provides a systematic empirical investigation of backtracking's role in RL post-training through controlled synthetic data construction. The use of depth-first search and heuristic methods to generate trajectories with precise backtracking counts (0, 1, 2, 3, 5, 10) offers a methodologically clean approach to studying this phenomenon. The finding that incorrect trajectories can benefit RL training (Section 5.1) adds an interesting dimension to understanding SFT-RL interactions.

**Quality:**
The experimental setup shows methodological rigor: (1) consistent use of Qwen2.5-3B across experiments enables fair comparisons, (2) evaluation spans eight reasoning tasks providing diversity, (3) rule-based rewards ensure reproducibility, and (4) inclusion of multiple training paradigms (cold-start RL, self-sampled SFT+RL, synthetic backtracking) allows for systematic comparison. The shuffled data ablation (Section 5.2) demonstrates attention to understanding what drives performance.

**Significance:**
The work contributes to understanding SFT-RL interactions for reasoning model post-training, a relevant topic given recent developments (DeepSeek-R1, O1). The controlled experimental approach and attention to ablations (incorrect trajectories, shuffled data) provide empirical insights into training dynamics that may inform future work in this rapidly evolving area.

### Weaknesses
**W1: Insufficient Motivation for Focusing Solely on Backtracking (Lines 92-93, Section 1)**

The paper abruptly introduces backtracking as the primary research focus without adequately justifying why this specific behavior merits exclusive attention over other potentially important reasoning patterns. Modern reasoning models exhibit multiple cognitive behaviors beyond backtracking, including:
- Multiple verification strategies  
- Alternative solution exploration

The manuscript lacks a principled analysis or empirical evidence demonstrating that backtracking is the dominant or most impactful behavior for reasoning improvements. This gap weakens the motivation for the entire study. The authors should either: (a) provide comparative analysis showing backtracking's relative importance, or (b) reframe the work as an investigation of one among several important reasoning behaviors.

**W2: Limited Novelty of Core Findings (Line 118, Abstract)**

The central claim that "initializing with synthetic SFT data containing an appropriate number of backtracks—matched to the difficulty of the task—consistently pushes the model beyond its base reasoning capabilities" lacks sufficient novelty given the current literature landscape (late 2025). Multiple recent works have established similar insights about the relationship between training data characteristics, task difficulty, and RL effectiveness. The authors should:
- Clearly articulate what is genuinely novel beyond existing findings
- Consider repositioning the contribution as a systematic empirical validation rather than a novel discovery

**W3: Undefined "Short CoT" Concept (Lines 251-263, Section 3.2)**

The paper introduces "short CoT" as a central experimental condition but never provides operational definitions or quantitative criteria. Critical missing specifications include:
- Token length thresholds
- Number of reasoning steps
- Comparison baseline (short relative to what?)
- Whether "short" is task-dependent

This ambiguity makes it impossible to reproduce the experiments or understand what precisely differentiates Section 3 from Section 4. The authors must provide explicit, quantitative definitions for all CoT categories used in the study.

**W4: Flawed Logic in Sudoku Analysis (Lines 268-269, Section 3.2)**

The conclusion that "tasks like Sudoku still posed substantial challenges, highlighting the limitations of short CoTs for deeper, combinatorial reasoning" is logically unsound given the experimental results:

*Problem A - Overly Assertive Conclusion:* Figure 3 shows both base model and pure RL achieve ~0% on Sudoku. Since pure RL itself fails to improve performance, attributing the failure to "limitations of short CoTs" is premature without ruling out alternative explanations.

*Problem B - Unexamined Alternative Hypotheses:*
1. The base model may simply lack fundamental capabilities for Sudoku regardless of training approach
2. The issue may be trajectory *quality* rather than trajectory *length* - if pure RL generates incorrect solutions, the sampled trajectories are fundamentally flawed, not merely short

*Problem C - Data Collection Paradox:* The authors claim to obtain "8 RL'ed models initialized from correct SFT, and the other 8 initialized from incorrect SFT" (Section 3.2). However, if pure RL achieves ~0% accuracy on Sudoku, how were sufficient correct trajectories obtained? The only plausible explanation (sampling easier problems) creates a train-test distribution mismatch that itself could explain poor performance.

This confounding undermines the paper's narrative about CoT length being the key factor.

**W5: Confounded Variables - CoT Length vs. Backtracking Presence (Section 4)**

The paper's core experimental manipulation conflates two variables simultaneously:
- Introduction of backtracking steps
- Increase in overall CoT length

When comparing 0-backtrack vs. 5-backtrack conditions, the trajectories differ in both backtracking behavior AND total token count. The experimental design lacks critical ablations to disentangle these factors:

*Missing Control 1:* Long CoTs without backtracking (verbose reasoning with more detailed steps but no corrections)

*Missing Control 2:* Short CoTs with backtracking (concise reasoning with efficient error correction)

Without these controls, the claimed benefits of backtracking could simply reflect giving the model more tokens/steps to reason, rather than the specific cognitive pattern of error detection and correction. This fundamentally undermines the paper's central claim about backtracking being the critical factor.

**W6: Contradictory Chapter Framing (Section 4 Title vs. Arc 1D Results)**

The chapter title "PUSHING BEYOND BOUNDARIES WITH BACKTRACKING" directly contradicts the Arc 1D findings (lines 327-329), where the authors explicitly state "Using zero backtrack is optimal." This contradiction reveals several issues:

1. The chapter title overgeneralizes findings that only apply to certain task types
2. The manuscript lacks nuanced discussion of *when* backtracking helps vs. harms
3. Arc 1D's "easy" classification and preference for zero backtracking actually supports a more sophisticated finding (simple tasks don't benefit from backtracking while complex ones do), but this insight is inadequately emphasized

The framing should be revised to accurately reflect that backtracking benefits are task-conditional, not universal.

**W7: Unjustified Causal Claims - Task Difficulty vs. Task Type (Lines 375-376)**

The conclusion that "optimal backtracking scales with task difficulty" commits a classic confounding error. The experimental design varies two factors simultaneously:
- **Task type:** Arc 1D (pattern recognition) vs. Countdown (arithmetic search) vs. Sudoku (constraint satisfaction)
- **Task difficulty:** Easy vs. Medium vs. Hard

The observed differences in optimal backtracking (0 for Arc, 1 for Countdown, 5 for Sudoku) could be driven by intrinsic task structure rather than difficulty:
- Arc's preference for no backtracking may reflect that grid transformations benefit from direct pattern matching
- Sudoku's need for extensive backtracking may reflect its nature as a constraint satisfaction problem requiring systematic search

**Missing Experiment:** To validly claim difficulty scaling, the authors must control task type and vary only difficulty (e.g., easy/medium/hard Sudoku instances) to observe whether optimal backtracking increases monotonically with difficulty within the same task family.

The current evidence only supports: "Different task types require different amounts of backtracking," which is far weaker than the claimed finding.

**W8: Unclear Relationship Between Section 3 and Section 4**

The manuscript fails to clarify whether Section 3's "short CoT" data contains any backtracking steps. This creates fundamental confusion:

*If short CoTs contain backtracking:* What differentiates Section 4's contribution? Is it merely a matter of degree?

*If short CoTs contain no backtracking:* Why not? Is this because:
- Cold-start RL naturally produces no backtracking?
- The authors filtered backtracking examples?

This missing information obscures the logical progression of the paper and makes it difficult to understand what each experimental section uniquely contributes.

### Questions
**Q1: Clarification on "Short CoT" Definition**
Please provide explicit, quantitative definitions for "short CoT" used in Section 3:
- What is the token length range?
- How many reasoning steps do these trajectories typically contain?
- Do any short CoT examples contain backtracking steps? If so, what proportion?
- What differentiates "short CoT" from the synthetic backtracking data in Section 4 beyond the number of explicit backtrack operations?

**Q2: Data Collection for Correct Trajectories in Low-Performance Regimes**
In Section 3.2, you report collecting correct and incorrect trajectories from cold-start RL models. However, Figure 3 shows that cold-start RL achieves ~0% accuracy on Sudoku. Please explain:
- How were sufficient correct trajectories obtained for Sudoku when the model succeeds on virtually no test instances?
- Were training problems sampled from a different (easier) distribution than test problems?
- If so, how might this train-test distribution mismatch affect the conclusions about short CoT effectiveness?

**Q3: Disentangling CoT Length from Backtracking**
Your Section 4 results show performance improvements when adding backtracking, but these modifications also increase total trajectory length. Please clarify:
- Have you conducted ablations with length-matched comparisons (e.g., verbose zero-backtrack CoTs with the same token count as 5-backtrack CoTs)?
- Can you provide evidence that backtracking specifically (not just longer reasoning) drives the improvements?
- What proportion of the performance gain is attributable to backtracking behavior versus simply having more computation/tokens?

**Q4: Task Type vs. Task Difficulty**
Your conclusion states that "optimal backtracking scales with task difficulty" (lines 375-376), but your experimental tasks differ in both type and difficulty simultaneously. Please address:
- Have you evaluated optimal backtracking within a single task family at varying difficulty levels (e.g., easy/medium/hard Sudoku)?
- Could the observed pattern (0 for Arc, 1 for Countdown, 5 for Sudoku) be better explained by task structure (pattern recognition vs. arithmetic vs. constraint satisfaction) rather than difficulty per se?
- What specific evidence supports difficulty as the primary factor?

**Q5: Backtracking in Section 3 Trajectories**
Please clarify the presence/absence of backtracking in your Section 3 experiments:
- Do the "short CoT" trajectories from self-sampling contain any backtracking steps?
- If no, is this because cold-start RL naturally doesn't produce backtracking, or did you filter such examples?
- If yes, approximately how many backtracking steps appear, and how does this compare to Section 4's synthetic data?

**Q6: Alternative Explanations for Sudoku Failure**
Given that pure RL achieves ~0% on Sudoku, how do you rule out that:
- The base model fundamentally lacks required capabilities (independent of training approach)?
- The failure is due to poor trajectory quality (incorrect reasoning patterns) rather than insufficient trajectory length?
- Section 3's conclusion specifically isolates CoT length as the limiting factor rather than these alternatives?

**Q7: Generalization of "Pushing Beyond Boundaries"**
Section 4's title suggests backtracking universally helps, yet Arc 1D performs best with zero backtracking. Please discuss:
- Under what conditions does backtracking help vs. hurt performance?
- Can you provide a principled framework for predicting which tasks benefit from backtracking?
- Should the claim be revised to acknowledge task-conditional benefits?

**Q8: Why Focus Exclusively on Backtracking?**
Please provide justification for studying backtracking to the exclusion of other reasoning behaviors:
- Have you analyzed the relative frequency of backtracking vs. other patterns (verification, exploration) in strong reasoning models?
- Is there empirical evidence that backtracking is the most impactful behavior to study?
- Or should this work be positioned as one investigation among several needed to understand different reasoning patterns?

### Soundness
2

### Presentation
2

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
This paper investigates the interaction between supervised fine-tuning (SFT) and reinforcement learning (RL) in enhancing large language model (LLM) reasoning abilities, particularly focusing on how backtracking behaviors contribute to reasoning performance. Through controlled experiments on eight reasoning tasks (e.g., Countdown, Sudoku, Arc 1D, Zebra Puzzles), the authors find that short chain-of-thought (CoT) SFT can moderately improve RL outcomes, but introducing synthetic backtracking traces during SFT produces more significant and stable gains—especially for more difficult, combinatorial reasoning problems. The paper proposes that the optimal number of backtracking steps scales with task difficulty, offering empirical guidance for constructing effective SFT+RL training mixtures

### Strengths
- The paper carefully explores the effects of different SFT warm-up strategies (no-SFT, self-sampled, synthetic backtracking, shuffled) on RL training, providing clear empirical comparisons 

- The tasks, metrics, and model configurations (based on Qwen2.5 family) are clearly described, and synthetic datasets are constructed in a principled way using DFS or heuristic search.

### Weaknesses
- Task scope is narrow: The evaluated tasks are mostly puzzle-style logical reasoning (Countdown, Sudoku, etc.), which limits generalizability to other forms of reasoning like mathematical proofs, symbolic integration, or commonsense reasoning.

- Limited model diversity: Experiments rely almost entirely on the Qwen2.5 family; it’s unclear whether the findings hold for other architectures (e.g., Llama).

- The general idea of combining SFT warm-up with RL is well-trodden; the specific contribution—varying backtracking depth—is interesting but may be viewed as an empirical refinement rather than a conceptual leap.

### Questions
No.

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
This work systematically analyzes the interplay between supervised fine-tuning (SFT) and reinforcement learning (RL) in enhancing reasoning abilities of LLMs, showing that short chain-of-thought (CoT) sequences moderately benefit RL warm-starts but their impact diminishes on harder tasks. By constructing synthetic datasets with controlled backtracking steps, experiments reveal that longer CoTs with backtracks improve RL training stability and effectiveness, and more challenging tasks require more backtracking during SFT. Results further show that RL training is largely insensitive to the correctness of long CoTs, focusing instead on structural patterns, providing practical guidance for optimizing LLM reasoning training strategies.

### Strengths
1. Very Good Research Question: How SFT & RL affects LLM reasoning
2. Good Experiment settings on SFT types / benchmarks etc.
3. Propose a practical method to boost reasoning

### Weaknesses
1. **Insufficient Experiment on model choices: Only Qwen-2.5-3B**. [1] have shown that Qwen has something specific when considering reasoning abilities, so I believe more **experiments on other series open-source models and sizes (maybe ~7B)** should be added.

[1] Spurious Rewards: Rethinking Training Signals in RLVR. https://arxiv.org/abs/2506.10947

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
