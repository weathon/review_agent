# Information Gap in Chain-of-Thought Induces Implicit Thinking that Fails in Length Generalization

- Decision: Reject
- Scores: 8, 4, 2

## Abstract
Recent work reveals that Chain-of-Though may not faithfully reflect the model’s actual reasoning, as their semantics can diverge from their underlying “implicit thoughts”. In this work, using a synthetic dataset with controllable complexity, we find signs of implicit thinking in models after supervised finetuning (SFT) on CoT rationales, that is, the models have internally identified all necessary variables to be solved before generating the actual CoT. This implicit thinking ability sharply degrades as the required CoT steps exceed those seen during training, hence preventing the model from generalizing to more complex problems. To understand why implicit thinking emerges during SFT on explicit CoT rationales, we first define “information gap” within a CoT based on the ratio of unexplored actions and all admissible actions at each state. We hypothesize that a large information gap (a lot of admissible but unexplored actions) force LLMs to justify the actions explored in golden CoT by looking for clues in its internal representation, hence leading to implicit thinking. We benchmark 4 types of CoT, each based on a different graph traversal heuristic, and observe a positive correlation between the magnitude of information gap in CoTs and the implicit thinking ability in models finetuned on these CoTs. We further support this hypothesis by showing that actively reducing the information gap by including multiple CoT trajectories per question can reduce implicit thinking and enhance generalization to more complex questions. Overall, our findings suggest rethinking the role of CoT in LLM reasoning and understanding the necessary condition of learning generalizable CoT.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates a critical failure mode of Chain-of-Thought (CoT) reasoning: the model's inability to generalize to problems requiring longer reasoning chains than those seen during training (i.e., length generalization).

The authors introduce a controllable synthetic math dataset called WORLDOFBOXES (WOB) to diagnose this failure. They find that models trained with standard supervised finetuning (SFT) on CoT rationales do not learn to reason explicitly. Instead, they learn an "implicit thinking" shortcut, internally identifying all necessary variables before generating the first CoT token. This implicit ability is shown to be static and fails when the problem complexity (e.g., graph depth) exceeds the training distribution, causing a collapse in performance.

The paper hypothesizes that this shortcut is induced by a large "Information Gap" in the- training data—a significant disparity between the many "admissible" reasoning actions at each step and the single "golden" path provided in the CoT. This gap, they argue, forces the model to find an internal justification for the specific path, leading to this non-generalizable implicit shortcut.

To validate this hypothesis, the authors provide a series of strong pieces of evidence. First, they use linear probing  to find that the models failing at generalization are precisely the ones that exhibit high "implicit thinking" on in-distribution data; this implicit ability then collapses when faced with out-of-distribution (OOD) data. Second, they show a positive correlation  between the quantified size of the "Information Gap" for different CoT types (e.g., FORWARD-COT vs. DFS-COT) and the degree of "implicit thinking" they induce. Finally, and most critically, they demonstrate through an intervention experiment (Intervention) that actively reducing the Information Gap does indeed suppress implicit thinking and significantly improves the model's  generalization.

### Strengths
1. The paper's primary strength is its  experimental design. The WOB synthetic dataset allows the authors to control for all confounding variables (domain knowledge, problem structure, complexity) and cleanly measure the variables of interest: CoT type, implicit thinking, and generalization.
2. The "Information Gap" is a novel concept that provides a compelling explanation for why models take "implicit" shortcuts. The authors demonstrate it can be measured and, more importantly, mitigated to produce  improvements.
3. The paper  connects the model's internal representations (measured via probing) to its external behavior (task accuracy). 
4. The paper is exceptionally clear. The logical flow from problem to diagnosis to hypothesis and solution is compelling and easy to follow.

### Weaknesses
The paper's weaknesses are minor and are largely limitations inherent to its (very strong) controlled methodology.

The core claims are validated exclusively on the synthetic WOB dataset. This control is a strength, but it leaves open the question of how this mechanism operates in "real-world" reasoning tasks (e.g., GSM8k, coding) which are messier and entangled with world knowledge.

### Questions
1. The connection to real-world tasks via DFT loss is intriguing. A more direct test would be to apply your core intervention (training on multiple CoT trajectories) to a real-world dataset. Have the authors considered creating or finding a dataset like GSM8k-multi-path, where the *same* math problem is solved using 2-3 different (but valid) reasoning chains? This would be a powerful demonstration that the Information Gap principle holds beyond WOB.
2. The experiments are run on 7B/8B models. Do the authors have any insights or preliminary results on how model scale affects this phenomenon? Does "implicit thinking" become more or less prevalent in much larger models (e.g., 70B+)? Or, is it a fundamental artifact of the SFT-on-CoT process, regardless of scale?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors investigate how supervised fine-tuning with different types of chain of thought traces affects performance on the WorldOfBoxes task, a simple GSM-type task involving object relationship and addition/subtraction of small integers. 

They define implicit thinking based on whether the internal weights of the model can be used to predict which boxes are important to the task, as measured by linear probing. By applying SFT with different CoT techniques, they are able to elicit different amounts of implicit thinking. They find that implicit thinking negatively impacts length generalization. Further, they hypothesize a relationship between implicit thinking and the ‘information gap’ of the chain of thought traces.

### Strengths
The problem investigated in the paper is timely and well-motivated.

The World of Boxes problem is a clever way to isolate different types of reasoning traces. Since it requires only simple arithmetic, the main variable being tested is finding the correct L2R/R2L path. 

The authors successfully connect SFT with different types of chain of thought traces to a concrete difference in implicit thinking as measured by linear probing. They also make this connection with length generalization, showing significant evidence that implicit thinking is associated with worse length generalization.

### Weaknesses
The paper could be improved by a formalization of ‘implicit thinking’. The authors measure it by using a linear probe to predict from the model’s weights whether a box is relevant to the WoB instance. If the probe is successful, they say that the model is performing implicit thinking. It would be nice to see an explicit definition like “model M is an implicit thinking model if (condition)”, or more likely “model M thinks implicitly on problem P if (condition)”.  

The central hypothesis of the paper is that this specific computable quantity, the information gap in chain of thought, induces implicit thinking. However, the second half of the hypothesis, $M^{\phi_1}$ has less implicit thinking (as measured by probing) than $M^{\phi_2}$, is not specific enough to be falsifiable.

Further, I don’t get a strong sense for why the authors believe that the information gap is the relevant variable. The experimental evidence is limited (there are only five data points for the information gap in fig. 4, and the experiment only shows correlation). This could be strengthened via some mathematical intuition – not necessarily a formal proof, but maybe showing the relationship between information gap and implicit thinking in some simplified setting.

### Questions
I noticed that the probing accuracy for RS-COT and DFS-COT is the same despite a large difference in the information gap. Why do you think this is?

Do you think of implicit thinking as a binary state (i.e., either the model solves the problem by thinking implicitly or it solves the problem by step-by-step reasoning)? Is there evidence that models can use different degrees of implicit thinking?

Typos:
Sentence 1 of abstract: “Chain of Though(t)”

In p.7 Hypothesis 1, the trajectories are labelled as $^{\phi_1} \tau$ and $\tau^{\phi_1}$? Do these have different meanings, or is it a typo?

Additionally wrote both $M_{\phi_i}$ and $M^{\phi_i}$ in Hypothesis 1 and 2.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper takes a closer look at the phenomena of "implicit thinking", where chains-of-thought (CoT) generated by models are post-hoc rationalizations i.e. they they do not causally determine the prediction and the model already has an internal representation of the answer.  To study this, the authors introduce a synthetic graph traversal dataset called WorldOfBoxes (WoB) with two variants:
* WoB-R2L (root-to-leaf): traverse from a root box down to a leaf node
* WoB-L2R (leaf-to-root): start at a leaf node and recover root's value

On top of this dataset they supervise four different chain of thought styles on the same underlying problems in order to determine which styles support out of distribution generalization. The four styles are forward, backtrack, random, and DFS chain of thought. They find that in the R2L setting backtracking based CoT traces generalize best and forward only traces generalize worst. In the L2R setting forward chain of thought appears to do best.  They find that in the R2L setting, backtracking-based CoT traces generalize best and forward-only CoT traces doing worst.

They then measure implicit thinking with a linear probe on frozen model representations. In the R2L setting the probe shows evidence of implicit thinking for models trained with forward chain of thought and some evidence for models trained with DFS CoT. In the L2R setting probe accuracy is high for models trained with backtracking, which they interpret as those models reconstructing the solution structure before emitting the CoT.

The authors then try to formalize some of this in terms of an information gap metric, which aims to measure the completeness of exploration at any point of the CoT trajectory.  If the information gap is large between two consecutive CoT steps, that means there are many possible nodes that can be explored at that step and the current trace has only explored a few by the end.  They show that this information gap quantity correlates with their probe based measure of implicit thinking and use this to motivate their overall hypothesis.

### Strengths
Overall, the ideas explored in the paper are interesting. The question of how much additional computation in the form of chain of thought actually contributes to determining the answer is a worthwhile one, and it is good to see it studied in a setting where the underlying structure is fully known.

I liked the synthetic setup. Using a controlled graph traversal task makes the discussion concrete, allows multiple chain of thought styles to be compared on exactly the same problem, and makes it possible to talk precisely about admissible actions, coverage and exploration.

The empirical results in the early part of the paper are interesting. The observation that some supervised traversal styles generalize to deeper trees while others fail, and that this depends on the direction of traversal and the branching factor, is of some value to the community. This part of the work shows that the choice of CoT supervision matters even when the underlying task is fixed.

### Weaknesses
My primary criticism of this work is that it is merging several related but distinct questions, and the paper would benefit from a clearer statement of which question it is actually trying to answer. There are at least three threads running through the current version:

1. Faithfulness: Are chains of thought in fact correlated with correct versus incorrect answers, or are they mostly post hoc rationalizations of an already formed answer?

2. Search/exploration: Which traversal style chains of thought actually help with length or depth generalization when the underlying structure is a tree or a graph?

3. Useful metrics for CoT: Can we define a notion of utility or completeness for a chain of thought that tells us when the model was forced to explore versus when it simply followed a narrow path?

Most of the experiments in section 4 are really addressing (2). They show that certain exploration traces, such as backtracking in the root to leaf case, survive depth increases better than forward only traces, and they suggest that this is tied to the branching structure of the graph. I believe some of these results have appeared in the literature (see, eg ref [1]), which makes the search and exploration angle the most solid part of the paper. However, the paper mostly frames itself in the language of (1), using terms such as “implicit thinking” and “post hoc” that belong to the faithfulness discussion. Those two lines of inquiry are related but not identical. If the central question is whether chains of thought are faithful, it is not obvious that length generalization on a synthetic tree task is the most appropriate environment to answer it. Conversely, if the central point is that some distilled search strategies are brittle while others are robust, then the search results should be analyzed more fully.

A second criticism is that the concept of "implicit thinking" is never fully defined.  In the synthetic setting, one could create a working definition such as "the model is implicitly thinking if a linear probe on its hidden state before CoT can classify whether each box is on the GT path."  That is in fact how the paper measures it in practice. If that is the working definition, it is unclear why an additional construct such as the information gap is needed. The current text uses the probe as a litmus test but does not clearly state what hypothesis is being tested, what constitutes strong evidence for implicit thinking, and how this is disentangled from the quality of the supervised search strategy. A weak or mismatched CoT trace will of course force the model to reconstruct missing structure from context, so detecting that the model has done so is not very informative by itself. A more interesting direction, which the current version only hints at, would be to analyze how different traversal algorithms affect convergence or confidence in the target answer.

Also, I am not convinced by the utility of the information gap metric.  It feels redundant with simpler explanations.  In the graph traversal case, one may just as well use a branching ratio or related graph theoretic metrics.  The metric feels ad hoc and hard to transfer for real math reasoning tasks where the admissible action set $C_i$ isn't known.  I think using the probe makes sense as a definition of implicit thinking, but it would have been nice to see some examples in the math reasoning outside of the synthetic setting.  It is also worth noting that everything is dumped into context. So some of what the probe is reading could just be “the model has encoded the structure of this context,” not “the model planned.”  The claim about using DFT to suppress gradients from high gap tokens and hence prevent models from developing implicit thinking is not supported by detailed results. It is not shown whether this improves depth accuracy or only lowers the probe. This part needs numbers.

Though I did not penalize for this, the related work section is not complete. Recent papers that study diversity and coverage in chain of thought should be cited, in particular arXiv 2504.07052, arXiv 2506.05744 and arXiv 2505.21825.

[1] https://arxiv.org/abs/2504.07052v2

[2] https://arxiv.org/abs/2506.05744v3

[3] https://arxiv.org/abs/2505.21825

### Questions
1. I am not fully clear on the probing task setup. At what point in the sequence is the probe applied, and what exactly is being classified. Is the probe asked, for every box in the serialized input, to predict whether it is on the ground truth path for the current query? If so, why is the chance baseline 50 percent? In a tree setting the positive class should be the minority unless you balanced it.

### Soundness
2

### Presentation
3

### Contribution
2
