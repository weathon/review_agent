# Towards Understanding Metacognition in Large Reasoning Models

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Large Reasoning Models (LRMs) have achieved strong performance on complex multi-step tasks. However, they often fail to assess difficulty, monitor uncertainty, or revise incorrect reasoning—posing fundamental challenges to reliability. We argue that these limitations reflect the absence of metacognition, the capacity to monitor and control one’s own cognitive processes. Building on insights from cognitive science, we present a structured study of metacognition in LRMs, focusing on both internal signals and observable behaviors. We first show that internal activations, attention patterns, and token-level confidences contain rich information predictive of reasoning correctness. We then design a set of evaluation tasks to test functional abilities such as difficulty awareness, confidence adjustment, task decomposition, and strategy revision across five widely used LRMs. Our results suggest that while LRMs exhibit partial signs of metacognitive behavior, these abilities are inconsistent and easily disrupted. We further explore two complementary approaches for strengthening metacognition: prompt-driven control and supervised training with structured metacognitive traces. Together, our findings highlight metacognition as a critical lens for diagnosing and improving reasoning in large models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates metacognition in LRMs. The paper first performs a series of empirical evaluations to measure metacognitive abilities in current LRMs, and then proposes two paradigms for enhancing models’ metacognitive abilities (through test-time prompting as well as parameter-level training).

### Strengths
The topic of metacognition is important and timely, and I think the paper makes interesting contributions, both in evaluating metacognition and improving metacognitive abilities.

### Weaknesses
I had a few concerns about the design of the self-awareness study in Section 4.1. The authors write that they “operationalize self-awareness as the model’s ability to accurately classify the difficulty of mathematical problems when explicitly prompted to do so” (l. 208-209). They find that the tested LRMS “exhibit a lack of capacity in perceiving task difficulty” (l. 240-241). There are several potential issues with this design. First, what do labels like “easy”, “medium”, and “hard” mean for models? Would humans even be able to perform this labeling task? Second, this task may not be a good measure of self-awareness, because there may be confounding factors that can be used to distinguish between the difficulty classes—for example, maybe the hard problems are longer on average, or contain certain key words or phrases that are not present in easy problems. It would be useful to see some control analyses here. Third, and related to the second point, Song et al. (2025) argue that true metacognition requires *privileged self-access*. If model B could predict whether a math problem will be easy/medium/hard for model A as well as model A can predict itself, then model A is not truly showing signs of self-awareness. This is a critical theoretical point that is not discussed anywhere in the paper.

Building off the point above, there are many related works that are not discussed. For example, on the topic of introspection and self-awareness: Binder et al. (2024; “Looking Inward: Language Models Can Learn About Themselves by Introspection”), Song et al. (2025a; “Language Models Fail to Introspect About Their Knowledge of Language”), (2025b; “Privileged Self-Access Matters for Introspection in AI”). 

In addition, the importance of intermediate planning has also been extensively studied (e.g., Valmeekam et al. 2023: “On the Planning Abilities of Large Language Models - A Critical Investigation”; Webb et al. 2025: “A brain-inspired agentic architecture to improve planning with LLMs”), so I’m unsure about the novelty of the task decomposition experiment (Section 4.3).

Finally, the paper has no discussion of limitations, and the discussion is very short.

### Questions
Could you address the points raised above in the Weaknesses section?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Existing Large Reasoning Models (LRMs) may exhibit fragile behaviors in their chain-of-thought reasoning, such as reasoning rigidity or illusory self-correction. The authors argue that if this deficit can be explained as an absence of meta-cognition, and ask whether current LRMs exhibit any analogous capabilities to those in meta-cognition.

### Strengths
At first I was skeptical of this paper's claim of positioning metacognition with LRM but I believe by page 7 I became more convinced after seeing the excellent results (e.g., Table 2, Table 3). The paper's motivation, intuition, problem statement, and organization are professionally done and acceptable for ICLR standards. Quality of the figures is also excellent. By far the best part of the paper is section 5. I wish more of the paper was like Section 5, but am also unsure how to edit it to accommodate that. I also appreciate the delicate and grounded approach the authors took in connecting LRMs with metacognition. I strongly urge the authors to review the identified weaknesses in my review as well as the questions I have asked, and see how they might revise this submission to improve upon it.

### Weaknesses
"we probe open-source LRMs to determine whether internal computational signals correlate with reasoning outcomes (§3). Our analysis yields a striking finding: signals spanning the entire Transformer architecture—from input-layer attributions to final-layer token probabilities—are highly predictive of answer success. Critically, we demonstrate that correct and incorrect reasoning traces generate statistically distinguishable internal signatures, providing the first empirical evidence that a machine-readable basis for metacognitive
experience exists within these models." <-- isn't this an obvious finding? It is a neural architecture that generates answers. So, of course, there would be intermediate calculations (latent representations) that indicate inevitable success as those intermediate calculations are used to inevitably produce the successful answers. Also, is this really a metacognitive experience? This sort of behavior can be found in the most basic of neural architectures like multi-layer perceptrons if I am understanding this claimed finding correctly.

Line 86-87 "desgining systems that not only..." should be "designing systems that not only..." and "...but cna also act..." should be "...but can also act..."

* What were the t-SNE hyperparameters (e.g., perplexity, iterations) for Figure 2?

* How many epochs were the linear classifiers (probers) trained for on these internal signals as features for predicting final correctness? Was the output actually a correct/fail value? If so, wouldn't it be more appropriate to use a logistic regression or similar? Table 1 AUC seem rather low to claim "...the probers can predict the final correctness of a reasoning trace with an AUC score significantly above chance". IF AUC = 0.5 is random guessing, then AIME2024 w/ Softmax probabilities, Fully-connected activations, and Integrated Gradients are the same or worse than random guessing. Same issue with AIME2025, but also potentially including self-attention scores now. "These results establish that the signals are both distinct and highly predictive, thereby validating our initial hypothesis." I disagree based on the AUC scores. Also, why no formulas or stats on the linear probers (e.g., p-value, R^2) or statistical investigation on if a linear relationship holds?

* Figure 2: The t-SNE of the internal signals do not appear to show different distributions to me. The data looks very embedded, but might only show a slight difference due to alpha of the plotted dots being not exactly 1 or overlap.

* "These token-level scores are then aggregated to produce a trace-level metric, termed of average trace confidence, for each complete solution." (lines 251-252) <- I am assuming arithmetic mean. Should probably avoid the use of "average" as it is less precise. However, the mean alone for describing trace confidence loses a lot of information about the distribution of confidences as its susceptible to outliers. Why not incorporate the standard deviation and other metrics into this analysis when capturing confidence dynamics? Did you investigate if outliers were present? Figure 4 bar plot: are those error bars standard deviation, standard error, or confidence interval?

I find the results of 4.2 for metacognitive monitoring to be lackluster. I am not convinced by Figure 4. And also, discretizing quantitative values into discrete categories (e.g., all confidence above/below a high threshold) although common practice is a bit of a misleading statistical analysis to exaggerate differences. Better practice is to directly analyze the quantitative data with no binning. Also, what are these high thresholds (line 255)? 

GRPO algorithm (line 410): write out before using acronym.

Conclusion: I would actually suggest revising this to say something more along the lines that the "...authors empirically demonstrate that while LRMs possess internal signals that *may* be predictive of reasoning outcomes, there is also variability in their usefulness across other domains (e.g., AIME2024, AIME2025); thus, they alone do consistently translate into effective monitoring or control behaviors. To address this gap, ..." Obviously word this better than I have, but I think that performance discrepancy we see on AIME2024 and AIME2025 can actually be addressed in such a way to support your overall proposed method!

### Questions
* What were the t-SNE hyperparameters (e.g., perplexity, iterations) for Figure 2?
* How many epochs were the linear classifiers (probers) trained for on these internal signals as features for predicting final correctness? Was the output actually a correct/fail value? If so, wouldn't it be more appropriate to use a logistic regression or similar? Table 1 AUC seem rather low to claim "...the probers can predict the final correctness of a reasoning trace with an AUC score significantly above chance". IF AUC = 0.5 is random guessing, then AIME2024 w/ Softmax probabilities, Fully-connected activations, and Integrated Gradients are the same or worse than random guessing. Same issue with AIME2025, but also potentially including self-attention scores now. "These results establish that the signals are both distinct and highly predictive, thereby validating our initial hypothesis." I disagree based on the AUC scores. Also, why no formulas or stats on the linear probers (e.g., p-value, R^2) or statistical investigation on if a linear relationship holds?

* "These token-level scores are then aggregated to produce a trace-level metric, termed of average trace confidence, for each complete solution." (lines 251-252) <- I am assuming arithmetic mean. Should probably avoid the use of "average" as it is less precise. However, the mean alone for describing trace confidence loses a lot of information about the distribution of confidences as its susceptible to outliers. Why not incorporate the standard deviation and other metrics into this analysis when capturing confidence dynamics? Did you investigate if outliers were present? Figure 4 bar plot: are those error bars standard deviation, standard error, or confidence interval?

* "The loop terminates when sufficient consistency is achieved (e.g., 5 consecutive verification passes) or persistent failure occurs (e.g., a 10-step failure streak)" <- How often did each of this termination conditions occur; what is the ratio?

* "...where $\mathcal{C}$ is a predefined confidence threshold." How is that determined? Hyperparameter?

* Also, what are these high thresholds (line 255)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a functional framework for metacognition in Large Reasoning Models (LRMs), separating information from abilities. It first shows that internal signals (final-layer probs, activations, attention, IG) predict success, then evaluates monitoring and control. To bridge them, it introduces (i) prompt-based and (ii) fine-tuning based methods.

### Strengths
Clear functional framing (information vs abilities) and a helpful suite of behavioral diagnostics; writing and figures are generally clear.

### Weaknesses
While the framing is tidy, the methods largely mirror existing practice:

- In-context scaffolding / multi-step prompting resembles established prompting frameworks (self-consistency, plan-and-solve, Tree-of-Thoughts, debate/verification variants).
- Fine-tuning on reasoning/process traces closely relates to STaR and subsequent process-supervision works.
- Predicting success from internal signals (log-probs/confidence/activations) is well-documented (“LMs mostly know what they know”).

Overall I view the conceptual clarity as good, but the technical originality as moderate: the core ingredients—prompting, self-verification, process-supervised fine-tuning, and internal confidence probes—are known. 

I think the paper would benefit from clarifying what is new beyond re-packaging.

### Questions
See above.

### Soundness
3

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
4

### Summary
This paper proposes two methods to understand the metacognition; (i) prompt-guided role-playing. It is to assign distinct metacognitive roles—such as ‘Planner’, ‘Solver’, and ‘Verifier’—to simulate a complete monitoring and control loop. (ii) enriches model’s
metacognitive knowledge through fine-tuning. They construct a dataset with explicit metacognitive annotations (e.g., plans, self-corrections) and fine-tune the model using a hybrid learning objective, directly embedding these capabilities into its parameters.

Specially, they
- evaluate the metacognition on a series annotated tasks, i.e., self-awareness, difficulty, task decomposition and flexibility.  based on annotated dataset, for example, difficulty-level dataset, then prompt the model to assess the  difficulty of the input task and record their performance as ACC.
- based on the observation that LLM fail to exhibit desirable metacognition, they use role-playing method to elicit their internal awareness
- they further improve LLM's metacognition by training LLM on annotated metacognition-aware trajectories.

### Strengths
1. It studies the metacognition in knowledge and experience, two aspects, and design four tasks to assess the LLM's capabilities.
2. It proposes two methods to enhance model's metacognition, both prompt-guided role playing and fine-tuning.

### Weaknesses
1. In section 4, the author design 4 tasks to evaluate the model's metacoginition, 
- while some of these experiments are already evaluated in existing literature, such as difficulty-level, and self-awareness.  "Do Large Language Models Know What They Don't Know?"
- the experiment design is not rigorous. for example, the task decomposition. It shows that decomposition can improve performance, are you evaluating for the same question, with/without decomposition? Also, it is similar to Chain-of-thought setting, which has already shown that with step-by-step thinking, the model can think better.
2. Both prompt-guided role and fine-tuning methods are not very insightful. 
- for prompt-based method, what is your vanilla setup? direct prompt? but your proposed method actually incorporate self-reflection loop, which has been shown to be effective than direct prompt or even CoT. so the performance boots for me is not very insightful.
- for fine-tuning methods, do you have any OOD evaluation? and what is your prompt in Qwen2.5-Math-7B when inference?

### Questions
See questions above.

### Soundness
3

### Presentation
3

### Contribution
2
