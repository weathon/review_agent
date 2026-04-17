# Comparing the learning dynamics of in-context learning and fine-tuning in language models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Pretrained language models can acquire novel tasks either through in-context learning (ICL)---adapting behavior via activations without weight updates---or through supervised fine-tuning (SFT), where parameters are explicitly updated. Prior work has reported differences in their generalization performance and inductive biases, but the origins of these differences remain poorly understood. In this work, we treat ICL and SFT as distinct learning algorithms and directly compare the learning dynamics they induce across medium-sized models, analyzing both the evolution of their inductive biases and the underlying internal representations. We find that ICL preserves rich input representations but imposes stronger priors inherited from pretraining, whereas SFT suppresses task-irrelevant features---potentially explaining its weaker generalization in few-shot regimes. These results highlight a mechanistic distinction between context-driven and weight-driven learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors apply both few-shot prompting ("ICL") and supervised fine-tuning ("SFT") to a toy 2D linear classification task, analyzing how performance and representations change as more examples are added.

### Strengths
* Demonstrates that while final performance is similar between ICL and SFT, (a) ICL has a prior towards 45 degrees while SFT has a prior towards 90 degrees, (b) ICL is sensitive to periodic orderings of example labels, and (c) SFT representations are more tightly coupled to the label space.
* Finds support for (a) across Llama3-8B, Qwen3-8B, Gemma3-12B, and Gemma3-27B

### Weaknesses
* Claims in the text are often not clearly linked to the corresponding portions of the dense and complex figures. For example, the claim "Both predictions were verified when comparing model performance across seeds for ICL" references "Fig.2A" when I believe the only relevant part of Figure 2A is column 4, the claim "we observed an overestimation (resp. underestimation) of the inferred task angle for θ = 30◦ (resp. θ = 60◦)"  references "Fig.2A" when I believe the only relevant part of Figure 2A is column 3, etc. Minimally, every subfigure within the current figures needs to be labeled somehow (not just the rows), and the appropriate subfigure needs to be referenced by label wherever it is discussed. More broadly, every claim should be coupled with text that explains how to read that claim off of the corresponding figure.
* The focus is on a single artificial task, and how these findings might generalize to other problems is not well discussed. For example, I don't know what prediction the identified bias towards a 45 degree angle would make for a real-world task like question answering.
* Most analyses are done on Llama3-8B, so we can't tell, for example, whether findings (b) and (c) listed in the Strengths generalize to other LLMs. (The appendix shows that GPT-OSS:20B can't even learn the task, so generalization of the findings across models is an important concern.)

### Questions
* I don't understand how to read the "previously seen feature bias" off the figures. Can you explain?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper directly contrasts in‑context learning (ICL) and supervised fine‑tuning (SFT) as learning algorithms on a tightly controlled 2‑D linear classification family. Using matched data, shot counts, and example orderings, the authors track accuracy, "smoothness," confidence, inferred boundary angle, and layer‑wise representational similarity (RSA). Both ICL and SFT reach similar held‑out accuracy, but with different inductive biases and internal representations: ICL preserves richer input structure yet shows stronger from-pretraining priors (e.g., diagonal "number‑comparison" bias and row/column reuse), whereas SFT aligns representations along label axes, yielding higher confidence but apparent representational collapse. Ordering effects reveal short‑horizon pattern‑following in ICL. Results qualitatively persist across several model families and in a semantic (adjective‑ordered) variant, though learning there is slower.

### Strengths
* The matched‑trajectory setup isolates algorithmic differences between ICL and SFT more cleanly than typical open‑domain benchmarks

* I think the comparison and framing of SFT and ICL as two distinct learning algorithms is a useful and interesting framing

### Weaknesses
* I think the main limitation, as mentioned by the authors, is scope: Add one or two richer tasks (non‑linear boundaries or 3‑class variants; a small real‑text task with controlled geometry) to test whether the same ICL/SFT differences recur.

* RSA uses last‑query‑token activations; alternative readouts (earlier tokens, attention heads, probing classifiers) could nuance the "collapse vs. preservation" story. Causality is not established

* Fig. 3 compellingly shows SFT’s label‑aligned compression vs. ICL’s input‑structured geometry, even when accuracies match. I thought this was cool

### Questions
Does representational collapse happen with LoRA or other PEFT approaches?

### Soundness
3

### Presentation
3

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
This work empirically compares the learning dynamics of in-context learning and finetuning on mid-sized language models on two synthetic tasks: a 2-D numerical classification task and a 2-D semantic classification task. This work demonstrates that, while both algorithms can learn the tasks, in-context learning is more subject to biases present in natural language training data (e.g., biases towards comparing numbers, rather than learning an arbitrary classification boundary). Furthermore, the authors demonstrate that the two learning algorithms yield radically different internal representations, with finetuning collapsing representations into two classes and ICL maintaining a greater amount of structure in representations.

### Strengths
This work empirically addresses a broad, pressing question in the field: what is the empirical relationship between in context learning and finetuning? 

The work appears empirically sound, the experiments are thorough, and the breadth of models tested (within the mid-sized scale) is fairly large.

The results regarding inductive biases of ICL are sensible, yet interesting. These results bolster broader arguments that ICL is selecting from a pool of functions learned in pretraining. The analyses of inductive bias in ICL add nuance to this discussion, indicating that there are systematic differences in the prior probability assigned to functions.

### Weaknesses
Why not study small models as well? Even if studying very large models is computationally challenging, one might be able to discover a scaling trend in ICL vs finetuning learning dynamics by systematically studying models on the order of 1B to 27B. This would increase the impact of this work.

The RSA of the finetuned models should be further fleshed out. One guess is that the task is so trivial that the model simply converges on the correct answer early on in its computation (i.e., at an early layer). Perhaps early on in finetuning, the model’s representations look more like what you find for ICL, or perhaps there is a one-to-many comparison to be made between the early layers of the finetuned model and all of the layers in the ICL model. This setup seems especially well suited for a finding like “finetuning results in a compressed version of the computation that ICL converges on”. Studying the dynamics of what is called “representation collapse” in the text would be a very valuable contribution.

### Questions
Why not study smaller models as well?

Did you try using other nonsense labels? Or even labels with pretrained semantics? Perhaps there would be interesting divergences between ICL and finetuning that hinge on the label.

### Soundness
3

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
This paper investigates the different inductive biases and learning dynamics of in-context learning (ICL) and supervised fine-tuning (SFT) in medium-sized language models. Using a controlled 2D linear classification task, the authors directly compare ICL and SFT across matched learning trajectories (i.e., same data and example ordering), analyzing both generalization patterns and internal representations. The authors find that while both methods achieve similar final accuracy, their strategies and inductive biases differ significantly. ICL is shown to exhibit strong inductive biases inherited from pretraining, such as a "comparison bias" (favoring diagonal decision boundaries, $\theta \approx 45^\circ$) and a "previously-seen feature value bias" (favoring axis-aligned boundaries). In contrast, SFT is shown to suppress task-irrelevant features, leading to internal states clustering primarily by task label. Conversely, ICL preserves more varied input-specific representations throughout its layers. These core findings are also shown to generalize to an analogous semantic classification task.

### Strengths
1. *Clear Presentation:* The paper does an excellent job of clearly presenting its results. The use of a minimal, controlled 2D linear classification task allows for a precise and direct comparison of the learning dynamics, generalization patterns, and inductive biases of ICL and SFT.
2. *Representational Analysis:* The representational similarity analysis (RSA) provides a clear distinction between the two learning regimes. It visually and quantitatively demonstrates SFT's representation collapse versus ICL's preservation of input structure (Fig. 3), adding a strong layer of evidence beyond simple task performance metrics.
3. *Strong Grounding in Literature:* The findings are well-situated within the broader literature, particularly in relation to Bayesian accounts of ICL, and contribute compelling evidence to the ongoing discussion challenging "ICL as gradient descent" mechanisms in larger models.

### Weaknesses
1. *Limited Scope of SFT Experiments:* The paper's central claims about SFT rely on fine-tuning experiments conducted primarily on a single model (Llama-3-8B), which the authors acknowledge as a limitation. While the ICL results are replicated across several models, the core ICL vs. SFT comparison would be greatly strengthened by a more comprehensive evaluation of the SFT condition (e.g., across more models, training hyperparameters, or regularizers) to ensure the "representation collapse" is a general feature of SFT and not an artifact of a specific setup.
2. *Novelty in Context of Prior Work:* While the direct comparison between ICL and SFT is valuable, many of the core results (e.g., ICL can be sensitive to pretraining priors, SFT can be brittle, etc) have been documented in related work. The paper could do a clearer job of articulating the specific, novel contribution of its findings beyond these effects. For instance, is the key novelty the direct demonstration of how the internal representations diverge under identical data, or the specific characterization of the "comparison" vs. "axis-aligned" biases?

### Questions
1. *Influence of SFT Hyperparameters*: The authors note that SFT hyperparameters were not exhaustively probed (Limitations, p. 9). Given that SFT is known to be sensitive to hyperparameter choices, how confident are the authors that the observed "representation collapse" is an inherent feature of SFT on this task, rather than an artifact of a specific hyperparameter regime (e.g., potential over-fitting)?
2. *Clarifying Novel Contribution*: The paper's discussion notes that ICL's sensitivity on pretraining priors and SFT's brittleness have been noted in prior work. Is the authors primary contribution the direct, matched-data demonstration of how ICL and SFT strategies diverge representationally, or the specific characterization of the competing inductive biases (e.g., "comparison bias" vs. "feature value bias")? A clearer framing of the novelty would help situate the work's impact.

### Soundness
3

### Presentation
4

### Contribution
2
