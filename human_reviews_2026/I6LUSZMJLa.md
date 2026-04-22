# HumanPCR: Probing MLLM Capabilities in Diverse Human-Centric Scenes

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
The aspiration for artificial general intelligence, fueled by the rapid progress of multimodal understanding, demands models to understand humans in diverse and complex scenarios, as humans manifests intelligence and embody the world. We propose HumanPCR, an evaluation suite for probing MLLMs’ capacity in human-centric visual contexts across three hierarchical levels: Perception, Comprehension, and Reasoning (denoted by Human-P, Human-C, and Human-R, respectively). Human-P and Human-C consist of over 6,000 multiple-choice questions evaluating 34 fine-grained tasks covering 9 essential dimensions. Human-R presents a manually curated challenging video reasoning test that requires integrating multiple visual evidence, proactively extracting implicit context beyond question cues, and applying human-like expertise. Each question includes human-annotated Chain-of-Thought (CoT) rationales with key visual evidence to support further research. Extensive evaluations on over 30 state-of-the-art models exhibit significant challenges in human-centric visual understanding, particularly in tasks involving detailed space perception, temporal understanding, and mind modeling. The analysis of Human-R further exposes a critical failure in reasoning: models struggle to proactively gather necessary visual evidence, instead showing a faulty reliance on query-prompted cues, with advanced techniques offering only marginal gains. We hope HumanPCR and our findings will advance the development, evaluation, and human-centric applications of multimodal models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces HumanPCR, a human-centric benchmark for evaluating multimodal large language models at three levels: perception, comprehension, and reasoning. It contains over 6,000 multiple-choice questions across 34 fine-grained human-related skills. These reasoning questions require the model to combine evidence from multiple moments in a video and notice important details that are not directly pointed out in the question. By testing 30+ state-of-the-art models, the paper shows that even the strongest systems still fall well short of human performance on fine-grained spatial understanding, temporal reasoning, and intent inference.

### Strengths
1. The paper is well written and organized. 
2. The graph in the paper is clear and beautiful.
3. The core idea of this paper is interesting and special. It fills the gaps in prior benchmarks, which were either too narrow or too broad without focusing on human-centric understanding.
4. The dataset of this paper is carefully built with multi-stage human review and strict quality control. Each Human-R question must rely on multiple visual clues and include at least one piece of proactively gathered evidence, and only about 20% of questions are accepted.
5. Extensive experiments have been conducted to demonstrate the effectiveness. The paper evaluates more than 30 state-of-the-art models and shows that humans still outperform the best model by over 15%, while many open-source models score below 30% accuracy on the most challenging Human-R reasoning tasks.

### Weaknesses
1. Evaluation setup is not perfectly fair. The paper evaluates many models, but they are run with different inference settings, so the comparison is not strictly fair. For comprehension, models only get 32 sampled frames and must answer directly, while for reasoning some models can use the maximum number of frames they can handle and are given stronger prompts plus tricks like Best-of-N sampling. 
2. Causal claims about failure modes are a bit strong. The paper concludes that proactive evidence extraction is the key bottleneck in complex video reasoning, based on manual analysis of 200 Human-R questions from top models; however, this is still correlational and not backed by controlled ablations.
3. The paper does not report the actual compute cost (runtime, GPU memory usage, or GPU hours) required to run the benchmark, which makes reproducibility and practical adoption harder.

### Questions
My questions have already been described in the Weaknesses section. I would like the authors to respond to these points in detail and clarify how they plan to address them.

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
The HumanPCR dataset was constructed by first defining a three-tiered human-centric skill framework (Perception, Comprehension, Reasoning), and then sourcing video and image data from both public datasets (e.g., Ego4D, ActivityNet, MOMA) and recent web videos (e.g., YouTube clips from the last two years) spanning 11 real-world domains. For Human-P and Human-C, the team created over 6,000 multiple-choice questions using a mix of template generation, LLM drafting (e.g., GPT-4), and human review. For Human-R, they developed 442 open-ended video reasoning questions that require multiple pieces of visual evidence—at least one of which must be proactively retrieved, not directly cued—each paired with expert-authored Chain-of-Thought rationales. Rigorous quality control was applied, including blind filtering of text-answerable items, expert meta-review.

### Strengths
1) A reasoning paradigm that enforces multi-evidence integration and proactive evidence seeking, with rigorous filtering to avoid shortcutting.
2) Most datasets just ask questions where the answer is directly visible in one frame or clip. This benchmark forces models to search multiple places in the video to figure things out—like humans would.

### Weaknesses
1) The paper says annotations will be CC BY 4.0, but the DUA also says “non-commercial academic only.” CC BY allows commercial use—this must be reconciled (e.g., CC BY-NC for annotations, or remove NC language from DUA).
2) All answers are scored by a single LLM, even if correlations with humans are decent. To be rigorous, they should show confidence intervals, try multiple judges, and share prompts/seeds for transparency.
3) Only ~10% of Human-R questions were answered by actual humans. To make a strong comparison, they should show how many annotators were used, what instructions they had, and how reliable those answers are.
4) Many tables just give single accuracy numbers. They should report error bars, task-level variances, and statistical tests to show if differences are meaningful.

### Questions
1) How do you define “proactive” vs. “referred” evidence? Is this consistently labeled across annotators?
2) Do models improve because they see more, or because they choose better?

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
3

### Summary
This paper presents HumanPCR, a hierarchical benchmark for human-centric visual understanding in MLLMs, consisting of Human-P/C (over 6k multiple-choice items across 34 tasks and 9 dimensions) and Human-R (442 open-ended video reasoning questions spanning 11 domains). A central design of Human-R is to require multi-evidence integration and at least one proactive (non-referred) visual evidence per question, with expert-annotated CoT rationales and rigorous filtering to avoid shortcutting via query cues. Evaluating 30+ SOTA models, the authors show that current MLLMs substantially underperform humans, with pronounced weaknesses in fine-grained spatial/temporal understanding and mind modeling, and that mere frame/context scaling yields diminishing or negative returns when evidence requirements grow. In contrast, reasoning-oriented models and test-time Best-of-N sampling provide consistent gains. The benchmark includes quality control, prompts, model configs, LLM-judge validation against human judgments, and ethical data terms.

### Strengths
The benchmark’s most distinctive contribution is Human-R’s explicit requirement for multi-evidence reasoning with at least one proactive, non-referred cue—an evaluation target largely missing from existing video QA benchmarks. This forces models beyond query-matching shortcuts and surfaces a realistic capability gap in long, complex, human-centric videos. The taxonomy across Human-P/C delivers both breadth and diagnostic depth; macro-averaging at task/dimension level makes failure modes actionable (notably posture, contact, procedure). The annotation and verification design is strong: expert-authored open-ended questions, evidence-grounded CoT rationales, blind filtering to remove textual shortcuts, and meta-review to ensure non-redundancy and proactive evidence. The empirical study is comprehensive: 30+ models, analyses of frame scaling, multiple context-extraction strategies, test-time scaling (Best-of-N vs self-refine), and a human-grounded error taxonomy highlighting proactive evidence omissions. LLM-judge reliability is quantified against 4k human ratings with high correlations and no detectable self-bias, supporting practical reproducibility. Overall, the work addresses an important and under-evaluated capability—human-centric multi-evidence reasoning—of clear significance to assistive systems, egocentric AI, and HRI.

### Weaknesses
While Human-P/C are valuable and thorough, their task space overlaps with prior perception/comprehension benchmarks; the strongest novelty lies in Human-R. The paper would benefit from ablations demonstrating added diagnostic value of Human-P/C design (beyond coverage), or clearer evidence that specific sub-tasks reduce confounds seen in earlier datasets. Human-R’s reliance on an LLM judge, albeit validated, still risks metric drift as judges evolve; a small, fully human-adjudicated gold subset would strengthen longitudinal stability and cross-paper comparability. The proactive-evidence constraint is compelling but mainly supported by qualitative/error analyses; controlled ablations—e.g., converting proactive evidence into referred cues, or masking question hints—could quantify its causal contribution to difficulty and discriminative power. Finally, given aggregation from public datasets and web videos, a more explicit analysis of potential pretraining overlap with popular MLLM corpora would help dispel contamination concerns and clarify generalization.

### Questions
Can the authors provide controlled variants of Human-R items to quantify the marginal difficulty induced by proactive evidence, for example by (i) turning proactive evidence into referred cues, and (ii) adding minimal hints? Reporting Δ accuracy across models would causally validate the role of proactivity and identify which architectures benefit most. To disentangle perception/extraction vs reasoning/knowledge, have the authors run oracle-evidence experiments that supply the required clip timestamps or bounding boxes? Such upper bounds would localize the dominant bottleneck and guide method development. For judge robustness, would the authors consider releasing a small human-adjudicated “gold set” (balanced across domains and reasoning types) with reference CoTs to calibrate future LLM judges and track metric stability over time? Also, did judge agreement vary systematically by reasoning type (e.g., planning vs causal) or domain (e.g., education vs sports)? Given that adding frames can hurt when evidence count is high, have the authors tried structured evidence-gathering policies—iterative timeline search, subgoal decomposition, or tool-augmented retrieval—to mitigate attention dilution? Even preliminary results would clarify whether the bottleneck is context management or search policy. Finally, since several tasks touch on emotions and social relations, have the authors conducted bias or robustness audits across demographics and cultural contexts? A brief analysis would strengthen ethical readiness for real-world, human-facing applications.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a fairly reasonable pipeline to curate complex VLM evaluation benchmarks across different difficulty levels (Figure 3).
The paper takes extra care to ensure that the quality and difficulty of the dataset is high. For instance, the questions solvable by blind LLM are filtered. In addition, the reasoning questions that one evidence, or lack of reasoning were not kept.  The paper tests a wide range of models in the open and closed world across several model scales. The fine-grained analysis reveals that only reasoning model o4-mini improves consistently with more visual context while the other models fail to do so.

### Strengths
1. The paper proposes a fairly reasonable pipeline to curate complex VLM evaluation benchmarks across different difficulty levels (Figure 3).

2. The paper takes extra care to ensure that the quality and difficulty of the dataset is high. For instance, the questions solvable by blind LLM are filtered. In addition, the reasoning questions that one evidence, or lack of reasoning were not kept. 

3. The paper tests a wide range of models in the open and closed world across several model scales. The fine-grained analysis reveals that only reasoning model o4-mini improves consistently with more visual context while the other models fail to do so.

### Weaknesses
1. The process of human participation is not very clear at different stages of the work. (a) who are domain experts and for what domains? (b) Where are these domain experts sourced from and how many of them were there? (c) Did the same folks also solve the task for human evaluation number in Table 3? (d) Is human eval taken from the same kind of domain experts that made the dataset or are they average human workers on some platform?

2. The writing-style and presentation could be improved. For instance, (a) it is pretty hard to read some figures like 7 and 8. (b) the spacing seems to be altered at many places – hardly any gap between paragraphs on page 3 and 8.

### Questions
Mentioned above

### Soundness
3

### Presentation
3

### Contribution
3
