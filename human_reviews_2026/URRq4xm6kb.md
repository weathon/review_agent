# Single Answer is Not Enough: On Generating Ranked Lists with Medical Reasoning Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
This paper presents a systematic study on enabling *medical* reasoning
models (MRMs)--which achieve SOTA performance on multiple-choice
benchmarks--to remain robust when producing alternative *answer
formats*. Answer formats define the structure of a final answer in a
generated response, such as an option, free text, or a ranked list.
Although clinical decision-making typically involves weighing multiple
plausible possibilities, current MRMs are trained to produce only one
answer, and their robustness beyond that format is not well studied. We
focus on the *ranked-list* format as an alternative that better reflects
clinical uncertainty. To address this gap, we evaluate *prompting* and
*fine-tuning* for enabling MRMs to generate ranked lists across common
medical benchmarks. While prompting provides a lightweight solution,
MRMs vary widely in their ability to follow such instructions. We
therefore explore supervised fine-tuning (SFT) and reinforcement
fine-tuning (RFT) as stronger adaptation methods. SFT trains models to
imitate ranked outputs, whereas RFT optimizes behavior through reward
functions; we introduce new rewards tailored to ranked-list generation
and analyze their effects through ablations. Our results show that
although some SFT models handle certain formats well, RFT yields more
consistent robustness across multiple answer formats. A case study on a
modified MedQA benchmark with multiple valid answers further reveals
that MRMs can recognize clinically sound alternatives even when
misaligned with a benchmark's preferred ground truth. To the best of our
knowledge, this is the first systematic investigation of adapting MRMs
to alternative answer formats such as ranked lists. We hope this study
lays the foundation for developing more flexible and clinically aligned
MRMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the alignment problem between the multiple choice question format and real world medical complexity where a single answer may not be enough to accurately represent the best next steps and uncertainty due to incomplete information.

The authors propose to transform existing MCQ evaluations into a ranked list and open QA format to compensate the limitations of the MCQ format. Using these conversions they perform multiple performance comparison and proceed to use different techniques including prompt engineering, SFT and RFT to study the impact of different methods to adapt existing models to new formats.

### Strengths
Moving from MCQ to more representative methods is well motivated. In addition, the authors provide a comparison of proprietary models, general open models and medical finetuned models. The three methods used to train models provide an interesting comparison of these techniques to adapt existing models and show how RFT generalizes better to new formats.

Overall the paper is well written and easy to follow with a clear graphical abstract that gives a good overview of the contribution and the structure of the paper.

### Weaknesses
While the motivation to move beyond standard accuracy on single choice MCQ is detailed and sound, the authors do not discuss prior work that introduced different testing methodologies (e.g. AI Hospital, MetaMedQA, MAI-DxO, HealthBench) [1-4]. The authors do not sufficiently position their work compared to other approaches to justify the use of ranked lists and possibly explain how ranked lists could be integrated in non MCQ formats.

The reliance on existing benchmarks such as MedQA and MedMCQA imports the known limitations of board style vignettes which mostly test pattern recognition with shortcuts [4, 5]. In addition, the authors do not discuss the alignment between board style vignettes and real-world clinical work, for instance, the absence of longitudinal and noisy data severely limits their relevance to real world data. As the authors explain in the limitations, the source datasets are single answer which limits the use of ranked lists where multiple options can be correct and their relative priority informs us on the understanding of appropriate next steps or diagnosis. Finally, these QA samples are not representative of medicine as a whole and priorities may shift depending on the context, for instance, in a high resource setting for a patient with a suspected STEMI we direct the patient to a cath lab immediately but in lower resource settings we would consider thrombolysis as a first step. A ranked list would fail to accurately represent two correct answers depending on context, these limitations are not discussed.

The ablation studies use relatively small models due to compute availability which limits generalizability of the results. A single experiment with a large model would help in demonstrating the scalability of this method.

[1] AI Hospital: Benchmarking Large Language Models in a Multi-agent Medical Interaction Simulator (Fan et al., COLING 2025)

[2] Large Language Models lack essential metacognition for reliable medical reasoning (Griot et al., Nature Communications 2025)

[3] Sequential Diagnosis with Language Models (Nori et al. Preprint 2025)

[4] HealthBench: Evaluating Large Language Models Towards Improved Human Health (Arora et al. Preprint 2025)

[5] Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine (Griot et al., ACL 2025)

[6] Language Models are Surprisingly Fragile to Drug Names in Biomedical Benchmarks (Gallifant et al., Findings 2024)

### Questions
Suggestions:

1) Improving the related work introduction to better position this paper in the boarder realm of medical evaluations for LLMs would strengthen the motivation for this particular approach.

2) A small (n = 50-100) benchmark designed by experts with multiple ranked options as a test evaluation would strengthen the findings and support the proposed methods.

3) Discussing existing limitations of the source datasets and how the authors compensate for these limitations to support their conclusions would improve the validity of the experiments.

4) Discussing the alignment problem between automated testing and real-world applications is necessary to position the work as the goal of application research is to translate to real-world benefits. For instance, training models to use a ranked list format is not motivated in relation to the applicability of these lists in real-world settings. When in clinical workflows would this be useful?

5) Scaling the ablation to a larger model would add credibility to the approach to specialize models to new formats.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors highlight concerns about the tendency of medical reasoning models to produce a single final answer even in open-ended settings. They explore prompting and fine-tuning methods in order to produce ranked lists of outputs, including reinforcement fine-tuning and supervised fine-tuning across several models.

### Strengths
There are a few strengths of this paper. The issue of overly narrow benchmarks failing to consider the full breadth of clinical responses in evaluation is a valid one (and is in line with broader concerns in the field about the limitations of MCQs). Better and more nuanced assessment of the quality of differential diagnoses and clinical reasoning is important. 

The authors apply appropriate technical rigor to the implementation of methods of supervised and reinforcement fine-tuning. They also engage with an interesting array of medical-specific models, even if frontier models are absent.

### Weaknesses
Speaking frankly, it is not entirely clear to me that the authors substantiate the concern underlying this paper. Both in my personal experience with these models as well as in the literature I am familiar with (e.g. https://arxiv.org/abs/2412.10849 and others), it is relatively trivial to instruct any of the state-of-the-art language models to produce a clearly ranked differential diagnosis list (e.g., multiple outputs). While it is true that overconfidence in models can be a problem (e.g. https://ai.nejm.org/doi/pdf/10.1056/AIdbp2500120), it remains much more complex than simply a limitation in the length of outputs. 

Perhaps these problems are more substantial for specific medical reasoning models like some of those models that the authors use (I have rarely seen these evaluated or used), but this does not appear to be an issue for the main frontier models. Unfortunately, the authors only include extensive evaluation of these various undersized models, and extrapolate conclusions more generally to LLMs as a class based on them. This is a severe limitation of this work, and although it can be cost-prohibitive, this work really should be performed at the frontier. 

Further, it is not clear that the ranked list generation methods that the authors employ actually map cleanly onto any of the aspects of list ranking that are valued by clinicians in reality. For example, a robust differential diagnosis should also include "rare but dangerous" considerations, or "rare but treatable" considerations in order that they are fully considered by the clinicians at hand, even if these would not technically maximize score based on rank list structure (see discussions of this in e.g. https://pmc.ncbi.nlm.nih.gov/articles/PMC3270234/ and https://www.nature.com/articles/s41586-025-08869-4/). Ultimately, the assessment of ranked lists is an important and nuanced question, that depends on the specifics of the clinical problem at hand. Basic probabilistic metrics, or knowledge-based metrics like their expanded medQA are insufficient. 

The use of LLM-as-judge is inappropriate without at least some degree of human validation by physicians. Medical answers and diagnoses are very complex and nuanced, and it cannot simply be assumed that the semantic equivalence matching is high-quality. While it is reasonable to use a validated LLM-as-judge as an extension, you must use core evaluation methods for any specific implementation. The entire section evaluating different judge models is incoherent in this context. You cannot state that "to change the judge model significantly impacts performance", because the underlying performance of the systems are the same. What has changed is the performance onf the judge model, which the authors have not appropriately validated.

### Questions
1. Are these concerns about the brevity of model answers substantiated in frontier models?
2. How do these concerns about answer length manifest across the range of various clinical tasks, beyond multiple choice? Does this apply to differential diagnosis alone? What of clinical decisionmaking in other realms?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper argues that clinical decision-making inherently involves considering multiple possible answers, yet current medical reasoning models (MRMs) are trained to produce only one. To address this limitation, the authors propose enabling MRMs to generate ranked lists of plausible answers through prompting, supervised fine-tuning (SFT), and reinforcement fine-tuning (RFT) with novel ranking-based reward functions. Experiments across medical QA benchmarks show that RFT models generalize more robustly across answer formats (MCQ, QA, list) than SFT models, and that ranking-oriented rewards (e.g., MRR, LLM-judge) improve list quality. A case study on a modified MedQA dataset with multiple valid answers further demonstrates that MRMs often recognize correct alternatives overlooked by single-answer benchmarks.

### Strengths
1. The evaluation is comprehensive, covering a wide range of models and datasets, which convincingly demonstrates the effectiveness and generality of the proposed methods. 

2. The case study on MedQA with multiple valid answers is particularly interesting and provides valuable insights into the limitations of single-answer benchmarks and the potential of ranked-list reasoning in clinical contexts.

### Weaknesses
1. Overstated novelty claim (“first systematic study”). Several recent works have already prompted LLMs to generate ranked differential diagnoses and evaluated them using top-k or position-aware metrics across diverse medical datasets and model families [1, 2, 3]. The paper should more clearly delineate its unique contribution, maybe emphasizing the reinforcement fine-tuning (RFT) reward design or curriculum learning over answer formats.

2. Limited analysis of SFT vs RFT generalization. The claim that “SFT-MCQ generalizes across formats while RFT generalizes both across formats and unseen examples” is intriguing but under-explained. The concept of answer format is not formally defined, and the paper lacks controlled ablations or causal evidence clarifying why RFT achieves superior transfer.

[1] Zhou, Yuxuan, et al. "Reliable and Diverse Evaluation of LLM Medical Knowledge Mastery." The Thirteenth International Conference on Learning Representations. 

[2] Lin, Tianwei, et al. "HealthGPT: A Medical Large Vision-Language Model for Unifying Comprehension and Generation via Heterogeneous Knowledge Adaptation." Forty-second International Conference on Machine Learning. 

[3] Lim, Seungseop, et al. "H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis." The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance.

### Questions
1. On novelty and positioning: How does this work differ concretely from prior ranked-differential evaluations

2. On generalization mechanics: Could you provide controlled ablations isolating why RFT transfers across formats—such as (a) keeping compute and data identical for SFT vs RFT, and (b) varying prompt templates to rule out prompt or distribution confounds?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper conducted a systematic study of steering Medical Reasoning Models (MRM) via prompting and fine-tuning for generating ranked-list outputs. 

The authors introduced an evaluation framework covering three answer formats and shown that many models know the correct answers but often fail to select the benchmark-preferred one.

They then conducted a comprehensive study of SFT and RFT under specific answer formats, finding that SFT with MCQ generalizes well across formats, while RFT models generalize both to unseen examples and across answer formats.

The paper also provided practical insights into RFT design choices, including reward design and curriculum-style sequencing of RFT to improve answer robustness.

### Strengths
This is a comprehensive systematic investigation of methods to enable MRMs to generate answers as ranked lists, a useful alternative to single answers in medical QA.

The ranked-list format aligns with clinicians’ real-world decision processes (e.g., weighing multiple differentials or treatments), which can reduce risk from over-reliance on a single LLM output. 

The paper also explored reward designs for RFT on ranked lists and conducted ablations to identify optimal design choices such as the training curriculum, which is are actionable for the community and likely to generalize to other medical reasoning settings.

### Weaknesses
While the ranked-list format is a promising alternative to single-answer outputs, if a few incorrect answers are included and passed to clinicians, wouldn’t that still mislead or contribute to diagnostic failures? Does your training discourage listing incorrect options (e.g., via loss shaping or negative rewards)? A discussion of error propagation and interpretive risks from ranked outputs would clarify the safety implications.

As noted in the limitations, the ranked lists do not reflect probability magnitudes. Do you have strategies, via prompting or training, to elicit calibrated probabilities (e.g., model-reported confidence scores) or to learn them?

How does a simple baseline that ranks options by model likelihoods over the option tokens (e.g., a clinician pre-select top 5 likely diagnosis options) compare to explicit ranked-list training in terms of answer accuracy? Adding this baseline would quantify the added value of ranked-list training over simply outputting top choices, especially given your claim that models often already “know” the correct answer.

Did you include qualitative examples in the appendix illustrating SFT vs. RFT behaviors and their output formats? Including such examples would help readers interpret results sections like “RFT-QA Exhibits List-like Behavior” and “Reward Function Effects With RFT-List.” (If present, please point to them; if not, consider adding.)

Minor: please fix small typos in the introduction. Sections 3 and 4 both describe experimental setups; reducing subsection titles and headings could improve readability.

### Questions
Just for clarification, in 'Reward Function Effects With RFT-List', when you say 'RFT performs robustly on non-list formats and 
generalizes better to unseen answer formats, what exactly do you mean: do you mean that RFT also obeys other formats you defined in training well other than ranked-list, or do you mean RFT sticks with ranked-list outputs even that there is instruction on other answer format?

### Soundness
3

### Presentation
2

### Contribution
2
