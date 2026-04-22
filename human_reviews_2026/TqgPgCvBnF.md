# PhysUniBench: A Multi-Modal Physics Reasoning Benchmark at Undergraduate Level

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Physics problem-solving is a challenging domain for large AI models, requiring integration of conceptual understanding, mathematical reasoning, and interpretation of physical diagrams. Existing evaluations fail to capture the full breadth and complexity of undergraduate physics, whereas this level provides a rigorous yet standardized testbed for pedagogically relevant assessment of multi-step physical reasoning. To this end, we present PhysUniBench, a large-scale multimodal benchmark designed to evaluate and improve the reasoning capabilities of multimodal large language models (MLLMs) specifically on undergraduate-level physics problems. PhysUniBench consists of 3,304 physics questions spanning 8 major sub-disciplines of physics, each accompanied by one visual diagrams. The benchmark includes both open-ended and multiple-choice questions, systematically curated and difficulty-rated through an iterative model-in-the-loop process. The benchmark's construction involved a rigorous multi-stage process, including multiple roll-outs, expert-level evaluation, automated filtering of easily solved problems, and a nuanced difficulty grading system with five levels. Through extensive experiments, we observe that current state-of-the-art models encounter substantial challenges in physics reasoning. For example, GPT-5 achieves only about 53.7% accuracy in the proposed PhysUniBench. These results highlight that current MLLMs struggle with advanced physics reasoning, especially on multi-step problems and those requiring precise diagram interpretation. By providing a broad and rigorous assessment tool, PhysUniBench aims to drive progress in AI for Science, encouraging the development of models with stronger physical reasoning, problem-solving skills, and multimodal understanding. The benchmark and evaluation scripts are available at https://anonymous.4open.science/r/PhysUniBenchmark-5784.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces PhysUniBench, a large-scale multimodal benchmark specifically designed to evaluate AI models on undergraduate-level physics problems. The benchmark contains 3,304 carefully curated physics questions spanning 8 major sub-disciplines (Classical Mechanics, Electromagnetism, Optics, Quantum Mechanics, Thermodynamics, Solid Physics, Relativity Physics, and Molecular/Atomic/Subatomic Physics). The benchmark has EN and CN language coverage and has MC as well as Open-Ended QA (OE) in terms of question format, stratified across 5 difficulty level and evaluated using a variety of frontier (M)LLM models.

### Strengths
1. Multimodal:  It's nice that each question has exactly one diagram, (altho it also raises the question as to sometimes in the real use case of AI4Science the ability to interpret multiple images and cross-ref between them could be a valuable capability to probe)
2. Difficulty Rating: The authors wrote that each problem is annotated with a fine-grained difficulty level ranging from 1 to 5, which then in turn shows a corresponding performance gradient in the evaluation phase (harder problems = lower scores) which is good for testing MLLMs on a difficulty gradient. (Note: I later realized the difficulty rating was also based on LLM performance, which may be considered circular reasoning and could be strengthened by e.g. human-AI cross-validation and other checks for construct validity)
3. I also noticed that the stratified difficulty has roughly similar number of questions around 660, which is a nice property for statistical analysis in general when one wants to test the model performance across different difficulty level without having to worry about sample size as a confounder

### Weaknesses
Disclaimer: I come from a mix of CS+Physics background and have at least graduate student level of physics knowledge, so I believe I am qualified to speak to the validity to an undergraduate level physics benchmark.

APS: American Physical Society (which is generally regarded as one of the most credible non-profit org on physics research)

1. Trivial mistakes in the Physics discipline: I found some of the mistakes made in this paper to be quite trivial:

For example:"Solid Physics" is not the correct term in physics, but it should be "Solid State Physics" or "Solid-State Physics" 
I encourage everyone to simply do a Google Search of the word "Solid Physics" you will find everything is about "Solid-State Physics" because "Solid Physics" is not really a term we use in Physics.

[Start quote from Wiki] Solid-state physics is the study of rigid matter, or solids, through methods such as solid-state chemistry, quantum mechanics, crystallography, electromagnetism, and metallurgy. It is the largest branch of condensed matter physics. [End quote]

(This is not some proprietary knowledge, but a simple Wikipedia search would yield this explanation/definition as to what it means and what the correct term is)

2. Therefore, I respectfully question if there were indeed "expert curation" going on by the expert of physics in constructing such a benchmark, how could these "experts" have not noticed such a basic mistake in the terminology? This gave me no choice but to question the validity of the claimed expert construction process, because it's hard to believe that if any of the annotation was actually done by experts, or even inspected/guided by experts, they would not notice this basic mistake.

3. Very similarly to the point I raised above,"Relativity Physics" is also not really a term we often use in Physics, because the Theory of Relativity has both "Special Relativity"(which applies to all physical phenomena in the absence of gravity.) as well as "General Relativity" (which explains the law of gravitation and its relation to other laws of physics) where GR is often classified into Gravitation, Cosmology and Astrophysics (this term is stipulated by APS). So it's also quite confusing to me as to why the authors choose to specifically list "Relativity Physics" in parallel on the same level of e.g. "Quantum Mechanics" because in Physics they are definitely not classified to be on the same level. (You can e.g. check PhySH published by the APS)

4. Apart from trivial mistakes in Physics, the coverage of this benchmark has a couple of flaws:

a. The domain distribution is quite uneven, where EM and CM constitutes nearly half and the rest are much smaller

b. There are domains which are totally not covered like Astrophysics and Cosmology, Quantum Optics, Quantum Info etc. depending on the curriculum you choose to follow I can often see some of these classes being offered to undergrad in many university-level institutions across various countries, especially to physics major in their final year(s).

5. Many of the details aren't super clear to me: For example, if we are trusting LLM with stratification and LLM-as-a-judge for OE questions, are there any validity check by experts to see if they actually do a good job? There should be a inter-annotator or human-AI cross-validation so as to see if there's any false positive/negatives and report the statistics accordingly

6. Also, there lacks a human baseline by different level of human candidates, or at least undergraduates?

7. For difficulty stratification: If you are relying on MLLM 16-rollout success rate to classify the questions, aren't you kind of circular reasoning here in terms of:

a. Use model accuracy to classify the questions from 0 to 5

b. Then say ok this classification is good because models indeed perform correspondingly, but that's actually by design, right? since it is indeed the standard by which you partition the questions in the very first place, so this point in discussion seems like circular reasoning to me. Granted you could say the partition is done by a single model and the eval on many models, but the pretraining corpora of these models are likely overlapping a lot, so I don't think this classification makes much sense to me as to how it's designed.

8. The related work coverage and comparison can be more comprehensive, some general benchmarks like ARB has a physics portion and more dedicated benchmarks like HiPhO and SeePhys and Multi-Physics should be more comprehensively compared in this paper

9. Also, the evaluated models could be more comprehensive (only 2 text-based LLM, maybe for multimodal benchmark you don't really need that? Or if you want to compare LLM against MLLM you should have roughly similar number of models evaluated on each side?) and more thorough error analysis could significantly strengthen this work. 

Therefore, I believe this work has significant improvement to be done that could better contribute to the community at large, I encourage the authors to take some more time for improvement. And while we appreciate the author's good effort in constructing this benchmark, it is essential to remember for all of us that AI4Science benchmark (not just in physics, but any other domains as well) should have some input from domain experts in respective field to ensure their construct validity.

### Questions
I've mostly put the questions inside the last section of weakness. Some additional ones

1. For evaluated models: Even for MLLMs, it's mostly GPT models, also o3 and o4-mini are o-series not GPT-series, why isn't Claude/Gemini/Grok/other series as comprehensively tested as GPT/OpenAI models? It would probably be nice to test scaling and time evolution of many families of models, or different size models of the same open-sourced families? In general I think the evaluation and discussion section can be improved significantly;

2. For the OE judge logs can you show some examples of model output against LLM judge verdict, so that we can see how exactly they perform in terms of judging the output? I fear rubric-based grading may not be very optimal given GPT-4o's limited capability.

3. Maybe the authors can better present the novelty of this work other than "multi-phase construction by advanced AI models" because one could always just use stronger models to get a much more difficult benchmarks in that case (e.g. replace Qwen-VL with Gemini-2.5-Pro or GPT-5-High?) then the benchmark would be much less prone to saturation over time comparing to the current version?

4. Also, can you better articulate how OE questions are judged? The current Figure 1 has some of them just "Steps" and then the answer is contained in the last step, but some others has a dedicated "Answer"? Are OE questions judged based on "Answer" entry and not on the intermediate process? If on the intermediate process maybe you could shed some light (e.g. by giving an example in the main text, to show this grading, which would imo strengthen this work a lot)

5. As previously mentioned could you better organize the appendix with a Table of Contents? It takes a lot of time for reviewers to read thru >20 pages of appendix without a clear structure.

6. Although this may not warrant an ethics review, how exactly have the authors complied with licensing/data protection laws in the process of curating this benchmark? This may warrant a clearer presentation in Section 3. Currently most of the descriptions are really vague and high-level only.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PhysUniBench, a large-scale, multimodal benchmark designed to assess undergraduate-level physics reasoning. The benchmark consists of 3,304 physics problems, each with accompanying diagrams, covering eight core physics subfields, including classical mechanics and electromagnetism. A key feature of this work is the dataset construction through a meticulous, multi-stage "model-in-the-loop" pipeline, which involves using a powerful multimodal model (Qwen2.5-VL-72B) for preliminary answers, filtering out simple problems, and applying a fine-grained five-level difficulty grading to the remaining problems. The benchmark includes both open-ended (OE) and multiple-choice (MC) questions. The authors used PhysUniBench to conduct an extensive evaluation of state-of-the-art large multimodal models (MLLMs). The experimental results show that even state-of-the-art models (such as GPT-5, discussed in the paper) face significant challenges in handling these problems, highlighting the shortcomings of existing models in high-level physics reasoning, particularly those involving multi-step reasoning and precise diagram interpretation. This work aims to advance the field of artificial intelligence by providing a rigorous evaluation tool.

### Strengths
(1) Significant and Well-Defined Research Gap:

This paper successfully addresses a significant gap in existing physics benchmarks: the lack of a fully multimodal, undergraduate-level benchmark. It strikes a good balance between K-12/Olympiad-level and text-based university-level benchmarks, providing a suitable platform for assessing the comprehensive reasoning skills crucial for future scientists and engineers.

(2) Rigorous Dataset Construction Process:

The dataset construction process described in the paper is very systematic and rigorous. 16 rollouts were performed using a model (Qwen2.5-VL-72B) to filter out "too easy" questions, and the difficulty of the questions was graded based on success rate, ensuring the challenging nature of the benchmark.

(3) Comprehensive Evaluation and Insightful Analysis:

The authors evaluated a large number of currently popular closed-source and open-source MLLMs and provide detailed results. The analysis goes beyond overall accuracy and includes in-depth analysis across sub-disciplines and difficulty levels, including comparisons of open-ended and multiple-choice questions. In particular, the ablation study on image vs. caption inputs is highly valuable, clearly highlighting the bottlenecks of current models in visual information extraction and physical scene understanding.

(4) Scale and Coverage: 

The benchmark is characterized by its scale (3,304 problems), broad disciplinary coverage (8 core areas), and fully multimodal nature (with accompanying images for each problem).

### Weaknesses
(1) Potential Bias in Difficulty Stratification:

The difficulty stratification of the entire benchmark relies entirely on the performance of a single model (Qwen2.5-VL-72B). This can lead to difficulty ratings that are biased by that model. A problem that is difficult for the Qwen model may not be difficult for a model with a different architecture or training data (e.g., the GPT series), and vice versa. This may affect the generalizability of the difficulty stratification.

(2) Contradiction in Multiple-Choice Question Design:

The paper converted the "hardest" open-ended questions, which the model failed to solve in all 16 attempts, into multiple-choice questions. While using the model's own incorrect answers as distractors is a clever design decision, it fundamentally reduces the difficulty of the questions—the model now has a 25% chance of success, similar to random guessing. This seems to contradict the original intention of retaining these questions as the "hardest challenges."

(3) Limitations of the Evaluation Method:

While using GPT-4o as a judge for open-ended questions is a common and scalable approach, its limitations are well known, including potential consistency issues, a preference for specific representations, and an inability to fully capture the nuances of complex physical reasoning. While the paper mentions the use of human evaluation in ambiguous cases, it does not quantify the extent of this human involvement.

### Questions
(1) Regarding the use of a single model for difficulty grading:

Difficulty grading is one of the contributions of this work, but it is entirely based on the performance of a single model, Qwen2.5-VL-72B. Can the authors elaborate on why they chose not to use an "evaluation committee" composed of multiple models with different architectures for difficulty calibration? Doesn't this run the risk of creating difficulty grading that is heavily biased towards the specific strengths and weaknesses of the Qwen model and thus fails to generalize well to a wider range of MLLMs? Have the authors considered using other state-of-the-art models (such as Claude-3.5-Sonnet or GPT-4o) to cross-validate the rationale for these difficulty gradings?

(2) Regarding the conversion of the most challenging open-ended questions:

The paper mentions converting the most challenging open-ended questions that Qwen2.5-VL-72B has never successfully answered into multiple-choice questions. I understand the idea of ​​using the model's error output to construct distractors, but this seems to fundamentally reduce the "absolute difficulty" of these questions, as the model can now achieve 25% accuracy by random guessing. Can the authors address this potential contradiction? How does this design choice ensure that the MC problem set does not reduce the difficulty of its OE type questions? Have other ways of dealing with these most difficult problems been considered?

(3) Analysis of the bottleneck of visual reasoning:

The analysis in Figure 4 (the model performs better when using text descriptions than when using original images) is a key finding. This suggests that there is a bottleneck in the current visual encoder.
This may be partly due to the fact that the text description (generated by the model) itself contains refined reasoning clues that are often not obtained from a single observation of the image.
Can the authors provide a specific example showing a physical diagram and its corresponding "long text description" (long caption) to help us understand the degree of abstraction and explanation contained in the text description?

(4) Suggestions for qualitative error analysis:

The paper provides excellent quantitative performance analysis. If some qualitative error analysis can be supplemented, it will greatly enhance the value of the paper.
Can the authors provide some typical failure analysis? For example, is the model more likely to make conceptual errors (misapplying physical laws), mathematical calculation errors, or multimodal alignment errors (misunderstanding the information in the diagram)? This analysis will provide more targeted guidance for future model development.

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
3

### Summary
The paper introduces PhysUniBench, a large-scale multimodal benchmark for undergraduate-level physics reasoning. It contains 3,304 problems across 8 sub-disciplines, each paired with a diagram image, with both open-ended and multiple-choice formats and five difficulty levels calibrated via model-in-the-loop rollouts. The authors evaluate a range of MLLMs. An ablation replacing images with auto-generated captions improves accuracy, suggesting current models’ weaknesses in physics-diagram understanding. The benchmark, construction pipeline, and evaluation scripts are released for reproducibility.

### Strengths
- The benchmark targets undergrad physics with calibrated difficulty and multilingual EN/ZH support
The paper uses a model-in-the-loop curation to remove trivially solvable items and to stratify difficulty; this is a thoughtful twist on dataset construction
- There is a clear dataset stats and coverage across different subfields, with balanced difficulty bins
- The caption ablation is a smart diagnostic revealing a likely bottleneck in visual/diagram understanding for physics
- The results tables disaggregate by sub-discipline and difficulty, highlighting where models fail

### Weaknesses
- Difficulty calibration (Qwen2.5-VL rollouts) and LLM judging with GPT-4o may inject model-specific biases; it’s unclear how sensitive results are to the choice of judge/rollout model or to prompt templates. A cross-judge analysis (or human spot-checks) would strengthen validity
- Potential data contamination: It is not clear where exactly the datasets are from. The paper mentioned that problems came from textbooks/exams/competitions, but many may already appear online. The paper does not quantify near-duplicate overlap with existing pretraining/test corpora (e.g., UGPhysics, PhysicsArena, PhysReason), risking optimistic or inconsistent estimates
- The “caption > image” result is interesting, but captions are generated by GPT-4o -- a strong prior that may inject solution-relevant structure (naming vectors/relations), not just a faithful description. In other words, the result doesn't necessarily prove that "text > images" for physics reasoning; it might just prove "having GPT-4o pre-analyze the problem helps."
- As the paper claims that having multilingual evaluation is one of their main contributions (what separates their benchmark from other similar benchmarks), the English/Chinese evaluation results should be included in the main sections of the paper

### Questions
- Where do the ground-truth answers come from? The authors mention that they evaluate against a “known gold solution” in the Appendix, then fall back to SymPy equivalence or an LLM judge (GPT-4/4o). It seems like the paper implies the gold solutions are from the source materials (textbooks/exams/exercise sets), but it doesn’t spell out a per-item list, or explain if there are specific cases where they had to involve humans for ground-truth answers
- Are there any overlap tests that were done to see how similar physics-related benchmarks (e.g., UGPhysics, PhysicsArena, PHYX, PhysReason) are related to PhysUniBench? How did the authors ensure that there were no duplicate questions (within PhysUniBench itself and also compared to other benchmarks)?
- Beyond just describing the accuracy results from the experiment tables, it would be nice to know what exactly caused the mistakes in each subcategory. For example, is it diagram misread? Using the wrong formula? Algebra slip? etc.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents PhysUniBench, a large-scale benchmark for evaluating undergraduate-level physics reasoning in multimodal large language models (MLLMs). It includes 3,304 human-verified problems with diagrams, five difficulty levels, and both open-ended and multiple-choice formats. Experiments show that current MLLMs perform poorly, underscoring the challenges of multimodal physical reasoning.

### Strengths
- First comprehensive multimodal benchmark for undergraduate-level physics reasoning.

- Rigorous curation process with difficulty calibration and quality control.

- Extensive evaluation across sub-disciplines provides clear diagnostic insights.

- Well-written and potentially impactful for advancing AI-for-Science.

### Weaknesses
### 1. Lack of actionable guidance for model improvement 
While the benchmark offers valuable insights into the limitations of current MLLMs, the paper does not sufficiently explore how architectural or training modifications (*e.g.*, physics-informed modules, structured reasoning layers, or symbolic integration) could help enhance physical perception and reasoning.

### 2. Limited analysis of reasoning failures
The error analysis mainly reports accuracy drops across sub-disciplines and difficulty levels, but lacks qualitative or process-level diagnostics showing why models fail (*e.g.*, confusion between symbolic manipulation vs. conceptual understanding). Such analysis could better guide future model design.

### 3. Benchmark orientation without methodological contribution
PhysUniBench is primarily an evaluation resource, and while comprehensive, it lacks a corresponding methodological or modeling innovation that demonstrates how benchmark insights could translate into improved multimodal reasoning systems.

### Questions
1. In Table 3, for Open-ended Questions, GPT-5 significantly outperforms all other MLLMs on RE and QM, while the rest of the models exhibit nearly identical results (2.1% in RE, 0% in QM except GPT-4o with 6.2%). This trend deviates sharply from the patterns observed in other evaluation dimensions, suggesting potential systematic issues or bottlenecks in data construction, evaluation protocols, or metric design for this sub-task.

Moreover, the reported 2.1% accuracy is mathematically inconsistent with a test set of 80 samples, raising concerns about statistical validity:

    - If 1 sample were correct, the implied dataset size would be 1 / 0.021 ≈ 47.6, which contradicts the stated size of 80;

    - If 2 samples were correct, the implied size becomes 2 / 0.021 ≈ 95.2, which exceeds 80.
Without clarification on additional scoring mechanisms (*e.g.*, partial credit, weighted averaging, repeated sampling, or a different denominator), this value lacks a consistent explanation. The authors are encouraged to provide the exact accuracy computation method or the raw number of correctly predicted samples.

2. I suggest including **difficulty × sub-disciplines evaluation statistics**. In my view, analyzing difficulty stratification within each sub-discipline is essential to better characterize the model’s capability boundaries and failure modes, and would substantially strengthen the empirical findings.

### Soundness
3

### Presentation
3

### Contribution
2
