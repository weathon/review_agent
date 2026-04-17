# OLIIV Benchmark: Does Your VLM Care What You Say, or How You Say It?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 8

## Abstract
Recent advances in vision-language models (VLMs) have improved their ability to perform multimodal reasoning. However, their capacity to consistently follow answer-specification instructions—explicit directives about how responses should be formatted, structured, or composed—remains largely unexplored. This ability is critical for improving user experience and enabling fair and reliable comparisons across models. To evaluate answer specification instruction following, we introduce OLIIV (Open Language-Image Input Variation), a benchmark designed to measure compliance with answer-specification instructions. OLIIV spans four representative task types: multiple-choice reasoning, binary question answering, structured output generation in JSON, YAML, and XML, and length-constrained image captioning. Each task is tested under systematically varied prompt formulations. This is to assess whether models maintain instruction following compliance when input phrasing changes, but the underlying task remains fixed. Results show that many models perform inconsistently across superficially different, but semantically equivalent, prompts. We found that models often behave differently when presented with Roman numerals versus letters in multiple-choice questions, or produce more compliant Yes/No answers than True/False ones—despite identical instructions. To evaluate adherence to length constraints, we introduce the Length Infidelity Score (LIS)—a deterministic, model-agnostic metric for quantifying over- or under-length responses. Structured-output evaluation further shows that models frequently produce syntactically correct but structurally invalid outputs, such as inserting empty fields or omitting required schema elements. Taken together, our findings reveal that all VLM's used in our experiment are highly sensitive to prompt variation. Such sensitivity limits the fairness of current benchmarking methods, OLIIV fills this gap by providing a structured framework to explicitly test how robust VLMs are to semantically equivalent variations in prompts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces OLIIV, a benchmark designed to test whether vision-language models follow answer-specification instructions consistently across semantically equivalent but syntactically different prompts. It covers four task types—multiple-choice, binary QA, structured output, and length-constrained captioning—and proposes the Length Infidelity Score (LIS) to quantify deviations from expected output length. The results show that even state-of-the-art models behave inconsistently under superficial prompt changes, highlighting gaps in instruction-following robustness.

### Strengths
This is a timely and well-motivated contribution that targets an underexplored aspect of multimodal model evaluation. The benchmark is clearly designed, with four intuitive task categories that reflect real-world interaction styles. The proposed LIS metric is simple yet interpretable, allowing for reproducible quantitative comparisons without reliance on human judgment. Experiments are broad, covering both open and closed models, and the structured error analysis provides concrete evidence of systematic failures. Overall, the paper makes a practical and diagnostic contribution to the field of multimodal instruction-following evaluation.

### Weaknesses
The set of tasks and prompt variations, while representative, is relatively narrow, which limits the generality of conclusions. The benchmark does not quantify task complexity or control for difficulty, making it hard to disentangle model sensitivity from intrinsic task variance. Reference to prior work such as Lei et al., 2024 (IWISDM) would help ground this aspect. The method of task generation from COCO and NoCaps is under-specified and lacks examples illustrating how visual content maps to instruction types. The LIS metric also appears incomplete, since the lower bound  is mentioned but unused in the equation, and it may fail to distinguish different types of instruction-following failures. Using a fixed temperature of 0 could underestimate model variability, and the paper does not report evaluation accuracy alongside compliance. Finally, some minor issues—like the typo on page 5, inconsistent results in Table 1 (where frontier models sometimes underperform open-source ones), and the lack of multi-choice reformulations (e.g., binary/MC unification)—slightly detract from presentation quality.

### Questions
- How were tasks generated from COCO and NoCaps, and could the authors show examples of the visual–instruction pairing process?

- Could the LIS be extended to reflect other failure types (e.g., format violations, semantic over-generation) rather than just length?

- How is task complexity controlled or estimated? Would including a metric like in IWISDM (Lei et al., 2024) change the interpretation?

- What is the evaluation accuracy of the models under temperature 0.0—do results differ when stochasticity is introduced?

- Why do some open-source models outperform proprietary ones in Table 1?

- Could multiple-choice and binary-answer questions be standardized into a unified evaluation format to improve comparability?

- Do the authors plan to extend this benchmark to text-only LLMs for cross-modal consistency?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Most existing benchmarks only focus on task-specific accuracy and lack quantitative measurements of the response constraints in instructions such as response formats, output structures, and length constraints. OLIIV is the first benchmark focused on evaluating VLM's ability to follow answer specification constraints, covering four common format-constrained tasks, each tested through semantically equivalent but superficially expressed prompts. This paper also proposes a quantifiable evaluation metric, Length Infidelity Score (LIS), to assess the deviation between model output length and instruction requirements. Through empirical research, it was found that models are highly sensitive to prompt mutations and often produce outputs that are syntactically correct but structurally invalid, which are interesting findings.

### Strengths
- 1. This work focuses on a neglected core issue: systematically addressing VLMs for explicitly measuring robustness to prompt variation and fidelity in structured responses. Clearly stating that current benchmark tests (such as MMMU, MathVista, etc.) rely on post-processing scripts to adapt to different model prompts, conceals the true differences in instruction-following compliance of the models, and affects fair comparison.

- 2. Multi dimensional task evaluation has been designed, covering four common and representative tasks (multiple-choice questions, binary Q&A, structured output, and length limited image description) to ensure accurate identification of model sensitive sources. And a new model independent metric, LIS, was proposed to quantify deviations from expected response lengths.

- 3. Through extensive experimental testing (covering almost all mainstream VLMs), counterintuitive phenomena were found, such as the model being more concise than multiple-choice questions with 'single choice requirements' in true/false problems without explicit length limitations, revealing that the model is more influenced by implicit formatting conventions of task types rather than explicit instructions. It also verified the strictness effect of instructions, such as strengthening instructions by adding prompts such as "STRICTLY REQUIRED".

### Weaknesses
- 1. The types of tasks designed are relatively basic and simple, limited to closed question answering and simple structured output, lacking testing for more complex instruction following scenarios, such as format constraints for multi-step reasoning tasks.
- 2. Single variation dimension: mainly testing surface format changes (Roman numerals vs letters), lacking systematic testing of the following key dimensions: deep reconstruction of instruction semantics (such as positive/negative expressions), processing ability of fuzzy instructions, the influence of cultural or language style differences, etc.
- 3. LIS did not consider the sensitivity differences of length constraints for different tasks.
- 4. In the experiment, there is no comparison between systems of different scales in the same model series (such as 7B vs. 70B parameters), and there is no model variant specifically optimized for instruction-following compliance.
- 5. Lack of deeper discussion, for example, although the sensitivity of the model to prompt variation has been found, it has not been analyzed whether this sensitivity comes from pre-training data bias or the alignment, and the relationship between training strategies (instruction-tuning, RLHF) and instruction-following ability has not been discussed.

### Questions
refer to Weaknesses

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Proposes a new benchmark to measure how well VLMs comply with response format instructions, spanning multiple-choice, binary, structure, and length constraints. Proposes a new metric to quantify deviations, and finds that VLMs can be sensitive across semantically equivalent prompt phrasings.

### Strengths
– The writing is clear and easy to follow

– The paper clearly motivates the problem it is trying to solve

– The finding about VLM sensitivity across prompt phrasings is interesting

### Weaknesses
– The experimental design is somewhat contrived, since most frontier models support structured outputs / constrained decoding at inference, wherein the desired structured output schema can be provided (as JSON/Pydantic schema) and near perfect adherence is guaranteed. The paper does not benchmark structured inference techniques at all, which limits the usefulness of its findings.

– The benchmark does not seem to be sufficiently challenging – simply making the prompt slightly more “strict” seems to lead to near perfect adherence across most proprietary models (LIS in Tables 1-2 is ~0 for most), despite the surveyed models not being the strongest/most-current iterations. Which suggests that the benchmark is already saturated.

### Questions
Please address the weaknesses listed above – in particular, I am not convinced that the benchmark is sufficiently challenging for frontier models nor is the experimental design (only free-form inference) appropriate.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the OLIIV (Open Language-Image Input Variation) benchmark, a new framework designed to evaluate how well vision-language models (VLMs) adhere to explicit instructions for answer formatting, structure, and composition. The benchmark tests VLMs across four representative tasks, including multiple-choice reasoning, binary question answering, structured output generation (JSON, YAML, XML), and length-constrained captioning (using systematically varied prompts to measure robustness). The authors found that all tested VLMs are highly sensitive to superficial prompt variations (e.g., performing differently on "Yes/No" vs. "True/False" questions) and frequently fail to comply with structural constraints, highlighting a key weakness in current models and the need for more robust evaluation.

### Strengths
1. The benchmark provides a necessary evaluation of VLM compliance with formatting instructions, a practical aspect of usability that is largely ignored by existing benchmarks focused on task accuracy.

2. OLIIV's core strength is its systematic variation of prompt phrasing for semantically equivalent tasks. This design effectively isolates instruction-following ability from simple pattern memorization.

3. The introduction of the Length Infidelity Score (LIS) offers a deterministic, model-agnostic, and easy-to-interpret metric for quantifying a model's failure to adhere to length constraints.

### Weaknesses
1.By design, the benchmark focuses on formatting compliance. However, the factual accuracy of the content is also important. While this isolates the target skill, it doesn't explore the potential interplay between a model's ability to be correct and its ability to be compliant simultaneously.

2. The paper's evaluation is confined to four specific task types. It does not assess compliance in more complex generation tasks, such as creating tables, diagrams, or other non-textual structured outputs. While the systematic variations are a strength, the authors acknowledge that future work is needed to explore a wider range of prompting conditions, including more linguistically diverse or even adversarial prompts, to fully test model robustness.

### Questions
In ICLR paepr format, to cite a paper, \citep should be used instead of [\cite]

### Soundness
3

### Presentation
3

### Contribution
3
