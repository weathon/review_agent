# Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Recently, Multimodal Large Language Models (MLLMs) have achieved exceptional performance across diverse tasks, continually surpassing previous expectations regarding their capabilities. Nevertheless, their proficiency in perceiving emotions from images remains debated, with studies yielding divergent results in zero-shot scenarios. We argue that this inconsistency stems partly from constraints in existing evaluation methods, including the oversight of plausible responses, limited emotional taxonomies, neglect of contextual factors, and labor-intensive annotations. To facilitate customized visual emotion evaluation for MLLMs, we propose an Emotion Statement Judgment task that overcomes these constraints. Complementing this task, we devise an automated pipeline that efficiently constructs emotion-centric statements with minimal human effort. Through systematically evaluating prevailing MLLMs, our study showcases their stronger performance in emotion interpretation and context-based emotion judgment, while revealing relative limitations in comprehending perception subjectivity. When compared to humans, even top-performing MLLMs like GPT4o demonstrate remarkable performance gaps, underscoring key areas for future improvement. By developing a fundamental evaluation framework and conducting a comprehensive MLLM assessment, we hope this work contributes to advancing emotional intelligence in MLLMs. Project page: https://github.com/wdqqdw/MVEI.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new annotation framework for MLLMs in the task of visual emotion estimation. The paper identifies limitations in the evaluation of MLLMs (namely, inaccurate responses, limited taxonomies, a focus on intrinsic image attributes, and reliance on majority voting) and first introduces a new task to address these limitations. Then, an automated annotation pipeline leveraging an ensemble of MLLMs is used to annotate 400k images and create fine-grained annotations that are used to produce statements with GPT4 and curated by a human expert. The labels and statements are further processed to produce additional coarse annotations (in Parrott-based hierarchy and positive/negative polarity). Finally, several popular MLLMs are benchmarked on the test set of the dataset, including an in-context learning and a finetuned version of Qwen.

### Strengths
- The problems identified by the paper are valid. The uncertain nature of human emotions (invoked, perceived or otherwise) is one of the key limitations in achieving high accuracy in the task, opposed to object detection, for example. Cultural barriers, inconsistent hierarchies, etc, are all issues in affective computing (and psychology for that matter).
- The proposed method is indeed scalable and can be replicated to extend the proposed dataset or correct the labels of existing benchmarks.
- With the rise of MLLMs, there is a growing interest in applying them to downstream tasks; however, the lack of annotations in natural language makes things more complicated, which is something this method addresses.

### Weaknesses
- One major weakness is the contradiction in motivation. Specifically, two of the key issues identified in the intro are a) MLLM's inaccurate responses when compared to human judgment and b) majority voting to address disagreement. The method then proceeds to create annotations by taking the majority voting of MLLMs. This is reiterated in the discussion (lines 448-9): "MLLMs may not yet be sufficiently competent for LLM-as-a-judge [...]". If this is the case, how are they sufficient as annotators? The method is contradictory to the problem it is trying to solve and thus is not convincing in terms of annotation accuracy. The one human expert curating statements post-hoc is probably not enough, as: a) there is already unconscious bias in the process as the expert is shown annotations rather than producing annotations, b) the task is uncertain even for experts, therefore, is one person really enough?, c) the method does not account for cultural differences and biases (both in the expert and the process as a whole). We also don't have an indication of how the humans (expert or otherwise) perform on the open-vocabulary compared to the LLMs. In general, I am unconvinced about the validity and motivation of the method.
- Looking at the human refinement process, the annotator agreement in Table 3 does not seem very strong, particularly for emotion interpretation and subjectivity, showing a (Cohen's?) Kappa of only 0.5. In fact, the only aspect where the human annotators agree is the scene context; for everything else, there is only weak agreement.
- For the fine-tuned results in Table 6, only 10k out of 462k images are selected. This seems a bit of an odd choice, as most large-scale affective datasets (including EmoSet) are much larger; therefore, it seems unlikely that the finetuned version will be able to compete, but at least performance gains from fine-tuning are shown.

### Questions
- How do the annotations generated by the proposed method in step [a] compare to the ones from EmoSet for the subset used in this work? Are the generated emotions in agreement with the 8 categorical used there?
- Using a small subset, is it possible to compare human open-vocabulary to MLLM open-vocabulary annotations? Are these in agreement or at least in the same region?
- The polarity underperformance may be due to the prompting used. Have you tried CoT reasoning? Polarity is a derivative, not a primary label, hence the suggestion.

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
4

### Summary
This paper introduces a new evaluation paradigm for assessing multimodal large language models (MLLMs) on visual emotion understanding. Instead of traditional emotion classification, the authors propose an Emotion Statement Judgment (ESJ) task, where models evaluate whether a given emotion-related statement about an image is correct. The paper further presents an automated pipeline (INSETS) to construct large-scale emotion-centric statements and builds the MVEI benchmark covering four dimensions: sentiment polarity, emotion interpretation, scene context, and perception subjectivity. Experiments show that current MLLMs demonstrate certain capabilities but still lag behind humans, especially in subjective and contextual emotional reasoning.

### Strengths
Comprehensive emotional dimensions, covering aspects that existing benchmarks ignore, such as scene context and perception subjectivity.

Extensive experiments on a wide range of MLLMs with meaningful insights and human comparison results.

### Weaknesses
The dataset is partially constructed using existing MLLMs, which may introduce inherited biases. Although human refinement is applied, its proportion remains limited.

The distinction from EEmo-Bench: A Benchmark for Multi-modal Large Language Models on Image Evoked Emotion Assessment is not sufficiently articulated.

The dataset is built on a single source domain, and its generalizability to other visual contexts is unclear.

### Questions
Please refer to the weakness.

### Soundness
3

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
This paper addresses the incompatibility of conventional emotion evaluation methods with Multimodal Large Language Models (MLLMs) by proposing the Emotion Statement Judgment (ESJ) task and the INSETS annotation pipeline. The work constructs MVEI, a benchmark with 3,086 samples covering four dimensions: sentiment polarity, emotion interpretation, scene context, and perception subjectivity. The authors evaluate 18 MLLMs and demonstrate that even top-performing models like GPT4o fall substantially short of human performance.

### Strengths
1. Benchmark Innovation and Scope: The ESJ task and MVEI benchmark fill a substantial gap in visual emotion evaluation for MLLMs, moving beyond rigid classification to nuanced, open-vocabulary, and multifaceted judgment tasks. This approach allows broader, context-aware, and fine-grained assessment.
2. Automated scalable pipeline: The INSETS pipeline effectively reduces human labor by automating emotion tagging and statement generation while maintaining a reported 90.6 % accuracy after human refinement.
3. Comprehensive empirical analysis: The benchmark evaluation spans >15 MLLMs, both open and proprietary, and compares against human baselines .
4. Grounding in Theory: The mapping to Parrott’s hierarchical emotion model and the use of extensive prompts/protocols demonstrate that the benchmark is theoretically informed, not ad-hoc.

### Weaknesses
1. Potential circularity in evaluation: Some MLLMs used to construct INSETS also participate in evaluation, which could inflate results and reduce benchmark independence.
2. Vague or Underspecified Mathematical/Labeling Procedures: The emotion label assignment, majority voting, and mapping to the hierarchical model are described mainly in prose and schematic diagrams, with some reliance on prompt outputs and “manual expert check.” The specific algorithms are insufficiently formalized.
3. Human annotation reliability and bias: Although inter-annotator agreement statistics are reported, there is limited qualitative analysis of ambiguous cases or bias due to annotator demographics.
4. Insufficient discussion of dataset bias and ethical risk: Bias acknowledgement is limited to a short ethics statement without empirical analysis of representation balance (e.g., emotional polarity, demographic context).
5. Limited analysis of failure modes: Sec. 5.3 identifies that "perception subjectivity shows only modest improvement"  but provides no error analysis. What types of subjectivity statements are most challenging? Are errors systematic? Figure 12 shows examples but no quantitative breakdown.
6. Absence of ablation or component analysis: There is no direct evidence quantifying how each component contributes to final benchmark quality.

### Questions
Please check the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the evaluation of visual emotional perception in Multimodal Large Language Models (MLLMs). The authors argue that prior evaluation methods—fixed label classification and coarse taxonomies—fail to capture the subjectivity, contextual dependence, and open-vocabulary nature of emotion perception. To remedy this, they propose the Emotion Statement Judgment (ESJ) task: MLLMs judge whether emotion-focused statements about an image are correct, enabling open-vocabulary, multifaceted assessment. They also introduce INSETS, an automated pipeline that generates emotion-centric labels and statement candidates with limited human intervention. Using INSETS they build a large automatically annotated corpus (INSETS-462k) and then refine a human-validated benchmark (MVEI) of 3,086 image–statement pairs covering four complementary dimensions: sentiment polarity, emotion interpretation, scene context, and perception subjectivity. Systematic evaluation of many MLLMs shows that models have notable strengths (especially in emotion interpretation and scene context) but still trail humans—particularly on sentiment polarity and perception subjectivity. The paper also explores lightweight adaptation methods (in-context learning, LoRA, full fine-tuning, GRPO) and finds that adaptation improves performance—most dramatically for sentiment polarity—while perception subjectivity remains challenging. The authors release code and data and discuss ethical considerations around dataset biases and model outputs.Three main contributions:1. Task formulation: Introduces the Emotion Statement Judgment (ESJ) task, a flexible, open-vocabulary framework that reframes visual emotion evaluation as a statement verification problem, reducing issues from rigid ground-truth answers and enabling richer, multifaceted assessment.2. Scalable annotation pipeline and corpus: Proposes INSETS, an automated pipeline for generating emotion labels and statements with minimal human effort, and uses it to construct INSETS-462k (462k statements across ~17.7k images), significantly improving scalability over prior labor-intensive datasets.3. Benchmark and empirical analysis: Curates MVEI (3,086 human-refined image–statement pairs) covering four evaluation dimensions (sentiment polarity, emotion interpretation, scene context, perception subjectivity), and provides systematic benchmarking of many contemporary MLLMs plus adaptation studies that reveal strengths, weaknesses, and directions for improvement.Three main shortcomings / limitations:1. Model scale coverage: The evaluation is limited mainly to MLLMs with fewer than ~10B parameters because of computational constraints, excluding larger open-source or closed models that might perform differently; this limits conclusions about upper-bound capabilities.2. Monolingual focus and potential biases: The current implementation and benchmark are monolingual (presumably English) and the automatically generated INSETS-462k may inherit biases from pretraining data and MLLMs; despite human refinement some problematic or culturally specific labels could persist.3. Fundamental difficulty with subjectivity: While adaptation improves sentiment polarity substantially, perception subjectivity remains poorly handled by MLLMs and shows only modest gains from fine-tuning; this suggests deeper architectural or objective-level limitations that the current pipeline and data do not fully resolve.

### Strengths
- Task formulation: Introduces the Emotion Statement Judgment (ESJ) task, a flexible, open-vocabulary framework that reframes visual emotion evaluation as a statement verification problem, reducing issues from rigid ground-truth answers and enabling richer, multifaceted assessment.
- Scalable annotation pipeline and corpus: Proposes INSETS, an automated pipeline for generating emotion labels and statements with minimal human effort, and uses it to construct INSETS-462k (462k statements across ~17.7k images), significantly improving scalability over prior labor-intensive datasets.
- Benchmark and empirical analysis: Curates MVEI (3,086 human-refined image–statement pairs) covering four evaluation dimensions (sentiment polarity, emotion interpretation, scene context, perception subjectivity), and provides systematic benchmarking of many contemporary MLLMs plus adaptation studies that reveal strengths, weaknesses, and directions for improvement.

### Weaknesses
- Model scale coverage: The evaluation is limited mainly to MLLMs with fewer than ~10B parameters because of computational constraints, excluding larger open-source or closed models that might perform differently; this limits conclusions about upper-bound capabilities.
- Monolingual focus and potential biases: The current implementation and benchmark are monolingual (presumably English) and the automatically generated INSETS-462k may inherit biases from pretraining data and MLLMs; despite human refinement some problematic or culturally specific labels could persist.
- Fundamental difficulty with subjectivity: While adaptation improves sentiment polarity substantially, perception subjectivity remains poorly handled by MLLMs and shows only modest gains from fine-tuning; this suggests deeper architectural or objective-level limitations that the current pipeline and data do not fully resolve.

### Questions
SEE WEAKNESS

### Soundness
3

### Presentation
3

### Contribution
3
