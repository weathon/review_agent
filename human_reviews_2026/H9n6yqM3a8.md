# LLM Output Homogenization is Task Dependent

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 4, 8, 6

## Abstract
A large language model can be less helpful if it exhibits output response homogenization. But whether two responses are considered homogeneous, and whether such homogenization is problematic, both depend on the task category. For instance, in objective math tasks, we often expect no variation in the final answer but anticipate variation in the problem-solving strategy. Whereas, for creative writing tasks, we may expect variation in key narrative components (e.g. plot, genre, setting, etc), beyond the vocabulary or embedding diversity produced by temperature-sampling. Previous work addressing output homogenization often fails to conceptualize diversity in a task-dependent way. We address this gap in the literature directly by making the following contributions. (1) We present a task taxonomy comprised of eight task categories that each have distinct concepts of output homogenization. (2)  We introduce task-anchored functional diversity to better evaluate output homogenization. (3) We propose a task-anchored sampling technique that increases functional diversity for task categories where homogenization is undesired, while preserving it where it is desired. (4) We challenge the perceived existence of a diversity-quality trade-off by increasing functional diversity while maintaining response quality. Overall, we demonstrate how task dependence improves the evaluation and mitigation of output homogenization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper describes work on proposing a task-specific perspective for evaluating and mitigating output homogenization in LLMs. The authors define output homogenization where LLMs produce similar outputs which can be problematic in scenarios that require high diversity (e.g., creative writing) but in ones that require verifiability (e.g, math problems), hence the need for a task-dependent approach. The authors introduce the notion of “functional diversity” where a user might find two or more responses to be useful based on the task they use an LLM for. The authors then propose a simple task-anchored sampling technique aimed to maximize functional diversity for task categories where homogenization is needed as well for tasks where it is not. The authors lay down an experiment setup that mainly uses commercial models (Claude, GPT-4o, Gemini) for both generation and response evaluation and sample prompts from five existing datasets. In terms of measuring functional diversity, results show that task-anchored sampling performs better than general sampling techniques and oftentimes reduces homogenization for crucial categories such as Encyclopedia Inquiry, Creative Writing, and Advice or Opinions. In terms of diversity-quality tradeoff, using task-anchored functional diversity removes said tradeoff which is evident if general diversity metrics are used such as vocabulary diversity.

### Strengths
The concept of output homogenization and its inherent nature of being both essential and to be mitigated based on task is an important study. One strength I found with the paper is how well the motivation for the work is laid out that anyone can readily appreciate and understand the problem that needs to be solved. I also appreciate the simplicity with the task taxonomy and task-anchored sampling technique that the authors propose in order to increase functional diversity from LLMs. I agree that the proposed method’s simplificty will be key in the ease of adoption of future work with output homogenization.

### Weaknesses
My main issue with the current study is the LLM-as-a-judge-centric evaluation for the concept of diversity in the experiments. The authors define functional diversity as “a user would perceive two responses as meaningfully different for a given task”, hence, I was fully expecting human evaluation as the main driver of the evaluation procedures across the task taxonomy rather than LLM-as-a-judge.  

Another issue I have is why did the authors not explore any open-weight models for both the functional diversity and diversity-quality tradeoff experiments and only used commercial ones? This reduces the reproducibility of the work that could continue to on focusing "why" (factors affecting) output homogenization happens as future work. The authors also mentioned that for all LLM-as-judge setups, the same models (GPT4o, Clause, GeminiFlash) are essentially evaluating their own responses even if they are averaged together. This setup sounds like bias may be introduced in the evaluation procedure of the study. What is the justification of the authors for this?

The authors mentioned that the work does not focus on answering the "why" output homogenization happens but the current results of the paper feels very limited and minimal at its current state. I would have appreciated it if the authors could have linked variations of functional diversity to factors such as model scale/size, language coverage, effect of preference optimization vs no preference optimization, etc. Especially that some of these factors like RLHF-ed models have been investigated by previous works to negatively affect output diversity. It would have been supplemental if the authors also investigated the same direction with the functional diversity whether this changes for non RHLF-ed models.

The paper should provide a clear disclaimer that this framework has only been tested with English-centric evaluation procedures. The results related to model output homogenization specifically functional diversity and diversity tradeoffs do not generalize in multilingual models unless otherwise tested sufficiently with the same level of rigor done for English.

### Questions
What are the justification of the authors for exclusively using commercial models and not open-source/open-weight models?

Were the prompts in Table 3 for each dataset and task category randomly sampled or handpicked?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies output homogenization in large language models, arguing that it should be evaluated and mitigated in a task-dependent way rather than uniformly across tasks. The authors develop an eight-category task taxonomy distinguishing settings where consistent outputs are desirable versus those where diversity is preferred. They introduce task-anchored functional diversity and a task-anchored sampling approach that adapts decoding to the task’s diversity requirements. Experiments on multiple LLMs suggest that this approach improves meaningful diversity without degrading overall quality.

### Strengths
- The paper raises an important and underexplored perspective by framing output homogenization as a task-dependent issue rather than a universal flaw of LLMs.

- The proposed task taxonomy and task-anchored sampling offer a clear and interpretable framework that connects conceptual reasoning about diversity with practical decoding strategies.

### Weaknesses
- The paper’s main conclusion that task-anchored sampling resolves the diversity-quality trade-off relies entirely on LLM-judge evaluations. Both “functional diversity” and “checklist-based quality” are assessed by aligned models under task-specific prompts, and only the former is partially validated against human judgments. Without independent human validation for the quality metric, the claim of “no trade-off” may reflect a biased artifact of LLM-based evaluation rather than a genuine improvement in human-perceived output quality. This circular setup weakens the paper’s main result.

- The eight-category task taxonomy that supports the framework is neither empirically grounded nor fully supported by data. Some categories, such as “Problem Solving or Design Subjective”, lack any corresponding dataset, while others have ambiguous or overlapping boundaries, such as “Encyclopedia Inquiry” and “Advice or Opinions”. In addition, the taxonomy itself is generated and validated by LLMs rather than humans, which risks reinforcing model-specific biases instead of revealing genuine task distinctions. This undermines the robustness and generality of the proposed task-dependent framing.

- The proposed task-anchored sampling depends on accurate task classification, yet this step is never evaluated. The experiments assume access to ground-truth task labels derived from model majority votes, meaning the system is tested under idealized conditions. In real-world scenarios, even moderate misclassification would cause the model to apply the wrong sampling behavior, reducing quality in exactly the tasks where consistency or factual accuracy is most important. The lack of analysis of this dependency leaves the practical reliability of the approach uncertain.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper makes the following key contributions. They observe a gap in the literature that prior work largely discusses the output diversity of LLMs in a task-agnostic manner. This might lead one to assume that homogenization is 'always an issue'. To address this gap, they propose to measure 'functional diversity' or the effective diversity, or the number of valid solutions generated by a system that is anchored in the expected answer format for the prompt/task. They also provide a comprehensive taxonomy of tasks, dividing them based on the expected answer format, which helps operationalize this definition into a formal evaluation. Through experiments with 3 frontier LLMs (GPT-4o, Gemini-2.5-Flash, Claude-4-Sonnet), they show that targeted task-anchored prompting strategies (basically a prompt that provides some functional definition of the expected output) can increase functional diversity for tasks where it is desired. The paper is well motivated, clearly written, and provides sufficient evidence for its claims.

### Strengths
1. The positioning and framing of this paper are really good. It notices an over-simplified finding in the literature (about homogenizing of output of LLMs as a single amorphous truth in all tasks), provides a clear and tractable method (by categorizing tasks) to create a more usable definition of functional diversity, and provides benchmarking of simple methods to mitigate homogenization as needed. 

2. The definition of functional diversity is clear, and also references past work that implements versions similar to it. 

3. The proposed taxonomy is extensive (and the authors also note that it is non-exhaustive in a refreshing change from recent ML discourse!), and the experiments provide clear evidence for the value of task-specific prompting techniques in domains where diversity is desired. Fig. 2 displays this result really well.

4. Section 4.3 contextualizes prior findings of a trade-off between diversity and output quality as one of having mismatched metrics and that when measured correctly, these are not always at odds with one another.

### Weaknesses
1. The definition in Def 3.1 should be extended to a set of responses $y \in Y$ since that is really the thing you're interested in, and what you measure in the experiments (L.313-317). Given a fixed sampling budget (an answer set size), what is the effective number of unique responses in it?

2. This is more for reproducibility, but I would advocate for experiments with an open-weight LLM since the models on which results are reported are all black-box and can be updated under the hood.

3. For Section 4.3, before making a strong claim of 'not seeing an effect', I think you should also evaluate on tasks with a very large set of responses (and not just 5). It could just be that the effect is not observed at the current sampling budget.

### Questions
1. Re footnote 2 on page 6 - Aren't MacGyver [1] or CoPoet [2] as used in [3] examples of a task in Category E, i.e., "Tasks to solve a problem with many verifiable solutions"? Basically, these involve coming up with solutions to real-world physical reasoning problems ('Iron a shirt with a coat hanger, steamer, and kettle') or creative instructions that have semantic constraints and stylistic variants.

2. In [4], they find that many diversity measures are correlated with simple information compression algorithms like GZIP. I'm curious how well functional diversity correlates with information compression. 

3. In Fig. 3, L.422, should the caption say (a) and not (a)-(c)?

4. It would be interesting if you could compare base and aligned models on functional diversity, given past findings [5, 6] that feedback tuning reduces diversity.

[1] Tian, Yufei, et al. "MacGyver: Are Large Language Models Creative Problem Solvers?." Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 2024.

[2] Padmakumar, Vishakh, et al. "Beyond Memorization: Mapping the Originality-Quality Frontier of Language Models." arXiv preprint arXiv:2504.09389 (2025).

[3] Chakrabarty, Tuhin, et al. "Help me write a poem: Instruction Tuning as a Vehicle for Collaborative Poetry Writing." Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2022.

[4] Shaib, Chantal, et al. "Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores." arXiv preprint arXiv:2403.00553 (2024).

[5] West, Peter, and Christopher Potts. "Base models beat aligned models at randomness and creativity." arXiv preprint arXiv:2505.00047 (2025).

[6] Padmakumar, Vishakh, and He He. "Does Writing with Language Models Reduce Content Diversity?." The Twelfth International Conference on Learning Representations.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper hypothesizes that quantifications and inductions towards output homogenization should be task dependent. They develop a task taxonomy enlisting which kind of variations is meaningful for that task and use that to construct prompts that describe task-dependent axes of variations to guide the generation more adaptively (to the task’s homogenization needs). Their task-anchored method increases diversity and does not reduce quality.

### Strengths
- The key hypothesis is unique and has high potential for explaining the shortcomings of several works that categorize the diversity of the LM generations 
- Highlights an interesting shortcoming about prior work using scalar rewards (task-agnostic) to assign quality scores.

### Weaknesses
- The task taxonomy and prompt design appear somewhat arbitrary and intuition-based, not derived from user studies or large-scale data.
A small ablation that applies mismatched task prompts (e.g., creative prompt for factual task) would strengthen confidence that each mapping is truly optimal.Since the sampling methods themselves are not entirely new, the value of the paper rests heavily on how sound and justifiable these task definitions are.

- All evaluations rely on LLM judgments. While the use of checklist-style grading for task-relevant quality is an improvement over scalar rewards, there are no human-based validations. Because the LLM-graded metrics show larger gains for the system-prompt variant and weaker discrimination elsewhere, a human diversity-quality study would help verify whether these numerical gains reflect actual perceptual differences.

### Questions
- What fraction of your evaluation prompts fell into this “uncategorized” case? Could you report the performance of the methods on these? 
- How accurate is the task-to-category classification? Did you run any verification (human or otherwise) to ensure that prompts were correctly labeled before applying task-specific system prompts?

### Soundness
3

### Presentation
3

### Contribution
3
