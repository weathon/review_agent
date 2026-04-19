# Simple synthetic data reduces sycophancy in large language models

- Decision: Reject
- Scores: 6, 6, 3, 5

## Abstract
Sycophancy is an undesirable behavior where models tailor their responses to follow a human user's view even when that view is not objectively correct (e.g., adapting liberal views once a user reveals that they are liberal). In this paper, we study the prevalence of sycophancy in language models and propose a simple synthetic-data intervention to reduce this behavior.

First, on a set of three sycophancy tasks where models are asked for an opinion on statements with no correct answers (e.g., politics), we observe that both model scaling and instruction tuning significantly increase sycophancy for large language models up to 540B parameters. Second, we extend sycophancy evaluations to simple addition statements that are objectively incorrect, finding that despite knowing that these statements are wrong, language models will still agree with them if the user does as well.

To reduce sycophancy, we present a straightforward synthetic-data intervention that takes public NLP tasks and encourages models to be robust to user opinions on these tasks. Adding these data in a lightweight finetuning step can significantly reduce sycophantic behavior on held-out prompts. Code for generating synthetic data for intervention can be found at https://anonymous.4open.science/r/sycophancy-intervention-F0D1/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the prevalence of sycophancy in LLMs and discovers that both model scaling and instruction-tuning can exacerbate this issue. In response, the authors propose a straightforward synthetic data construction method designed to mitigate sycophancy in LLMs. Experimental results demonstrate that their training approach effectively generalizes to prompts that the models have not seen during training.

### Strengths
**Clarity:** The paper is very well-written and easy to follow, with a smooth flow that makes it enjoyable to read. The figures are clear and effectively help in understanding the problems being addressed and the approach employed.

**Motivation:** The paper is well-motivated, addressing the critical issue of sycophancy in LLMs. This phenomenon poses a significant challenge, as it resembles reward hacking and may undermine the effectiveness of the RLHF process, potentially hindering the further development of LLMs.

**Implications:** The paper highlights that both model scaling and instruction tuning can lead to increased sycophancy in LLMs. These important findings serve as a reminder of the problem's significance.

**Soundness:** The proposed approach is logically sound, and the experimental results convincingly demonstrate the effectiveness of their training method, particularly in generalizing to prompts that the models have not seen during training.

### Weaknesses
**Comparison:** It would be helpful to compare the model's performance using alternative prompting techniques, such as system 2 attention [1]. If these prompting methods effectively reduce sycophancy, it could challenge the applicability of the current training approach, which appears to require more efforts. Additionally, it remains unclear how the current approach would generalize to scenarios involving more open-ended questions where there are no set of gold answers to choose from. In comparison, prompting techniques could generalize to those open-ended questions more easily.



[1] System 2 Attention (is something you might need too)

### Questions
See Weaknesses part. Could you add a comparison to some prompting-based techniques to see whether the training based approach works better or not?

### Soundness
3

### Presentation
3

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
The paper investigates sycophancy in LLMs, where models tend to tailor responses to align with user opinions, even when those statements are incorrect. To measure this behavior, the authors first examine sycophancy on three classification tasks with no definitive correct answers and one simple addition classification task involving incorrect statements. They find that both model scaling and instruction tuning amplify sycophantic tendencies, and sycophancy still exists even when the statements are apparently wrong. To mitigate this issue, the authors propose a synthetic data intervention designed to encourages more robust responses to potentially incorrect user beliefs. They demonstrate that simply finetuning on this dataset can significantly reduce sycophantic responses on held-out prompts. Further evaluations reveal that this intervention does not impact performance on established benchmarks, such as MMLU and Big-Bench Hard, indicating that the approach maintains model accuracy while reducing sycophancy.

### Strengths
This work effectively highlights the issue of sycophancy in LLMs, and conducts evaluations across three model sizes—8B, 62B, and 540B. This finding that sycophantic behavior becomes more pronounced as model size increases provides a valuable insight into how scaling influences sycophancy.

The synthetic data intervention method is straightforward and effective, making the intervention potentially easy to replicate across different models.

The proposed method is tested on two popular benchmarks, MMLU and Big-Bench Hard, showing the effectiveness without compromising the model’s performance on established tasks or affecting the existing learned knowledge.

### Weaknesses
While the paper offers insights about sycophancy in language models and a method for reducing it, further experiments could enhance the robustness and generalizability of the proposed finetuning method:
1. Although three models of varying sizes were tested, the evaluation is limited to a single model type. It would be beneficial to examine sycophancy across a wider range of both open-source LLMs, such as LLaMA -- which has been widely studied in research and also offers multiple size options -- and closed-source models like GPT or Gemini. Expanding the evaluation to diverse model architectures would also help verify whether the synthetic data intervention remains effective across different types of models.
2. The paper acknowledges limitations in prompt diversity and task scope, as the experiments are largely restricted to classification tasks with a narrow range of prompt formats. Testing the fine-tuned model’s performance on more varied, out-of-distribution samples would provide a stronger assessment of the intervention’s generalizability. For example, the user's belief is subtly embedded within a description of their actions, serving as contextual information.
3. In Section 3, the study on objectively incorrect statements is limited to simple addition tasks, which constrains the ability to generalize the claim that “models are sycophantic for objectively wrong answers.” Expanding this evaluation to include a broader array of tasks involving objectively false statements—such as those related to misinformation (e.g., "Apple cider vinegar is a miracle cure for cancer"), fake news (e.g., a famous celebrity supports xxx but actually not), or conspiracy theories (e.g., "the Earth is flat")—would lend stronger support to this conclusion.

Writing clarity:
Table 1 and Figure 1 are nearly duplicated.

### Questions
What proportion of questions is filtered out for each model during the filtration process? Given the differences in model sizes, I suppose there would be a trend where larger models retain a greater number of synthesized training examples.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper studies the prevalence of sycophancy in language models and puts for a synthetic data based approach to reduce sycophantic behavior in LLMs.

### Strengths
- the synthetic data intervention step leverages openly available datasets, as well as a good variety of such datasets at 17 total
- well fleshed out limitations section, indicating a paper that is grounded it what it purports to provide evidence for.

### Weaknesses
- the set of models that are used for experiments are quite limited.
- the intervention to reduce sycophancy requires fine-tuning, which may not be feasible for all use-cases. For example, when access to the model is limited by openness or resource constraints.
- single prompt format in all experiments

### Questions
1. is there any reason why larger or different models were not chosen? For example, the Mixtral models or Llama family?
2. In cases where fine-tuning is not an option, what are other ways to reduce sycophantic outputs?
3. were there attempts with different prompt formats, e.g., with some ablation studies? If so, what were the results? If not, why not?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The authors aim to address the phenomenon of sycophancy in language models, where models tend to align with user opinions even when they are incorrect or subjective. This phenomenon is observed even for large language models up to 540B parameters. The authors present an approach to mitigate this by using synthetic data in a lightweight fine-tuning process. This synthetic data is generated by reformatting publicly available NLP tasks, intending to decouple truthfulness from user opinions in model responses. The intervention effectively reduces sycophancy, especially in scenarios involving incorrect statements, and shows generalization to models of varying sizes.

### Strengths
1. The paper is well-structured, with a clear explanation of sycophancy, its implications, and how the proposed intervention addresses this problem.
2.  The fine-tuning process is lightweight, making this approach accessible and adaptable for large-scale language models with limited computational resources.
3. The intervention's impact is demonstrated with comprehensive results across multiple models and tasks, showing clear reductions in sycophantic responses.

### Weaknesses
1. The sycophancy evaluations are primarily limited to multiple-choice tasks. It would be beneficial to explore if the intervention works in generative settings where response options are more diverse.
2. The smallest model used (Flan-LLM-8B) did not respond well to the intervention, highlighting a potential limitation in the effectiveness of the approach for smaller models.

### Questions
1. How well does the intervention generalize to generative tasks where the model isn’t limited to choosing from predefined responses?
2. Is there any evidence to suggest that the intervention could be adapted for smaller models to improve performance?
3. Could this approach be extended to improve the model’s adherence to factual information when user opinions align with correct statements?

### Soundness
3

### Presentation
3

### Contribution
3
