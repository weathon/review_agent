# EXPLORING RESPONSE UNCERTAINTY IN MLLMS: AN EMPIRICAL EVALUATION UNDER MISLEADING SCENARIOS

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Ensuring that Multimodal Large Language Models (MLLMs) maintain consistency in their responses is essential for developing trustworthy multimodal intelligence. However, existing benchmarks include many samples where all MLLMs exhibit high response uncertainty when encountering misleading information, requiring even 5-15 response attempts per sample to effectively assess uncertainty. Therefore, we propose a two-stage pipeline: first, we collect MLLMs’ responses without misleading information, and then gather misleading ones via specific misleading instructions. By calculating the misleading rate, and capturing both correct-to-incorrect and incorrect-to-correct shifts between the two sets of responses, we can effectively metric the model’s response uncertainty. Eventually, we establish a Multimodal Uncertainty Benchmark (MUB) that employs both explicit and implicit misleading instructions to comprehensively assess the vulnerability of MLLMs across diverse domains. Our experiments reveal that all open-source and close-source MLLMs are highly susceptible to misleading instructions, with an average misleading rate exceeding 86%. To enhance the robustness of MLLMs, we further fine-tune all open-source MLLMs by incorporating explicit and implicit misleading data, which demonstrates a significant reduction in misleading rates

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the issue of response uncertainty in Multimodal Large Language Models (MLLMs), which can be problematic when models encounter misleading information. To tackle this, the authors propose a two-stage pipeline: first, gathering MLLM responses without misleading information, followed by collecting responses influenced by specific misleading instructions. They effectively evaluate model uncertainty by measuring the misleading rate and tracking shifts between correct and incorrect responses. They introduce the Multimodal Uncertainty Benchmark (MUB), which uses explicit and implicit misleading instructions to assess MLLM vulnerability across various domains. To improve robustness, they fine-tune open-source MLLMs using misleading data, substantially reducing misleading rates.

### Strengths
The strengths of this paper include:

- This work focuses on an important issue: the robustness of Multimodal Large Language Models (MLLMs) when faced with misleading instructions. This is a compelling research topic that addresses a gap in the current field. 

- The paper is well-structured, with a clear framework, and the authors present three research questions that are thoroughly examined through extensive experiments involving 12 models. 

- This work contributes to the community by introducing the Multimodal Uncertainty Benchmark and providing a fine-tuning dataset, demonstrating improved model robustness against misleading instructions.

### Weaknesses
The weaknesses of this work include:
- The paper lacks a discussion on whether the models are calibrated. It does not address whether more consistent (more certain) model outputs correspond to more accurate answers. The results in the paper are primarily based on misleading rate (MR) and average consistency rate (ACR), without showing metrics like model accuracy. There is a lack of analysis on utility.
- The authors fail to analyze the impact of instruction tuning on model usability. It is unclear how much the model’s performance on different tasks changes before and after fine-tuning. This lack of explanation limits the understanding of the benchmark’s functionality. Researchers are left uncertain whether fine-tuning causes models to generate consistent but incorrect responses.

Suggestions:
The images in the paper are quite blurry, especially Figure 2. The authors should check the image quality. There are also some typos, such as mixed-use of InternVL-Chat-V1-5  and Internvl-chat-v1.5.

### Questions
Refer to Weakness. The analysis of utility and calibration is important for such work.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper dives into how MLLMs often fail to perform well when faced with misleading prompts. To tackle this, the authors set up a benchmark called the Multimodal Uncertainty Benchmark (MUB), which first gathers standard responses and then throws in misleading inputs to see how often the models get tripped up. They then fine-tuned open-source models with a mix of straightforward and subtle misleading data, cutting down the rate of being misled, while keeping the models’ overall accuracy intact.

### Strengths
The attempt of addressing the response uncertainty in MLLMs is an interesting and important task. The proposed method seems largely sound to me in addressing at least parts of the problem. The paper is written in a structured and esay2understand manner -- quite straightforward. The MUB benchmark could be useful and the benchmarking results are generally informative. The effort of finetuning the MLLMs with misled data adds some more insights into how the problem could be mitigated.

### Weaknesses
My main concerns are

- This work only evaluates/tackles VLLM instead of MLLM as claimed multiple times in title and throughout paper, though I could maybe see the way to extend to other modalities.

- Having the implicit misleading information generated by GPT-4o seems like a "fighting fire with fire" approach -- I think it is better to have at least a subset of implicit ones written by human annotators so that we can see whether there is any difference between the human-generated ones and GPT-4o generated ones.

- During finetuning, a random set of explicit and implicit misled samples are used for finetuning, yet I am afraid the explicit misleading info has a too obvious and unique pattern due to how it's designed, hence too easy to pick them up, making the improvement after finetuning not too surprising.

- Instead of finetuning, I would recommend the authors to simply systematically prompt the MLLMs, such as "The questions might contain misleading information, you should try to answer the question correctly despite of those misleading information ..."; another version could even give it two examples (one explicit and one implicit). I would guess/assume, simply doing this extra prompting will make the results much better.

- The questions only include multi-choice and T/F styles, which certainly makes the metrics calculation easier (reflected in equation 1 and 2), yet probably losing the delicacy in the type of Q/A addressed?

### Questions
- The misleading information was only added to the textual questions, why not consider altering the image to inject misleading information?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies uncertainty measurement for responses from MLLMs. The main novelty is a novel uncertainty measurement based on how MLLMs' response shifts after injecting misleading instructions. Empirically, the paper developed Multimodal Uncertainty Benchmark (MUB), and systematically evaluates most major MLLMs’s uncertainty; The result suggests a dominant issue of uncertainty in MLLMs, with an average misleading rate exceeding 86%. The author experimented with fine-tuning MLLMs with targeted misleading data, which notably improves robustness.

### Strengths
- I think the effort of proposing novel and more efficient metrics to is very relevant and helpful. Previous metrics, such as self-consistency rate are widely used but in practice I found it to be very unreliable on big models.
- The experimental evaluation is comprehensive, covering most of the commonly used closed and open-sourced models.

### Weaknesses
- Soundness of measurement (Major): While I very much appreciate the effort on better and efficient measurements for uncertainty, currently I still have some doubts about whether adding misleading information can measure the uncertainty in model’s response. I’ll explain my concerns and perhaps the authors can clarify:
    - The measurement might be dependent to the misleading information themselves: the content, position, length, etc might all influence this metric. Moreover, since the implicit misinformation is generated by GPT4o, which is also evaluated on the benchmark, will it incur evaluator bias?
    - Implicit scenarios seem better defined; But for explicit scenarios (e.g. telling the model true answer), the model behavior might be inherently undefined: i.e. shall the model follow the user’s “instruction” (e.g. “the true answer”), or answer the question and ignore user instruction.
- Task (Minor): The study is confined to multiple choice question. I am curious about how would the definitions, measurements, and findings generalize to open-ended question. But I don’t think this is a major point, because most current VLM benchmarks are multiple-choice only.

### Questions
See weakness section.

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a dataset of misleading instructions for multimodal language models. This is done in two ways: through a template (telling the model that the answer is "X", where X is wrong), and through a language model (for instance by adding evidence or reasoning that contradicts the true answer). It is shown that models have lower consistency on instructions that are successfully misleading, and that fine-tuning can improve this.

### Strengths
The paper tackles an interesting problem, and the idea of adding misleading evidence to a prompt is a nice way to test for robustness. I also thought it was interesting that consistency decreases.

### Weaknesses
I'm not sure I buy the overall motivation -- of course if you tell the model the answer is wrong, it will flip some fraction of the time. But is this going to affect real users in any way? Arguably this is even an intended feature to avoid getting into fights with users.

As a result of this, I don't think the "Explicit" misleading prompts are really a meaningful benchmark. The "Implicit" ones are more interesting, but need more detailed treatment: for instance, at least give several examples and enough information to assess data quality. The evaluation of the "Implicit" setting is also strange -- using best-of-5 sampling (even though "Explicit" is best-of-1) which inflates the success rate.

A separate issue is that, throughout, there is not enough information to understand the data or experimental setup in detail. The paper says that they "fine-tuned on separate data", but there are not many details that would let a reader verify or reproduce the experiments. (This is also part of the problem with the "Implicit" setting -- not enough details to fully understand.)

I think the authors are tackling an interesting problem, and have made a good start on it, but in my opinion the experiments and writing should be tightened up before it's ready to be accepted to ICLR.

### Questions
Why best-of-5 sampling, and why only for "Implicit"?

Can you provide several random samples from the "Implicit" setting?

### Soundness
2

### Presentation
2

### Contribution
3
