# High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 10, 4, 4

## Abstract
Large Language Models (LLMs) currently respond to every prompt. However, they can produce incorrect answers when they lack knowledge or capability -- a problem known as hallucination. We instead propose post-training an LLM to generate content only when confident in its correctness and to otherwise (partially) abstain. Specifically, our method, HALT, produces capability-aligned post-training data that encodes what the model can and cannot reliably generate. We generate this data by splitting responses of the pretrained LLM into factual fragments (atomic statements or reasoning steps), and use ground truth information to identify incorrect fragments. We achieve capability-aligned finetuning responses by either removing incorrect fragments or replacing them with "Unsure from Here" -- according to a tunable threshold that allows practitioners to trade off response completeness and mean correctness of the response's fragments. We finetune four open-source models for biography writing, mathematics, coding, and medicine with HALT for three different trade-off thresholds. HALT effectively trades off response completeness for correctness, increasing the mean correctness of response fragments by 15% on average, while resulting in a 4% improvement in the F1 score (mean of completeness and correctness of the response) compared to the relevant baselines. By tuning HALT for highest correctness, we train a single reliable Llama3-70B model with correctness increased from 51% to 87% across all four domains while maintaining 53% of the response completeness achieved with standard finetuning.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The authors are addressing the issue of LLMs generating responses they are not confident in which can lead to hallucinations. They only want the LLM to generate responses it is confident in otherwise it'll stop generating certain responses at will output "Unsure". 

The authors propose a post-training method called High Accuracy, Less Talk (HALT). This method involves prompting the LLM to generate responses for certain domains (wiki, math reasoning, etc). They then break these responses down into fragments and use an evaluator (separate LLM) to determine which fragments are correct / incorrect. They then create a new dataset where the responses are only the correct responses. 

After finetuning the authors show that their model improves correctness on all their testsets along with maintaining a reasonable response completeness.

### Strengths
1) The method laid out is clearly explained and takes a principled approach. I think the different ways to assess fragments based off the domain is good and it is good that the authors are acknowledging future work would need to involve dependency graphs.

2) The baselines provided seem to be good to compare their method against.

3) The study in the section "Finetuning on Few-Shot Prompted Responses is comparable to Finetuning on Ground Truth
Responses" was important to run to trust this method. Even though it's been shown in previous that LLMs do not acquire novel capabilities during finetuning since the few-shot prompting was a key component of the method was good to show that results wont diminish much.

### Weaknesses
1) One thing that is not clear is how does response completeness affect user experience? Obviously we want correct responses for the user but is a completeness score of 51% low? What's the best tradeoff?

2) There's some lack of discussion. In Figure 4 why is there a sharp dropoff for the MATH dataset but not for Wikibios. I might be missing something but it would be good to have this clarified.

### Questions
Questions

1) Why are you using RandomTrim as a baseline? It's not clear what purpose it serves as a baseline. 

2) Since you are sampling responses when creating the finetuning testset how often are the responses similar to each other? Does sampling help then if you are not getting diverse responses?

3) Have you considered using the negative fragments to move the model's response away from those?

Suggestions / Typos

1) I think you meant to say APPS in line 334

2) I suggest putting figures closer to where you mention them like Figure 3

### Soundness
3

### Presentation
4

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
HALT fine-tunes large language models to answer only when confident and to abstain when uncertain, outputting "Unsure from here". It trains models only on content they can reliably generate, aligning outputs with internal confidence rather than forcing full responses. HALT decomposes answers into factual fragments, verifies each with an evaluator LLM, and keeps only correct ones. Without adding inference cost, it improves factual accuracy and balances precision and recall. Across LLaMA3, Gemma2, and Mistral models, HALT outperforms standard finetuning, producing more reliable and self-aware responses.

### Strengths
1. The core idea of this paper is intuitive and clearly presented.
2. The "Unsure from here" mechanism makes the model’s uncertainty explicitly interpretable, improving user trust and enabling controllable trade-offs between correctness and completeness.

### Weaknesses
1. The method depends on an additional evaluator model to assess fragment correctness, which may introduce bias or inconsistency depending on the evaluator’s quality and alignment.
2. The method also relies on fragmentation, which feels somewhat heuristic to me.
3. The training process is complex, requiring additional time and computational cost.

### Questions
1. What does the statement f in line 147 refer to? I assume it corresponds to the "fragment" mentioned in the Introduction section, but please clarify this.
2. Why do you sample four prompt–response pairs and concatenate them with the target prompt?
3. Using Llama3-405B as the evaluator or for fragmentation appears quite costly. How sensitive is the proposed method to the evaluator model’s size and capability?

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
4

### Summary
The authors propose HALT, a method to train language models to abstain from providing certain fragments of their long-form responses when they are uncertain. Given an LM, the approach obtains several samples for each training input, splits each sample into fragments and fact-checks them, and finally replaces incorrect fragments with abstention statements. By collecting samples with more or fewer correct statements into different supervised finetuning splits, HALT is able to navigate the correctness-completeness tradeoff for long-form generation, unlike prior work. Experiments on four datasets across general-domain factuality, math, etc., demonstrate HALT's navigation of this tradeoff.

### Strengths
- The paper is generally well-written and clear.
- The motivation is strong; abstention and uncertainty quantification at claim-level are an important approach to decrease LM hallucination while retaining usefulness.
- The method is simple and clever, in particular the idea to use multiple samples given an input along with their number of correct claims to control the correctness-completeness tradeoff. I suspect that this work could emerge as a straightforward, effective baseline for training LMs to abstain in their long-form generations.
- The experiments are reasonably thorough (four datasets in different domains, evaluation of multi-task transfer, and extension to uncertainty quantification).

### Weaknesses
- The authors should better contextualize with respect to prior work on finetuning language models to express their verbalized uncertainty [1] [2]. That setting is in some sense more challenging than the final experiment in the present paper, which marks fragments as uncertain instead of omitting them.
- A number of heuristic choices are made which are not clearly ablated. For example, I would expect that many non-mathematical tasks have some causal dependency in consecutive sentences, and therefore simply replacing an arbitrary subset of fragments with an abstention phrase will harm coherence. Can you evaluate HALT on some "guardrail" instruction-following tasks which would test how coherence and helpfulness are affected by your finetuning, e.g., AlpacaEval or similar?

[1] Band et al., Linguistic Calibration of Long-Form Generations. ICML 2024. https://arxiv.org/abs/2404.00474
[2] Stengel-Eskin et al., LACIE: Listener-Aware Finetuning for Confidence Calibration in Large Language Models. NeurIPS 2024. https://arxiv.org/abs/2405.21028

### Questions
See above for the key feedback on better contextualization with respect to previous "LM finetuning for calibration" works, and evaluating coherence with a guardrail task.

### Soundness
3

### Presentation
2

### Contribution
2
