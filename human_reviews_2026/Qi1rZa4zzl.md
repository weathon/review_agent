# Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-Tuning and Can Be Mitigated by Machine Unlearning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Recent vision language models (VLMs) have made remarkable strides in generative modeling with multimodal inputs, particularly text and images. However, their susceptibility to generating harmful content when exposed to unsafe queries raises critical safety concerns. While current alignment strategies primarily rely on supervised safety fine-tuning with curated datasets, we identify a fundamental limitation we call the "safety mirage", where supervised fine-tuning inadvertently reinforces spurious correlations between superficial textual patterns and safety responses, rather than fostering deep, intrinsic mitigation of harm. We show that these spurious correlations leave fine-tuned VLMs vulnerable even to a simple one-word modification-based attack, where substituting a single word in text queries with a spurious correlation-inducing alternative can effectively bypass safeguards. Additionally, these correlations contribute to the over-prudence, causing fine-tuned VLMs to refuse benign queries unnecessarily. To address these issues, we show machine unlearning (MU) as a powerful alternative to supervised safety fine-tuning, as it avoids biased feature-label mappings and directly removes harmful knowledge from VLMs while preserving their general capabilities. Extensive evaluations across safety benchmarks show that under MU-based alignment reduces the attack success rate by up to 60.27% and cuts unnecessary rejections by over 84.20%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper uncovers a “safety mirage” in safety fine-tuning of MLLM, which is caused by spurious correlations between questions and safety labels in training data. On the attack side, the authors show these models are still jailbreakable via a simple one-word substitution that exploits these correlations, and they link the over-prudence (benign refusals) to the same bias. On the defense side, they propose machine unlearning to remove the influence of unsafe responses without relying on spurious shortcuts. Extensive experiments across multiple safety benchmarks and VQA tasks validate the mirage phenomenon and show that MU-based fine-tuning mitigates spurious correlations and reduces over-prudence while preserving utility.

### Strengths
1. This paper identifies an important issue in safety fine-tuning of MLLM: spurious correlation between rejecting/answering behaviours and certain keywords in the query. The authors demonstrate this clearly through investigating the word frequency in the training dataset and one-word jailbreak/over-prudence experiment. 

2. The adopted unlearning objectives achieves better safety and mitigates over-prudence

### Weaknesses
1.Lack of explanation and motivations for the unlearning objectives. The authors are doing a good job identifying the problem of spurious correlation certain reject/answer behaviours and certain key words in the query. However, it is unclear why the use of machine unlearning objectives could mitigate the spurious correlation problem. 

2. Without a clear motivation of the unlearning objectives, the reviewer can only link the effect of unlearning to regularization. The spurious correlation might result from the MLLM overfitting to the safety fine-tuning datasets. What if the authors apply certain regularization techniques such as lager weight decay, smaller lr or less training steps? In addition, data augmentation strategy that expands the query space might also be useful. 

3. Possible side-effects of the unlearning objectives. The adopted RMU intends to erase the model's representation of unsafe queries/responses. However, it is unclear how would the fine-tuned model behave when seeing similar queries. Would the model output nothing or generate irrelevant texts?

### Questions
1. What are the inputs to the MLLM when conducting the K-shot attack in Figures 3, 4 and in other evaluation? Are the image paired with K versions of the paraphrased queries or paired with one of the version at a time? Why do the authors choose K-shot attack throughout this paper?

2. Can the authors provide examples of one-word jailbreaking and over-prudence? I.e., how is the keywords are integrated in to the original query. 

3. When performing NPO, how are the loss mask set? In normal instruction tuning, we would mask the prompt/instruction tokens and only compute loss on the response tokens. In the case of NPO, if only the queries were provided (lines 342-343), would the prompt tokens be masked?

### Soundness
2

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
5

### Summary
This paper explains that existing fine-tuning-based safety alignment methods do not truly enhance a model’s safety capabilities. Instead, they rely on spurious correlations triggered by special tokens to induce refusal behaviors. Building on this observation, the authors propose a one-word jailbreak, showing that simply replacing and repeating a single word before answering can efficiently jailbreak safety-aligned models. To address this issue, the paper introduces a machine unlearning approach and demonstrates its effectiveness using NPO and RMU evaluations.

### Strengths
1. The paper is well-organized and clear
2. The content is rich, containing both analysis, jailbreak, and defense of vision-language model safety. The analysis of "spurious correlation" is interesting and insightful
3. The proposed one-word jailbreak and machine unlearning methods are effective and achieve strong performance across various benchmarks

### Weaknesses
1. The paper only analyzes the spurious correlation between the first word in the query and the rejection responses. However, this is not the only possible factor, there may also be correlations involving other words in the query that are not in the first position.
2. After unlearning, the utility performance of models degrades more than that of safety fine-tuning methods.
3. All the experiments are conducted on LLaVA-1.5 models, which are relatively old models (released 2 years ago). If the author can provide some analysis based on experiments on a recent model (Qwen series), it will further strengthen the findings and analysis.
4. The current results show that the ASR and utility of the model after unlearning (before jailbreaking) are inferior to those of the safety fine-tuned model. Would it be possible to perform unlearning on the safety fine-tuned model to eliminate the previously analyzed correlations?
5. The “Safety Mirage” described in the paper seems to lack properties specific to vision-language models; it appears to be a phenomenon similar to what has been observed in LLMs. A further discussion on how visual information influences model safety would enhance the completeness

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper first identifies a “safety mirage” problem in current fine-tuning approaches, where vision-language models (VLMs) develop biases due to spurious correlations between textual queries and safety labels in the training data. Such biases make the models vulnerable to one-word attacks, in which simply replacing a highly frequent query word with one associated with benign outputs can bypass safety mechanisms. Moreover, fine-tuned models often exhibit an over-prudence issue, leading them to reject normal or harmless queries. To address these problems, the authors propose using machine unlearning (MU) as a direct method to mitigate the effects of spurious correlations. Experimental results on several benchmarks demonstrate that MU-based safety fine-tuning effectively alleviates both the spurious correlation and over-prudence issues.

### Strengths
Revealing the “Safety Mirage” Phenomenon: The author systematically investigates the existence of spurious textual features and correlations by analyzing word frequency patterns across various safety fine-tuning datasets. This analysis reveals that certain frequent words are strongly correlated with safety labels. To validate this bias, the author conducts one-word attack experiments, confirming that the presence of bias-inducing words is crucial for the attack’s success. Furthermore, even a single insertion of a rejection-related word (e.g., “Share”) can trigger over a 90% rejection rate for otherwise safe text–image queries, highlighting the severity of the over-prudence issue.

Proposing Representation Misdirection Unlearning (RMU) and Negative Preference Optimization (NPO): To address these biases, the author introduces two complementary methods. RMU maps the intermediate representations of unsafe data to randomized feature spaces, effectively erasing learned spurious associations. Meanwhile, NPO encourages the model to deviate from the reference model’s behavior specifically on unsafe inputs. Together, these techniques preserve the model’s general capabilities while mitigating spurious correlations.

Comprehensive Evaluation Framework: The author adopts a multi-dimensional evaluation framework encompassing safety, over-prudence, and general capability metrics, benchmarking the proposed method across diverse datasets. The experimental results confirm the effectiveness of MU in improving performance across all three dimensions. Additionally, token-level saliency analyses—such as masking the words “What” or “Share”—provide further evidence of the causal influence of spurious correlations.

### Weaknesses
Limited Model Scope and Generalizability: The paper evaluates its methods exclusively on the LLaVA-1.5 model series, which is now relatively outdated. By not extending experiments to more state-of-the-art (SOTA) vision-language models (VLMs), the study leaves two critical questions unresolved: (1) whether the “safety mirage” phenomenon is a widespread issue across modern VLMs or merely an artifact of LLaVA-1.5’s limited language generalization; and (2) whether the proposed Machine Unlearning (MU) method maintains its advantages over standard supervised fine-tuning (SFT) when applied to models with stronger intrinsic anti-bias mechanisms. This narrow model validation thus limits the generalizability and external validity of the study’s conclusions.

Lack of Comparison with Data-Level Bias Mitigation: The study identifies "training data bias" (spurious token-label correlations) as the root cause of "safety mirage," yet it omits a critical baseline: mitigating data bias via LLM-driven data diversification. This gap prevents a comprehensive assessment of whether MU is truly a necessary solution, or if simpler data-level interventions could already alleviate the problem.

Dependence on Existing Dataset Paradigms: Despite its conceptual novelty, the proposed MU-based framework remains grounded in existing safety fine-tuning datasets, without addressing the deeper issue of how these datasets themselves are constructed and labeled. Consequently, the method offers a pragmatic yet incremental improvement rather than a fundamental solution to the broader challenge of VLM safety alignment.

### Questions
1. For SOTA VLMs with stronger cross-modal alignment (e.g., Qwen2.5-VL), do they still exhibit "safety mirage" (e.g., high ASR after one-word attacks) when fine-tuned with traditional SFT?
2. If the safety training dataset is diversified via LLM-driven paraphrasing (e.g., using GPT to rewrite queries while preserving semantic intent, weakening token-label correlations), would traditional SFT on this diversified data achieve ASR and over-prudence levels comparable to MU?
3. Compared to MU, how does the computational cost of "LLM-based data diversification + SFT" compare? Is MU still a cost-effective solution if data diversification can already mitigate safety mirage?
4. If we combine "LLM-driven data diversification" with MU (i.e., apply MU on a diversified dataset), would this further improve safety-utility tradeoffs? Or would the two methods be redundant, given that diversification already reduces spurious correlations?

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
3

### Summary
This paper examines the effectiveness of machine unlearning for improving the safety of Vision-Language Models (VLMs).  It is shown that existing supervised fine-tuning (SFT) defenses exhibit two main problems: 1. Brittleness to one-word attacks, where replacing just a single word in a harmful prompt can bypass the defense, and 2. Over-prudence, where replacing just a single word in a benign prompt can lead to an unncessary refusal. The paper shows that this is due to spurious correlations that are learned from the fine-tuning dataset. To overcome this, the paper proposes using label-free Machine Unlearning (MU) approaches, and shows that they are much more effective at mitigating the spurious correlation issues than SFT-based approaches.

### Strengths
1. Provides convincing evidence that unbalanced datasets (in terms of the initial word) cause spurious correlations to be learned when performing SFT for safety enhancement in VLMs: The frequency of initial words is analyzed (revealing highly skewed distributions), which are then used to create an effective one-word jailbreak attack as well as to demonstrate over-prudence.
2. The use of machine unlearning techniques to address the spurious correlation problem is well-motivated, as they are label-free. Moreover, they are shown to be much more effective than SFT-based defenses at reducing the one-word attack success and mitigating over-prudence across multiple VLMs, with minimal minor impact to model utility.

### Weaknesses
1. The attacks evaluated in this work are very limited (just the proposed one-word attack). It would strengthen the paper to evaluate against other kinds of attacks, such as those mentioned in section 2.
2. Building off the previous weakness, it is unclear whether the findings would remain consistent under attacks that manipulate the image (e.g., gradient-based adversarial image generation), since the observations were based on spurious correlations on just the text component of the input.

### Questions
1. It seems the one-word attack evaluation uses adversarial words found via the analysis done on the various training datasets in Figure 2. I’m wondering whether there are adversarial words “outside” of these datasets that may be just as effective, and whether or not the machine unlearning approaches can generalize to such words. These words could be discovered through something similar to GCG ([1]) (e.g., swapping a single token to maximize a harmful target).

[1] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.

### Soundness
3

### Presentation
4

### Contribution
3
