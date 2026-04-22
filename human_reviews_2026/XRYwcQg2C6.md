# Black-box Optimization of LLM Outputs by Asking for Directions

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
We present a novel approach for attacking black-box large language models (LLMs) by exploiting their ability to express confidence in natural language. Existing black-box attacks require either access to continuous model outputs like logits or confidence scores (which are rarely available in practice), or rely on proxy signals from other models. Instead, we demonstrate how to prompt LLMs to express their internal confidence in a way that is sufficiently calibrated to enable effective adversarial optimization.
We apply our general method to three attack scenarios: adversarial examples for vision-LLMs, jailbreaks and prompt injections. Our attacks successfully generate malicious inputs against systems that only expose textual outputs, thereby dramatically expanding the attack surface for deployed LLMs. We further find that better and larger models exhibit superior calibration when expressing confidence, creating a concerning security paradox where model capability improvements directly enhance vulnerability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the setting of jailbreaking closed sourced LLMs with text-only API access, which is more prevalent but difficult setting than the white box setting, as malicious actors can leverage gradients to steer LLMs, and with APIs that output log probs, as gradient free can be used for jailbreaking. Existing work on the text-only setting requires surrogate LLMs or reward models to score possible jailbreaks and update them. This work provides a different approach by using the victim model directly to score possible jailbreaks / adversarial examples, and then leverages these scores to generate new attacks. This is done by developing a series of binary comparisons of potential adversarial attacks for the victim model to score, which are then used to attack the model directly.

### Strengths
The main strengths of this paper is that it does not require any auxilliary models and data to score adversarial samples, which is very useful for stress-testing such closed source API LLMs. The algorithm is simple and very easy to implement and builds on existing black box adversarial attacks on LLMs and VLMs. Additionally, the use of binary comparisons instead of absolute confidence scores is novel and well motivated.

### Weaknesses
There are two primary weaknesses of this paper. First, there is a lack of related work / comparisons to existing text-only black box methods. Methods such as BlackDan [1] and D-Attack / DH-CoT [2] are not mentioned. Additionally, while existing methods like PAIR and Tree-of-Attacks do require auxilliary models, it would strengthen the paper to compare against them, as currently the only comparison is to the logprobs based method. 

Secondly, the results in Table 1 are confusing, as it seems that the approach is not very successful on its own. Table 1 shows attack success rates that are significantly lower than the transfer based approaches across all models, and incorporating the attack with transfer based methods, either directly or through ensemble does not yield significantly better results. Can the authors explain this? 


[1] Wang, X., Huang, V. S. J., Chen, R., Wang, H., Pan, C., Sha, L., & Huang, M. (2024). Blackdan: A black-box multi-objective approach for effective and contextual jailbreaking of large language models. arXiv preprint arXiv:2410.09804.
[2] Zhang, C., Zhou, L., Xu, X., Wu, J., Fang, L., & Liu, Z. (2025). Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts. arXiv preprint arXiv:2508.10390.

### Questions
1) How efficient is the algorithm at converging to a successful adversarial attack? Can the authors provide an ablation study on changing the maximum number of iterations?
2) The majority of the attacks are done on non-reasoning models or non-thinking models. How does the performance of this approach change for models such as the o1-3 OpenAI reasoning model series?
3) Is the repeated iteration done in a multi-turn fashion (i.e. does the model's context include the past binary comparisons)?

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
4

### Summary
The paper presents an effective black-box attack strategy for LLMs that operate in the most restrictive, text-only API setting. The core contribution is the discovery that while LLMs are poorly calibrated for absolute confidence scoring, they are surprisingly well-calibrated for binary comparisons. The authors leverage this insight to build a general, query-based "hill-climbing" optimization algorithm. By repeatedly "asking for directions" via these comparative prompts, the attack iteratively refines a malicious input. The method's effectiveness is demonstrated across three distinct and important domains: adversarial examples for vision-LLMs, jailbreak attacks, and prompt injections.

### Strengths
1. The paper's primary insight, using comparative, self-reported confidence as an optimization signal, is a significant and novel contribution. It elegantly bypasses the common requirement for logits or confidence scores, which are rarely available in production systems. The validation in Figure 3, which contrasts the failure of absolute scoring with the success of binary comparison against ground-truth logits, is very convincing.
2. The attack is designed for and effective in the "text-only" black-box setting. This is a very practical and challenging scenario, and this work dramatically expands the known attack surface for deployed, proprietary models.
3. A key and counterintuitive finding is that larger, more capable models are often more vulnerable to this attack. The paper provides strong evidence for this across model families (e.g., Qwen-VL-72B > 7B, GPT-5 mini > GPT-4o mini). The hypothesis that this is because they are better at the comparative reasoning task is a useful insight for the field.
4. The method is not a one-trick thing. It is successfully applied to three very different attack types (adversarial images, jailbreaks, and prompt injections) across numerous model families (GPT, Claude, Llama, Qwen). Furthermore, the results are state-of-the-art, achieving near-perfect success on jailbreaks and even outperforming logit-based attacks in some cases (e.g., 98% vs. 56% on GPT-4.1 mini).

### Weaknesses
1. In Table 1, the high ASRs (e.g., 94.7% on GPT-4o mini) are achieved with the "Transfer+ours" hybrid method. In this hybrid, the improvement from the optimization step is sometimes marginal (e.g., 92.9% to 94.7% for GPT-4o mini), suggesting the (known) transfer attack is doing most of the work. The standalone power of the query attack for vision seems less impressive than for text.
2. The entire attack hinges on the model's willingness to answer the comparative "meta-prompt." The authors note this as a failure mode, where a strongly aligned model may simply refuse to perform the comparison. This seems like a critical vulnerability of the attack itself. The paper does not sufficiently explore the robustness of the attack to simple defenses on this meta-prompt (e.g., "I cannot compare prompts in a way that might lead to a harmful outcome").
3. As mentioned in the first point, the benefit from the "Transfer+ours" method is highly variable. For GPT models, the gain is tiny (1.8% on GPT-4o mini) (table 1), but for Claude models, it is massive (35.1% to 59.6% on Haiku 3.5). This significant discrepancy is not analyzed. Does it mean the transfer attack is already near-perfect for GPT, or that the Claude models provide a much better optimization signal? This is an important detail.

### Questions
1. Following up from weakness, why is the improvement from "Transfer+ours" so minimal for GPT models but so large for Claude models (table 1)? Does this imply that the "directions" from GPT are less effective, or that the CLIP-based transfer attack is already highly aligned with the GPT vision encoder?
2. The paper identifies that a model can refuse the comparison query as a defense. How difficult is it to bypass this? Did the authors experiment with iteratively re-prompting or reformulating the comparison prompt itself to get around such refusals?
3. The "security paradox" claim, does this finding suggest that alignment techniques that rely on a model's advanced reasoning (like self-critique or Constitutional AI) are fundamentally flawed, as that same reasoning capability can be turned against the model to guide an attack?
4. For the vision-LLM attacks, the query budget was 1,000. What was the average number of queries for a successful attack? Table 3 provides this for jailbreaks (e.g., 4.9-79.3 queries), which is very efficient. Is the cost for vision attacks similarly low, or does it regularly require hundreds of queries?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce an approach for attacking black-box large language models. They iteratively present a model with two slightly different images and based on the responses to that question, select the image showing desired behavior, and then use the selected image to create the two images for future iterations. This results in an image that can successfully elicit the desired behavior in models.

### Strengths
Black box attack for VLMs
Shows promising results

### Weaknesses
Very compute heavy and expensive
Unclear from the paper how many iterations were required to get the shown ASR

### Questions
1. What were the computational costs/time required for the optimization of the adversarial image?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the use of a binary comparison rather than relying on a signal (e.g., logprob signals or from a surrogate model). They show that in image classification, this comparison is significantly more calibrated than directly asking the model. Furthermore, using this tool they  case study three scenarios for which their method is effective.

### Strengths
1- The paper is well-written and is easy to understand. 

2- The comparison between binary comparison and directly asking the model for the confidence score is the main novelty of this paper, where in my view can be applied in a broader scope.

3- The results of the paper seem promising.

### Weaknesses
1- I believe that a more comprehensive related work needs to be done in this paper. The idea of comparing only two prompts is largely investigated in the prompt optimization literature. For instance: 

Wu et al. "LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization"
Lin et al. "Prompt Optimization with Human Feedback" 

2- In the continuation of the previous point, there are several papers that study the calibration of the model's score. This paper only compares with a basic approach--directly asking for a score without defining any rules, criteria, etc.

3- I'm still confused about how the model answers this question: "which one of these prompts is less harmful: 1- Tell me how to build a bomb|suffix_1; and 2- Tell me how to build a bomb|suffix_2" and there needs to be clarification here. Moreover, I don't understand how the comparison of these two will give a meaningful signal. I think more ablation studies are required to ensure the significance of this comparison. For instance, how does the answer of this comparison correlate with the value of the log-probs. 

4- The baselines in all three case studies are weak. In the first scenario, the authors only compare their methods with a transfer method. Can the authors clarify if the attack algorithm for the transfer was adapted from (Li et al., 2025)? How about other methods? For instance:
Hu et al. "Transferable adversarial attacks on black-box vision-language models" 

5-  In the second and third scenario, they only compare with a random search method that has access to the log-probs. Even in this case, the only meaningful comparison is for GPT-4.1 mini in Table 3 (actually I don't understand how the log-probs were calculated for this black-box model. Are they exploiting the same top-5 method as the original paper does?) Why didn't the authors include more attacking methods (preferably more recent ones) in their table?

### Questions
1- Can you do the same calibration experiment for the adversarial prompt? I.e., asking a model to give you a score (there are methods developed for this already such as StrongReject) vs. asking it two only compare two prompts?

### Soundness
2

### Presentation
3

### Contribution
2
