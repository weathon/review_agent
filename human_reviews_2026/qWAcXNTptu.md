# Beyond Binary Evaluation: Measuring Language Model Hallucinations Through Distributional Correctness

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Common evaluation paradigms for language models focus on scoring single responses through accuracy metrics or proper scoring rules, failing to capture the full richness of a model's belief state. Recent work illustrates that language models hallucinate in-part because they are optimised to be good test-takers under binary scoring schemes that reward any answer over abstention. While this insight naturally leads to penalty-based approaches, they ignore crucial distinctions in how models distribute uncertainty, for example between hedging toward incorrect answers versus hedging toward ``I don't know'' responses. We introduce a novel evaluation metric to solve this problem of not considering a model's entire probability distribution over answer choices. Our metric naturally distinguishes between harmful overconfidence in wrong answers and uncertainty expressed through abstention, providing scores in an interpretable default range. Through theoretical analysis and illustrative examples, we demonstrate our metric offers a more nuanced and aligned evaluation paradigm that incentivises models to express genuine uncertainty rather than guessing. We then adapt 12 existing evaluation benchmarks to our metric's variants and measure performance on six language models, showing that for half of the tested benchmarks scores are *negative across all tested models*, indicating significant tendencies towards hallucination.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Under binary scoring (+1 for correct, 0 for incorrect), there is never a reason to abstain from answering or say "I don't know": it's always better to make a guess. It is then surprising that LLMs trained to perform well under binary scoring learn to "hallucinate", i.e., make a best guess even when they know. The paper proposes a new metric called the Distributed Correctness Score (DCS) that rewards abstention when the model is uncertain. The authors discuss similarities and differences between DCS and existing uncertainty-aware metrics like confidence-weighted accuracy. The authors evaluate 6 LLMs on 12 benchmarks and show that even modern LLMs perform quite poorly with respect to DCS: for 6 of the benchmarks, all LLMs have negative scores.

### Strengths
I think the issue of binary scoring incentivizing guessing is crucial, not just for hallucination but for all forms of caution and overconfidence in LLMs. Incorporating a correctness metric which penalizes overconfidence is a natural countermeasure. The selection of models and benchmarks to test is reasonable. Scores are reported in a clear way, with standard error included. The writing is quite clear throughout.

### Weaknesses
### **Concerns about the DCS metric**

I have major concerns about the DCS metric which the whole paper revolves around. I agree that this metric solves the core issue with binary scoring, but it seems to me that proper scoring rules solve this issue better. For the area chair, DCS is defined as $(\ell_c p_c - \ell_W p_W) (1- p_{IDK})$ where $p_c,p_W, p_{IDK}$ are the probabilities the model respectively assigns to the correct answer, to all of the wrong answers in aggregate, and to the "I don't know" answer. $\ell_c$ and $\ell_W$ are user-chosen parameters that are usually treated as 1.

1. It seems to me that a rational agent will always choose either p_{IDK} = 0 or p_{IDK} = 1. Specifically, it is optimal to choose p_{IDK} = 0 iff the model believes that $(\ell_c p_c - \ell_W p_W) > 0$. As such, DCS does not elicit the true belief state of a rational agent.

2. Why is it natural to include an explicit IDK option, rather than eliciting a distribution over the actually plausible answers and then deciding whether to abstain based on that distribution? The model knows that IDK is not the correct answer, so the model's true belief state over the correct answer should answer 0 probability to IDK.

3. The authors critique proper scoring rules by saying that they "fail to capture the full richness of a model’s belief state": specifically, proper scoring rules only consider the max probability of any answer and ignore how the rest of the probability mass is distributed. However, I am unconvinced that the rest of the distribution matters. The authors' argument seems to be that we care how much probability is assigned to IDK. But this issue goes away if IDK is not included as an answer option to begin with.

4. The authors also argue:
> Unlike forecasting tasks where the ground truth is a stochastic label, language model evaluations present deterministic facts of the matter. The ‘true’ conditional distribution is a point mass, so it is meaningless to demand that a model report the frequency of correctness for each option, as for example the Brier score rewards...Our objective is not to elicit calibrated probabilities but to measure trustworthy epistemic behaviour.

While the authors mention one interpretation of the Brier score, it is perfectly well-defined to still award of a score of 1 - (p-c)^2 where p is the max probability of any answer and c in {0,1} indicates whether that answer is correct. Furthermore, this definition maintains the desirable property that a rational agent should always report its true probability distribution over answers. DCS does not have this property, as mentioned in Point 1 above. It's also unclear why calibrated probabilities are not satisfactory as "trustworthy epistemic behaviour", or why DCS induces more trustworthy epistemic behaviour.

**What I would find convincing.** The main argument I see for DCS over proper scoring rules is that if we want to finetune models to sometimes abstain, then it makes sense to include an explicit IDK option in the finetuning dataset. This argument makes sense to me, but no finetuning experiments are performed to see whether DCS effectively serves this purpose. Also, a simpler scoring rule like "+1 for correct, 0 for abstain, -1 for wrong" could suffice for the finetuning application. See [Kang et al (2025)](https://aclanthology.org/2025.naacl-long.183/), which performs RL finetuning on this simpler scoring rule and shows that it successfully teaches the model to abstain. I think it's totally plausible that finetuning on DCS is more effective at teaching models to appropriately abstain. But if that's the main claimed benefit of DCS, then I think the paper needs to show that experimentally.

## **Other issues**

1. No proper scoring rules are included in the experiments as baselines.
2. The authors mentioned that in two cases, a model got 0% accuracy. The authors suggest that "these models failed to understand the minimally-adjusted multiple-choice instruction format." But if the authors are extracted answer probabilities directly from the LLM output probabilities (i.e., what is p("A"), what is p("B"), etc, then normalize), format following shouldn't really be an issue? Also, is it even possible for accuracy can be 0 and DCS to be positive? If DCS is positive on a given question, then the probability assigned to the correct answer is greater than the sum of probabilities assigned to incorrect answers, so the correct answer has the max probability and thus accuracy should be positive?

Overall, I think this is a very important problem and the ideas in the paper are promising, but I'm not sure that the current approach lives up to the motivation of the paper.

### Questions
It would be great if the authors could clarify my questions about the 0% accuracy issue. I also am open to being convinced about the benefits of the DCS metric.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces the Distributional Correctness Score (DCS), a theoretically grounded metric that evaluates full probability distributions, and incorporates abstention as a vital component. This metric is designed to mitigate the pitfalls of the traditional binary evaluation metrics for language models, as they fail to capture epistemic uncertainty. The metric is evaluated across a wide range of datasets.

### Strengths
* The proposed DCS metric is novel and interesting and makes intuitive sense.
* The paper includes extensive examples of the DCS metric across various cases.
* The paper includes evaluation across of diverse range of datasets.
* The paper is well written and easy to understand.

### Weaknesses
* The paper does not clarify how the DCS metric can be applied to problems without options. E.g., question from QSM8k where the answers could be integers in the range $[-\infty ,\infty ]$.

*  The paper should also discuss other metrics that consider the distribution over answers: "Enhancing Hallucination Detection through Noise Injection, arXiv Feb 2025".

* Section 6 includes interesting results across models. However, the paper does not provide any explanations as to why DCS score is low or high for a specific model. Why is Llama3.1 8B Instruct DCS score highest on MMLU? 

* The evaluation should consider newer models such as Qwen-2.5 or Qwen-3. 

* There is also no analysis of the effect of model size on the DCS score. It would be interesting to show the DCS score for the same family of models from small to large, e.g., Qwen-2.5 from 0.5B to 72B.

### Questions
* The paper should include more extensive analysis across model sizes.
* The paper should include a discussion of the applicability of the DCS metric when questions do not a set of options as answers.
* The paper should include a more extensive discussion of prior work.

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
2

### Summary
This work introduces the Distributional Correctness Score (DCS), a novel metric that evaluates a model’s entire probability distribution rather than the maximum predictions. This new metric considers the model's uncertainty for correct answers, incorrect answers, and abstentions.
The authors prove a theoretical analysis to demonstrate that  DCS incentivises the desired behaviour: confidence in correct answers, uncertainty when knowledge is lacking, and preference for abstention over confident incorrectness. 
Experimental results across 12 existing benchmarks indicate that (1) many language models exhibit systematic epistemic overconfidence;
(2) all models hold negative DCS scores

### Strengths
* this work identifies that current metrics focus on a single argmax answer while ignoring the distribution across the space of possible responses, which might be useful for future studies
* this work provides both theoretical and empirical studies. 
* extensive experiments across 12 benchmarks
* this proposed DCS is working, which is able to mitigate the overconfidence issue

### Weaknesses
* I did not fully understand the motivation of this metric. (1) The probability assigned to a specific answer is computed by a softmax layer, which accounts for the logits over different answers, including abstention as well. (2) proper scoring rules and other metrics, e.g. entropy, can also depict this. It would be useful if the authors clarify why the DCS is necessary and stress the difference compared to existing metrics
* It is tricky to see the benefit of using the proposed DSC. From Figure 2 and Table 2, we can observe that DCS is consistently lower and remains negative in most cases, but why should we stick with DCS? Under which conditions (tasks), we should select DCS as the metric.

### Questions
* what is the meaning of a concrete value of DCS? If the value is negative or positive, how to explain it?
* In Figure 2, I did not understand how to compare DCS to other baselines. I observe that DCS is consistently lower than other baselines, but what does it mean? It is hard to understand the benefits of using DCS in this figure
* In Table 2, it is tricky to derive the main findings.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the Distributional Correctness Score (DCS), an evaluation metric for language models. The authors argue that standard binary accuracy metrics incentivize hallucination because they reward guessing over abstention. Unlike simple penalty-based metrics, DCS evaluates the entire probability distribution over answers, including an explicit "I don't know" (IDK) option. The paper presents experiments on 12 benchmarks with 6 LLMs that show that many models achieve negative average DCS scores, revealing overconfidence that is masked by traditional accuracy metrics.

### Strengths
I see the main strengths of the paper as follows:

- problem formulation: the paper identifies a key socio-technical issue: current evaluation metrics encourage models to game the system by guessing rather than abstaining; the distinction between "error-hedging" and "abstention-hedging" is a useful conceptual contribution
- exposition: the paper is well-written and the motivating examples effectively illustrate the flaws in current metrics that DCS aims to fix
- empirical evaluation: testing across 12 diverse benchmarks and 6 models of varying sizes provides a reasonably comprehensive picture of how DCS behaves in practice compared to standard metrics

### Weaknesses
I believe there are a few weaknesses:

- implementation: the reliance on log-likelihoods for all answer options plus a canonical "IDK" response is a meaningful barrier to adoption as many API don't provide these
- parameterization: the introduction of $l_c$ and $l_w$ parameters adds flexibility but seems somewhat arbitrary and there is no explanation as to how these parameters ought to be chosen
- sensitivity to "IDK" phrasing: the method assumes a single, canonical "IDK" string represents abstention, but language models might distribute uncertainty across many synonyms (e.g., "I'm unsure", "Unknown", "Cannot determine"); there is no evaluation regarding the sensitivity to the specific choice of this string
- multiple choice question-answering: the formulation is tightly bound to multiple-choice or closed-set classification and as such does not address the more common real-world problem of open-ended long-form evaluation

### Questions
- Have you evaluated the sensitivity of DCS to the specific string used for abstention (e.g., "I don't know" vs. "Not sure")? Does using a set of abstention phrases and summing their probabilities improve robustness?
- How would you recommend applying DCS to a black-box API model that only returns generated text without log probs? Is there a sampling-based approximation you considered?
- Could you provide an explanation or a small study on how to set $l_c$ and $l_w$ for different risk profiles (e.g., medical advice vs. creative writing)?

### Soundness
3

### Presentation
3

### Contribution
3
