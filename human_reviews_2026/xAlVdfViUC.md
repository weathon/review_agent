# The Privacy-Hallucination Tradeoff in Differentially Private Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 6

## Abstract
While prior work has studied privacy tradeoffs with utility and fairness, the impact of privacy-preservation on factual consistency and hallucination in LLM outputs remains unexplored. Given that privacy-preservation is paramount in high-stakes domains like healthcare, the factual accuracy of these systems is critical. In this study, we uncover and investigate a privacy-hallucination tradeoff in differentially private language models. We show that while stricter DP guarantees do not distort knowledge acquired during standard pre-training, they hinder the model's ability to learn new factual associations when fine-tuned on previously unseen data, as a result of which the model tends to hallucinate incorrect or irrelevant information instead. We find that the proportion of factual texts generated drops by 17-24% when models are fine-tuned on the same data using DP (epsilon = 8), compared to the non-DP models, and on average, the factuality scores differ by at least 3-5%. This disparity is further pronounced when pre-training with DP, where we find a 43% drop in the number of factually consistent texts. Our findings underscore the need for more nuanced privacy-preserving interventions that offer rigorous privacy guarantees without compromising factual accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies how differential privacy (DP) affects factual accuracy in large language models, revealing a privacy-hallucination trade-off. For DP fine-tuning, the authors find that stricter privacy budgets lead to more hallucinations for facts in fine-tuning data, while pre-training knowledge remains largely unaffected. For DP pre-training, the authors find that it causes much larger factual degradation compared to non-DP pre-training. The work highlights the need for privacy-preserving methods that protect data without compromising factuality.

### Strengths
- The paper studies an underexplored question at the intersection of DP and factuality in LLMs

- The empirical studies are well-presented and cover both DP fine-tuning and DP pre-training

### Weaknesses
- Lack of conceptual novelty. The central finding is unsurprising given the well-known privacy-utility trade-off. DP inevitably reduces the model’s ability to fit the training data, and thus degrades utility, including factual accuracy. In the context of LLM fine-tuning, this manifests as poorer next-token prediction on factual tokens within the private dataset. Without analyzing overall utility (e.g., perplexity) alongside factuality, the paper's framing around a “privacy-hallucination” trade-off risks being a rebranding of the well-known privacy-utility trade-off, rather than a new phenomenon.

- The study examines only two privacy budgets (16 and 8), which are far too loose to be meaningful for most DP applications. This makes the findings difficult to interpret for practitioners concerned with realistic privacy guarantees. Moreover, the evaluation is confined to a single factuality metric (FactScore) and a narrow set of Wikipedia-style generation tasks, without testing robustness across alternative metrics, datasets, or domains. As a result, it remains unclear whether the reported privacy-hallucination effect is general or simply an artifact of this specific setup.

- The paper primarily reports an empirical observation that stricter privacy budgets increase hallucination rates, without offering a principled explanation or proposing methods to mitigate this trade-off. As a result, the work feels more like a measurement study, and does not meet the level of technical depth typically expected for ICLR.

### Questions
I don't have further questions.

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
2

### Summary
This paper studies the effects of differentially private fine-tuning on factuality/hallucination.

Three sets of experiments/evaluations are performed:

- Fine-tune (using DP-SGD) GPT-J model (pre-trained on data from before 2020) on new Wikipedia articles to generate articles from title.  Then, during evaluation, the model is prompted to generate articles from articles titles from a hold-out set.
    - Using automated evaluation (FactScore), they found models fine-tuned on larger privacy budgets (or no DP/standard FT) have higher FactScore compared to those with tighter budgets, leading to the conclusion that DP impacts model hallucination.  This suggests that DP hinder model's ability to acquire and generalize new knowledge.
    - Human evaluation also shows similar trends.
- They subsequently prompted the LLM to generate from titles on Wikipedia articles from before 2020 (likely in the pre-training set), and find equal performance among models trained at all privacy budget.
    - This suggests that downstream training on new knowledge with or without DP does not affect knowledge acquired from pre-training.
- They also compare and evaluate VaultGemma, which can be viewed as a DP pre-trained variant of Gemma 3-1B, and found the former to have a lower FactScore (i.e., higher hallucination)

### Strengths
- The paper is well-motivated and timely, and the experiments and methodology are carefully setup (including the handling of deduplication of generated facts).
- The paper adds to a cluster of recent explorations on the interaction between DP fine-tuning, regurgitation/memorization, and downstream performance. In terms of originality, however, one may argue that hallucination is just another way of viewing/framing memorization and generalization performance.

### Weaknesses
- It is not obvious to the reviewer whether real issue is hallucination, or the fact that the model cannot acquire real knowledge due to the DP noise.
    - It would be better informed if the author could also index the fine-tuning set and check whether the generated fact belongs to the fine-tuning set or not, so one could check whether the ability/failure to generate fact is related to memorization (which is known to be influenced by DP) or the ability to "reason" over new knowledge.  One interesting experiment would be to compare the ability to generate multiple-hop facts by chaining together one-hop facts in the fine-tuning set.
    - Could the authors comment on the frequency at which models makes statement (factual and hallucination combined), before dedup?  And if it differs across DP/non-DP models?
    - A common remedy for hallucination is to learn to abstain; if the model has learned to abstain before/after FT, what would be the influence of DP?
- A minor note is that, although the setup compares DP-SGD at different eps settings, a recent paper (https://arxiv.org/pdf/2503.12314, and references therein) shows that different DP-SGD hyperparameters calibrated to the same eps setting can have different performance (both utility and memorization).  The reviewer is curious whether the FactScore is also sensitive to the choice of hyperparameters at the same eps; if so, should we instead report the max FactScore achieved among hyperparam settings all calibrated to the same eps instead?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the interplay between differentially private (DP) training of LLMs and the frequency of hallucinations (producing statements contradictory to facts present in the training data).
The authors consider both DP fine-tuning and DP pre-training, investigating the increase of hallucinations separately with respect to the fine-tuning or pre-training data.

The private data is chosen to be Wikipedia articles created after 2020, when the pre-training data ends. The privacy unit is chunks of 1024 tokens. The articles are either Science, where the information may appear in other forms in the pre-training data, or AI, where the information does not exist before 2020. The data is padded also with random Wikipedia articles in order to have a sufficiently large dataset. The actual "private" data is either 231 (Science) or 124 (AI) articles. Factual evaluation is done by asking the (privately-trained) LLM to produce an article given a title only. Factual claims are evaluated with respect to the true Wikipedia article using LLM-as-a-judge. Human evaluations were also done by showing CS PhD graduate students text produced by private and non-private LLMs and asking them to evaluate veracity. Finally, semantically repeating hallucinations that appear multiple times are identified through a heuristic clustering approach.

The results are split into three parts.

### Fine-tuning DP training, Fine-tuning hallucinations

In this setting, the LLM was fine-tuned on data absent in the pre-training data. The LLM was then asked to generate articles corresponding to the new data and evaluated through the LLM-as-a-judge FactScore (FS) as well as human evaluation. Results are reported for $\epsilon \in \{8,16,\infty\}$ as well as with no fine-tuning at all. The general trend is for the FS to decrease with $\epsilon$ by 3 to 5 percentage points comparing $\epsilon = \infty$ to $\epsilon = 8$. This general trend also appeared in the human eval. Curiously, the performance at $\epsilon=8$ is sometimes less than the model which has not been fine-tuned at all.

### Fine-tuning DP training, Pre-training hallucinations

Here, the LLM was fine-tuned as before, but the generated articles are from the pre-training data. Varying $\epsilon$ did not seem to make a large or consistent difference in FS performance in this case.

### Pre-training DP training, Pre-training hallucinations

Here, there is no fine-tuning. The authors compare 3 open source pre-trained models: Gemma, VaultGemma, and GPT-2. Gemma and GPT-2 are non-DP models where Gemma has been trained more recently than GPT-2. VaultGemma is a full DP trained version of Gemma. The models are asked to generate Wikipedia articles created after the training data cutoff of GPT-2. Across several domains, Gemma does better than VaultGemma which in turn does better than GPT-2. The differences range from less than 1 percentage point to 15 percentage points.

### Strengths
- This paper investigates the question: do privately trained models hallucinate more? Intuitively, this should be the case as there is a tension between memorization and privacy. However, the authors take the steps to empirically evaluate this claim in several settings.

- The three settings are interesting cases to study the tradeoff between privacy and hallucinations.

- While using somewhat loose measurements of hallucinations/factual validity (for example, using an LLM-as-a-judge), the authors use (limited) human eval and some ablations to show consistent results.

### Weaknesses
Two high-level comments. First, the empirical setup has some issues making it hard for me to completely buy the takeaways in the discussion/conclusion section (see specifics below). Second, and perhaps more important, it is hard for me to understand how this study departs from studying the privacy-utility tradeoff of a LLM to specifically the privacy-hallucination tradeoff (see final bullet below).

- Error bars showing variance of all the results are missing and are very important in this study. The gaps between reported FS are sometimes quite small and there is currently no way to evaluate the significance of these gaps.

- Dataset size is really very small with only several hundred articles being used as the data containing new facts. Given such few data, the guarantees of DP almost directly imply that hallucination on such data must be similar to that of the base model. If AlphaEvolve is only mentioned in say a couple articles, the model simply cannot learn what it is. 

- The recurring hallucination analysis does not make it clear to me whether the private models have more recurring factual inconsistencies in particular, or just make more factual errors overall. The recurrence rates do not differentiate from many claims repeated twice or one claim repeated many times.

- The phenomena where the DP fine-tuned model performs worse than the base model is strange. There is another control test that would be interesting to run in the first research question. To separate the effect of DP dynamics from the training data, DP fine-tuning can be run on, say Wikipedia Science, but evaluted on generating articles from Wikipedia AI. This would also help separate effects from training on Wikipedia style articles in general vs. specific content.

- The solution to hallucination is not simply to know more. Rather, the desired behavior of a model is to cite sources and acknowledge what it knows and does not know. It would be very interesting to see how DP interplays with such mitigations, but that is not investigated in this paper. Rather the models are forced to produce a list of factual claims about a small dataset it has been privately trained on, and we see that cannot reproduce those factual claims. This is inherent to privacy, but I am not sure that this is really the full story of private models and hallucinations.

### Questions
- Why is the FS so low on non-privately fine-tuned models? Experience with LLMs would suggest that if trained directly on a Wikipedia article for a given topic, it should be able to return a correct article perhaps with some limited hallucinations.

- Some details seem to be missing in Appendix G about exactly how the different components of the FS algorithm are implemented (fact extractor module, claim verification model, knowledge source, prompts, etc.). I understand that Llama-3.1-8B-Instruct is used, but how exactly is not specified.

- I think the bolding in the first column of Table 6 is incorrect. Also, there should be bolding in Table 7.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper outlines a study of the impact of differentially private (DP) training on large language model (LLM) hallucination. They find that when models are fine-tuned using differential privacy, they exhibit more tendency to hallucinate when prompted with information from the fine-tuning set. In contrast, they don't observe the same levels of hallucination for information from the pre-training set. Finally, the test the effects of DP on pre-training, observing that it produces more hallucination than fine-tuning. They conclude that hallucination is another tradeoff / risk of DP training.

### Strengths
The problem they explore is valuable. While most prior work has examined accuracy-privacy tradeoffs, this work explores hallucination-privacy tradeoffs at multiple stages of training. This is an important consideration for the community. The analysis they provide gives some insights into how DP impacts hallucination, and may point toward future work in the area. Generally the writing is clear, and the structure of the paper is sound.

### Weaknesses
Some points are somewhat challenging to follow, and I have concerns about some of the claims made, which I outline below. If these can be addressed I would be open to raising my score.

1. The annotator-model agreement scores, particularly for support, and for DP-16 are very low. While it is claimed that humans generally agree with the model produced scores, some of these kappa values are low enough to almost indicate a lack of agreement. I would ask that the phrasing around this be changed to reflect this.

2. The text claims that models trained with eps=8 output "fewer recurring supported claims [...] and more recurring unsupported claims". However, table 4 indicates that for Wikipedia AI, models with eps=16 output more recurring unsupported claims, which is not indicated in the text or the bolding of the table.

3. While recurring hallucinations are a theme that is addressed in the introduction, it isn't discussed in depth in the paper. The analysis in table 4 is somewhat challenging to follow, and there is no analysis of what types of claims tend to be repeatedly hallucinated. I would like to see a better analysis of this in order to support the claim that models are hallucinating the same pieces of information.

4. While is it not strictly required, it would be nice to see some discussion of how this tradeoff might be improved, even as a small point in the discussion section.

### Questions
1. Do you have ideas for how to improve these tradeoffs?

2. What kinds of recurring hallucinations did you observe?

3. Can you clarify how the analysis in table 4 was done?

### Soundness
2

### Presentation
2

### Contribution
3
