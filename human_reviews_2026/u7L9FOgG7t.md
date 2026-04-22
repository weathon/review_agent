# Midtraining Bridges Pretraining and Posttraining Distributions

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Recently, many language models have been pretrained with a "midtraining" phase, in which higher quality, often instruction-formatted data, is mixed in at the end of pretraining. Despite the popularity of this practice, there is little scientific understanding of this phase of model training or why it is effective. In this work, we conduct the first systematic investigation of midtraining through controlled experiments with language models pretrained from scratch and fine-tuned on supervised finetuning datasets in different domains. We find that when compared after supervised fine-tuning, the effectiveness of midtraining is highest in the math and code domains, where midtraining can best reduce the syntactic gap between pretraining and posttraining data. In these cases, midtraining consistently outperforms continued pretraining in both in-domain validation loss as well as pretraining data forgetting after posttraining. We conduct ablations on the starting time of the midtraining phase and mixture weights of the midtraining data, using code midtraining as a case study, and find that timing has a greater impact than mixture weights, with earlier introduction of specialized data, yielding greater benefits in-domain as well as preserving general language modeling better. These findings establish midtraining as a domain adaptation technique that compared to continued pretraining yields better performance through reduced forgetting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper measures the effectiveness of changing the pretraining distribution mid-training. The problem is formalized as picking a starting point for mid-training and a mixing parameter between the mid-training distribution and the initial pretraining distribution. The experiment explores these settings for 4 test domains and 3 model sizes.

### Strengths
* Clear presentation of the motivations, an important topic for LLM training.
* Experiments are performed over a variety of domains.

### Weaknesses
* The paper does not report a joint hyper-parameter search over the two hyper parameters of midtraining (fraction of pretraining data in the second phase, fraction of total tokens in the second phase). The reader cannot understand how the two parameters interact, e.g. could a shorter 2nd phase might be compensated by seeing less pretraining data in the second phase? It is not even clear if any pretraining is necessary, i.e. could the best result be obtained with a single training phase that mixes in-domain and pretraining data (single mixture).
* The paper ignores prior work on injecting pretraining data during supervised fine tuning which is a lot like your definition of mid-training, including 
   * Liu et al. Improved fine-tuning by better leveraging pretraining data. Advances in Neural Information Processing Systems, 35:32568–32581, 2022.
   * Kang et al Get more for less: Principled data selection for warming up fine-tuning in llms. arXiv preprint arXiv:2405.02774, 2024.
   * Ibrahim et al Simple and scalable strategies to continually pre-train large language models, 2024.
   * Bethune et al. Scaling laws for forgetting during fine-tuning with pretraining data injection. arXiv preprint arXiv:2502.06042, 2025.
* The paper provides only limited scale experiments and does not attempt at extracting a scaling law to predict behaviour at scales larger than their computing budget.

### Questions
* L018 “Syntactic gap” what do you mean? Could you define it?
* L133 “models have stabilized" what is a stabilized model? Could you define it?
* The legend is missing on Figure 4. Could you include it?
* The font in Table 1 is smaller than the main text font. 
* Figure 3 is too small, esp. the figures are not tall enough to be readable. Most of the labels (titles, axis names and legends) are repeated several times: space can be used more efficiently.
* In Table 3. How were the fraction of starcoder/math selected? How was the amount of pretraining vs mid-training selected? How do these two parameters interact?
* In Figure 3b and 3c, it seems that the 20% mix of starcoder is not particularly good for both model sizes. Why did you pick 20% for the other experiments?
* In Table 2 and Table 8: it seems that the best data source for LIMA is different for all three model sizes, how do you explain this inconsistency?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper systematically examines midtraining, an intermediate phase that mixes specialized data into pretraining to improve posttraining outcomes. Through controlled experiments , the authors test various midtraining domains (code, math, instructions, QA) and analyze effects on downstream performance and forgetting. They find that midtraining is most effective in domains far from web-text data, acting as a bridge between pretraining and fine-tuning distributions. Early introduction and mixed data outperform late or domain-only approaches. The study provides an empirical foundation for understanding midtraining as a structured domain adaptation technique.

### Strengths
- The paper provides a clear and well-executed study of midtraining, a timely and underexplored topic in large language model development. Its originality lies in empirically evaluating this training phase across domains. 

- The experiments are thorough, well controlled, and convincingly analyzed. 

- The writing is clear and structured, with results that directly support the claims. 

- The work offers meaningful insights into how midtraining bridges pretraining and fine-tuning, making a valuable and practical contribution to understanding domain adaptation in modern model training.

### Weaknesses
- While the study is thorough, its scope is limited to relatively small models (up to 410M parameters), leaving uncertainty about how the findings scale to contemporary billion-parameter systems. The paper would benefit from stronger evidence that the observed trend particularly regarding timing sensitivity and mixture effects hold at larger scales.

- The analysis focuses on supervised fine-tuning and does not test whether the same principles apply to reinforcement learning or preference-based posttraining, which are now standard in practice. Including even limited experiments or discussion in this direction would strengthen the generality of the conclusions.

- The paper would benefit from citing and briefly discussing related work that also explores interventions during pretraining to improve posttraining adaptability. In particular,  “Improving Language Plasticity via Pretraining with Active Forgetting” (NeurIPS 2023) proposes a complementary approach that modifies pretraining dynamics to enhance model plasticity. While the mechanism differs from midtraining, both works share the motivation of bridging or improving transitions between pretraining and fine-tuning phases. Adding a short comparison (e.g., in the Related Work section) would help position this paper more clearly within the broader literature on training-phase modifications for improved adaptation.

### Questions
1. How might the effects of midtraining, particularly the importance of timing, extend to larger models? 
2. Would the same bridging behavior be expected if posttraining were done through reinforcement learning or preference optimization rather than supervised fine-tuning?
3. How do the authors view midtraining in relation to other ways of shaping pretraining dynamics, such as active forgetting or data reweighting? A brief reflection on their connections would help clarify its conceptual role.

### Soundness
2

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
The paper study the impact of midtraining on the final performance, in particular through the lens of the similarity between pretraining set and midtraining set, and between midtraining and posttraining (SFT). Realistic high quality datasets are used, such as Staroder or DCLM. The model range remains 410M, using the Pythia family.

The performance is evaluated on some specialized domains using the validation loss

### Strengths
### Timely problem

The problem by the paper is timely, as midtraining is becoming increasingly important in learning pipelines. 

### Realistic datasets

These datasets have been used in the past for pretraining or midtraining of sucessful models.

### Experiments of Sec. 6

I think the experimental setup of Sec. 6 is very promising, as the dilemma "mixture VS time" is paramount to these considerations.  

### CKA

The tools of Section 7 (CKA) are also interesting, and might provide good explanations.

### Weaknesses
### Lack of accuracy-based benchmarks

The model scale limits the possibility of relying on downstream providing an accuracy, since models below 410M typically operate at the random level on many tasks. However, a non-random accuracy can be expected on tasks like ARC-Easy, so it could be interesting to measure performance through that lens.

This weakens impact of the paper, as improvement/alignment as maily measured through distributional similarity at the token logits level. This measure is based on adhoc cosine + Jaccard + Jensen-Shannon combination, but it is unclear why this metric makes sense. KL divergence looks like a more sensical choice.  

### Unsurprising conclusions

The similarity measure is a bit adhoc, and some conclusions (like finding 1) are expected.  

> Finding 3. Maintaining a mixture with general data in midtraining is preferable to continued
pretraining on specialized data alone.

This is also a finding documented in: 

"Ibrahim, A., Thérien, B., Gupta, K., Richter, M.L., Anthony, Q., Lesort, T., Belilovsky, E. and Rish, I., 2024. Simple and scalable strategies to continually pre-train large language models. arXiv preprint arXiv:2403.08763."

> Finding 4. The timing of midtraining is more critical than the mixture weight; early intro-
duction of specialized data in the code domain leads to stronger in-domain gains and better
retention of general capabilities.

I think we miss more ablations to conclude this: other midtraining/pretraining/postraining mixtures could have yield a different conclusion. See "questions".

### Questions
**Q1:** What is the rational between the similarity measure?

**Q2:** do you have a downstream benchmark accuracy (like ARC easy) results for some of these models?

**Q3:** in section 6. can you report results as total number of tokens seen from each domain, instead of training time? At equal number of midtraining tokens, which setup wins between early and late midtraining (playing on relative ratios to achieve this)?

**Q4:** in Sec 7., is there a correlation between CKA and the metric you define in Appendix C?

### Soundness
2

### Presentation
3

### Contribution
1
