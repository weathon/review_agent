# Language Bottleneck Models: A Framework for Qualitative Cognitive Diagnosis

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Accurately assessing student knowledge is central to education. Cognitive Diagnosis (CD) models estimate student proficiency at a fixed point in time, while Knowledge Tracing (KT) methods model evolving knowledge states to predict future performance. However, CD and probabilistic KT models represent knowledge states via quantitative estimates of knowledge concept mastery, limiting expressivity, while deep learning-based KT methods prioritize predictive accuracy at the cost of interpretability. We propose Language Bottleneck Models (LBMs), a general framework for producing textual knowledge state summaries that retain predictive power. LBMs use an encoder LLM to produce minimal textual descriptions of a student’s knowledge state, which a decoder LLM then uses to reconstruct past responses and predict future performance. This natural-language bottleneck yields human-interpretable summaries that go beyond the quantitative outputs of CD models and capture nuances like misconceptions. Experiments show zero-shot LBMs rival state-of-the-art CD and KT accuracy on synthetic arithmetic benchmarks and real-world datasets (Eedi and XES3G5M). We also show the encoder can be finetuned with reinforcement learning, using prediction accuracy as reward, to improve summary quality. Beyond matching predictive performance, LBMs reveal qualitative insights into student understanding that quantitative approaches cannot capture, showing the value of flexible textual representations for educational assessment.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Language Bottleneck Models (LBMs) for cognitive diagnosis. 
An encoder LLM compresses a student’s interaction history into a free-form text summary of their knowledge state; a decoder LLM then uses only this summary to predict future performance. This work instantiates LBMs, evaluate zero-shot and with RL (GRPO) fine-tuning of the encoder using decoder accuracy as reward, and report competitive accuracy vs. strong baselines on a synthetic arithmetic benchmark and two real datasets.

### Strengths
- Casting knowledge-state modeling as a language bottleneck is interesting and novel, different from fixed concept spaces in CD and learning embeddings in KT. The diagram on page 2 (Fig. 1) clearly positions LBMs against CD/KT and motivates the bottlenecked interface.
- The design choice to freeze the decoder and focus learning on the encoder (rewarded by downstream accuracy) is elegant. Showing near-perfect decoding when given oracle summaries isolates where the difficulty lies.

### Weaknesses
- The contribution “we cast knowledge state modeling as an inverse problem” overclaims a bit: this framing is long-standing in cognitive diagnosis/psychometrics. Section 2.2 centers CD and deep KT but does not mention at all about classic Bayesian KT (BKT/IRT/HKT/PSIKT)s that already posit latent states generating responses. 
- Figure 4 is overloaded (many colors/markers/linestyles) and almost impossible to read given that is the main result figure, making the sample efficiency story hard to parse. Please split into subplots (by backbone or family), or group KT vs. CD vs. LBM, and simplify. 
- I am a bit confused by those quite different evaluations. The decoder is prompted to output “Yes/No”, and for open-source models AUC is computed from logits of those tokens, whereas closed-source models are parsed from text. This apples-oranges setup can bias AUC and calibration. Recommend a unified probability extraction (verbalizer sets or logit bias) and reporting ECE/Brier in addition to accuracy/AUC. Also, CD is evaluated with same-student 80/20 splits, while KT/LBM use unseen-student evaluation and fixed |Y|=4. This makes cross-family comparisons too tricky to understand. 
- LLMs further evaluation
    - LLMs require question text, many KT/CD datasets provide only IDs. The authors acknowledge this limitation but did not test robustness to degraded text. Please add controlled paraphrase/noise/ablation tests (e.g., ID+short stem, masked tokens) to quantify sensitivity and deployment feasibility on ID-only platforms.
    - I assume LLMs' produced summary is hugely dependent on some shallow statistics/temporal correlation in the data. Please test whether LBM-identified “understands X/Y, not Z” correlates with simple signals (e.g., per-concept past success rates, transition matrix) or add human-rater studies on faithfulness/actionability. If strong correlations exist, I am not quite sure what LBMs add beyond well-tuned regressions. 
    - Without explicit priors/dynamics, LLM summaries may be temporally inconsistent (e.g., “mastered X” then “forgot X” within a short window). Classical Bayesian models enforce coherence via latent dynamics. Consider at least check the consistency, to make sure the inconsistency is more coming from prompting/LLMs randomness. 
- The trained-encoder experiment (GRPO) shows more gains on synthetic data, but fine-tuning can be expensive. Do you have a continual/online training strategy (e.g., periodic LoRA updates, replay buffers, cost budgets) as student data grows, and how much is the marginal gain per additional student?

### Questions
Could you please answer each point in the weaknesses?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors suggest going through an text-based intermediate representation in natural language to provide interpretable knowledge tracing. They show their method matches or underperforms existing approaches on 1 synthetic and 2 real datasets.

### Strengths
I felt that the proposed approach was interesting.

### Weaknesses
The authors keep stating that existing approaches just provide "quantitative skill mastery estimates" or "uninterpretable latent representations" but the proposed approach, which does rely on embeddings that are similar latent representations, may hallucinate, and the authors do not provide any qualitative assessment of the generated explanations.

The authors write:
> "Finally, recent LLM-based approaches have shown promise for knowledge tracing tasks (Li et al., 2024a; Wang et al., 2025), but they generally remain opaque, either treating LLMs as black boxes or relying on
model-generated explanations susceptible to hallucination."

I don't see how the proposed approach is not also treating LLMs as black boxes nor wouldn't be susceptible to hallucination.

"rigid predefined KC taxonomies" is too vague (I suspect this is generated by LLM), and repeated over the text. I assume the authors mean that the q-matrix needs to be provided, but as the authors state it themselves, some neural approaches for cognitive diagnosis can learn the q-matrix.

"unintepretable latent representations" But nothing prevents the authors from trying to interpret a posteriori a learned vector by an existing deep learning approach for knowledge tracing.

> "The largest performance gap arises on XES3G5M. However, this dataset has an average accuracy of 85%, implying that even a constant predictor would achieve 85% accuracy."

One way to avoid this is looking at AUC, which is what is done in Table A1.
It also means that the proposed LLMs are performing worse than a constant predictor (with respect to accuracy).
Table A1 seems to indicate that models like gemma-3-27b seem to have 0.78 AUC which is among the top AUC, while their accuracy is .33 among the worst one. This should be discussed.

In the appendix:
> Compare to Cognitive Diagnosis which assumes a constant knowledge state, Knowledge Tracing method aim at estimating evolving knowledge states as students answer questions. We similarly review

This sentence seems incomplete. Also it should start with "Compared". Another typo: "[KT] methods".

Minor: in section 1358 the authors put IKT and QIKT in the same paragraph but those models are very different, and QIKT is not meant to be interpretable.

### Questions
In the synthetic dataset, who wrote the ground truth knowledge? Is it curated by a human or yet another LLM?

### Soundness
3

### Presentation
3

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
This paper presents a project on creating a model -- a language bottleneck model, for analyzing students' skills and misconceptions through a combination of two large language models. The first model serves as the encoder, encoding the students' past learning history into natural language explanations, and then uses another LLM to decode it into the original submissions as well as predictions on the next submissions. They performed predictions on a synthetic dataset, as well as two public datasets, and showed that overall the results are better in most base models.

### Strengths
+ Overall this paper is relatively easy to follow and read. The motivation is easy to understand.
+ The design of the model is straightforward. Under the context provided by the authors, the design makes sense.

### Weaknesses
- One major issue of the paper is about evaluation. 
 -> 1) While the design of the model is centered around interpretability (multiple places are showing this, including introduction, discussions, and the design considerations, etc.), there is no systematic evaluation of this perspective. While the case studies give a peek at the performance, and it is looking good, it still lacks formal evaluations. In some cases, it might be a case that some natural language summarization could be incorrect or not interpretable, but still decoded correctly. A more careful look is needed.
 -> 2) The result of accuracy, if interpreted correctly, seems to be similar to CDM, even with large language models. Although it saves the total number of seen questions, it is still not a major improvement motivated in this work.
Overall, the work is interesting, and the results may use better presentations to be more relevant to the motivations of the work.
- There are also some minor issues with the narratives of the work, listed in the questions below.

### Questions
- Line 12: The goal of KT, though, is still to estimate students' skills.
- Line 15: This only applies to DKT models. For BKT models, they have clear interpretability.
- Line 41: Again, the one you cited from Corbett is BKT, and it does not have vector representations for knowledge -- it's just a set of statuses representing whether students know certain knowledge or not. It is quite interpretable.
- Line 50: The following statement should be fine even outside of the CD domain.
- Line 170: This is not convincing -- since the decoder part can produce good results already, then why wouldn't we make it better?
- Line 177: At this point, readers start to wonder what exactly the knowledge state will look like in natural language. For some tasks, like open-ended problems, it is just hard to reconstruct the exact same answers, no matter how good the LLM is.
- Line 306: Preprocessing of datasets should not be in Appendix as it is necessary for replication. It is an integral part for a research paper to be validated.
- Line 469: So the quality of the summary should be systematically evaluated.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Language Bottleneck Models (LBMs) for representing student knowledge states in natural language. The encoder LLM produces concise textual summaries of a student’s knowledge, and a decoder LLM reconstructs past responses and predicts future performance solely from that text. This transforms the traditionally quantitative representations used in cognitive diagnosis and knowledge tracing into interpretable textual summaries. The authors evaluate LBMs on synthetic, Eedi, and XES3G5M datasets, showing that zero-shot LBMs achieve performance comparable to state-of-the-art KT/CD models while offering interpretability and qualitative insight into misconceptions. The paper further explores reinforcement learning to refine summaries and demonstrates steerability through prompt or reward shaping

### Strengths
1. The encoder–decoder LLM design is interesting, especially that the decoder reconstructs past responses and predicts future performance solely from the textual bottleneck.
2. The work demonstrates a strong theoretical framing that connects cognitive diagnosis, knowledge tracing, and language bottlenecks in a coherent way, supported by extensive experiments across diverse datasets.

### Weaknesses
1. The knowledge state representation defined by coarse textual categories such as Mastered, Fails on, and Misconceptions (Figure 5) may lose important intermediate information—for instance, differences between mastery levels of 0.6 and 0.7. Moreover, extracting precise concept-level interpretations from free-form text can be ambiguous due to synonymy and linguistic variability.
2. The compared CD and KT baselines do not include recent LLM-based variants[1,2,3], which may lead to an incomplete assessment of the proposed method’s relative performance and limit the fairness of the comparison.
3. There has limited analysis on scalability and cost of the two-stage LLM setup in real deployments.
4. Some sections (e.g., 5.4) could include more statistical rigor on variance and significance.
5. Missing user or teacher evaluation of the interpretability claims (qualitative human study).
6. The code is not released, which may lead to difficulties in reproducing the paper

```
[1] Wang Z, Zhou J, Chen Q, et al. LLM-KT: Aligning Large Language Models with Knowledge Tracing using a Plug-and-Play Instruction[J]. arXiv preprint arXiv:2502.02945, 2025.
[2] Dong Z, Chen J, Wu F. Knowledge is power: Harnessing large language models for enhanced cognitive diagnosis[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(1): 164-172.
[3] Li H, Yu J, Ouyang Y, et al. Explainable few-shot knowledge tracing[J]. Frontiers of Digital Education, 2025, 2(4): 34.
```

### Questions
1. How does the method handle long student histories given LLM context limits?
2. Would joint training of encoder and decoder yield better interpretability or stability?
3. Could LBMs extend to evolving knowledge states (non-static) for longitudinal modeling?
4. What are the compute and cost implications compared to KT baselines for large-scale deployments?

### Soundness
2

### Presentation
2

### Contribution
2
