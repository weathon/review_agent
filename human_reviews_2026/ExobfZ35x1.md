# Uncertainty Quantification via Reasoning–Explanation Symmetry in LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Uncertainty quantification (UQ) for large language model (LLM) outputs has attracted increasing attention, as it is crucial for hallucination detection and selective generation; however, existing semantic methods based on cross-output consistency require multiple sampling and thus incur additional cost. We hypothesize that, for reliable answers, LLMs exhibit consistent forward reasoning and backward explanation paths. Building on this, we propose Reasoning--Explanation Symmetry (RES) to quantify uncertainty from the answer itself without multiple sampling: for each question, we first generate structured reasoning and an answer, then condition on the answer to generate a structured explanation; bidirectional natural language inference (NLI) assesses the semantic entailment between the two to construct a symmetry score. RES yields more accurate estimates with small sampling counts and offers stronger interpretability. We evaluate RES on six datasets for both uncertainty quantification and best-answer selection, and the results demonstrate significant advantages on complex reasoning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to use the Reasoning–Explanation Symmetry scores to quantify the uncertainty of an LLM from its answers and reasoning chains. Specifically, the authors sampled three answers with reasoning chains from the language model to a question, and then prompt the language model to explain the answers given the questions. The authors use the RoBERTa model pre-trained on the NLI task to calculate the entailment probabilities of the reasoning–explanation pairs and aggregate them as an evaluation of the confidence of LLM to its answer.

### Strengths
The paper is straightforward and easy-to-follow. The proposed idea of using Reasoning–Explanation Symmetry scores to evaluate the uncertainty/confidence of a language model is interesting and could be inspiring to some of the researchers in the community. Overall, the proposed method, experiment setup and result analysis make sense. The proposed idea does introduce certain performance gain on some datasets.

### Weaknesses
The motivation of this paper is not well-defined. The authors claims "uncertainty serves as a signal for hallucination detection and selective generation" but I personally do not understand how AUROC and TOP1-AUC are good metrics for hallucination detection and selective generation. I'd appreciate some explanation here.

The authors also claim the proposed method can "quantify uncertainty without multiple sampling". But from the following description it looks like the method sampled 2*k (=6) responses almost independently from the language model after all. I don't understand why this is not "multiple sampling".

In lines 034 the authors claim "studies have shown that these measures (token probs) correlate weakly with generation quality". This is not the case. Also I did not find where the reference (Chen et al., 2023) made or supported this claim. The authors may checkout some recent works [1--3] discussing related methods.

Regarding the methodology, I think it is more like a minor tweak to directly generative 2k=6 answers and calculate the pair-wise entailment probabilities. The authors can add related ablation studies to support the effectiveness of the "explanation" step.

Regarding the selection of datasets, I wonder why the authors choose a subset instead of using the entire BBH dataset. Also some benchmarks from other domains such as MATH or GSM8k can be adopted in the experiments.

Figure 4b indicates that the latency of RES is at least 2X of LN-PE but Figure 4a shows the opposite. Considering that the case in Table 5 shows longer explanation paths I would appreciate some explain on this difference.

Table 2 shows that the AUROC of some methods falls below 0.5 which indicates that incorrect answers are associated with higher confidence. This deviates from the results in their original papers and recent survey papers [3]. Please explain.

Misssing reference in line 260.

[1] Zhang, Tunyu, et al. "Token-Level Uncertainty Estimation for Large Language Model Reasoning." arXiv preprint arXiv:2505.11737 (2025).  
[2] Li, Yinghao, et al. "Language model uncertainty quantification with attention chain." arXiv preprint arXiv:2503.19168 (2025).  
[3] Abbasli, Toghrul, et al. "Comparing Uncertainty Measurement and Mitigation Methods for Large Language Models: A Systematic Review." arXiv preprint arXiv:2504.18346 (2025).

### Questions
Please refer to "Weaknesses"

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Reasoning-Explanation Symmetry (RES) to quantify the consistency of LLM answers for hallucination detection.
For a given question, first a structured reasoning is generated, followed by a final answer.
Then given the question and the answer, a structured explanation is generated.
An NLI model is this used to compare reasoning and explanation, computing a consistency score based on the predictions of the NLI model.
The resulting score is utilized for hallucination detection and best-answer selection tasks.
RES is efficient at relatively small sampling counts and the extra generated reasoning and explanation offers interpretability in the generated answer.

### Strengths
- The approach of structured reasoning and explanations to cope with the limitations of the NLI model is very practical also beyond this work.
- A wide variety of datasets were used for the evaluation.
- Having reasoning and explanation as additional tool for interpretability warrants the additional computational costs incured by using this method for hallucination detection.
- The treatment of prior work (except the very closely related BSDetector which should be discussed in more detail) is extensive and the selection of baselines is reasonable.
- Interesting ablation studies, especially the role of structured prompting and NLI model alternatives.

### Weaknesses
- Unfortunately, I have to disagree on a fundamental premise of the paper, namely that consistency or the lack thereof is the same as uncertainty in the prediction. Let me give a counterexample to this intuition. For a given input the model could be uncertain about the reasoning as there are two possible ways to answer the question, but the answer itself is certain given one of the two reasoning paths. Thus for a given answer, the model is also very certain about the explanation, indicating high certainty overall. However, would one generate a second time, chances could be 50:50 that the other reasoning would be applied, leading to a very different answer, yet again a consistent explanation would be generated for this answer. In essence, the suggested relationship may only hold if there is only a single true answer to the question. However, I still find the premise of the approach convincing to generally assess the reliability of LLM answers and detecting hallucinations, yet I would prefer if it would be marketed as such. Alternatively, a more convincing argument why consistency and uncertainty should be the same or conditions under which this may hold would be necessary.
- The idea of the paper is very close to the BSDetector algorithm in using the NLI model for the final score. This would warrant a more extensive treatment and delineation from RES in the related work section.
- One of the main selling points in the abstract is that RES does not do multiple sampling (line 16/17), which is not how it is evaluated in the experiments where k=3 samples are drawn, later on there is also an ablation with a minimum of k=2. The method is never evaluated on a single sample k=1. Note that also baseline methods such as LN-PE or SAR (TokenSAR) could be evaluated on just a single sample. I would tone down this argument or remove it completely. What the results in the ablation study in section 5.3 show is, that is more effectively utilizes samples (k), yet one has to bear in mind that samples for RES are more expensive than for the baseline methods.
- The actual score calculation is handed off to a secondary BERT model trained on the NLI task, thus is completely reliant on the fact that the NLI model generalizes to the particular domain that RES should be used in. This limitation should at least be acknowledged to a sufficient extent, as done e.g. in the BSDetector paper.
- The weakness stated in line 45-49, namely that semantic metrics are only useful on a question rather than a answer is not entirely true. The work of Qiu and Mikkulainen (2024) tackles excactly that issue.
- Using ROUGE-L as a correctness metric has recently been shown to be problematic by various independent papers on the matter (Santilli et al., Ielanskyi et al. and Janiak et al.). Their findings suggest that using LLM-as-a-judge approaches, best with different prompting strategies and aggregation, would lead to much more dependable results.
- Furthermore, PRR (Malinin & Gales) would be preferable as an evaluation metric, complimenting the AUROC used so for in the results. PRR is usually preferable, as it is less impacted by the correctness of predictions by the LLM.

Remarks:
- While reading, I found it unintuitive that the resulting score is higher for more reliable (certain) samples, which is opposite of what is standard for a general uncertainty score. For example line 254 reads strange due to that, where samples with highest UQ score are selected for best-choice accuracy. For hasty readers, this might be irritating.
- Is the SoRE method in Figure 3 an earlier naming of RES? I couldn't find SoRE referenced anywhere else and RES results would be missing otherwise.
- In line 39, the SDLG method is called "Semantically Diverse *Language* Generation"

---
Qiu, Mikkulainen (2024) Semantic Density: Uncertainty Quantification for Large Language Models through Confidence Measurement in Semantic Space, NeurIPS

Santilli et al. (2025) Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results, ACL

Ielanskyi et al. (2025) Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation, ArXiv 

Janiak et al. (2025) The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs, EMNLP

Malinin, Gales (2021) Uncertainty Estimation in Autoregressive Structured Prediction, ICLR

### Questions
- I would be curious about using - mean(c_{i,fwd}, c_{i,bwd}) as scoring function, would this be beneficial over the *penalized* score?
- Regarding the case study, how can the confidence scores in Table 5 be so high? This would imply that every token of the answer is predicted with (near) certainty - a probability very close to 1 - or how is the confidence calculated for this example?
- Interestingly, the performance of RES is weakest on the TriviaQA and CoQA datasets, which are pretty standard in prior work on uncertainty estimation for LLMs. Do you have any insights into why the method struggled on those datasets or is it rather that baselines were already tuned well for them?
- Recently, Aichberger et al. (2024) showed that the likelihood of the greedily decoded sequence can be a very strong measure of uncertainty. Due to it being basically for free, I would like to know how well it performs as uncertainty measure on the considered tasks.
- In Figure 3, what temperature was used in (a) and what k in (b)? I would guess t=0.7 and k=3 from the description at the beginning of the experimental section, but it should be stated when describing this experiment.

---
Aichberger, Schweighofer, Hochreiter (2024) Rethinking Uncertainty Estimation in Natural Language Generation, ArXiv

### Soundness
2

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
This paper tackles the cost and instability of semantic uncertainty estimation in LLMs that rely on many sampled outputs. It proposes Reasoning–Explanation Symmetry (RES): for a question, the model first generates structured reasoning and an answer, then, conditioned on the answer, generates a structured explanation; a bidirectional NLI judge computes mutual entailment across aligned sections to yield a symmetry score as an uncertainty proxy. Experiments demonstrate that RES improves AUROC for uncertainty quantification and boosts best-answer selection.

### Strengths
- Firstly, I found the overall presentation, including organization and writing of the paper, making it relatively easy to follow their narrative.
- The authors articulate the core problem they're addressing and the motivation to reduce the cost and instability of multi-sample semantic UQ is straightforward. The methodology is intuitive, leveraging a concise concept of internal consistency that makes sense for identifying specific types of errors.
- The experimental evaluation covers a diverse set of tasks and models, which lends some support to the general applicability of their approach. The efficiency gains reported, particularly the need for only a few samples to achieve their results, are a practical advantage that stands out.

### Weaknesses
- My main concern is that RES does not address epistemic uncertainty: answer-conditioned explanations can rationalize confident but incorrect outputs, yielding high symmetry and thus underestimating uncertainty when the model simply lacks knowledge. In other words, RES measures process consistency rather than factuality; however, on knowledge-heavy or retrieval-dependent tasks, RES might fail without a fact-checking component.
- The model suite is too small for a reasoning-centric paper: results are confined to ≤8B open models (Llama and Qwen 8B) and a mini closed model (GPT-4o-mini), with no larger size or modern reasoning specialists (contemporary 30–70B or reasoning-tuned variants like QwQ-32B and Deepseek-R1), limiting external validity.
- Scalability to long chains remains untested, as the NLI judge's length limits and section-wise segmentation are potential bottlenecks. Additionally, there is a lack of sensitivity studies on the impact of reasoning/explanation token length on performance.
- In this paper, RES scores pairwise, section-aligned entailment, and then aggregates (by min/mean/penalized). This symmetry aggregation is coarse and overlooks cross-sectional dependencies, allowing global contradictions or step-level failures to slip through. For example, answer-conditioned explanations tend to rationalize the given answer, so R and E can be mutually entailing even if both are self-consistent but wrong relative to the original premise, which RSE (a per-section average) won't penalize.
- The related work discussion didn't cover recent reasoning or explanations focused on UQ directions [1-5], which are directly relevant to this framing.

[1] Mei et.al, Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know, arXiv 2025.

[2] Zhang et.al, CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought, ACL 2025.

[3] Becker et.al, Cycles of thought: Measuring llm confidence through stable explanations, arXiv 2024.

[4] Da et.al, Understanding the uncertainty of llm explanations: A perspective based on reasoning topology, COLM 2025.

[5] Mo et.al, Tree of uncertain thoughts reasoning for large language models, ICASSP 2024.

### Questions
Based on the weaknesses I've outlined, here are my main questions for the authors:

- How do the authors distinguish RES’s signal from epistemic uncertainty? Or can the authors give me some evidence that answer-conditioned rationalization doesn’t yield low-uncertainty scores on “confident but wrong” cases?

- Can the method generalize to larger and reasoning-tuned models (30–70B class and reasoning variants), and is there any evidence of size-sensitivity?

- What happens to RES when reasoning/explanation chains get long—do NLI truncation and section segmentation degrade performance, and can the authors provide token-length sensitivity curves?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces RES - Reasoning-Explanation Symmetry as an uncertainty estimation method for natural language generation.
The method roughly consists of generating Evidence and Premises for a Question Answer pair and computing the symmetry score.
The several variations of this score can further be used for ucnertainty estimation.
The authors perform several ablations of their method and compare it to several prominent uncertainty estimation algorithms for autoregressive generation.

### Strengths
The method is novel, although highly heuristic.
The evaluation encompasses several multi-step datasets which is a good practice.
Authors perform ablations on the components of their method.

### Weaknesses
1. Minor:
    1. Line 038: Semantically Diverse Likelihood Generation (SDLG) - the name is disambiguated incorrectly, it's language, not likelihood.
    2. Evaluation includes TriviaQA and CoQA, I do not see how the method would be applicable to these given their short answer lengths.
    3. Answer length not reported for the evaluation suite. 
    4. Reasoning nature of GPT-4o-mini is not treated. The model produces CoT which it does not disclose.
2. Major:
    1. [1][2] say that ROUGE metrics can be quite unreliable for assessing the selective prediction performance of uncertainty estimation algorithms for NLG. Would be nice to show comparison with judge LMs or and average thereof. 
    2. PE without length normalization can make a significant difference [3], yet not included in the evaluation.
    3. The proposed pipeline is not well motivated theoretically and has little in the way of integrating it with the theory of preceding uncertainty estimation literature.


### References
1. Ielanskyi, M., Schweighofer, K., Aichberger, L. & Hochreiter, S. Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation. Preprint at https://doi.org/10.48550/arXiv.2510.02279 (2025).
2. Santilli, A. et al. Revisiting uncertainty quantification evaluation in language models: Spurious interactions with response length bias results. in Proceedings of the 63rd annual meeting of the association for computational linguistics (volume 2: Short papers) (eds Che, W., Nabende, J., Shutova, E. & Pilehvar, M. T.) 743–759 (Association for Computational Linguistics, Vienna, Austria, 2025). doi:10.18653/v1/2025.acl-short.60.
3. Aichberger, L., Schweighofer, K. & Hochreiter, S. Rethinking Uncertainty Estimation in Natural Language Generation. Preprint at https://doi.org/10.48550/arXiv.2412.15176 (2024).

### Questions
1. Why does this method work? What theoretical uncertainty quantity if your method closest to? 
2. The previous question but on a more applied level: could you compute correlation plots between the RES and other uncertainty estimates for each dataset, could give some interesting insights. 
3. How does your method fit in the framework of aleatoric and epistemic uncertainty?
4. What is the computational overhead of the additional generations (e.g. Premises and Evidence)?

### Soundness
2

### Presentation
2

### Contribution
2
