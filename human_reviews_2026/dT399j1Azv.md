# Reward Models Inherit Value Biases from Pretraining

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 8

## Abstract
Reward models (RMs) are central to aligning large language models (LLMs) with human values but have received less attention than pretrained and post-trained LLMs themselves. Because RMs are initialized from LLMs, they inherit representations that shape their behavior, but the nature and extent of this influence remain understudied. In a comprehensive study of 10 leading open-weight RMs using validated psycholinguistic corpora, we show that RMs exhibit significant differences along multiple dimensions of human value as a function of their base model. Using the "Big Two" psychological axes, we show a robust preference of Llama RMs for "agency" and a corresponding robust preference of Gemma RMs for "communion." This phenomenon holds even when the preference data and finetuning process are identical, and we trace it back to the logits of the respective instruction-tuned and pretrained models. These log-probability differences themselves can be formulated as an implicit RM; we derive usable implicit reward scores and show that they exhibit the very same agency/communion difference. We run experiments training RMs with ablations for preference data source and quantity, which demonstrate that this effect is not only repeatable but surprisingly durable. Despite RMs being designed to represent human preferences, our evidence shows that their outputs are influenced by the pretrained LLMs on which they are based. This work underscores the importance of safety and alignment efforts at the pretraining stage, and makes clear that open-source developers' choice of base model is as much a consideration of values as of performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates whether reward models (RMs) inherit value biases from their pretrained language model backbones. The authors conduct a systematic analysis across 10 open-weight RMs based on either Llama or Gemma architectures and show that these models consistently differ along established psycholinguistic dimensions of human value—particularly the "Big Two" axes of agency and communion. They demonstrate that Llama-based RMs exhibit stronger preferences for agency-related words (e.g., freedom, ability, success), whereas Gemma-based RMs prefer communion-related words (e.g., love, friendship, care). Importantly, the paper traces these differences back to the log-probability structure of the base models themselves and formulates an “implicit reward model” defined by log-probability deltas between two LLMs that captures such differences. Experiments with RM training further show that these biases persist even after training RMs on large preference datasets.

### Strengths
- **Relevance and potential impact**: The work tackles an important and underexplored problem in the alignment literature—understanding how pretraining choices influence downstream reward models. Even though the result may not be entirely surprising, demonstrating it empirically and rigorously is valuable for the community.

- **Methodological rigor**: The experiments are well-designed and carefully controlled, with sound statistical analyses (e.g., mixed-effects models, permutation tests, Bonferroni correction). The use of validated psycholinguistic corpora lends interpretability and psychological grounding to the findings.

- **Comprehensive empirical validation**: The authors examine both real-world open-weight RMs and controlled in-house replications, providing converging evidence for the bias inheritance effect.

### Weaknesses
- **Potential domain mismatch**: The preference datasets used for RM training (e.g., HelpSteer, UltraFeedback, HH-RLHF, Argilla-Math) are focused on instruction-following, helpfulness, honesty, and truthfulness, not the kinds of moral or social values represented in the psycholinguistic test sets. Thus, it is not clear whether the persistence of biases reflects insufficient training data volume or an out-of-distribution (OOD) evaluation setting. 
- **Formatting issue**: I believe the font used in the submission violates the ICLR template. Please revise this in the updated pdf.

### Questions
1. Could the observed persistence of biases be primarily due to the psycholinguistic test corpora being OOD relative to the RM training data? For example, do preference gaps narrow more substantially on in-distribution prompts aligned with RM training datasets (e.g., helpfulness, truthfulness, safety)?

---

Overall, I find this an impactful and a well-executed study. I am willing to increase my score if the authors address my question regarding the OOD nature of the test sets relative to the RM training data.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates how reward models (RMs) inherit value biases from the pretrained LLMs they are initialized from. Using validated psycholinguistic corpora ("Big Two" and MFD2), the paper finds Llama-based RMs consistently show a preference for "agency" values (e.g., freedom, success, ability), while Gemma-based RMs show a preference for "communion" values (e.g., love, family, friendship). This bias is traced directly back to the log probabilities of the instruction-tuned and even the original pre-trained models. The paper also shows that while the bias in RMs can be mitigated with sufficient preference data, it does not fully disappear.

### Strengths
**S1.** The paper clearly traces moral-value biases (agency vs. communion) from the outputs of trained RMs back to the log-probabilities of the base pre-trained models, which provides a clear takeaway that the choice of base model for RM training is also a critical decision that will have downstream value implications.

**S2.** The paper evaluates multiple open-weight RMs based on psycholinguistic validation through controlled ablations on data and base model selections.

### Weaknesses
**W1.** The central claim that reward models (RMs) inherit biases from their base pretrained LLMs already feels intuitive and largely expected, given prior research demonstrating bias propagation across fine-tuning and alignment stages [1, 2]. Therefore, this makes the contribution of the paper primarily observational since it does not provide mechanistic interpretability, analysis of latent representations, or deeper causal insight into why such biases emerge. Furthermore, it doesn't offer any actionable mitigation strategy beyond suggesting careful base model selection.

**W2.** The analysis is limited to a narrow experimental scope, focusing only on two base model families (Llama and Gemma; three if we consider Qwen in the Appendix) with two parameter scales (2B and 3B with LoRA), and primarily the agency/communion axis from the Big-Two framework despite referencing MFD2.

## References

[1] Fulay, S., Brannon, W., Mohanty, S., Overney, C., Poole-Dayan, E., Roy, D., & Kabbara, J. (2024). On the relationship between truth and political bias in language models. arXiv preprint arXiv:2409.05283.

[2] Xiao, J., Li, Z., Xie, X., Getzen, E., Fang, C., Long, Q., & Su, W. J. (2025). On the algorithmic bias of aligning large language models with rlhf: Preference collapse and matching regularization. Journal of the American Statistical Association, (just-accepted), 1-21.

### Questions
In terms of experimentation, it may be interesting to model scaling law behavior rather than just observe it, such as how increasing preference data or compute (model size) affects the persistence of inherited biases, along with proposing a concrete methodology for mitigating such biases at larger scales.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents case studies that investigate how reward models (RMs) inherit systematic value biases from the base LLMs on which these RMs are instantiated. The authors examined the preferences on a token level along dimensions characterised by two psycholinguistic corpora (Big Two - Agency vs. Communion, and Moral Foundations Dictionary - 5 further aspects). By analysing token preference differences on 10 open-source RMs, the authors validate that Gemma- and Llama-based RMs have systematic differences. Further investigations on the Big Two corpora shows the differences can be traced back to the token probabilities of pre-trained Gemma and Llama models. Experiments tracking Big Two token preferences during RM training show that the inherited biases persist.

These insightful findings draw attention to the choice of base LLMs for RM training and thus the entire LLM pipelines, urging the research community to reflect on current standard practices.

### Strengths
- This paper offers some fundamental insights for the research community on choosing base LLMs for RM training, which is underexplored. A focus shift from pure performance metrics to more fine-grained details on value biases is much needed these days.
- The investigations done in the paper make sense and are quite novel, providing solid evidence of the inheritance traces of value biases. Experiments also cover diverse aspects.
- Clarity is excellent - clear motivation, adequate and in-depth discussions, good coverage of related work, and discussions on limitations.

### Weaknesses
- The RMs used in Sections 3 and 4 are quite small (2B and 3B), somewhat limiting the significance of results and the validity of relevant claims.
- Sections 3 and 4 focus on a binary value distinction between "Agency" and "Communion". This seems a bit arbitrary. It is also obvious that different types of LLMs (Llama vs. Gemma) would have systematic differences. I would assume that if I randomly choose two common value aspects to repeat the same investigations, I would observe different preferences anyway. Could the authors comment on this choice?
- I have the impression that the findings this paper presents are an instantiation of a common phenomenon, model multiplicity, that we can obtain machine learning models that perform similarly but differ in their internals for the same task, in the realm of reward modelling and LLM training. How is your finding different from something like, "for a tabular classification task predicting loan default, I find one neural network prefers feature AGE more, and another neural network prefers feature LOAN AMOUNT more"?

### Questions
See weaknesses.
- One additional comment: there are also RMs that are trained to explicitly predict scores along certain axes (helpfulness, verbosity, coherence, etc., see datasets like HelpSteer) in a multi-regression style. These prediction signals could potentially be a better playground to perform your investigations.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors studied the problem of how base model induces inherent bias in the reward model that was fine tuned from them, using methods from psycholinguistics. Using two existing psycholinguistics datasets from domain experts, the authors tested RM from several based models and concluded they indeed induce inherent biases. They then traced the source of the biases and find log probability can explain these biases from the base model.

### Strengths
- The authors took an interdisciplinary approach in viewing the question --- we should indeed borrow existing domain knowledge from disciplines that studies human values. 
- The statistical tests are careful in e.g., FDR control
- The framing of model difference as reward difference is an interesting view point

### Weaknesses
Exhaustive token search might itself have limitations, I am not sure how the prompting scheme have an influence in this process.

### Questions
## Statistical analysis
- The difference can be significant but not large, and a small p-value can be due to large sample size. I am not very sure what is the best way to interpret the differences the authors reported in Fig.1, they do not seem to be large in some cases. Especially since the authors showed error bar using standard errors rather than standard deviations. I have no doubt the differences are *statistically significant*, but are they really *meaningful*? E.g., median rank of 1000 and 1001 might not be that meaningful even if the difference is significant because the estimation error is small.

-  Fig.2 provides a bit more insight since the density plot. 

## Implicit reward 

- Is it fair to say that the author would also view KL between two models to be reward that can make model A to B if trained with RL? 

## Prompting
- Does prompting have an influence in vocab search?

### Soundness
3

### Presentation
3

### Contribution
3
