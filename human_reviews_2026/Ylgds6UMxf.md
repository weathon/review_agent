# Chat-CBM: Towards Interactive Concept Bottleneck Models with Frozen Large Language Models

- Decision: Reject
- Scores: 2, 6, 6, 4, 2

## Abstract
Concept Bottleneck Models (CBMs) provide inherent interpretability by first predicting a set of human-understandable concepts and then mapping them to labels through a simple classifier. While users can intervene in the concept space to improve predictions, traditional CBMs typically employ a fixed linear classifier over concept scores, which restricts interventions to manual value adjustments and prevents the incorporation of new concepts or domain knowledge at test time. These limitations are particularly severe in unsupervised CBMs, where concept activations are often noisy and densely activated, making user interventions ineffective. We introduce Chat-CBM, which replaces score-based classifiers with a language-based classifier that reasons directly over concept semantics. By grounding prediction in the semantic space of concepts, Chat-CBM preserves the interpretability of CBMs while enabling richer and more intuitive interventions, such as concept correction, addition or removal of concepts, incorporation of external knowledge, and high-level reasoning guidance. Leveraging the language understanding and few-shot capabilities of frozen large language models, Chat-CBM extends the intervention interface of CBMs beyond numerical editing and remains effective even in unsupervised settings. Experiments on nine datasets demonstrate that Chat-CBM achieves higher predictive performance and substantially improves user interactivity while maintaining the concept-based interpretability of CBMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Chat-CBM proposes an integration of LLMs into the CBM framework by adopting a semantic bottleneck (i.e. textual description of the concepts) which is consequently passed through the LLM. As such, the LLM replaces the target classifier. This framework enables a lot of flexibility for interventions, where they are not bound to the concept set of the concept encoder, and can interact with a user directly with language, which is a benefit. The LLM is queried with two important additions: (i) Class Concept Semantics Prior Integration, which captures the class-level concept structure that is present in e.g. CUB, and thereby gives the LLM context about the data structure, and (ii) In-context learning, which gives the LLM data examples (incl. predicted concepts) for each candidate class, thereby aiding the model in understanding potential shifts of the concept encoder.
Experiments are very extensive for datasets with ground-truth concepts and discovered concepts.

### Strengths
Since the era of LLMs, multiple works have tried to leverage them in combination with CBMs. The direct usage of LLMs as target predictor is neat, as it allows for a lot of flexibility at the intervention stage. As such, I see the main contributions of this work to be the smart setup of the LLM prompting for giving meaningful context about the task, as well as the design choice of using LLMs directly, thereby enabling flexible intervenability. One additional strength is that for user interventions, it is more convenient for users to intervene in language space rather than numerical concept space, thereby reducing the adoption barrier of Chat-CBM.

The paper is nicely written, clearly explained, formalized, and outlined. The experiments are impressively vast, spanning supervised and unsupervised datasets, interventions, and ablations.

### Weaknesses
To summarize my main critique in one sentence: I see this work as mainly adapting LLM terminology to CBM terminology.

The usage of LLMs and in-context learning for classification is common[1]. Similarly, the text interventions are the same thing as having a normal conversation with a chatbot; a direct consequence of using LLMs.
As I see it, the main novelty of this work is adapting few-shot LLMs to fit a CBM framework by smart prompt design that incorporates dataset priors. While an art in itself, I do not think great prompt engineering warrants acceptance.

Beyond the aforementioned main point, I do have additional critiques:
* The common reason for adopting a linear classification head is its interpretability, since the mapping from concepts to target are clear. By replacing this layer with a LLM, the reasoning becomes opaque. While a linear layer indicates directly that the concepts are being used for prediction, which is the desired behavior, the few shot LLM seems to rather perform a prototype-based prediction from the in-context examples, but it is unclear if that is being done and what contributes how (note that I do not think an LLM's reasoning output is interpretability[2]).
* I do not fully agree with some of the conclusions drawn from the results. (i) For accuracy on concept-supervised datasets, ChatCBM is compared to baselines that employ a linear head. As such, I wonder if the performance gain from ChatCBM is not merely due to the introduction of a non-linear classifier that can better model the concept--class mapping. An effect that could also be achieved by using an MLP instead of a linear layer. Also, evaluating a 70B LLM versus a linear layer does not seem like a fair comparison for performance.
(ii) For accuracy on concept-unsupervised datasets, the authors argue that a fair comparison would be to compare 2-shot ChatCBM with 2shot baselines. I disagree. To me, it the 2-shot limitation of ChatCBM is an inherent downside of the method's design. As such, the other methods do not need to be artificially worsened, if they do not have this limitation. Comparing 2shot ChatCBM to allshot baselines leads to worse performance of ChatCBM. (iii) Intervention on unsupervised datasets: The assistant LLM seems the ground truth class label, which is an unfair advantage. If a user had access to the class label, there would not be any reason for intervening on the model. An alternative setup might be to provide the assistant LLM the ground truth knowledge for the concepts. Alternatively, standard intervention policies such as random / uncertainty-based could be employed, where no LLM is required. (iv) Similarly, for the wikipedia example, using the class's description, rather than what is visible from the image, is not a realistic intervention setting.
* It is stated that the unsupervised methods struggle from a noisy dataset, which I agree with. However, given that ChatCBM also bases on these methods' concept annotations for their approach, they naturally inherit all these issues instead of solving them. 
* No code is provided, thereby limiting reproducibility.
* The semantic bottleneck of ChatCBM does not take into account the predicted concept uncertainty, thereby omitting valuable information for its considerations. Also, absent concepts (i.e. c=0) are omitted instead of being added to the prompt as being absent.
* The flexibility to add new concepts for interventions at test time is not new. For example, [3] targets exactly this problem.
* It is unclear whether there might be some test set leakage due to the LLM likely being trained on it.


[1] Brown, Tom, et al. "Language models are few-shot learners." Advances in neural information processing systems 33 (2020): 1877-1901.

[2] Barez, Fazl, et al. "Chain-of-thought is not explainability." Preprint, alphaXiv (2025): v1.

[3] Laguna, Sonia, et al. "Beyond concept bottleneck models: How to make black boxes intervenable?." Advances in neural information processing systems 37 (2024): 85006-85044.

### Questions
I invite the authors to respond to any of the previously stated weaknesses and potential misunderstandings. In addition, I have a few minor questions

* Will reproducible code be provided?
* In App. A.1, the SCBM model is stated as a baseline, but it does not appear in the results. Can the authors add it to the results or explain why they are not showing it?
* Contrary to the statement on p.4 "The top-N choice (of unsupervised concepts) is discussed in appendix A.1", it is not discussed (or I can't find it). Would the authors mind adding this section?
* What intervention policy (i.e. choice of concept to intervene on) is currently used for Fig. 4? Also, I was thinking that maybe an additional benefit of having an LLM classifier would be that the LLM could decide on which concept to intervene on next, i.e. query the user based on information that would help it make a better decision.
* I did not understand the explanation of what D-GT means in Table 3. Can the authors try to explain it in more detail? Thanks!

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Chat-CBM, a novel approach to Concept Bottleneck Models that replaces traditional score-based classifiers with language-based classifiers using frozen LLMs. The key innovation lies in reasoning directly over concept semantics rather than numerical activations, which enables richer interventions including concept correction, addition/removal, and high-level reasoning guidance. The authors evaluate their approach on nine datasets, demonstrating improvements in both supervised and unsupervised CBM settings while maintaining interpretability.

### Strengths
1. Well-motivated problem: Clearly articulates limitations of existing CBMs (restricted interventions, struggles with unsupervised settings) and how the proposed approach addresses them.

2. Comprehensive evaluation: Nine datasets spanning supervised/unsupervised settings, multiple baseline comparisons, and extensive ablations demonstrate thoroughness.

3. Practical intervention capabilities: Figure 3 and Figure 5 convincingly show that conversational interventions work in practice and can substantially improve performance.

4. Honest limitations discussion: Appendix B provides an assessment of computational costs, incomplete concept challenges, and safety concerns with LLM knowledge.

5. Strong performance on key metrics: The intervention curves (Figure 4-5) show Chat-CBM can approach or exceed fully-supervised baselines with just a few interventions. Preview TeX is supported

### Weaknesses
1. The paper's central value proposition is enabling "richer and more intuitive interventions" through natural language, yet all experimental validation uses automated procedures without any human participants. Section 4.2's intervention experiments (Figure 5, lines 374-398) rely on an assistant LLM with ground-truth labels rather than actual users, making it impossible to verify whether real humans find these interactions genuinely intuitive or efficient. The authors should conduct even a small user study with 15-20 participants comparing Chat-CBM against traditional CBM interventions, measuring metrics like time-to-correct-prediction and user satisfaction to validate their core claims about interaction quality.

2. The paper experiments exclusively with large LLMs (8B-72B parameters in Table 5) without exploring whether compact models like 1-3B parameter LLMs could achieve acceptable performance with feasible resource requirements. The authors should add experiments establishing a performance-efficiency trade-off curve with smaller models and provide concrete deployment recommendations, or clearly acknowledge that the approach is currently limited to research settings rather than practical interactive systems.

3. The paper demonstrates Chat-CBM works when concept predictions are accurate (93-98% in Table 1) but provides no systematic analysis of robustness to concept errors or

minimum quality thresholds needed for effectiveness. The paper only shows cherry-picked success cases in Figures 3 and 7-11 without any failure analysis, leaving practitioners unable to assess when they can trust the model's reasoning. The authors should add experiments with artificially corrupted concepts to establish robustness boundaries, include a failure analysis subsection with concrete error examples, and provide guidelines like minimum concept accuracy thresholds so users understand the approach's reliability limits.

4. The experimental design gives Chat-CBM advantages not available to baselines, most notably using validation set statistics to construct class priors θ in unsupervised settings (Section A.4, lines 802-829), which is only revealed in the appendix rather than transparently discussed in the main paper. Additionally, the paper mentions XBM and CB-LLM as conceptually similar work using language models for interpretable classification (Section 2.2, lines 126-137) but provides no empirical comparison. The authors should include direct comparisons with XBM and CB-LLM on several datasets, add ablations where baselines also receive class prior knowledge to ensure fair comparison, and explicitly state in the main paper what information each method accesses.

### Questions
1. Concept selection: How sensitive is performance to the choice of top-N concepts (you use top-10 for unsupervised)? Have you experimented with learned concept selection based on uncertainty or relevance?

2. Figure 5 shows sequential interventions, but does the model maintain coherent reasoning across turns? Can conflicting interventions cause degradation?

3. How does Chat-CBM handle strongly correlated concepts? Do LLMs naturally account for concept dependencies, or does this require explicit modeling?

4. What happens when users introduce concepts at test time that are semantically distant from training concepts? Is there a notion of "valid" interventions?

### Soundness
4

### Presentation
3

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
The authors propose Chat-CBM, a model that leverages a large language model (LLM) as the classifier in the prediction phase. Chat-CBM integrates concept semantics directly into the prediction process: labels are inferred through reasoning over the semantics of concepts rather than through activation scores. This design preserves the interpretable aspect of Concept Bottleneck Models (CBMs) — namely, the concept layer.

This modification also enables richer forms of intervention, since interventions can now be expressed directly in natural language within the LLM. In contrast, a simple linear layer only allows interventions on the numerical scores of each concept. Furthermore, the flexibility of the LLM makes it possible to incorporate prior knowledge about concept semantics.

The authors identify several types of possible interventions: traditional numerical interventions as in standard CBMs, but also natural language interventions that allow users to correct a concept, add or remove concepts, or provide higher-level information.

Experiments are conducted in both supervised settings (three datasets with concept labels) and unsupervised settings (six datasets without concept labels). Two LLMs are evaluated: LLaMA3-Instruct and Qwen2.5-Instruct.

### Strengths
- The idea of replacing the classification layer with an LLM is not highly original, but the experiments and comparisons with existing models are interesting and well executed.
- The authors explore several types of interventions that are not possible with standard CBMs, highlighting the added flexibility of their approach.
- Chat-CBM generally achieves higher accuracy than baseline models in the supervised setting across the three tested datasets, though this improvement is observed mainly in the 1-shot and 2-shot scenarios.
- Concept correction interventions lead to a faster improvement in accuracy compared to the other models evaluated.

### Weaknesses
- The question of whether the overall interpretability of the model is preserved after introducing an LLM is not addressed.
- The authors do not discuss the contribution of the LLM’s prior knowledge to the observed accuracy improvements, in comparison to the contribution of the concept layer itself.
- The use of different LLMs for the supervised and unsupervised settings is not clearly justified, although an ablation study is provided for part of the unsupervised experiments.

### Questions
The paper presents an interesting approach by replacing the classification layer of a CBM with an LLM, which enables new forms of intervention. However, several points would benefit from clarification or deeper discussion:

- The authors should mention explicitly that they limit their experiments to image inputs, even if this does not undermine the generality of the proposed approach. Is the method easily transferable to non-visual modalities?
- What is the relative contribution of the frozen LLM to the prediction accuracy, compared to that of the concept layer?
- To what extent is interpretability preserved at the LLM stage, especially if the LLM is not guaranteed to be faithful in its reasoning?
- In Table 1, what is the difference between the two rows labeled Chat-CBM?
- Why is LLaMA3 used in some experiments and Qwen2.5 in others? The motivation for this choice is not clearly explained, although an ablation study is provided for part of the unsupervised setting.

Minor Comments (no impact on the evaluation score):
- The best-performing model’s score is not always highlighted in bold in the tables.
- Line 776: there seems to be a character encoding issue.

### Soundness
3

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
The paper proposes Chat-CBM, an interactive concept bottleneck framework that replaces the standard score-based label head with a language-based classifier over concept semantics using a frozen LLM. The idea is to preserve CBM interpretability (predictions pass through a concept set) while enabling richer interventions: concept correction, adding/removing concepts, injecting external knowledge, and high-level reasoning guidance. The method targets both supervised and unsupervised CBMs (where concept activations are noisy and dense). Experiments on nine datasets show accuracy gains over conventional CBMs, and interaction studies (including an assistant-LLM that performs automatic interventions) indicate that a few edits can surpass “all-shot” baselines. Ablations explore knowledge injection, ICL shot counts, and LLM size, and a “new concepts at test time” setting (including Wikipedia text) illustrates flexibility.

### Strengths
1 CBMs are interpretable but often brittle, and their linear heads constrain user intervention.  
2 Experiments are comprehensive, including both supervised/unsupervised settings, plus ablations on (i) knowledge priors, (ii) ICL shots, (iii) LLM families/sizes.  
3 The writing is easy to follow

### Weaknesses
1 The central interaction result relies on an assistant LLM that edits concepts. As such, the quality of concepts may be dependent on the LLMs used.  
2 Comparisons focus on classical CBMs (linear heads, activation-based). Missing are stronger non-linear concept heads.  
3 The paper is similar to the paper "Interpreting Pretrained Language Models via Concept Bottlenecks". Can the authors distinguish their work from this one?  
4 The datasets used seem to be simple and not very complex, using CIFAR and ImageNet.

### Questions
How is this paper related to the paper I mentioned in the weakness? Can the work be generalized to more complex tasks like QA?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to integrate LLMs into the CBM architecture. The LLM serves to augment the mapping from concepts to labels through reasoning over concept semantics. Chat-CBM enhances the concept bottleneck layer by providing additional information such as the top-N most likely class given the concept activation, learning examples, and concept priors for relevant classes. Moreover, Chat-CBM allows a novel way to intervene at test time—through conversation with the LLM. End users can correct, add, or remove concepts and provide higher-level strategy guidance. The authors run experiments on both supervised and unsupervised concept datasets.

### Strengths
- Novel and interesting idea
- Variety of benchmarks in experimental section
- Figures are mostly well designed and help readers

### Weaknesses
I don't think directly comparing existing CBM architecture with a linear concept bottleneck layer to Chat-CBM is fair. Chat-CBM is augmented with a significantly more capable model (with billions of parameters). As a result, I am not convinced that the authors have shown that
>  these challenges primarily stem from the reliance on score-based label predictors (line 59-61).

What it's really showing is that a linear classifier mapping concepts to labels is not as performant/expressive as a larger model; this is a pretty trivial result. 

To me, Chat-CBM's value (and novelty) is in test-time intervention -- the ability to intervene with natural language. In addition to concept correction (analogous to numerical intervention in traditional CBMs), the authors state that we can intervene with Concept augmentation/removal or High-level strategy guidance. However, Section 4.2 does not investigate the effects (or gains) of these intervention strategies that is unique to Chat-CBM.

Furthermore, there are a few unsubtantiated claims throughout this paper. For example:
> This process allows users to understand and take control of the decision pipeline (line 298-299).

> The introduced prior knowledge effectively enhances Chat-CBM’s robustness against noise in concept predictions (line 430-431).

In both cases, I am left asking "how exactly?"

Overall, it would help if the authors (1) elaborate on findings/results (i.e., what significance they have) and (2) back claims with evidence.

### Questions
- How would Chat-CBM perform on tasks where there is little public knowledge? I imagine LLMs have a substantial amount of text/corpus from the tasks in the benchmarks.
- How exactly do you intervene on datasets without concept labels? There is no "true" label here.
> employ an assistant LLM that selects concepts to emphasize, remove, or augment based on the conversation history of Chat-CBM, the top-20 predicted concepts, and the ground truth class label.

How does one know the ground truth class label at test time?

- Is $\hat{y} = \textrm{arg} \max_y P(y | D, \theta, \hat{s})$ correct? The LLM does not output numerical probabilities...
- Do the authors think that natural language interventions are more "costly" for the end user at test time? It seems like one may have to put in more thought into formulating their intervention.

### Soundness
3

### Presentation
3

### Contribution
2
