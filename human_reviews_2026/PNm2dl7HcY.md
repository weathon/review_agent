# Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
In the past year, distillation has seen a renewed prominence in large language model (LLM) pretraining,
exemplified by the Llama-3.2 and Gemma model families. While distillation has historically been
shown to improve statistical modeling, its effects on new paradigms key to modern LLMs—such as
test-time scaling and in-context learning—remain underexplored. In this work, we make three main
contributions. First, we show that pretraining with distillation yields models that exhibit remarkably
better test-time scaling. Second, we observe that this benefit comes with a trade-off: distillation
impairs in-context learning capabilities, particularly the one modeled via induction heads. Third, to
demystify these findings, we study distilled pretraining in a sandbox of a bigram model, which helps
us isolate the common principal factor behind our observations. Finally, using these insights, we shed
light on various design choices for pretraining that should help practitioners going forward.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores the impact of distillation-based pretraining on various aspects of model performance, namely in-context learning and test-time scaling. The authors conduct large scale experiments and discuss the pros and cons of distillation in the context of these two aspects. They also propose a theoretical support of their empirical observations, and experiment in a controlled synthetic setup to validate their findings. 

Their conclusions underline that pretraining with distillation leads to models with strong abilities in terms of test-time scaling (as measured with pass@k experiments), but with weaker ICL performance. They propose to fix this by dropping the distillation objective on low-entropy tokens to recover strong copying mechanisms, which partially bridges the gap with standard pretrained counterparts.

### Strengths
This paper is well-written and offers an interesting insight into the interplay between distillation and LLM inference behaviors (context retrieval abilities and test-time scaling).
- **Interesting and novel interplay analysis**: the analysis of the effect of distillation on key strengths of LLM (ICL and test-time scaling) at large scale is novel to the best of my knowledge. It provides relevant insights on the strenghts and weaknesses of distillation beyond pure language modeling performance as measured by perplexity. It is also relatively well grounded in theory and explained by the authors in a convincing way.
- **Sound experimental design**: the experiments are conducted in a controlled setup. The IsoData comparisons disentangles the effects of additional data and of distillation.

### Weaknesses
This paper lacks some crucial details and some experimental results are analyzed in ways that leave room for confounding factors. It also frames terms in confusing ways and lack proper background for some parts
- **Training reproducibility & related conclusions**: The authors only describe the training dataset they used in the Appendix, and do so in a very broad way. The paper is also not mentioning many details such as (in order of relevance): sequence length, architecture, batch sizes, or learning rate warm-up strategy. These details are crucial for reproducibility but also to better analyze the ICL results that rely on long-context experiments. For instance, we currently have no clue whether the Needle-in-the-haystack experiments fit within training context length or not. It is also impossible to tell whether 36% for the 8B teacher model is a strong performance. It would be interesting to see training dynamics and to compare the model performance to pretrained counterparts or at least to similar open models (e.g. OLMo-2) at equivalent training stages.
- **Validity of some of the IsoData conclusions**: The IsoData setup is important to assess the efficiency of distillation in a dataset-equivalent regime between teacher and student training. However, the authors frame this setup in these terms: "student is shown the same amount of data as the teacher", which may not be exactly fair. Although the student only trains once on the 1T dataset, it immediately benefits from an estimate of the full training distribution through the teacher. As a result, at every training step, the model is not only shown the current sample, but also a compressed view of all related samples to come through the distillation loss. It can thus be wondered if working in a more data-constrained setup (e.g. having several training epochs) could affect the outcomes.
- **Missing discussion on the impact of temperature(s)**: A crucial parameter in the distillation setup is the distillation temperature. The authors mention a grid search, but do not report the corresponding results, which matter both for the final performance of the models and for the impact of temperature on ICL performance and the induction heads. As a result, it seems possible that a different temperature would lead to a different trade-off between benchmark scores and ICL / diversity abilities of the model. Even though this test is costly at scale, it could be at least conducted on the synthetic data experiment. It is also unclear what the generation parameters are the pass@k experiments, and whether they affect the pretrained and distilled models differently.
 - **Inaccurate/unclear terms**: My understanding of In-Context Learning (ICL) is that it mostly entails methods that design prompts that steer the model towards the desired behavior, the most basic example being few-shot evaluation where correct examples of a task are provided in the prompt. Referring to abstractive/extractive context-based QA or in-context retrieval (NIAH) tasks as ICL tasks does not fit this definition. The term "in-weight learning" would also benefit from contextualization and a corresponding reference.

### Questions
- How do you distinguish between high-entropy and low-entropy in section 4?
- The scale of teacher models used in distillation is usually much larger. These largers models should be better at predicting low-entropy probabilities accurately. Do you think this will have an impact on the inductive abilities of student models? 
- You compare your method to self-distillation (in the IsoData setup). What justifies that comparison? Do you think strict self-distillation would result in similar outcomes?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the impact of distilled pretraining on test-time scaling and in-context learning.
It is found that, compared to the usual pretraining, distilled pretraining enhances test-time scaling while hurting in-context learning.

The main reason for these observations are that distillation causes diversity in the output distribution: in-context learning requires memorization; diversity hurts this, while test-time scaling is enhanced by this diversity.
The paper demonstrates this intuition via a bigram model.
Furthermore, the paper proposes a simple method to overcome the drop in in-context learning.

### Strengths
1. The paper is well written, with clear motivations and convincing arguments for the behavior of distilled p retraining. A solution is also provided, which promotes the adoption of distilled pretraining. 
2. The paper also provides complementary details, e.g., changing the k of the top-k logit for distillation, to back up the arguments.

### Weaknesses
1. I think that saying that distilled pretraining is better at test time scaling is an overclaim. At pass@1 it is worse (figure 1a) and only when at large k of pass@k the performance is better (otherwise, we could say that the base model is better than an RL-trained model at reasoning, as in [2504.13837]).
It is just that the distilled model can sample from a more diverse distribution as claimed in the paper, which does not imply better test time scaling in general.
To say that it is better at test time scaling I would also expect a comparison in terms of an accuracy vs thinking time plot (e.g., those presented in figure 1 of [2501.19393]).

2. With the above in mind and the memorization property of in-context learning, the presented results are perhaps not surprising, as distilled pretraining provides soft labels which are diverse in its supervision, and this naturally leads to better test-time scaling (more diverse sampling) and worse in-context learning (diversity affects direct memorization).

3. Another possible improvement would be vary the student/teacher model size to check the validity/make more interesting observation from it.

### Questions
1. Since distillation requires extra compute from the teacher to calculate the logits, and compute is an important consideration for LLM that requires a lot of compute, why do you think that it is not correct to compare the perfomance at the iso-compute setting, as written in line 478? Particularly, the compute would be high when the whole soft label distribution is used in the calculation of distillation loss.

2. Is there any in-context learning task that does not rely much on memorization, such that there is exceptions to the claim that distilled pretraining degrades in-context learning?

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
The goal of this work is to investigate the connections between distilled pre-training to in-context learning and test-time scaling. To this The larger goal of the work is to probe how distilled pretraining (DPT) ties into in-context learning and test-time scaling. The paper shows that DPT gives noticeably better test-time scaling, yet hurts in-context learning, especially tasks usually handled by induction heads. The authors argue this comes from a bias toward high-entropy tokens during distillation, which down-weights the low-entropy bits that teach pattern copying and other ICL skills. They back this up with a clean sandbox using a bigram model that isolates a single governing factor behind both effects. Finally, they offer a simple fix: token-routing, where you drop the distillation loss on x% of lowest-entropy tokens by 
which recovers in-context performance.

### Strengths
The paper is very clearly written. The main idea, experiments, and analysis all feel solid and well thought out. The bigram model is a nice touch that builds intuition without overcomplicating things. The proposed fix is simple, sound, and gives clear gains, which makes it both practical and easy to appreciate.

### Weaknesses
Please see Questions section.

### Questions
1. While I find the experiments and analysis solid, I’m still a bit unsure about some of the practical aspects of DPT. It would help to know roughly how its compute requirements (excluding the teacher’s pretraining cost) compare to standard pre-training. The teacher’s soft-label generation likely adds some overhead. If we account for that, how much performance would an equivalent amount of standard pretraining compute buy, and how does that compare to DPT performance?
2. There also seem to be some easy extensions worth discussing. For instance, since the paper drops the distillation loss for x% of low-entropy tokens, one could imagine dropping the hard-label loss for x% of high-entropy tokens from the teacher, which might further improve test-time scaling. Did the authors try or consider this? Similarly, could a similar “token-routing” idea help in standard hard-label pretraining, e.g., using the model’s own entropy to guide which tokens to update or skip?

### Soundness
4

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
This paper investigates the effect of soft-label distillation during pretraining on the downstream performance of language models. They differentiate between tasks that rely on in-context learning and tasks that can be test-time scaled. For the first group, they notice a performance drop due to distillation and they propose a justification for this drop. On the other hand, they show that distillation helps test-time scaling.

### Strengths
- The authors did many experiments, using a thought-out and controlled setup and relatively large language models. The results give novel insights into the effect of distilled pretraining.
- Using the findings of the controlled experiments, the authors than propose a novel "token routing" method that is a simple modification of traditional soft-label distillation that should mitigate (to some degree) the negative properties (bad ICL performance) of naive distillation.
- The paper shows how distillation can help to generate more diverse outputs that can substantially help with test-time scaling.
- I really like all the additional experiments that you have done and that unfortunately did not fit into the main text. The Appendix G could be a very useful practical contribution and I feel like it is a bit of a shame that it's so hidden -- but I of course understand that you wanted to have a nice narrative arc in the main text and that the page limit is a concern. In particular, the finding that RL-aligned models are better teachers for base models is surprising and noteworthy (Appendix G.3).

### Weaknesses
- I'm not very convinced by your explanation of the poor ICL performance. Looking at the average ICL performance in Figure 3(d), it looks like distillation actually leads to very good ICL performance! It is able to match standard pretraining with just 50% of the training samples. However, the performance suddenly drop when trained on 100% training samples. This behavior is never explained in the paper. The theory of induction heads that the authors propose does not match the data -- it says that the distilled model should always be worse (because it does not robustly learns how to copy), but that is not what the results show.
- The paper claims multiple times that the teachers of Gemma-3 and Llama-3.2 were trained on "far more data than the students", and the fundamental difference of this paper is that the students and teachers are trained on the same amount of data. I'm not sure this is really true in practice. The Gemma-3 students were trained on up to 14T tokens, but there is no information about the teacher. Llama-3.2 models were trained on 9T tokens with teachers trained on 15T tokens. But note that the 3.2 models are very small (1B and 3B parameters), so 9T tokens is way past the compute optimality. So in practical terms, I believe that training can be thought as approximately "isoData", it is certainly an overstatement to say "far more data than students" (line 39), "significantly more data than students" (line 139) or "way more data than [students]" (line 467).

### Questions
- Please correct me if I'm wrong about the sudden ICL performance drop. Can that be explained by your theory of not properly learning the induction head circuits?
- Why do you suddenly use Llama-3.1 as a teacher for results in Figure 4, while everything else is done in the more controlled "IsoData" setting?

### Soundness
3

### Presentation
3

### Contribution
3
