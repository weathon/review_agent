# FSPO: Few-Shot Optimization of Synthetic Preferences Effectively Personalizes to Real Users

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Effective personalization of LLMs is critical for a broad range of user-interfacing applications such as virtual assistants and content curation. Inspired by the strong in-context capabilities of LLMs, we propose few-shot preference optimization (FSPO), an algorithm for LLM personalization that reframes reward modeling as a meta-learning problem. Under FSPO, an LLM learns to quickly infer a personalized reward function for a user via a few labeled preferences. FSPO also utilizes user description rationalization (RAT) to encourage better reward modeling and instruction following, recovering performance with the oracle user description. Since real-world preference data is challenging to collect at scale, we propose careful design choices to construct synthetic preference datasets for personalization, generating over 1M synthetic personalized preferences using publicly available LLMs. To successfully transfer from synthetic data to real users, we find it crucial for the data to exhibit both high diversity and coherent, self-consistent structure. We evaluate FSPO on personalized open-ended generation for up to 1,500 synthetic users across three domains: movie reviews, education, and open-ended question answering. We also run a controlled human study. Overall, FSPO achieves an 87% Alpaca Eval win-rate in generating responses that are personalized to synthetic users and a 70% win-rate with real human users in open-ended question answering.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Few-Shot Preference Optimization (FSPO) for fast, user-specific alignment from a small number of preference examples. It utilizes the idea of meta learning by viewing each user as a task and adapts the model by conditioning on the user’s few-shot preferences. The method also uses user description rationalization to infer a short natural-language profile from those examples. Experiments show consistent gains over non-personalized and standard preference-tuned baselines. Overall, the paper offers a clear formulation of few-shot personalization with a simple, practical recipe and strong results.

### Strengths
This is a solid paper. The FSPO approach is novel, and the problem, effective user-specific preference adaptation, is crucial for LLMs. Empirically, the gains are consistent across settings and baselines, with good sample efficiency on small backbones and few shots. Overall, the writing and ablations are clear and support the main claims.

### Weaknesses
1. In the experiment, the method adds test-time steps (especially with RAT), but there are no latency/throughput numbers—only a note that RAT or caching could mitigate context overhead.
2. Minor weaknesses in experiments. Heavy reliance on LLM-judged win rate; PRISM “transfer” is narrow in the Appendix; only the single 3B backbone is considered—no size scaling. 
3. In equation (5), I feel the notation is ambiguous. For the historical data, an additional superscript may solve the potential misunderstanding.

### Questions
1. As noted in the weaknesses, what is the inference-time computation cost of RAT? 
2. The experimental design and results are strong. However, why is the standard pairwise response selection task not included? There are many ready datasets and more baselines to compare.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Few-Shot Preference Optimization (FSPO) treat personalization as a black-box meta-learning problem where each user = a task. Given a small set of that user’s preference pairs (preferred vs dispreferred responses), the model must answer a new query in that user’s style. Instead of classic RLHF that averages over users and washes out individuality, FSPO keeps user IDs in the preference data and optimizes a DPO/IPO-style objective conditioned on that user’s few-shot prefs. 

A key ingredient is RAT (User Description Rationalization): first, from the few-shot preferences, the model generates a natural-language user description (a synthetic persona); second, it generates the final response conditioned on that description. This makes the user representation more explicit and interpretable, and it’s like spending extra inference compute to do better reward modeling. 

Because real personalized data is expensive, the paper builds a structured synthetic data pipeline: generate multiple viewpoints per question, refine personas iteratively when they’re underspecified, and enforce both diversity and structure so the model can transfer to real users (sim2real). They evaluate on three personalization-style tasks (Reviews, ELIX, Roleplay) and show that RAT + FSPO outperforms simpler few-shot prompting and standard preference learning, and argue this pipeline also helps cold start and privacy-sensitive settings.

### Strengths
Thoughtful synthetic data design (diversity + structure). The pipeline doesn’t just random-sample dialogues; it explicitly generates multiple viewpoints and refines personas to avoid underspecification.
Casting “each user = a task, few-shot prefs = task context” is a neat, well-motivated transfer of meta-learning ideas to RLHF-style preference learning. It directly addresses the criticism that population-level preference aggregation destroys individual preferences.

### Weaknesses
Sim2real is argued well, but the paper could show more quantified transfer: e.g. how much do metrics drop when you remove diversity, or structure, or persona refinement? Right now, some of the pipeline choices feel plausible but not fully ablated.
Judge / metric bias. Since the setup is preference-style, are judges from the same model family? Are they sensitive to verbosity / style? We don’t see a deep cross-judge analysis.

### Questions
RAT sensitivity. How sensitive is performance to incorrect or partially correct user descriptions? Can you show a curve where you corrupt 0%, 25%, 50% of the RAT description and see task win-rate drop?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FSPO, a meta-learning approach for LLM personalization. FSPO uses a few user-specific preference examples to quickly infer a personalized reward function for a user, potentially with a user description rationalization (RAT) step that in addition infers a natural language persona summary to further improve personalized generation. This paper also constructs  a large synthetic preference dataset consisting of 1M synthetic personalized preferences. Experiments across three tasks and a real human study shows that FSPO outperforms standard prompting and SFT.

### Strengths
1. Conceptually sound meta-learning framing with RAT, which is practical and interpretable	
2. Well-designed synthetic data pipeline with large scale dataset as a result
4. Actual human study

### Weaknesses
1.A core desideratum for personalization is interactive, continual adaptation: as the LLM interacts with the user, it should actively infer, refine, and remember their preferences in context. Therefore, one major concern is over the fixed k-shot evaluation missing the crucial dynamic evolution. Specifically, there are a few dimensions to this
i. when the models are trained on k-shot optimization, but at test time, the user interactions are not k, but either less than or more than k. 
ii. the dynamic of how well the model does in terms of continually learning user preference as the model interacts with the user further, i.e., it might be the case that the model trained is better at eliciting preferences when interaction is sparse, but fails to remember well when interaction gets longer compared to context-engineering at inference time only (no training). It is informative to provide how the model performs across interaction turns.
iii. metric could be some sort of a regret-style curve over turns (win-rate or utility vs. step), not a single snapshot. We want to know how the performance scales from under-conditioned to over-conditioned regimes. Ideally, adaptation is monotone and data-efficient: small k yields robust gains, larger ks keeps helping without overfitting or context bloat.

2. Out-of-distribution: conversations naturally shift topics. It'd be informative to see evaluation over topics that are not seen during training and how good the model is at eliciting preferences given ood interaction histories.

### Questions
1. What if we are at a cold start problem? How is the model trained to pick up preference in a most efficient way.
2. The few-shot interaction that is provided, what if you have the chance to choose the questions to ask? Is there discussion over active preference sampling. The assistant should really selectively elicit the next most informative pair/question to shrink uncertainty in the user latent with minimal burden.

### Soundness
3

### Presentation
2

### Contribution
2
