# Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples

- Decision: Reject
- Scores: 4, 6, 4, 4, 4

## Abstract
Poisoning attacks can compromise the safety of large language models (LLMs)
by injecting malicious documents into their training data. Existing work has
studied pretraining poisoning assuming adversaries control a *percentage* of the
training corpus. However, for large models, even small percentages translate to
impractically large amounts of data. This work demonstrates for the first time that
poisoning attacks instead require a *near-constant number of documents regardless
of dataset size*. We conduct the largest pretraining poisoning experiments to date,
pretraining models from 600M to 13B parameters on chinchilla-optimal datasets
(6B to 260B tokens). We find that 250 poisoned documents similarly compromise
models across all model and dataset sizes, despite the largest models training
on more than 20 times more clean data. We also run smaller-scale experiments
to ablate factors that could influence attack success, including broader ratios of
poisoned to clean data and non-random distributions of poisoned samples. Finally,
we demonstrate the same dynamics for poisoning during fine-tuning. Altogether,
our results suggest that injecting backdoors through data poisoning may be easier
for large models than previously believed as the number of poisons required does
not scale up with model size—highlighting the need for more research on defences
to mitigate this risk in future models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper’s central claim is that the success of poisoning attacks depends on the absolute number of poisoned samples rather than their proportion within the training dataset. In other words, even a nearly constant number of poisoned examples can effectively compromise models, regardless of dataset size. The study comprises two main experimental parts. First, the authors conduct denial-of-service (DoS) poisoning attacks on language models ranging from 600 million to 13 billion parameters, demonstrating that approximately 250 poisoned documents are sufficient to degrade performance across all model sizes. Second, they perform smaller-scale experiments to explore factors influencing attack effectiveness, such as poisoning rate, sample order, and poison density per batch, across multiple attack types, including language-switching backdoor and harmful-request compliance backdoor attacks.

### Strengths
1. Originality:

The paper challenges a long-standing assumption in data poisoning research that attack success scales with the percentage of poisoned samples in the training data. Instead, it advances a novel and thought-provoking claim: the effectiveness of poisoning depends primarily on the absolute number of poisoned samples, independent of the total volume of clean data. This reframing of the problem represents a significant conceptual shift in understanding the scaling behavior of poisoning attacks.

2. Quality and Clarity: 

The paper is well-structured, and its exposition of threat models, attack methods, and experimental setups is generally clear and systematic. 

3. Significance: 

The work addresses a fundamental and practically important question in data poisoning attacks research: how to measure the amount of poisoning samples required to launch successful attacks. By clarifying this, the paper provides valuable insights that could influence both future attack designs and the development of more robust defense strategies.

### Weaknesses
1. I have reservations about the generality of the paper’s main claim that successful poisoning attacks require only a constant number of poisoned samples, regardless of the training dataset size. In Section 3, the paper shows that 250 poisoned samples are sufficient to compromise models ranging from 600M to 13B parameters in DoS attacks. Based on the Chinchilla scaling law, the total number of training tokens is roughly 20 times the number of model parameters. Under this assumption, 250 poisoned samples correspond to approximately 0.00016% of the training tokens for the 13B model and 0.0035% for the 600M model.

However, this evidence may not conclusively support the claim of size-invariant poisoning effectiveness. An alternative explanation is that attack success might still depend on the percentage of poisoned samples relative to the total training data—but this threshold decreases with increasing model size. This would be consistent with prior findings suggesting that larger, more capable models are generally more sensitive to data poisoning.

Furthermore, the experiments only test models up to 13B parameters, whereas training 30B+ models is common nowadays and modern production-scale models often exceed 100B parameters. If the claim holds universally, 250 poisoned samples should also suffice to compromise models of that scale. Yet, this remains unverified. It is also possible that 250 samples do not represent the minimal effective percentage of poisoned data, and thus may fail to affect substantially larger models. Without additional experiments at larger model scales, the true scope and validity of the paper’s claim remain uncertain.

2. In Section 4.2, the paper states that “[Figure 3 shows that] despite differences in poisoning rate, all configurations achieve similar attack success rates once they have encountered the same absolute number of poisoned examples.” Figure 3 plots the absolute number of poisoned samples encountered during training (x-axis) against the attack success rate (ASR) (y-axis). Each data point represents a distinct experimental configuration, potentially differing in the percentage of poisoned samples per batch and the frequency of such batches during training.

If I interpret the figure correctly, comparing these configurations requires examining vertical slices at specific x-values—that is, comparing ASR values across different settings for the same absolute number of poisoned samples. However, when looking at such a vertical slice—for instance, around 1,000 poisoned samples—there appears to be substantial variance in ASR among points of the same color (e.g., the green dots representing 1% poison density). ASR values range roughly from 40% to over 80%.

This wide spread suggests that factors beyond the absolute number of poisoned samples may significantly influence attack success in language-switch backdoor attacks. Therefore, Figure 3 does not seem to fully support the claim that the absolute number of poisoned samples alone determines ASR? 

3. In Section 5.2 and Figure 7, the paper claims that “Fine-tuning GPT-3.5-Turbo via the OpenAI API with different amounts of clean data (colour) randomly intermixed with different amounts of poisoned samples (x-axis) has minimal effect on ASR (y-axis).” But taking a closer look at Figure 7, and if we fix the number of poisoned samples, for instance, at 200, the attack success rate (ASR) appears to vary noticeably across settings with different fine-tuning dataset sizes (i.e., different amounts of clean data). In this region, ASR fluctuates from below 60% to around 75%, suggesting a non-trivial dependence on the quantity of clean data?

### Questions
1. In section 4, does one training step correspond to one processing one batch of data?

2. In section 4 line 249 – 250, the paper states “we train for 100 steps …” whereas in line 268 – 269, the paper mentions “we resume training for 300 steps …” Could you clarify the difference between the two different numbers of steps? 

3. In section 4 line 261 – 262, how to measure the similarity between “a similar-looking but distinct trigger” and the actual trigger?

4. The paper mainly studies poisoning attacks in pre-training, would the main conclusion -- that attack success depends on the absolute number of poisoned samples--still hold if the poisoned model were subsequently fine-tuned with clean data?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper studies backdoor **data poisoning** for LLMs and argues that attack success depends primarily on the **absolute number of poisoned samples** rather than the poisoned **percentage** of the training set. The authors pretrain 600M–13B parameter transformers on **chinchilla-optimal** token budgets and report that as few as **~250 poisoned documents** (or 420,000 poisoned tokens) trigger a denial-of-service (DoS) backdoor across model scales (perplexity inflation on triggered generations), with similar dynamics during fine-tuning (e.g., Llama-3.1-8B-Instruct; GPT-3.5-turbo). They also present ablations on poison density/frequency, poisoning timing, and limited analyses of persistence under continued clean training and post-training "alignment." Overall, they conclude that poisoning *does not get harder with scale* if the adversary can insert a near-constant number of poisons.

### Strengths
**Clarity.**
The paper is well written and easy to follow and makes a well supported point.

**Clear central claim, demonstrated across regimes.**
The constant-poison hypothesis is clearly stated and repeatedly tested: (i) pretraining DoS backdoor across 600M–13B with 250–500 documents, (ii) pretraining language-switching backdoor with Pythia checkpoints, and (iii) supervised fine-tuning backdoors (Llama-3.1-8B-Instruct; GPT-3.5-turbo). The "250 docs" DoS result is a memorable headline.   

**Large-scale pretraining experiments under realistic token scaling.**
The authors train at **chinchilla-optimal** tokens (~20× parameters), making the *percentage* of poisons shrink with model size, yet success remains similar—supporting their absolute-count claim. 

**Careful ablations of data mixture effects.**
The paper disentangles *per-batch poison density* vs *frequency* and suggests that success tracks the **number of sequential gradient steps on poisoned data**, not merely density, offering a mechanistic hypothesis worth follow-up. 

**Initial evidence on persistence and (limited) mitigation.**
The paper shows continued clean training can degrade attacks and that small amounts of **alignment SFT** reduce ASR for certain settings, hinting at a potential mitigation without over-claiming universality.

### Weaknesses
**Some methodology details are imprecise.**
Section 3.1 describes poisons using `random(0,1000)` and `random(400,900)` for prefix lengths and gibberish texts. Please **replace with formal variables and distributions** (e.g., $ (L_{\text{prefix}}\sim U[0,1000]), (T_{\text{gibberish}}\sim U[400,900]) $ ) and specify tokenizer sampling for gibberish precisely. This improves clarity and reproducibility. 

**Counter-intuitive poison-density finding needs sharper evidence.**
The paper states that "at higher per-batch poisoned density, attacks need more poisoned samples to succeed," hypothesizing a need for **sequential** poisoned steps. This is interesting but currently **speculative**. Please elaborate and add metrics to quantify this effect. 

**Persistence under realistic post-training is under-explored.**
The paper rightly flags that persistence through **post-training** (instruction fine-tuning, safety fine-tuning) remains open; Appendix I shows SFT can often reduce ASR to near-zero in particular settings, but the main claim centers on **pretraining** backdoors at moderate scales. Stronger evidence would include: (i) persistence through **instruction SFT + preference learning/RLHF** on larger models, and (ii) tests that the **DoS** and **harmful-compliance** backdoors survive realistic safety pipelines (not only language-switch).  

**Defenses are not evaluated.**
Given the practical framing, please run **simple filtering baselines** (e.g., perplexity/entropy filters, style/trigger heuristics, ...) on the training mixture, plus **continued clean training** schedules, to quantify how many poisons evade a realistic pipeline and how quickly ASR decays. Even if imperfect, it grounds the risk.

**Unrealistic gibberrish threat model.**
As these poisons would include high perplexity texts, it would be easy to simply filter the poisons out of the training data, making them too few in the training set to actually poison the models. How would you make your poisons harder to detect?

**Reproducibility limits (no released code/data; many moving parts).**
The paper claims very strong results but currently lacks public code/data; even small choices (tokenizer, sampling temperature, exact trigger placement) may affect outcomes. A minimal release (data constructors, prompt/trigger templates, training configs, exact LR schedules) would strengthen credibility. 

**Tables are oversized.**
Please consider fixing width of table 1 and 3.

### Questions
How was the 50+ perplexity threshold set?

Additionally see weaknesses field.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies backdoor data poisoning attacks during both pretraining and instruction fine-tuning of LLMs. The results show that, regardless of model size, dataset scale, or training stage, the absolute number of poisoned samples, instead of their proportion, mainly determines the success of backdoor attacks in LLMs.

### Strengths
This paper presents a comprehensive and systematic study of backdoor data poisoning attacks on LLMs. The key findings and strengths are: 
1. The results hold across multiple models, sizes, and training stages, showing the robustness and generality of the “fixed-number” result.
2. It features systematic experiments, carefully controlling data scale and poisoning conditions.
3. Includes cross-model validation, confirming the consistency of results across architectures.
4. Attacked models maintain high clean accuracy while still showing near-perfect responses to backdoor triggers, indicating that such malicious behaviors are covert and hard to detect.

### Weaknesses
1. This paper lacks theoretical explanation as mainly it shows an empirical result but does not explain why it would occur.
2. The triggers used are simple and easily noticeable phrases, not natural or contextually meaningful ones. Poisoned samples are randomly inserted across the dataset, which might not be realistic for real-world poisoning attacks.
3. The paper does not explore localized or domain-specific poisoning scenarios.

### Questions
1. Do the findings observed on models ranging from 600M to 13B parameters still hold for larger, frontier LLMs (say 70B)?
2. Could the study’s conclusions change if more realistic poisoning settings are used? For instance, as also pointed out in Weakness section, how about natural, contextually meaningful triggers and non-uniform, domain-specific injection of poisoned samples instead of simple phrases inserted uniformly at random?

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
2

### Summary
This paper investigates data poisoning and backdoor attacks against large language models (LLMs). Contrary to the common assumption that poisoning success depends on the fraction of corrupted data, the authors find that only a near-constant number of malicious samples (≈ 250) is sufficient to implant effective backdoors across models of vastly different sizes (from 600 M to 13 B parameters) and dataset scales (6 B–260 B tokens). The experiments cover multiple attack types (denial-of-service, language-switching) and demonstrate that the attack success is independent of model scale or data ratio.

### Strengths
- First large-scale demonstration that poisoning success depends on absolute sample count, not ratio. The discovery that only a constant number of poisoned samples is needed fundamentally challenges the common assumption that increasing dataset size naturally enhances robustness. 
- Systematic study across model scales, data sizes, and architectures (pre-training and fine-tuning). This scale-aware methodology ensures that results are not artifacts of training under- or over-parameterized models, strengthening the validity of the conclusions.
- Methodology clearly described and well-controlled (Chinchilla-optimal scaling). The paper clearly describes its training settings, poisoning injection procedures, and evaluation criteria (e.g., per-token perplexity increase, trigger activation rates). The inclusion of both pre-training and fine-tuning scenarios makes the work reproducible and relevant to real-world LLM development.

### Weaknesses
- The paper mainly evaluates simple trigger-based behaviors (DoS and language switching). It would be valuable to test more complex or stealthy objectives, such as factual corruption or conditional bias injection.
- While the paper identifies vulnerabilities, it does not propose or evaluate potential countermeasures, such as poisoned data detection or post-training sanitization.
- The paper lacks a theoretical explanation for why the poisoning effect saturates at a constant number of samples. A theoretical explanation could strengthen the contribution and provide deeper understanding beyond empirical observation.

### Questions
Please refer to the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies how the amount of data needed for backdooring large language models does not scale with the training set size and instead remains nearly constant. The authors explore backdoors during both pretraining and finetuning.

### Strengths
- The paper is well-written and polished.

- The main observation is interesting and valuable for the research community.

### Weaknesses
- The title is somewhat misleading, as the paper only investigates backdoor attacks (which is a specific type of poisoning attack) rather than poisoning attacks in general.

- There are some limitations in the experimental setup. For the main pretraining experiments, the attack used (gibberish generation) is quite simple. While it's still interesting to study, it may not be very relevant in practice. For the other attack types, such as the language switch and safety instruction finetuning, the ASR quickly reaches nearly 100%, making it unclear whether the finding (the number of poisons needed for success stays constant) truly generalizes.

- It's also unclear how ASR is computed for the safety instruction finetuning experiment. Is it evaluated across all test samples or only for some targeted ones? (I assume it's for all test samples.) If so, what do these samples look like in both training and testing? Are they similar in structure or content? The fact that the attacker needs only 20 poisoned samples feels surprisingly small if the dataset is diverse.

- It would be interesting to explore how defenses (e.g., Latent Adversarial Training) interact with the number of poisons. For example, do the attacks achieve similar (but lower) ASRs after applying defenses, regardless of poison count?

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2
