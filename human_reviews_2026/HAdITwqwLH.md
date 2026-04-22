# Can SAEs reveal and mitigate racial biases of LLMs in healthcare?

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8, 8

## Abstract
LLMs are increasingly being used in healthcare. This promises to free physicians from drudgery, enabling better care to be delivered at scale. But the use of LLMs in this space also brings risks; for example, such models may worsen existing biases. How can we spot when LLMs are (spuriously) relying on patient race to inform predictions? In this work we assess the degree to which Sparse Autoencoders (SAEs) can reveal (and control) associations the model has made between race and stigmatizing concepts. 

We first identify SAE latents in gemma-2 models which appear to correlate with Black individuals. We find that this latent activates on reasonable input sequences (e.g., "African American") but also problematic words like "incarceration". We then show that we can use this latent to "steer" models to generate outputs about Black patients, and further that this can induce problematic associations in model outputs as a result. For example, activating the Black latent increases the risk assigned to the probability that a patient will become "belligerent". We also find that even in this controlled setting in which we causally intervene to manipulate only patient race, elicited CoT reasoning strings do not communicate that race is a factor in the resulting assessments. We evaluate the degree to which such "steering" via latents might be useful for mitigating bias. We find that this offers improvements in simple settings, but is less successful for more realistic and complex clinical tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper aims to understand how LLMs encode patient race (white vs. Black) when prompted with clinical information, and how they use race when generating text about a patient. To do so, they identify SAE latent features in Gemma 2 models that predict race, and they show a few things with these SAE latents:
1. The latent that is most predictive of race = Black also activates on other words that suggest problematic associations learned by the LLM,
2. Steering via this latent (increasing its value and plugging the representation back into the model) leads to a higher rate of the model outputting problematic associations (e.g., belligerence),
3. Steering the latent reduces assignment of Black race to patient vignettes in conditions correlated with race (cocaine abuse, gestational hypertension, uterine fibroids),
4. Steering the latent does not substantially change the logits on whether the patient is recommended pain medication or not based on their clinical history.

### Strengths
The main strength of the paper is in attempting to understand how race is encoded in clinical settings - this is a laudable goal of interest to the community, since we don't want real applications of LLMs in healthcare to inadvertently cause harm. Coarse encodings of race are one way this can happen. So, the main strength of this paper is the choice to study this problem at the intersection of health, interpretability, fairness, etc.

The paper executes a fairly reasonable set of experiments, and they are explained clearly. 

In terms of originality, I was most unsure about how much this paper adds to Ahsan et al. (2025) [1]. That paper was first posted in Feb 2025 (and will be presented at EMNLP soon). I hadn't seen this paper before, so I read it closely. After reading, here is my view on the originality of the present submission:
1. It uses SAEs, while [1] directly intervenes on the MLP activations.
2. The current paper has slightly more emphases on steering experiments, e.g. the pain medication experiment, while [1] only has one steering experiment on clinical vignette generation.

See further discussion under Weaknesses, because ultimately I am not convinced that this paper adds enough actionable insight to warrant acceptance. I'm giving a 4 to encourage the authors to try to improve this work, but I don't think it should be accepted in the current form.

[1] Ahsan et al: "Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare" https://arxiv.org/pdf/2502.13319

### Weaknesses
There are three major weaknesses with this paper:
1. The tasks and evals are too toy. We can consider them one by one:
- Belligerence: Is this a type of task that people are doing (predicting whether a patient will be belligerent from their clinical notes)? If this task should matter on its own, there should be references; if the task should matter because it says something about other tasks, that link should be explicitly drawn out.
- Patient vignette generation: Why are we interested in this task? If it is for, e.g., generating a diversity of vignettes for clinical education, we should compare against an even simpler prompting baseline: "Please generate a vignette for a white patient." I know the Zack paper uses a similar setup but I believe it's an issue with that paper too.
- Clinical tasks (logit diff): Similarly, for this task, ignoring the fact that we do not see substantial differences, even if we did, we would need to show more specifically that the LLM is somehow using race to make *incorrect* predictions, not just that the predictions change.
2. Novelty compared to Ahsan et al. I would be willing to overlook this point if the paper found important new results by using SAEs (which is their main novelty). But, seeing as SAEs did not enable a new type of result or a new discovery of unexpected problematic associations, etc., the use of SAEs alone isn't sufficient novelty.
3. Taken together, (1) and (2) illustrate that the results of this paper aren't actionable. What is the failure mode here, and how this paper help solve it? More specifically, it is not surprising that there is a link in activation space between "Black" and "uterine fibroids", and this paper does not illustrate why this might cause harms in a real setting. This is an especially important question for this paper to answer, because the methodology or line of inquiry is not fundamentally new relative to Ahsan et al. (and possibly other papers I'm missing).

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper uses Sparse Autoencoders (SAEs) to investigate racial bias in healthcare LLMs. SAEs successfully reveal problematic associations (e.g., Black patients being "belligerent") that Chain-of-Thought reasoning hides. However, SAE-based mitigation only works on simple tasks, failing in complex clinical scenarios.

### Strengths
* **Reveals Hidden Biases:** SAEs effectively identify problematic associations between race and stigmatizing concepts like "incarceration" and "cocaine."
* **Exposes Unfaithful CoT:** Demonstrates that Chain-of-Thought (CoT) explanations are unfaithful, hiding the model's reliance on race.
* **Domain-Specific Interpretability:** Highlights the necessity of re-interpreting SAE latents for the specific clinical domain.
* **Causal Steering:** Uses steering to causally confirm that activating the "Black latent" increases predictions of patient belligerence.

### Weaknesses
* **Weaker takeaways:** The authors are applauded for taking a very honest stance on their findings and avoiding making strong claims not supported by their findings. However, the paper's takeaways, especially for ICLR, seem kind of vague. Previous studies have applied SAEs to LLMs, and before reading studies, a reader would have expected a similar relevance if they were used for studying fairness. The only major finding seems to be showing SAE's outperforming COTs in "some" cases. 
* **Other methods:** The paper seems to only compare SAEs to prompting-based methods of mitigation. It is not clear how other methods would compare. 
* **Mitigation Fails:** The proposed mitigation (ablating race latents) fails to reduce bias in realistic, complex clinical tasks.
* **Limited Model Scope:** The analysis is limited to only the gemma-2 model family.
* **Limited Demographic Scope:** The study focuses only on "Black" and "white" racial categories from a single hospital dataset.

### Questions
See above, please.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors tackle the problem of racial stereotypes in LLMs for clinical text. Specifically, they test whether sparse autoencoders can surface and control race-linked latents in clinical LLMs by identifying a "Black latent" in Gemma-2. The authors show that steering this latent with a patient belligerence prompt leads to a causal increase in "Yes" predictions, and reduced stereotyping in generated clinical vignettes, but ablating race latents do not improve downstream task biases significantly.

### Strengths
1. The authors tackle the problem of bias in clinical LLMs through a mechanistic interpretability perspective, which is an important real-world problem.

2. The authors conduct a fairly thorough set of tests to probe the "Black latent" that they discover.

### Weaknesses
1. The paper is essentially a case study on one specific type of bias (stereotypes against Black patients) in two specific open-source LLMs. It is unclear whether these findings would translate to biases against other demographic groups, or whether there would be a corresponding latent for all such biases. Further, it is unclear for what categories of demographics and clinical concepts the latents are disentangled.

2. I'm not convinced by the authors' argument in 4.2.1 that all of the biases probed in the paper are biases that we would actually want to eliminate. For example, gestational hypertension is more common in Black patients, and as long as race provides a signal in the real-world, I don't see any issue with predicting this disease with higher probability for Black patients. Intervening on this effect can actually worsen predictive accuracy for Black patients, which the authors should evaluate.

3. The authors study two relatively small open-source models. I would imagine both of these models have fairly poor performance on clinical prediction tasks, and so their real-world deployment is limited. The authors should report overall model performance and try a bigger model, e.g. Qwen 2.5 72B. It would also be interesting to see whether these findings hold for self-reasoning models.

4. The final conclusion of the paper, that SAEs can reveal racial associations but don't help with mitigation, is not very satisfying. The authors briefly speculate that entanglement may be the issue, but there are further investigations that can be done e.g. some of Anthropic's work on superposition, circuit tracing, etc, which would allow an explanation of why the discovered Black latent doesn't help with mitigation, and suggest other mitigation strategies.

### Questions
1. How can the authors be sure that the identified latent is a Black latent that spuriously relates with cocaine/gunshot/hypertension, instead of a hypertension latent that spuriously correlates with Black?

2. How does the Black latent relate with a potential White latent? Does one activating deactivate another? Is there a single race axis, or multiple distinct latents for each race?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates whether sparse autoencoders (SAEs) can help interpret and reduce racial bias in large language models (LLMs) applied in healthcare settings with a use case of the MIMIC dataset. They apply SAEs  to identify latent units that correlate with race (specifically Black vs White patients) when processing discharge summaries. They find a “Black latent” (for two variants of a model, gemma-2-2B-it and gemma-2-9B-it) that activates not only on mentions of “African American” but also on stigmatizing concepts (incarceration, gun-shots, cocaine use) in the clinical text. They perform steering experiments: by intervening on the latent (increasing its activation) they show that model outputs shift such that the model assigns a higher “risk of becoming belligerent” when the patient is steered to be “more Black”. They test whether ablating the race-latent (or a set of race-related latents) reduces bias in downstream tasks: (1) a vignette generation task and (2) more realistic clinical prediction tasks (risk prediction, pain management). They find that in the controlled vignette generation setting the SAE intervention reduces bias more than prompting; but in realistic tasks, the mitigation effect is very small.

### Strengths
I really like the use of race-correlated latents to investigate bias and also appreciate that the authors show that CoT is not as useful in this regard. 
I also appreciate the steering experiments, which shows some notion of causality here.

### Weaknesses
-One major limitation is using just two models from the same family. In addition, I am curious why they used the Gemma family instead of the medgemma family, which specifically was trained for medical tasks. 
-It would be nice to show some examples of the CoT which failed to catch the bias in the appendix
-While there is an ethics section, emphasize how latent‐steering tools could be misused (e.g., malicious “race injection” in model inputs) and how to guard against that.
-Some of the effect sizes are very small; possible to bootstrap and include confidence intervals?

### Questions
Could this also be tested on other open source models?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates whether sparse autoencoders (SAEs) can help interpret and reduce racial bias in large language models (LLMs) applied in healthcare settings with a use case of the MIMIC dataset. They apply SAEs  to identify latent units that correlate with race (specifically Black vs White patients) when processing discharge summaries. They find a “Black latent” (for two variants of a model, gemma-2-2B-it and gemma-2-9B-it) that activates not only on mentions of “African American” but also on stigmatizing concepts (incarceration, gun-shots, cocaine use) in the clinical text. They perform steering experiments: by intervening on the latent (increasing its activation) they show that model outputs shift such that the model assigns a higher “risk of becoming belligerent” when the patient is steered to be “more Black”. They test whether ablating the race-latent (or a set of race-related latents) reduces bias in downstream tasks: (1) a vignette generation task and (2) more realistic clinical prediction tasks (risk prediction, pain management). They find that in the controlled vignette generation setting the SAE intervention reduces bias more than prompting; but in realistic tasks, the mitigation effect is very small.

### Strengths
I really like the use of race-correlated latents to investigate bias and also appreciate that the authors show that CoT is not as useful in this regard. 
I also appreciate the steering experiments, which shows some notion of causality here.

### Weaknesses
-One major limitation is using just two models from the same family. In addition, I am curious why they used the Gemma family instead of the medgemma family, which specifically was trained for medical tasks. 
-It would be nice to show some examples of the CoT which failed to catch the bias in the appendix
-While there is an ethics section, emphasize how latent‐steering tools could be misused (e.g., malicious “race injection” in model inputs) and how to guard against that.
-Some of the effect sizes are very small; possible to bootstrap and include confidence intervals?

### Questions
Could this also be tested on other open source models?

### Soundness
4

### Presentation
4

### Contribution
3
