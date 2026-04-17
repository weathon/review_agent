# Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

- Decision: Accept (Poster)
- Scores: 8, 2, 4

## Abstract
Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. Currently, there is not yet a robust science of open-weight model risk management. Existing safety fine-tuning methods and other post-training techniques have struggled to make LLMs resistant to more than a few dozen steps of adversarial fine-tuning. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine-tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text — outperforming existing post-training baselines by over an order of magnitude -- with no observed degradation to unrelated capabilities. However, while filtered models lack internalized dangerous knowledge, we find that they can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach. Overall, these findings help to establish pretraining data curation as a promising layer of defense for open-weight AI systems

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors investigate if removing pretraining data from LLMs can make them more robust to different adversarial attack scenarios (finetuning, evasion-based, etc.). They find that filtering considerably increases robustness in all threat models. Moreover, they analyze limitations of their methods, such as robustness to in-context learning attacks. Overall, this paper presents one of the few works addressing how open-source LLMs can be made safer for public release.

### Strengths
- White-box attack settings in open-source models are generally underresearched in the literature. A lot of emphasis is put on the safety of increasingly capable models. How open-source threat models (that are incredibly hard to make save and at the same time very capable) fit in this scenario is mostly ignored. 
- The potential safety risk of open-source models is well-motivated
- High rigor (extensive information is provided in the appendix regarding the investigations performed in the paper)
- New approach is compared and combined with the most promising orthogonal methods already investigated in the literature (adversarial training, representation engineering-based training approaches)
- Extensive investigation (and honest reporting) of limitations (e.g., to in-context-learning)
- To the best of my knowledge, the most relevant papers have been referenced. Moreover, the authors specifically discuss their results in context with seemingly conflicting evidence from previous papers.

### Weaknesses
Major:
- The takeaway/conclusion of the paper is a bit to optimistic with respect to the experiment results. There are multiple aspects to consider: The academic view: Your approach considerably improves upon the baseline and seems to be a promising direction to explore / an orthogonal defense to many already investigated approaches. Practical view: The results given in Figure 5 show some practical promise in improving the safety of open-source models (harmful data will not always be readily available for fine-tuning).. Yet, finetuning a model for 300m tokens is feasible for basically everyone with more than >=10 dollars, and in most cases, dual-use / harmful data can be scraped from the internet. Thus, currently open-source models do not appear to be safe from misuse at all. In my opinion, this should be more clearly articulated in the paper. 

Minor:
- Section 2 provides a lot of very different information in an unstructured way (e.g., one subsection follows after another and addresses very different topics from method design to experiment setup). I believe this could be slightly improved by motivating the structure at the beginning or by clearly separating method design from experiment setup. For example, for somebody not familiar with the literature, the circuit breaker / LAT subsection in 2.3 will be lacking context. Similarly, 2.6 already provides experiment results, mixing three generally separate topics in one section (method, setup, results).
- Summarize the attack threat models in one place (e.g., Latent attacks and Input-space attacks). All of those are evasion attacks.
- The scope of the degradation in "accuracy" evaluation is fitting for this work, but I would assume that a lot of subtle differences/degradations might be difficult to spot, and performing evaluations on a few utility benchmarks might not be sufficient to spot those. A discussion could be added to the paper

### Questions
- Do you have an explanation regarding the non-trivial capabilities of the model trained with "strong-filtering" on WMDP, specifically concerning the considerably lower accuracy of the CB+LAT approach (e.g., Figure 3)? I would have expected filtering to be a strictly stronger approach. To phrase it differently, do you think WMDB might not be a suitable benchmark for dangerous capabilities? Based on the results, I would assume that it might be easy to answer some questions of the benchmark in context without any knowledge about the topic (specifically considering the observations provided in Appendix C1) 
- The authors acknowledge the limitations of their approach regarding ICL attacks. What exactly do they mean that defending against this kind of threats would require a "defense-in-depth approach". I personally have a hard time coming up with possible defense strategies in these settings.
- Do the authors think that finetuning attacks that try to elicite knowledge with a few training steps might be a relevant threat model in the future against "tamper resistant" models?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates an innovative approach to enhance the tamper resistance of open-weight models through pretraining data curation. The authors propose a scalable multi-stage data filtering pipeline and pretrain multiple 6.9B-parameter models from scratch based on this methodology.

### Strengths
1. The study presents comprehensive experimental validation.

2. The paper is easy to understand.

### Weaknesses
1. The paper suffers from disorganized structure, failing to follow the standard methodology-experiments-analysis framework. The presentation appears arbitrary, significantly hindering comprehension.

2. Inadequate baseline comparison: Only Circuit Breaking (CB) and Latent Adversarial Training (LAT) are included. The experimental design should incorporate more baseline methods.

3. Limited dataset evaluation: Sole reliance on the DCLM dataset prevents meaningful assessment of method generalization.

### Questions
1. The LaTeX version used by the authors appears inconsistent with the standard ICLR template. For instance, the first-page elements such as "Anonymous authors" and "Paper under double-blind review" are centered in the submitted manuscript, whereas they are left-aligned in the official template. It remains unclear whether other template parameters (e.g., font sizes, spacing adjustments) have been modified.



2. The paper's presentation quality is substantially below acceptable standards, requiring comprehensive revisions to achieve readability.

### Soundness
2

### Presentation
1

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
This paper studies the effect of filtering harmful/dual use data from the training dataset in order to prevent unwanted capabilities. In particular, they propose a simple and automated filtering pipeline using keyword filtering and (optionally) a classifier to remove potential biothreat data from the *pretraining* dataset and train 6.9B-parameter models from scratch. The key findings are that their models maintain utility on standard benchmarks, while having much lower performance on biothreat proxy metrics (WMDP-Bio), and also being fairly resistant to fine-tuning the models on such harmful data. However, they do also find that these ‘filtered’ models are still capable of leveraging the harmful information when provided in context.

First, I think it’s fair to say the results are in line with what would be expected: it’s fairly obvious that training a model with a certain subset of data filtered results in a model that is significantly less performant in that domain. That said, I do believe it is useful to precisely quantify what difference we can expect in practice, so I am generally appreciative of this work. However, I have a few concerns with the results and the framing; I’ll mention my main point below, and mention the rest under the weaknesses/questions sections. 

One of the main claims of the paper is “state of the art tamper resistant with respect to fine-tuning”;  I think it’s better to frame this as an empirical study on the relationship between pretraining and fine-tuning data rather than calling this a safeguard. With regards to the adversarial fine-tuning setting, I’m not sure if this is really a good evaluation of whether you are **resistant** to **adversarial** fine-tuning. You fine-tune on `cais/wmdp-bio-forget-corpus`, which contains about ~150M tokens. However we should compare that to the amount of pretraining data filtered: the base model had 300B tokens, with 8.42% filtered (42B), and 50B during the annealing phase with either 9.36% (4.68B) or 4.96% (2.48B) tokens filtered for the strong and weak filter respectively. These are orders of magnitude fewer tokens on these subjects that the filtered model has seen.  I think the missing experiment here is seeing how some addition of the filtered pretraining data would close the gap if incorporated to the fine-tuning phase, as this would be more representative of the “truly worst case” adversarial setting. This is especially relevant because the paper makes the claim that the model is “resistant to adversarial fine-tuning for up to ~305M tokens/10k steps”, while the cloze attacks recover the base LLM’s performance in about 4k (weak filter) and 5.5k (strong filter) steps. 

Beyond this, the paper shows that their method is both resistant to input space and latent space attacks attacks (as the knowledge is missing), but also highlights that the lack of knowledge does not prevent the model from using the knowledge if provided in context from other sources. This suggests that additional approaches are needed to mitigate harmful propensities beyond their knowledge-filtering method; the authors show that LAT and CB are effective in this setting. Data filtering thus appears to be a complementary and synergistic component of broader safety strategies.

I currently am leaning towards reject, but I am very open to increasing my score upon discussion and potentially clarifying some of the framing of the paper.

### Strengths
- Comprehensive and extensive experiments, including full pre-training of medium scale LLMs, ablations on their data filtration method, comparison with other LLM safety training techniques (LAT/CB).
- Regardless of whether the results are positive, negative, or obvious, large scale empirical studies like this are very valuable to the community
- The paper is well written and clear; there is a lot of information both in terms of intuition as well as technical references to allow users to reproduce their setup if desired. I appreciate how thorough the authors were; most of the times where I had a question about the implementation details, I was able to find it either in the main body of the paper or the appendix.

### Weaknesses
- I have several concerns about the evaluations done, in particular the adversarial fine-tuning case (which is one of the main selling points of the work). I have discussed them in the summary section.
- I don’t think I agree with the current framing of how strongly this approach is being sold as a safeguard/robustness technique to adversarial fine-tuning; I see this more as an empirical study on the relationship between the pretraining data and fine-tuning data which is a slight change in the narrative, as well as (in my opinion) missing a few experiments to solidify the results/conclusion.

### Questions
- Why are cloze style evaluations marked with the 25% random chance line? My understanding is that this should only apply to the MCQA evaluations?
- Is your adversarial fine-tuning evaluation protocol, which fine-tunes on `cais/wmdp-bio-forget-corpus`, really a fair comparison to what was removed from the base model? What would the results be if you fine-tuned on all the available data, or some interpolation between the regimes?
- How do you think the results would compare with far more powerful/larger pretrained models? Would you expect them to be able to learn harmful data with fewer fine-tuning tokens? If so, would this be a potential issue with scaling this approach?

### Soundness
3

### Presentation
4

### Contribution
3
