# Jailbreaking as a Reward Misspecification Problem

- Avg Score: 5.75
- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a new perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark  against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper approaches red teaming from a new perspective by using reward misspecification instead of target loss. The ReGap metric to quantify misspecification shows strong performance in creating red teaming prompts as well as good performance when transferring from a white box to a black box setting. The paper provides novel insights into safety alignment vulnerabilities.

### Strengths
The approach via reward misspecification is novel and highly effective. The experiments are sound, well presented, and extra effort was put into looking at practically relevant situations such as the impact of defense models on ReGap attacks.

### Weaknesses
Experiments on larger models would be of interest to show how scalable and economical the method is for real-life usage

With the focus on suffixes, the ability of the method to be used for other classes of jailbreaks is missing which would be important for a method to be used for a holistic jailbreak evaluation in practice to ensure a model is protected across a diverse set of jailbreaking strategies.

### Questions
Can the authors please elaborate on possible extensions of their method into multi-modal settings as large models are increasingly becoming multi-modal?

Can the authors elaborate on diverse jailbreak attacks going beyond suffixes with their method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper considers the problem of jailbreaking. The authors propose a metric, which they claim correlates with/tracks the alignment/susceptibility to adversarial attacks of LLMs. The authors then use this metric to devise an automated red-teaming scheme. They provide a thorough set of experiments to evaluate their approach.

**Review summary.** Overall, this is paper makes an interesting contribution, but it needs to be revised significantly to more clearly identify its contributions and to differentiate itself from past work. I have outlined this in detail in the weaknesses section of this review. I don't necessarily think that more experiments are needed. Rather, it's more important to get a sense for whether this method offers a clear algorithmic improvement over AdvPrompter, or whether it constitutes a few tweaks on top of this method that leads to a few points better improvement on AdvBench.

### Strengths
**Experiments.** The experiments are comprehensive. The authors include a number of ablation studies, and indeed, the proposed method improves significantly over AdvPrompter, particularly in the ASR @ 1 metric. For ASR @ 10, the methods do more or less the same on most of the models (except for Llama3.1, although this comparison isn't complete as the other baselines were completely omitted). The authors use a wide range of models, although it would make a stronger case if the authors could show improvements on more robust models (e.g., the Claude or Cygnet model families).

**Writing.** The writing is generally of high quality.  The paper is also structured nicely, and the experiments in Section 5 are clearly presented.

### Weaknesses
**Experiments.**
* A worthwhile question here is whether AdvBench is the right dataset to focus the analysis on. It is well-known that the train/test split of AdvBench is contaminated in the sense that both splits contain the same behaviors (e.g., asking for instructions to build a bomb). It may be worth focusing on HarmBench (which is used for the OOD experiments) or JailbreakBench.
* The experiment in Section 3.2 is not well explained. 
    * The authors omit most of the details regarding how the trigger is planted in the model, and from what I can tell, the authors assume that they have access to the group truth trigger, so they are acting both as the adversary and as the defender here.  
    * There are no details in the main text (and, from what I can tell, nor are there any details in the appendix) regarding what the five models $M_1,\dots,M_5$ are. Even if this is drawn from (Rando et al., 2024), the authors are still responsible for specifying the full details of the experiment that they are reproducing. 
    * The authors should derive why 0 is the correct threshold to use in the MR metric. It's not a complicated derivation, but it involves several steps, and readers will likely get lost here wondering why this threshold was chosen.
    * It's not clear why the "averaged result" is a better metric than using a randomly generated trigger. Why should we expect a trigger from one model to be good/bad for another model? As a reader, it's not possible to answer this question, both because the authors do not give details regarding what the models are and because it's not clear how the triggers are chosen in the first place. 
    * This leads to a more general point/question: What are the reference and aligned models throughout the paper? Is it reasonable to assume that we have access to both? Are we assuming that we always have a fine-tuned/aligned version of the same model?
* There are missing values in the last row of Table 1. This is a major weakness of the paper; it is not a fair comparison to omit these algorithms for some models and include them for others.


**Presentation.**
* The authors call their approach "novel" six times in the paper. I think the principle of "show, don't tell" would be beneficial here. Rather than repeatedly claiming that your approach is novel, it is more than sufficient to show that the approach is novel by contrasting it with work that's come before. And in this respect, the paper could improve. Specifically, the approach seems like a close variant of AdvPrompter, which, to my understanding, uses essentially the same algorithm as is presented in Algorithm 1 in the appendix. Given the generous 10-page page limit, it's reasonable to expect the authors to more clearly distinguish their work from AdvPrompter. This would be particularly helpful in the the context of the experiments, as it's not clear why the proposed method improves to the extent that it does over AdvPrompter. In other words, what is the main insight(s) that led to the improved performance, given that the algorithm and threat model are quite similar?
* The authors often use language that isn't justified, in my opinion. 
    * For instance, they claim that their approach "has the potential for comprehensive jailbreaking," but they do not explain what is comprehensive about it or why previous methods fail to be comprehensive. 
    * They claim that the "target loss alone is not a good proxy for jailbreaking," but they don't present a clear definition of what "good" means here, and the experiments don't seem to offer a more compelling definition (see the discussion around Figure 3). 
    * The authors talk quite a bit about rewards being "misspecified," but it's unclear what the definition of being "well specified" is here. 
    * A similar argument can be made about why $\pi(y|x)/\pi_\text{ref}(y|x)$ is an *implicit* reward; what is *implicit* about it, and/or what does it mean to be "implicit" here? Moreover, why does this represent a "comprehensive" measure of alignment? 
    * I could go on, but I think the point is clear: The paper would be much improved if the authors made claims that were (a) backed up by clear definitions, which (b) would facilitate the verification of these claims.
* I found the analysis in Section 2.1 hard to follow. 
    * One way to improve this would be to actually plug (2) into (1) and show that the interpretation of the reward function as being proportional to the ratio of the log-probs of the aligned and reference model holds.  
    * It's also unclear why "any" fine-tuning procedure could be viewed in this way. What if I fine-tune on objectives that don't consider the KL divergence (see, e.g., https://openreview.net/forum?id=2cRzmWXK9N)? Then, it's unclear how the objectives would coincide.
* The experiments in Section 2 are presented in an unclear way. The page limit should allow for a full description regarding how the experiment is set up. Some comments: 
    * The authors do not make a clear argument as to why the adversarial loss is "flawed." Should we use Figure 3 as evidence, considering that it's not clear (from the main text) what the evaluation criteria was for determining whether a jailbreak occurred? 
    * What does it mean to "elicit a target behavior" as opposed to "merely generating specific target responses." You could certainly view generating a response as a valid way of eliciting the behavior, and ultimately, in the experiments in Section 5, this is (from my understanding) what the authors end up doing. So if you don't want to generate a specific response, it's important that the paper *clearly* define what the criteria is for evaluation.  
    * Also, why is the suffix model the best model here? The authors say that they "follow previous work," but there is plenty of other previous work that one could also follow that does not use suffixes. For example, the authors cite (Chao et al., 2023) throughout. This paper proposes a non-suffix-based jailbreaking scheme.
* The main method could be presented in stronger/more clear way. 
    * In the current version, the authors say that the proposed approach "diverges from traditional methods that focus on minimizing the loss for specific target responses." But following this, the authors go on to explain how their method minimizes a particular loss that is writing in terms of two target responses $y^+$ and $y^-$. Now, the method may not be trying to elicit these responses in particular, but still, the fact remains that the method is more or less following the same blueprint that the authors claim not to use.
    * It's not clear what an "unlikelihood term" is/means. The authors should explain.
    * The paper would be clearer if the authors referred to their method offering "better performance" until they define what they mean by "good performance," i.e., define the metric by which they will compare methods.
    * It's not clear how $y^-$ is chosen. More details on this would help complete this section.
    * The authors claim that the $\alpha$ factor leads to better performance, but they don't make an evidence-backed case as to why. Further, the value of $\alpha$ is not specified in the experiments. This lack of rigor is a significant drawback.

**Related work.**
* It's not clear what the so-called "out-of-distribution issue" referenced on line 162 is. The authors should elaborate in their discussion of related work, since this problem is analyzed in the experiments later on.
* The authors say that the "reward misspecification problem is common in existing alignment pipelines, as evinced by relatively low reward accuracy." However, they have not yet made the case that this is true. There is little reference up until this point to "current alignment pipelines," and the claim that they suffer from poor accuracy is not backed up by evidence. Indeed, the vagueness of the claim extends to the fact that it's not clear what "accuracy" is being referred to. This is another instance in which making a crisp claim would really strengthen the paper.

### Questions
See the "weaknesses" section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Large language models (LLMs) are susceptible to adversarial attacks due to reward misspecification during alignment. This misspecification results in misalignment, where the models can produce undesirable or harmful outputs. The paper views such an issue through the lens of reward misspecification. The paper proposes **ReGap** to quantify this misspecification and introduces **ReMiss** to jailbreak the LLM, using reward misspecification as a guide to generate adversarial prompts. The paper conducts extensive empirical studies and justifies the effectiveness of the proposed method.

### Strengths
- The ReGap provides novel and valuable insights into the LLM misalignment, and the proposed ReMiss justifies the effectiveness of the revealed vulnerability.
- Extensive experiments have been conducted to provide a good insight into the components of the proposed method.
- The paper is generally well-written, with clear illustrations and tables.

### Weaknesses
Implementing the **ReMiss** involves stochastic beam search and requires significant computational resources compared to prompt-based methods like CihperChat [1] or language model-based methods like PAIR [2]. The significant resource requirement to conduct the ReMiss attack could be considered a defense, which reduces the proposed method's misused risk.
The idea of utilizing the reference model to jailbreak target LLM is interesting. However, ReMiss's effectiveness relies on the accurate modeling of rewards. Incorrect or missing reward modeling can lead to inefficiencies (Tab.3). 
- The ReGap only focuses on discovering suffixes to jailbreak the model. It would be more attractive to understand the effectiveness of the prompt-based jailbreak method.

[1] Yuan, Youliang, et al. "Gpt-4 is too smart to be safe: Stealthy chat with llms via cipher." arXiv preprint arXiv:2308.06463 (2023).
[2] Chao, Patrick, et al. "Jailbreaking black box large language models in twenty queries." arXiv preprint arXiv:2310.08419 (2023).

### Questions
1. Discuss the effectiveness of the prompt-based jailbreak method based on the proposed ReMiss.
1. Since one could use the same open-sources model as a reference model to attack, would ReMiss still be effective in using a small language model as the reference model to attack GPT?

### Soundness
3

### Presentation
3

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
The authors propose that vulnerabilities in large language models (LLMs) stem from reward misspecification during the alignment process. They introduce a metric, ReGap, to quantify this misspecification and a system, ReMiss, for generating adversarial prompts to exploit these vulnerabilities.

### Strengths
1.	The paper presents a novel perspective on the vulnerabilities of large language models (LLMs) by framing them as a reward misspecification problem, which adds depth to the discourse on model safety.
2.	The introduction of the ReGap metric is a significant contribution, providing a new method to quantify the extent of reward misspecification, thereby enhancing our understanding of how misalignment occurs.
3.	ReMiss effectively generates adversarial suffixes with low perplexity, indicating that these prompts can evade detection by perplexity-based filters, showcasing the practical implications of the authors' approach.

### Weaknesses
1.	The baselines in Table 1 have selected weaker settings from [1], with GCG and AutoDAN opting for the universal rather than individual settings, and AdvPrompter not utilizing the warmstart setting. What considerations led to these choices? Additionally, I could not find details on how the ASR measurements in Table 1 were obtained. Specifically, was the ASR for ReMiss derived from generating a suffix for each harmful instruction individually, or was it based on a single universal suffix? If ReMiss generates a suffix for each harmful instruction separately, then the data for GCG and AutoDAN using the universal setting would represent an unfair comparison.
2.	In Table 1, the system prompt for Llama2 is listed as empty; however, the data for GCG is significantly lower than the value of 56.0 reported in the original GCG paper. The original GCG study utilized the Legacy system prompt, which raises questions and contradicts the observation in Figure 6 that indicates the legacy prompt should be more challenging to successfully attack than the N/A prompt.
3.	Lines 304-308 state that the ASR metric used in Table 1 employs a keyword-matching method, which [2] notes has limitations that can "lead to false positive and false negative cases." If the authors wish to establish that ReMiss is state-of-the-art (L20), incorporating the GPT-4 judge—validated in [2] as a more comprehensive and accurate metric—would lend greater credibility to their claims. Furthermore, regarding the conclusion of state-of-the-art performance, was there a comprehensive comparison with all relevant methods, e.g. [3]?

[1] AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs

[2] Fine-tuning aligned language models compromises safety, even when users do not intend to!

[3] Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

### Questions
1.	Due to the aforementioned weaknesses, I recommend providing additional experimental details to support the claims in Table 1.
2.	For Figure 5, which suffixes were used for the transfer attacks on the tasks from HarmBench, and what method was employed to calculate the ASR shown in the figure?
3.	In Figure 6, the impact of suffix length on ASR shows considerable fluctuations with minimal consistent trends. How many samples were averaged to obtain the ASR for each length? Could the probable high variance be attributed to a limited number of samples?
4.	What is the overhead associated with ReMiss, including both training and testing phases?

I will reconsider my score if all these problems are adequately addressed.

### Soundness
2

### Presentation
2

### Contribution
3
