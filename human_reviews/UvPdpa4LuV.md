# Protein Language Model Fitness is a Matter of Preference

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Leveraging billions of years of evolution, scientists have trained protein language models (pLMs) to understand the sequence and structure space of proteins aiding in the design of more functional proteins. Although they have shown ability to improve efficiency in engineering, it remains unclear under what conditions they will succeed or fail. We aim to predict the circumstances in which pLMs can successfully perform zero-shot fitness estimation. Our work demonstrates the trends observed over hundreds of deep mutational scans across multiple different fitness objectives. We find that the likelihood, or abstractly, implicit preference of a certain protein sequence imbued during pretraining is predictive fitness prediction capabilities. Both over-preferred and under-preferred wild type sequences harm performance. Generating a causal link between training data and likelihood, we show a power law tail over what data increases protein likelihood which is tied to training sequence homology. Lastly, proteins of low likelihood can be remedied by unsupervised finetuning. In sum, the zero-shot fitness estimation abilities of pLMs can be predicted by the likelihood of the engineered sequence, thus suggesting when pLMs should be deployed in protein maturation campaigns and a way to improve their performance under circumstances of low likelihood.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper examines how protein language models (pLMs) perform in predicting protein fitness, specifically zero-shot fitness estimation. It argues that a model's success in such tasks can be predicted by the likelihood or implicit preference of a protein sequence learned during pretraining. The study finds that both high and low preferences for a sequence can negatively impact prediction accuracy, but using influence functions to trace these preferences back to specific training data reveals a power law distribution in influential data points. The research suggests fine-tuning on underrepresented sequences as an approach to improving prediction capabilities and provides insights into how training data shapes model performance, guiding more effective use of pLMs for protein engineering.

### Strengths
1) The paper offers a perspective by framing pLM behavior in terms of implicit preferences derived from training data, shedding light on limitations and potential improvements for zero-shot fitness prediction.

2) The paper shows a rigorous analysis of pLM performance across multiple datasets and uses influence functions effectively to understand the causal effects of training data.

3) The structure is clear, with theoretical explanations, empirical validations, and practical recommendations for pLM deployment in protein engineering.

### Weaknesses
1) The reliance on influence functions has known limitations, such as sensitivity to model hyperparameters, which could affect the validity of conclusions.

2) The paper heavily focuses on likelihood as a key predictor of performance without sufficiently exploring alternative metrics or features that may also contribute to zero-shot success.

3) The paper's experimental setup could be enhanced by including more diverse datasets or varying conditions to test the robustness of the evo-tuning technique.

### Questions
1. Could you elaborate on how the choice of loss function and measurable value impacts the influence function results? Specific examples could enhance understanding.

2. Could the authors provide a more extensive comparison across different architectures (e.g., autoregressive vs. masked models) to see if the observed trends are generalizable?

3.  Would results vary significantly with different pretraining datasets, especially those curated with diverse or balanced sequence representation?

4. Have the authors considered methods to explicitly model epistatic interactions in conjunction with the proposed likelihood-based framework? This could potentially address the limitations associated with linear fitness predictions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates how protein language models (pLMs), which learn from evolutionary data, can predict protein fitness in zero-shot scenarios. The study highlights that the likelihood or preference for certain protein sequences, as captured during pretraining, significantly affects the performance of these models. Key findings are as follows. (1) High or low sequence likelihoods can both impair the predictive performance of pLMs. Sequences with moderate likelihoods perform better in predicting mutation effects. (2) The training data's composition affects sequence likelihoods, and influence functions reveal that the impact of individual data points follows a power law, with homology playing a key role in driving model preference. (3) Fine-tuning (evo-tuning) on homologous sequences can enhance performance for low-likelihood sequences but may degrade it for high-likelihood sequences. This approach addresses the limitations in traditional pLM deployment.

### Strengths
1-The paper provides a detailed investigation into the performance of pLMs across various fitness prediction tasks, utilizing data from hundreds of deep mutational scans. This breadth of analysis allows for a more generalizable understanding of pLM behavior, as it encompasses diverse proteins and fitness objectives. By examining both high- and low-likelihood sequences, the study offers insights into the conditions under which pLMs perform well or poorly, making the findings applicable to a wide range of protein engineering challenges.

2-The study employs influence functions to causally link training data with model outputs, uncovering how individual training sequences impact the likelihood assigned to protein sequences. By showing that the influence follows a power law distribution and is closely tied to sequence homology, the paper offers a way to identify critical data points that drive model preferences, guiding the selection of training data and finetuning strategies.

### Weaknesses
The study investigates only two models, ESM-2 and ProGen-2, which, while both powerful and relevant, represent a narrow slice of the available landscape of protein and sequence-based models. Including a broader set of models, such as ProtBERT, ProtGPT-2, could lend further robustness to the findings. Expanding the selection of models would provide a more comprehensive assessment and potentially lead to stronger, more generalized conclusions about model performance across biological tasks.

Line 230, Section 4.2 is not at the Appendix.

### Questions
Line 347, why can only 10K protein sequences from UniRef50 approximate the distribution of training data of EMS-2?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper studies masked language models used for zero-shot prediction of the effect on fitness of a mutation to a protein. It argues that biases in protein LM pretraining data may result in a skew in the model's likelihood scores, which are observed to correlate with a model's zero-shot fitness prediction accuracy. The paper draws on influence functions to motivate revisiting an old fine tuning technique (evo-tuning) and shows that this increases performance on a standard suite of deep mutational scanning (DMS) datasets.

### Strengths
The paper's core observation, that there is a correlation between the overall likelihood of a protein sequence and the accuracy of zero-shot fitness prediction for variants of the sequence, is interesting. I think a number of proteins+ML researchers will be inspired by this to think about potential solutions.

The paper offers a concrete modeling suggestion that provides mild improvements on the DMS datasets. The technique, evo-tuning, was contributed in prior work, however. It just hadn't been revisited for these DMS problems.

### Weaknesses
I raise a number of questions below. My chief concerns are that (1) the presentation on a new algorithm for constant-time computation of sequence likelihood has some serious flaws, the (2) influence function analysis occupies a lot of space in the paper and relates to some cool recent work, but doesn't drive the paper's narrative enough to being included, and (3) the overall story about 'preferences' could be reframed in terms of a basic story about underfitting vs. overfitting during pretraining.

### Questions
======
The U-shaped relationship between fitness prediction accuracy and sequence likelihood is interesting, but I struggled to see a concrete argument about why this is reflecting some specific issue regarding the structure/balance of protein LM pretraining datasets instead of just a basic impact of undertraining/overtraining. For wildtype sequences with low likelihood, these may come from regions in sequence space with low frequency in the pretrianing data, which means that model is under-trained for this part of sequence space, and similarly it is over-trained for sequences with high likelihood.  Perhaps I'm missing something. What part of this observation is specific to proteins and why does it rely on a notion of 'preference'?

I understand that 'preference' is a trendy word due to things like DPO and ELO, but in this paper I found it to be a fairly lightweight rephrasing of the likelihood of a sequence. There is no notion of external supervised preference data (like DPO uses) or in some global ordering like in ELO. In Ding et al, 'preference' was important, because it provided a way to compare neighboring protein sequences, and the paper had a clear story about a confounding factor (species) that drives this preference. 

======
I found sections 4.1 and 4.2 (on predicting the pseudo-likelihood of a sequence in a single pass) to be highly confusing.

First of all, I found the theorem statement to be poorly defined. It says "Under a mask-consistent masked language model that sets masked tokens to a random token with probability alpha and keeps them unchanged with probability beta…". A model doesn't 'set' a token. Also, what does it mean for a model to 'keep' a token 'unchanged.' Do you mean something along the lines of "given a masked language model that was trained on data with the following distribution…"?
Finally, you need to define what 'mask consistent' means in the actual theorem statement.

Besides the details of how the theorem is presented, I actually don't understand the core statement either. Overall, the statement is of the form for 'task of computing quantity A, we have a new algorithm that is O(1), whereas prior methods for computing A were O(L)'. However, what exactly is A and do the new and old approaches provide the exact same output for all possible models, or only when models have maximized the likelihood of training data from a particular distribution? 

My impression is that the baseline O(L) approach would compute the pseudolikelihood by taking L forward passes of the model, where each pass replaces the ith token in X with [MASK] in order to compute the probability of x[i] given the rest of X. However, if I understand correctly, your approach passes X into the model with no mask. Given that the actual inputs to the model are different, which means that the activations are different, it appears to me that the approaches would provide numerically different outputs, albeit they may be similar if models have been trained for sufficiently long on a certain training distribution. I'm assuming that this is what is reflected in the lack of perfect correlation in Fig 2. This level nuance is certainly not present elsewhere in the text of sec 4.1 and 4.2.

Also, why is the single-pass approach actually necessary in your experiments? In my understanding, there are roughly a million distinct protein sequences in ProteinGym, so you would need to do one million forward passes of the model. Given that you had the compute resources to independently evo-tune the model on 200+ datasets, I'm assuming that one million forward passes was doable.

Also, you should add a discussion of the relationship between your technique and  "Predicting a Protein’s Stability under a Million Mutations."

 ======
I found the discussion of protein 'fitness' to be too vague. The paper focuses on DMS datasets that measure different attributes of a protein, such as stability, binding, etc, yet it uses assay-independent models based on evolution that predict a single number for a protein sequence, regardless of what attribute of the protein is being measured. Therefore, there is a fundamental ceiling on DMS accuracy for these assay-independent models, since different attributes of a protein can be anti-correlated in practice (e.g. an enzyme's catalytic activity and stability). 

Note that different types of fitness functions may be easier to predict using ML than others (e.g., I can see an argument that stability would be easier to predict than catalytic activity). This speaks to a potential confounder in your analysis: there is a correlation between the protein sequence's content and what kinds of DMS studies it appears in. What if, for example, the correlations you are seeing are due to the fact that low-likelihood proteins are from the sort of species where the DMS datasets are based on high-level observations about organismal survival, instead of low-level molecular biology measurements like binding? Perhaps predicting organismal survival is a fundamentally more difficult task.

======
I think influence functions are cool and I enjoyed seeing that you used them to study protein LMs, but I don't think the inclusion of the influence function results drives the paper's story enough to be included. Section 5.1 just demonstrates that the established power-law distribution of influence function values appears for protein LMs (extending prior work doing this for other LMs. I found these sentences confusing: First, we reproduce the power law tail observed for traditional LMs in Grosse et al. (2023), suggesting similar data dynamics between pLMs and LMs. From a biological perspective, Qin & Colwell (2018) finds a power law tail in the covariance of phylogenetic protein systems, which might lend a way to understand this result." Can you please explain the connection to Qin and Colwell in more detail? Besides the fact that the paper is about proteins and discusses power laws, is there more of a connection?

Can you elaborate on the result of sec 5.2? How is this observation actionable? How does it inform the other parts of the paper?

====
I didn't understand figure 1B. Points i, ii, and iii on the left reflect proteins with different likelihood scores for the wildtype. On the right, these have different sized boxes for the set of 'plausible' mutations. I'm assuming the set of plausible mutations is the set of mutations that don't significantly decrease the sequence's likelihood. Why would the overall likelihood of the wildtype be correlated with the size of this set?


====Update after the author's response period====
I appreciate the healthy back-and-forth we had. The new framing of your approximation algorithm is much more sound. I have raised my score. If the paper is accepted, there is not official mechanism to enforce that you change the exposition to reflect our discussion. However, it would be greatly appreciated that you switch to the new exposition, as it is more technically correct (in the sense that it is up-front about the approximation being an approximation), and more user-friendly to readers.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the effectiveness of Protein Language Models (pLMs) in predicting protein fitness in a zero-shot setting. In essence, it explores whether pLMs can capture treu biological patterns or artifacts of the training data. It analyzes hundreds of deep mutational scans through the likelihood-based preference and finds that both overly high and overly low sequence likelihoods decrease prediction accuracy. It further leverages influence functions to trace the origins of these likelihoods and proposes "evo-tuning" as a remedy to improve pLM performance on low-likelihood proteins.

### Strengths
This paper introduces a novel perspective by framing the predictive capacity of pLMs as a "preference" rooted in sequence likelihood. This framing is innovative, though this kind of framing is already common in RLHF for LLM. 

It is among the first to rigorously employ influence functions to trace the impact of training data on pLM predictions, effectively combining causal inference with protein sequence modeling. 

There are too many methods and tricks for improving performance on different kinds of datasets which simply apply most state-of-the-art language model methods and tricks into protein but lack more careful inspection into the problem. This paper thinks more deeply into the question and carefully examine the pLMs through implicit preference.

### Weaknesses
The experiment would be better if it includes more datasets to fully support some claims, for example, evo-tuning pLMs could improve fitness prediction. 
Also, I am wondering if Bert-like esm-3 or other smaller Bert-like pLMs may be examined.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
4
