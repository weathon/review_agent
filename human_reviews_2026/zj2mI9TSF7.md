# Mechanistic evaluation of Transformers and state space models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
State space models (SSMs) for language modelling promise an efficient and performant alternative to quadratic-attention Transformers, yet show variable performance on recalling basic information from the context. While performance on synthetic tasks like Associative Recall (AR) can point to this deficiency, behavioural metrics provide little information as to \textit{why}---on a mechanistic level---certain architectures fail and others succeed.
To address this, we conduct experiments on AR, and find that only Transformers and Based SSM models fully succeed at AR, with Mamba and DeltaNet close behind, while the other SSMs (H3, Hyena) fail. We then use causal interventions to explain why.
We find that Transformers and Based learn to store key--value associations in-context using induction. By contrast, the SSMs seem to compute these associations only at the last state using a single layer. We further investigate the mechanism underlying the success of Mamba, and find novel evidence that Mamba \textit{does} implement induction: not via the SSM, but instead via short convolutions.
Further experiments on a new hierarchical retrieval task, Associative Treecall (ATR), show that all architectures learn the same mechanism as they did for AR. Furthermore, we show that Mamba can learn Attention-like induction on ATR when short convolutions are removed.
These results reveal that architectures with similar accuracy may still have substantive differences, motivating the adoption of mechanistic evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper performs a mechanistic study on how SSMs and Transformers solve associative recall (AR) and associative treecall (ATR) tasks. The authors use causal interventions as the main tool. They find that for AR, Transformers and Based use induction mechanism, whereas SSMs such as Mamba rely on the convolution component --- without which Mamba fails to learn AR. They also find that similar mechanisms hold for ATR, except that Mamba can also use the Attention-like two-layer induction mechanisms in absence of the convolution component.

### Strengths
1. It is an interesting idea to use causal intervention as a mechanistic understanding tool to compare SSMs with transformers

2. The paper is clearly organized, with illustrative figures supporting the main claims.

### Weaknesses
1. Some claims are not well supported. For example, in line 75-76, the authors claim that ``(Mamba) fails to learn AR at all without this (short convolution component)'', but the experiments (Fig.4) only investigate the Mamba convolution kernel size no greater than $4$, unclear whether Mamba fails with long(er) convolution kernel. 

2. Lack of architecture details and analysis. Although the authors provide source code, the paper lacks concrete definitions of the architectures used for study, and thus provides little explanation why certain mechanisms present in one architecture but not the others.

### Questions
1. Sec 4.4 Convolution moves information to the next key (line 317-322): This claim arises from the results in Fig.5, where restoring next key at layer 0 gives the best result. But why does the restoration necessarily imply the convolution component moves information to the next key? If my understanding of restoration is correct (which is not very precisely defined in paper, but loosely on Fig.2), restoration means we can arbitrary set any token at the next-key position. Then one easy solution is to set the next-key to be the corrupted key, which does not rely on convolution moving information to the next key. In addition, if the next key is arbitrary far away from the previous key, a short-convolution by definition cannot move information to the next key. Can the authors clarify?

 2. I appreciate the interesting use of causal intervention to mechanistically evaluate the AR mechanisms, but the findings seem to mostly corroborate the existing known results (e.g., short convolution are key to association in SSMs as mentioned by the authors in related work), including some provable solution mechanisms in missing related works [1] [2]. Does the mechanistic study offer novel insights (e.g., identify unknown mechanisms, provide more fine-grained analysis on how  the AR mechanisms interact with the choice of architecture and optimization set-up)?

 References:

 [1] Bietti et al. "Birth of a transformer: A memory viewpoint." NeurIPS 2023.

 [2] Huang et al. "Understanding Input Selectivity in Mamba: Impact on Approximation Power, Memorization, and Associative Recall Capacity." ICML 2025.

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
5

### Summary
The paper at hand presents a study of associative recall (AR) and associative tree recall (ATR) on various sequence models: from simple convolutions to Mamba and Transformers. The authors claim these models use different methods to solve AR, which they study by intervening on input sequences.

### Strengths
Studying basic performance tasks on transformers vs new sequence models is interesting. The authors train a large set of models and carry out the analysis also on ATR, which is a less common setup that I did not know before, but I find quite insightful. 
The paper is also pleasant to read and schematic, which helps deliver the message. Plots are clear.

### Weaknesses
The findings in the paper have quite a few overlaps with previous works:

- Zoology: https://arxiv.org/abs/2312.04927
- Based: https://arxiv.org/abs/2402.18668
- Convolution-augmented transformers: https://arxiv.org/abs/2407.05591
- Revisiting Associative Recall: https://openreview.net/pdf/f7e9f322ba15e88dcc818ab70866648650a5e319.pdf
- H3 : https://arxiv.org/pdf/2212.14052

In light of the findings in the papers above, I did not find the paper very surprising. The authors cite all the papers above, but do not discuss or compare their results to previous literature:

1) performance on AR is reported in (1) "Zoology", and (2) with a much finer LR grid in "Revisiting Associative Recall". In the latter, the authors claim that LR sensitivity is a big issue in Mamba and Hyena. The authors do not discuss this issue nor seem to take any action to perform careful evaluations. Additionally, betas = (0.9, 0.999) in Adam is the default, but it is not what people typically use in language models. beta2 is too high (0.95 is default in many repos). How do you make sure your results depict what "each model can achieve"?

2) It is a bit hard to follow what the authors mean by "induction". I think, despite 1k years of philosophical debates, the definition can be a bit arbitrary. I was confused while reading your claims. Please define what you mean! I had an approximate idea at the end of the paper, but this is not formal enough.

3) The reader is not prompted to read the figures correctly, and the tasks are not well defined. Let us consider Figure 2: if you change A to B, the eval always returns A, and 2 returns "???". This is very unclear to me. I do not understand the ground truth and I find little explanations in the text. Furthermore, Figure 3 has a similar issue: you never formally define any of the tasks; what is "Restored @ Key"? You discuss how these tasks resemble interventions, but I cannot determine their respective importance. What are they individually supposed to test?

4) The role of convolution has been studied thoroughly, in "power of convolution-augmented transformers" but also in the H3 paper. In H3, they place a shift + gate exactly to enhance recall (https://arxiv.org/pdf/2212.14052, Fig 1). Again, I do not find surprising the claim about convolution, given also the induction head standard mechanism where the first transformer layer indeed represents a shift (e.g. proof in the Jelassi paper on the copy task, and Figure 1 above).

All in all, I do not see the level of novelty here to be at the level of acceptance. I ask the authors to please specify which new insights are presented in the paper, and to clarify what their causal interventions are precisely testing.

### Questions
See weaknesses.

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
The paper introduces a generalized Associative Recall task named Associative Treecall and provide mechanic interpretability for the success of Mamba.

### Strengths
1. comprehensively and mechanically evaluate common linear models and transformer.
2. clear and good writing.

### Weaknesses
1. mechanic metric are not used to provide guidance for the design of architecture but can only help understanding. thus its use is limited.

### Questions
1. can you provide an example of how mechanic metric helps the design of architecture?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces mechanistic evaluations (causal interchange interventions) to analyze the impact of different architectural components in solving Associative Recall (AR) task, along with a new retrieval task, called Associative Treecall (ATR). The authors show that on AR, Attention and Based architectures exhibit induction while Mamba and DeltaNet perform direct retrieval. The main ablation revolves around the size of the convolution kernels, where the ablations show that short convolution kernels are critical for AR on Mamba and Based architectures. With a detailed analysis on the two tasks, the authors claim that comparable accuracies can hide different internal mechanisms.

### Strengths
- The intervention protocol pinpoints the mechanism - more specifically, the QKV positioned restorations at layer input/outputs disambiguate induction from direct retrieval.
- The kernel size ablation study demonstrates that short receptive-field convolution kernels implement the association needed for AR. 
- With the new task (ATR), the paper verifies that the mechanism transfer to a harder task without positional dependence.

### Weaknesses
- The experiment results seem quite noisy. For figure 4, in particular, it is unclear why model dim 128 and Mamba Conv. = 2 suddenly fails. Also, it is unclear why Restored @ Key value for this configuration is notably high compared to other entries. A similar trend is observed for Figure 5, where model configurations that perform well can drastically fail, (~0% accuracy) with learning rates that are slightly modified. This result, unless it can be justified empirically or theoretically, raises a serious concern whether the results are valid conclusions or are due to insufficient sweep in the hyperparameter space.
- Potential confounds with the parameter count on the models. The models have different parameter counts and FLOPs, and hence presenting the results in one of these dimensions would be much more valuable.

### Questions
- Please correct me if I misunderstood, but ATR induces unequal pair frequencies, which may induce a generative prior, allowing the corrupted-key accuracy to stay above 0 in most cases. 
- Replacing Based's short convolution kernel with an implicit long convolution kernel significantly harms AR. Why might this be? Providing heatmaps for the long convolution kernel case to show where association fails could help better understand this phenomenon. Also a more detailed discussion about this phenomenon would strengthen this manuscript.
- The authors note that Hyena includes a short convolution kernel but it performs poorly on AR. This signals that a convolution kernel is not sufficient to solve this task of AR. Which downstream component could be preventing the convolution kernel from implementing the association step?

### Soundness
3

### Presentation
3

### Contribution
2
