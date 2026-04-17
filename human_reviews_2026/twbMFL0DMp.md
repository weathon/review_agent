# Muon Outperforms Adam in Tail-End Associative Memory Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
The Muon optimizer is consistently faster than Adam in training Large Language Models (LLMs), yet the mechanism underlying its success remains unclear. This paper demystifies this mechanism through the lens of associative memory. By ablating the transformer components optimized by Muon, we reveal that the associative memory parameters of LLMs, namely the Value and Output (VO) attention weights and Feed-Forward Networks (FFNs), are the primary contributors to Muon’s superiority. Motivated by this associative memory view, we then explain Muon’s superiority on real-world corpora, which are intrinsically heavy-tailed: a few 'head' classes are extremely frequent, while a vast number of 'tail' classes are individually rare. The superiority is explained through two key properties: (i) its update rule consistently yields a more isotropic singular spectrum than Adam; and as a result, (ii) on heavy-tailed data, it optimizes tail classes more effectively than Adam. Beyond empirical evidence, we theoretically confirm these findings by analyzing a one-layer associative memory model under class-imbalanced data. We prove that Muon consistently achieves balanced learning across classes regardless of feature embeddings, whereas Adam can induce large disparities in learning errors depending on embedding properties. In summary, our empirical observations and theoretical analyses reveal Muon’s core advantage: its update rule aligns with the outer-product structure of linear associative memories, enabling more balanced and effective learning of tail classes in heavy-tailed distributions than Adam.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies why Muon often outperforms Adam in LLM-pretraining. The central claim is that Muon’s advantage concentrates in associative-memory components, specifically the Value/Output (VO) attention matrices and FFNs. Muon's update rule yields a more isotropic singular spectrum, which in turn improves learning on tail classes in heavy-tailed distributions. The authors support this with 
1. component-wise optimizer ablations in Transformers,showing that VO+FFN accounts for most of the validation-loss gains for Muon.
2. measurements of more isotropic singular-value spectra under Muon than under Adam. 
3. a knowledge-intensive QA task reflecting heavy-tailed data where Muon shows larger gains on tail examples.
4. a theoretical analysis of a one-layer linear associative memory under class imbalance showing Muon’s superiority.

### Strengths
1. **Focused component ablations.** The blockwise ablation is informative and directly isolates VO/FFN as primary beneficiaries, supporting the associative-memory lens.

2. **Heavy-tail perspective with tail-class gains**. The paper emphasizes tail performance and links it to spectral isotropy, providing a plausible causal chain: Muon → more balanced singular spectrum → better tail learning.

3. **Theory that matches the story.** The one-layer associative-memory model with class imbalance theoretically explains why Muon outperforms Adam under heavy-tailed class distributions, through the model is simple.

4. **Muon wins over Adam, and community does not know why!**
5. **LLM settings are diverse.** Covers both gated and non-gated 160M/700M LLM.

### Weaknesses
1. **Multiple kinds of heavy-tailness** The paper presents servral kinds of heavy-tailness: data, knowledge, parameter and update. There are evidence in the paper to connect the heavy-tailness of data with the heavy-tailness of knowledge, there are also evidence to connect the heavy-tailness of parameter with the heavy-tailness of update. However, it remains unclear how to connect the heavy-tailness of data and knowledge with the heavy-tailness of parameter and update in the LLM.
2. **$W_V$ as associative memory** Prior work rarely frames $W_V$ as an associative-memory parameter. Although the paper argues $W_V$ and $W_O$ play symmetric roles, that symmetry breaks under MQA/GQA. Treating VO jointly is understandable as a number of literatures[2][3] say that VO(Value-Output) are similar and can be considered together in practice even under the context of MQA/GQA, but a restatement is needed. 
3. **Observation 2 seems to be known in literature** The figure 4 in [3] already points out Observation 2 that Muon consistently yields more isotropic weight matrices with broadly distributed spectral energy than Adam.
4. **FFN over QK may be the result of parameter counting** FFN over QK may be the result of parameter counting, not because of associative memory.
5. **Reproducibility** The `model` are missing from the released code, making the work not fully reproducible.

[1] Lin, C. H., Gao, S., Smith, J. S., Patel, A., Tuli, S., Shen, Y., ... & Hsu, Y. C. MoDeGPT: Modular Decomposition for Large Language Model Compression. In The Thirteenth International Conference on Learning Representations. \
[2] Wang, J., Wang, M., Zhou, Z., Yan, J., & Wu, L. The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training. In Forty-second International Conference on Machine Learning. \
[3] Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., ... & Yang, Z. (2025). Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982.

### Questions
1. **Capacity control** If you equalize parameter counts (e.g., only optimize $W_{in}$ parameter in some depth by Adam), do Muon-over-Adam gaps persist in QK vs. FFN?
2. **Multiple kinds of heavy-tailness** Can you connect the heavy-tailness of data and knowledge with the heavy-tailness of parameter and update in the context of LLM?
3. **Logic gap on associative-memory mechanism** The paper does not yet provide strong causal evidence that Muon surpasses Adam *specifically* because of associative-memory effects. The superiority on the knowledge-intensive QA task might not be attributable to associative memory per se. VO+FFN may contribute more ('may' is due to weakness 2 and 4), but the mechanistic link to associative memory remains suggestive rather than conclusive.

Overall, the paper discusses a vital problem in the optimizer community. However, currentlymy tentative recommendation is borderline Weak Accept/Accept. I’m open to revising my score if these concerns are addressed.

### Soundness
3

### Presentation
4

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
The authors mainly did an abalation study of Muon. They showed two major and novel findings: 1. applying muon only on VO+FFN can recover the full muon training. 2. Muon narrows the head-tail performance gap when learning highly imbalance datasets. They also provided a preliminary theory to explain their findings.

### Strengths
1. The ablation study of this work is sound to me. Muon only works for VO and FFN. This finding is somewhat interesting and might inspire following works to promote Muon's ability over QK.
2. The heavy-tail findings is also interesting, which promotes the appliablity of Muon in practice.

### Weaknesses
**The novelty and contributions of this paper could be further improved.**
- Three observations are provided. Among these, the second observation is already presented in Liu et al. (2025). Though the differences between them are noted by the authors, such as Liu et al. (2025) conduced experiments on MoE models whereas this paper focues on Dense Models, these reasons cannot convince me about the novelty. 
- The first observations are good, yet I expect more analysis. A central question is "Why Muon is more effective on VO+FNN than QK", yet this paper didn't address this question. They attribute this phenomenon to the associative memory property of VO and FFN, but the causal relationship between these is not clear. Is the heavy-tail task ability of Muon relate to the effectiveness of Muon on VO+FNN?
- In addition, via the third observation, the authors further highlighted the connection between Muon and the associative memory. However, I also doubt that if the third observation can fully supports that Muon acquires knowledge more evenly than AdamW. Muon indeed prompts the performance on heavy-tail tasks, but it might restrict to knowledge acquring tasks. I expect more experiments across various tasks.

**The toy theory seems to be intuitive**
- The theory seems to be intuitive. Muon produces updates evenly spread over all directions compared to GD and SignGD. Then Muon must give even/uniform weight matrices, and thereby in this simple associative memory modeling, mitigating the unbalance learning problem.

### Questions
N/A

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
4

### Summary
The paper studies what transformer components benefit most from Muon as compared to Adam. The authors find that the main benefit comes from the VO and FFN blocks, which correspond to the associative memory stores. The authors use a heavy-tailed knowledge task to compare Adam and Muon and provide theoretical insights on one-layer models.

### Strengths
- The imbalanced knowledge experiment is a nice set up and the ablations over which layers most benefit from Muon are interesting
- Provides an interesting connection between optimizer performance and prior work on associative memory 
- The writing and research motivation are clear

### Weaknesses
- Experiments are limited to a synthetic heavy-tailed knowledge task. It would be interesting to see in addition a task that relies heavily on QK to support the claim that Muon is not as important to apply to QK.
- Language modeling results are limited to 160M parameter transformers. It would be interesting to see the main ablation (Muon only on VO+FFN) at a larger scale and performance measured on downstream tasks.

### Questions
- Do you have any intuition on why benefits from Muon seem to diminish with scale (/if this pertains to associative memory)?
- It seems like the Newton-Schulz application could be an interesting knob to turn here -- have you tried varying the Newton Schulz coefficients/iterations to vary how much of the spectrum Muon is capturing and test how this affects associative memory?
- What LR schedule did you use for Adam and Muon respectively?

### Soundness
3

### Presentation
3

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
The submission investigates the reason for the improved performance of Muon over Adam on transformers. It establishes empirically that the weights of a transformer architecture that benefit most from Muon-style updates are the non-key/query parameters of the attention modules and the parameters of the feedforward networks. To explain this behaviour, the authors propose an explanation based on different frequencies of "fact" representations in an associative memory module, and show theoretically that Muon-style updates lead to more uniform convergence across different frequencies.

### Strengths
To my knowledge, this is the submission that attempts to provide empirical evidence for the source of the benefit of Muon updates over Adam on language models. The identification of specific weights that benefit most from Muon and the corresponding explanations help us understand the difficulties of optimization in transformer-based architecture, and hopefully design better methods. The experiment on the synthetic dataset with imbalanced facts is well-designed an provides good evidence for the claim that Muon helps most on associative memories. 

I am generally positive about the submission. My initial score for the review is low due to the issues listed below. Those should be addressable with some revisions to the presentation and an additional experiment. I am expecting to increase my score if that can be added.

### Weaknesses
My main issue is with the presentation of the results, specifically in Figure 1, 2, 4 and §4.1.Those are dense and require a lot of care on the reader's part to understand what is being shown. This complexity does not appear necessary. A pass on the issues are detailed below could improve the clarity of the presentation significantly.

On the evidence, the main argument of the submission, that Muon helps with imbalance in "facts" stored in an associative memory. It appears difficult to distinguish between this explanation and the more general imbalance in token frequencies in text data for the success of Adam. See below of a possible additional experiment that would strengthen the argument of the submission.

The submission would be much stronger with those changes, and I would be happy to increase my score if they are addressed.

## Distinguishing between frequency imbalance in "facts" vs. general imbalance in text

The argument presented by the submission is that the benefit of Muon over Adam stems from an imbalance in the frequencies of different "facts" stored in an associative memory.
The controlled experiment with a synthetic dataset in §4.1 is a very nice illustration of this point. However, we could also expect this imbalance to stem simply from the distribution of tokens in text data, which follow Zipf's law. Those have been shown to lead to a better performance of Adam over SGD (Kunstner et al., 2024) primarily due to the update in the first and last embedding layers (Zhao et al., 2024; Zhang et al., 2024). One hypothesis could be that Muon helps on additional layers for the same issue of imbalanced tokens, not specifically facts in associate memories. Adam cannot correct these imbalances beyond the embedding layers because they are no longer axis-aligned, while Muon could. 

The main difference I am drawing between those two hypotheses is the whether the main difficulty is due to an imbalance in token frequencies or an imbalance in the frequencies of "facts" to be stored in associate memories. As facts are also represented as tokens, there is some overlap between those ideas, but there is a way to disentangle these two explanations. If the arguments of the submission is correct, that Muon target imbalances in "facts" and associate memories beyond mere imbalance in token frequencies, I would expect the performance gap between Muon and Adam to decrease as the fact imbalance is reduced. The synthetic dataset prepared by the authors could be modified to exhibit various levels of imbalance, varying the imbalance continuously between the approximate power-law distribution in Figure 3 to a uniform distribution. The imbalance in token frequencies should still follow a power-law (assuming the different templates are rich enough to approximate natural text, which should be checked), but the imbalance in "facts" would be controlled. If the performance gap between Muon and Adam shrinks as the "fact" imbalance becomes more uniform, this would give a lot more strength to the proposed fact-based explanation.



## Presentation of the results

The specific recommendations below are only one possible way to improve the presentation, meant to help communicate the difficulties I encountered in understanding the results. The authors know their work best and might have other ideas on how to improve the presentation that are more appropriate. 


**Figure 1**:  
There are too many lines to be able to easily tell which combination of architecture/optimizer is performing best.
This detailed plot could be useful to readers wanting to see the behaviour over time, 
but the main point of Figure 1 would be better illustrated by a simple bar plot of the final performance, 
using more legible distinctions (colors/makers) or having the description of the setup directly above the bar. 

The figure also combines gated and non-gated variants of the architecture, 
but it is not clear why this distinction is important for the point being made.
The plot could be made much simpler by focusing on one of the two variants, 
and pointing the reader to "the same happens on the gated version, see Appendix X".

**Figure 2**  
suffers from a similar issue.
The plot shows 4 metrics measuring the spread of the eigenvalues, 
some which should be small while other should be large, 
for 2 weights (VO, $W_{\mathrm{out}}$) on 2 architectures (gated, non-gated).
This makes it hard to extract the main point of the figure at a glance.
A simpler variant of the figure could focus on one architecture and one metric,
relegating the others to the appendix with the comment that a similar behaviour is observed,
or use this space to show the full distribution of the spectrum over time.

An interesting addition to Figure 2 could be to show the QK weights. The Muon update is apparently less helpful for those weights; does the spectrum give a hint as to why?

**Figure 4 (b, c)**:  
I had a hard time understanding those figures, and what those metrics tell us about the update. 
I am not sure which is the best way to improve the presentation, but here are things I found confusing: 

The figure 4b is implicitly parameterized by a step-size which is being varied. This can only be understood from the main text (not the caption), which should be fixed. Even with this information, it is not easy to see the "step-size" on the plot. This might be because the lines "start" at the bottom right ($\eta = 0$ is at $(10^{.8}, 0)$) and go to the top left as $\eta$ increases, going against the standard left-to-right/top-to-bottom reading direction. Flipping the x and y axes might help. Another option could be to use a plot with 2 y-axes (or two subplots on top of each other) that would show the $\Delta$ vs. step-size and the corresponding loss vs. step-size. The same treatment could be applied to Figure 4c (with number of updates as the x-axis). Currently, 4b and 4c have a different experimental setting (varying step-size vs. varying time) but it is not clear from the figures/caption that they differ in this way.

**Theoretical results in §4.2**  
Assumption 4.2 seems overly complex for its uses in Theorem 4.3. The parameters $\alpha$ and $\beta$ appear to impact the results only through the ratio $r(\alpha,\beta)$. Since it happens to be $\frac{\min_i p_i}{\max_i p_i}$, wouldn't it be easier to just define the degree of imbalance as $r = \min_i p_i / \max_i p_i$ directly, and avoid the extra notation of $\alpha, \beta$? If not, writing the overall loss after Assumption 4.2 as a function of $\alpha, \beta$ might be helpful.

The only place where I see their use is in the proof for Adam, but the range of behavior observed under the $(\alpha,\beta)$ assumption would also be observed on a model with imbalance ratio $r$ since it's strictly less restrictive. If I'm missing something and the $\alpha, \beta$ are important for the proof, they should still be mentioned, but an explanation of what $r(\alpha,\beta)$ is should be given around Assumption 4.2 to help the reader.

Could some of the notation be simplified by assuming that the frequencies are sorted? For example $\min_{k\in[K]}$ and $\max_{k\in[K]}$ in Eq. 4.1 could be replaced by $p_K$ and $p_1$ if we assume the probabilities are decreasing.

The description of the stopping condition would also benefit from a slightly more wordy explanation. The start of §4.2 takes a while to parse. Something like "For each optimizer, we take one step with step-sizes $\eta$, selected such that the accuracy of the most frequent class reaches $1-\epsilon$. We then look at the accuracy achieved on the least frequent class."


## Additional references 

The submission should cite and discuss the following works which have done similar ablation studies on Adam to justify its performance on language models and identify that they help most with the first and last embedding layers. This does not detract from the contributions of the submission, but those works should be acknowledge.$

- Deconstructing What Makes a Good Optimizer for Language Models
  Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, Sham Kakade
  https://arxiv.org/abs/2407.07972
- Adam-mini: Use Fewer Learning Rates To Gain More
  Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P. Kingma, Yinyu Ye, Zhi-Quan Luo, Ruoyu Sun
  https://arxiv.org/abs/2406.16793


## Minor/Typos

- "intrinsically heavy-tailed: a few classes (tail classes) appear far less frequently than others"
  Is this a typo? Presumably if only a few classes are rare, their impact on the overall loss is small and Muon would not help much.
- It is not clear where Theorem 4.3 ends. Could the paragraph breaks "For GD, ... For Muon, ... For Adam, ..." be placed in an itemize block to help that?
- "highly sensitive" as a description of Adam's behavior in Figure 2 seems exaggerated. "More sensitive" would be appropriate.

### Questions
**Why does Muon help more on the VO/FFN than on the QK weights?**  
This is the main distinction found in the first set of experiments. 
The theory tries to explain why the VO and FFN weights are helped by Muon by drawing on the connection to associative memories, 
but I have some reservations about this explanation.

The QK weights are also part of the attention mechanism and form an outer-product based on the representation of the tokens. Why would the associative-memory explanation not apply to them as well? The FFN weights do not seem to follow this outer-product structure, so why is the associate-memory explanation relevant for them?

It might be that the associative-memory explanation is most relevant for the VO weights and do not explain the benefits on FFN weights or the reduced benefits on QK weights.
But if that is the case, this should be acknowledged more explicitly in the submission.

Beyond this theoretical explanations, it would be helpful for future work if the authors could look at the experiments and data they have collected to see if there is some empirical difference between the QK weights and the VO/FFN weights that could explain the different behavior, for example a different spread in the spectrum of the singular values of the inputs of each layer? This is not strictly necessary but even a speculative explanation (based on some empirical observation) might be helpful for future work,

**Why the focus on Gated vs. Ungated FFN?**  
Is there a-priori a reason to believe that Muon would behave differently? 

**How is the step-size set in Figure 1?**  
It is not clear to me whether the results presented in Figure 1 use the same step-size for both Adam and Muon, 
or use a tuned step-size for each optimizer. Could the authors clarify this point?

The later would seem more appropriate as it is not clear that the same step-size is optimal for both optimizers, especially when used at the same time for different weights.
Although it would then require a comparison against using Adam for both subset of weights with a different step-size for each subset to account for the possibility that the improvement comes from using different step-sizes.

### Soundness
2

### Presentation
2

### Contribution
3
