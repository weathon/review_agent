# Emergent Mixture-of-Experts: Can Dense Pre-trained Transformers Benefit from Emergent Modular Structures?

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 8, 3

## Abstract
Incorporating modular designs into neural networks demonstrates superior out-of-generalization, learning efficiency, etc.
Existing modular neural networks are generally $\textit{explicit}$ because their modular architectures are pre-defined, and individual modules are expected to implement distinct functions.
Conversely, recent works reveal that there exist $\textit{implicit}$ modular structures in standard pre-trained transformers, namely $\textit{Emergent Modularity}$.
They indicate that such modular structures exhibit during the early pre-training phase and are totally spontaneous.
However, most transformers are still treated as monolithic models with their modular natures underutilized.
Therefore, given the excellent properties of explicit modular architecture, we explore $\textit{whether and how dense pre-trained transformers can benefit from emergent modular structures.}$
To study this question, we construct $\textbf{E}$mergent $\textbf{M}$ixture-$\textbf{o}$f-$\textbf{E}$xperts (EMoE).
Without introducing additional parameters, EMoE can be seen as the modular counterpart of the original model and can be effortlessly incorporated into downstream task tuning.
Extensive experiments on various downstream tasks (visions and languages) and models (from 22M to 1.5B) demonstrate that EMoE effectively boosts in-domain and out-of-domain generalization abilities.
Further analysis and ablation studies suggest that EMoE mitigates negative knowledge transfer and is robust to various configurations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores whether dense pre-trained transformers can benefit from emergent modular structures, which are spontaneously formed during the early pre-training phase. The authors propose Emergent Mixture-of-Experts (EMoE), a method that externalizes the emergent modularity into explicit Mixture-of-Experts without introducing additional parameters. EMoE is evaluated on various downstream tasks, models, and configurations, with a total of 1785 models fine-tuned. The results show that EMoE effectively boosts in-domain and out-of-domain generalization abilities, mitigates negative knowledge transfer, and is robust to various configurations. It also demonstrates higher data efficiency of up to 20% compared to the vanilla fine-tuning and competitive performance with the strong baseline GMoE.

### Strengths
1. This paper appears to be the first to systematically investigate the emergent modular structures within the Feed-Forward Networks (FFNs) of pre-trained Transformers, contributing valuable insights to the field.

2. The authors propose a straightforward yet effective method for Mixture-of-Experts (MoE) initialization that outperforms existing solutions across both vision and natural language processing tasks, demonstrating its practicality and usefulness.

3. The experimental setup is comprehensive, with a total of 1785 models fine-tuned, showcasing the thoroughness of the investigation and the robustness of the proposed method in various scenarios.

### Weaknesses
1. The choice of N and k in EMoE differs significantly from existing Mixture-of-Experts (MoEs) models, which may potentially diminish some of the advantages typically associated with MoEs. It would be beneficial to further explore the implications of these choices and their impact on the performance of the method. (Refer to Question 1)

2. Some aspects of the experimental settings could be clarified or expanded upon to ensure a thorough understanding of the results and their significance. (Refer to Question 2, 3)

3. The presentation of results in tables and figures could be improved to enhance readability and better convey the key findings of the paper. Clearer visual representations would help readers grasp the main takeaways more effectively. (Refer to Question 4, 5)

### Questions
1. In previous MoE works, k is often set to {1, 2} to achieve high time efficiency, while the value of k in EMoE is {16, 32}. How does this choice affect time efficiency? It would be helpful if the authors could provide a comparison in terms of GMACs for Transformers, MoEs, GMoEs, and EMoEs.

2. The results demonstrate that EMoE outperforms EMoE-learn when the number of fine-tune iterations is the same. This might be attributed to the gate of EMoE having a better initialization point and converging faster. It would be interesting to see the performance of EMoE-learn with longer finetune iterations compared to EMoE.

3. In GMoEs, auxiliary loss functions (as described in Appendix C.4 of the GMoE paper) are employed to achieve good performance. Were these loss functions used for the GMoE baselines in this study? Additionally, could these loss functions improve the performance of EMoE?

4. The font size of "Office" and "Terra" in the figures appears to be smaller than that of the other labels. It would be helpful to ensure consistency in font sizes across all labels.

5. The right figure in Figure 4 seems to lack data points. Could the authors clarify if this is intentional or if there might be an issue with the figure's presentation?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a method to leverage the modularity inherent in pretrained backbone models, demonstrating improvements in both in-domain and out-of-domain datasets by inducing modularity in the pretrained backbones. The methodology clusters vectors in lower Feed-Forward Networks (FFN) into multiple experts, activating a few experts per input, which in turn specializes these experts for different types of inputs.

### Strengths
1. The proposal to exploit modular structures within pretrained transformers presents a novel approach towards enhancing model performance.
2. The paper provides an interesting analysis illustrating the emergent structure in the FFN layers demonstrating the modular nature of the pretrained backbones.

### Weaknesses
1. The improvements noted over Vision Transformer (ViT) and (GMoE) are marginal and fall within the noise range for the provided datasets, as evidenced by the data in Table 8, Appendix 4.1.
2. The method proposed is derived from previous work by Zhang et al. (2022 b), and does not represent an original contribution from the current paper.
3. The paper's method shares similarities with the GMoE method, yet the proposed method achieves comparable performance.
4. Despite the method increasing wall-time by 10% and demanding more memory as stated in Appendix A.3, there is no justification provided to substantiate why the method is beneficial, especially given the negligible performance improvements and the added overhead in hardware implementation.

### Questions
1. Could you elaborate on why the EMOE-learn approach did not outperform the avg-k method?

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper uses a recent method (MoEfication; Zhang, 2022) to convert a pretrained transformer-based model into a mixture-of-experts model and evaluates the transfer performance of the resulting model after fine-tuning on a variety of tasks. It finds improved performance and argues that that this is due to deactivating neurons with negative transfer effects.

### Strengths
The paper presentation is clear and the investigated research question is interesting and potentially impactful. The experiments are conducted on a wide range of benchmarks and performance appears to be good.

### Weaknesses
The results presented in the tables, spider plots and bar plots in the main text are missing standard errors. Unfortunately this makes it very difficult to judge whether improvements are statistically significant, especially since the means are often very close across methods. I would suggest to add these.

Since the paper uses an existing method, the question of novelity is a bit more subtle. As the authors explain, the main difference of this paper is the focus on understanding transfer performance instead of improving inference efficiency. This puts a stronger weight on the analysis and understanding how EMoE might improve performance. I find some of the evidence presented in the Analysis section not entirely convincing. In particular, I am concerned that the claim "Expert Selection Changes During Training" is not well supported in which case the interpretation of the effectiveness due to modularity (rather than for example pruning) might not be justified. Please see my questions below in this regard.

### Questions
1. Given that top-k/N is typically half, I am wondering if this is always the same half of modules activated? Figure6/8 seem to show that some experts are barely selected but this would somehow contradict the claim of the paper that "Expert Selection Changes During Training". Could you please clarify whether I am misunderstanding the figures? Does the fact that a lot of experts are consistently white over the course of training not indicate that they are rarely used in the forward pass? 
2. Following up on the previous point, a possible ablation I would have found very informative is to prune the experts based on the selection frequency during training, i.e. fix the gating after training based on the selection frequency for various fractions. I am wondering if the performance gains of EMoE are effectively due to pruning and not due to modularity.
3. I am not sure I understand the ablations conducted to support the claim "EMoE masks neurons with negative transfer impacts.". If we limit the model to use modules with small gating score I don't find it surprising that this performs worth than those with large gating scores as the former might just be mostly zero. Have you checked for this?
4. Following up on the previous point, I would find a more informative ablation to contrast the performance of top-k with selecting all modules i.e. removing the gating entirely. Would it be possible to run such an experiment?

> Differently, we treat these decomposed units as heterogeneous modules to exploit their modular properties in the framework of MoEs, and observe significant improvements over monolithic models.

5. It is not clear to me what it means to "treat decomposed units as heterogeneous modules", could you please clarify this sentence?
6. Why do the hyperparameter grids differ between EmoE and GMoE? How can we exclude the possibility that GMoE potentially also benefits from larger N and/or top-k?
7. I find the point on which layers are selected for the MoEfication puzzling. Why does it make sense to only pick every second layer in the second half of the network? 


Typos

- Page 2: "can benefit from its ermgent modular structure"

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes Emergent mixture-of-Experts, for constructing a modular architecture from a pre-trained transformer (post pre-training), without introducing any extra parameters to the model. The focus of this paper is on measuring generalization performance and if the proposed method can provide improved generalization in different settings. This work aims to show the generalization benefits that can be obtained by modularity that “emerges” after pre-training, rather than the modularity that is “baked-in” a model from the start or after pre-training (as done by other methods).

### Strengths
1. Strong empirical study: experiments are sufficient and conducted over different modalities and model architectures, ablations are sufficient and extensive as well.
2. Clarity: This work has made clear their contributions, and the paper is easy to understand and follow.
3. Interesting results: EMoE is shown to match or improve the performance of GMOE which requires more parameters for its construction, this result shows that you don't really need to have those extra parameters. Results are presented across DomainBed, and GLUE and over fine-tuning and lora-tuning settings. EMoE also improves over LORA which is an interesting result.
4. Analysis: The analysis part of the paper is also quite interesting. Showing that EMoE masks neurons which cause negative transfer of knowledge, I have some doubts on the other analysis which is done in the paper (see Q 1. below).

### Weaknesses
There are no major weaknesses that I see in this study. The work is sound in terms of experimentation and through various results and analysis is able to sufficiently address the main question of the paper as to whether EMOEs are beneficial for downstream tasks from the generalization perspective. There are some limitations as described in Section 6, that we are not sure how much the results hold for larger LLMs and for more involved tasks. As made clear by the paper, this work does not provide any new methods (for MOE construction), however, the analysis and results give a lot of insight and hence I believe novelty is not an issue with this paper.

### Questions
1. I am not sure if I understand the first analysis of Sec. 5. Completely. Are the conclusions of this analysis as follows: a) since LoRA2EMoE does not lead to much difference with respect to LoRA, hence sparse activation does not have much impact during testing. And b) Since EMoE2LoRA matches EMoE and leads to performance gains over LoRA, we can say that having modularity actually helps while fine-tuning. If my understanding is correct, I feel this section can be better written with more explanation in simple terms. If not, then I would like to know from authors a more detailed explanation and flaws in my understanding.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work introduces a new neural network layer which can immediately replace the dense layers within transformer modules. To train the layers first a dense layer is pre-trained. The incoming weight matrix to the hidden layer is then clustered into $N$ centroids such that all hidden neurons with the same features are grouped in a module. The average feature is then used as a key to identify when this module should be used for inference. At inference the $N$ modules keys are compared to the input (the query) and the top $k$ modules are used to forward propagate the input. The outputs of these modules are then used in a weighted sum where the weighting is determined by a gating function. A hand-crafted gating function as well as learned gatings are explored empirically but the hand-crafted function is found to be superior. A number of datasets are experimented on with proposed Emergent Mixture of Experts model (EMoE) and an ablation is also conducted to determine the utility of each part of the layer's computation.

### Strengths
## Originality
This work is draws heavily on prior work, particularly GMoEs and MoEfication, but remains fairly original. It is clearly stated what the intended differences are from prior work. For GMoEs this is the absence of additional parameters or training. In the case of MoEfication the decomposed units are treated as "heterogeneous modules" which avoid the drop in performance seen from MoEfication.

## Quality
The experiments employed in this work are extensive and span a number of practical domains, such as vision and text based tasks. Challenging datasets are also used and the algorithm is compared to a number of challenging baselines. In addition the ablation study is interesting and helpful for understanding the various components of the EMoE architecture. Overall the evaluation of EMoE aims to be thorough and indeed challenges EMoE.

## Clarity
The figures are a particular high-point of this work, especially Figure 1 which is quite helpful for understanding the proposed EMoE layer, while Figure 2 is a neat and visually intuitive summary of the results.

## Significance
The proposed layer is indeed significance and would provide a clear contribution to the field. Specifically, a mechanism to zero-shot identify modules within a dense network that improves the generalizability of the overall network and are more interpretable. I could also see future work continuing to build on the proposed layer and introduce more sophisticated ways of identifying modules or calculating the gating function for example. The ablation study will indeed help with guiding this future work.

### Weaknesses
## Quality
My primary concern of this work lies in the interpretation of the results and the intended message which they aim to convey. The authors claim that there is a benefit from EMoE and I cannot see how this is consistently the case across Tables 1, 2 and 3. While there is some improvement over the various baselines in each case (ViT, BERT, GPT2 and LoRA) it still appears marginal and fairly inconsistent. This is  made worse by the fact that standard deviations are not reported or commented on (while they are referenced in the caption of Table 4.1). On top of this, there is even less benefit when comparing to GMoE, however the authors mention in the discussion that being competitive towards GMoE is not the intended purpose. However, it is still stated that "Overall, EMoE outperforms ViT and GMoE". Which then highlights my second concern on quality. It is unclear to me what the intended purpose is of this work. It is stated in the introduction that the work aims to "explore whether and how standard pre-trained transformers could benefit from emergent modular structures". But this question seems to have been answered by the two prior works. GMoEs would indicate that there is a generalization benefit, and MoEfication demonstrates that there is a computational benefit. At stages it seems that EMoEs are being proposed because they do not add parameters or require more training like GMoEs, which I agree would be important. However, at no point is this said and at no point is an substantive evidence given to demonstrate a computational benefit for EMoEs. So based on the experiments this does not seem to be the case. A lot of my confusion here may be secondary due to the issues with clarity which I will now outline. Finally, while I appreciate the long ablation and it is indeed a helpful part of this work, determining the number of centroids in the clustering and number of modules to use are two key hyper-parameters which are not considered or spoken about at all. These two parameters can determine entirely how the module generalizes or under-fits and its overall expressive power. They are also two parameters which must now be tune -- together. Is this addition of complexity worth the benefit over using the vanilla baselines? The answer to that question is likely up to the reader, but failing to mention this or provide the reader with evidence on this appears to be a significant omission. I appreciate that Figure 5 does touch on the expert selection fraction but this is not exactly what I am suggesting and this experiment only uses EMoE architectures and compares to a dense network (it is part of the ablation so I understand that in this context limiting to these comparisons make sense, but still this leaves a gap in the experimentation and paper as a whole).

## Clarity
Firstly, in many cases this work is vague. Even in the abstract using the phrase "...superior out-of-generalization, learning efficiency, etc." is strange, particularly due to the use of "etc" with little context with which to determine what is being referred to. A second example, and the most severe in this work is it is the statement which describes how EMoEs differ from MoEfication is "we treat these decomposed units as heterogeneous modules to exploit their modular properties in the framework of MoEs". What is a "heterogeneous module" in this context. Surely these modules are heterogeneous by some metric by definition since they are identified by a clustering algorithm? But then why would MoEfication have homogeneous modules? A final example, which is similarly important for understanding how EMoEs work says "EMoE and dense model only differ in FFNs activations". Surely the EMoE differs by the fact that it is now gated with an enforced sparsity?

A second note on clarity, I relied on Figure 1 to understand EMoEs as overall Section 3.1 is difficult to parse. Having the transpose on the weight matrix $K^T$ just makes it difficult to follow. Additionally, the output of the $x \cdot K^T$ computation is a $d$-dimensional vector but you try multiply it on the right by a $h \times d$ matrix $V$. So $V$ must also be transposed here. Further, vectors are usually assumed to be column vectors when they are being operated on by matrices and so to following the matrix multiplication of $x$ you have to implicitely transpose it. This is a lot to try and keep in memory while trying to ground the work with the proposed network layer. As a minor extension of this, when would $d$ represent the dimension of the hidden layer and $h$ represent the dimension of the data? Also the first usage of $K_i$ does not bold the $i$ but it is bold everywhere else, so that should be corrected (that's not a critique, just wanted to point it out). Finally, while I appreciate that this was aiming to be grounded within the transformer literature , rephrasing a normal neural network forward pass in terms of keys, queries and values is also just unhelpful. Especially when the work is inconsistent on what the key is. Section 3.1 says the keys are the rows of $K$ but then it is more accurate to interpret the centroids of the clusters of $K$ as the keys as this is what is being compared to the query (input) to determine which module to use.

Finally, the other architectures and algorithms which are compared against are not introduced sufficiently. This makes it very difficult to assess the contributions of EMoE and its practical applicability. Moreover it makes it difficult to interpret the results. For example, why are EMoEs given significantly more modules to train (a larger $N$) and use (a larger $k$) than GMoEs? Does this make a comparison unfair? I also feel that the main take-away of the paper: "Instead, the goal is to demonstrate that dense pre-trained transformers can benefit from emergent modular structures" does not quite summarize accurately what is shown. This would demonstrate that the emergent modular structure is a phenomenon which lasts in transformers, not some structure which must be algorithmically extracted and enforced. But this is a final minor critique on clarity.

Overall I would be inclined to increase my score. Particularly if the results incorporate standard deviations and the discussion is updated to more accurately analyze the results. Unfortunately it seems EMoE does not merely out-perform the baselines. Secondly, I would also substantially increase my score if the clarity concerns are addressed above. I recommend cutting down on Sections 3.1 and 3.2 as they are not particularly helpful and Figure 1 does a great job expressing the same thing. I also think Section 2 could be shortened for a more in-depth background section focusing on GMoE and LoRA and potentially MoEfication which focuses on how EMoE differs. Finally, a general check of the grammar and vague phrasing is in order. While I believe we should be sympathetic towards use of language in these reviews, in this case it makes understanding the paper very difficult. I am also open to revising my score if it is shown that I have indeed missed a crucial point of this work or its presentation.

### Questions
I have asked a number of questions in the Weaknesses section above where they came up naturally within the context of my review. I would appreciate if these questions were covered. I do not believe I have any further questions for this section at this time.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
