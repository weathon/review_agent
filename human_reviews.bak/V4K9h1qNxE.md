# Attention as a Hypernetwork

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 8, 8

## Abstract
Transformers can under some circumstances generalize to novel problem instances whose constituent parts might have been encountered during training, but whose compositions have not.
What mechanisms underlie this ability for compositional generalization?
By reformulating multi-head attention as a hypernetwork, we reveal that a composable, low-dimensional latent code specifies key-query specific operations.
We find empirically that this latent code is predictive of the subtasks the network performs on unseen task compositions, revealing that latent codes acquired during training are reused to solve unseen problem instances.
To further examine the hypothesis that the intrinsic hypernetwork of multi-head attention supports compositional generalization, we ablate whether making the hypernetwork-generated linear value network nonlinear strengthens compositionality.
We find that this modification improves compositional generalization on abstract reasoning tasks.
In particular, we introduce a symbolic version of the Raven's Progressive Matrices human intelligence test, which gives us precise control over the problem compositions encountered during training and evaluation.
We demonstrate on this task how scaling model size and data enables compositional generalization in transformers and gives rise to a functionally structured latent space.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an interpretation of multi-head attention as a "hypernetwork", in the sense of Ha et al. (2017)---a network that generates weights that parameterize another network. The main insight comes from writing MHA as: $\sum_{h=1}^{H} W_h^{out}\sum_{k=1}^{K} a_{h,q,k} W_h^{val} x_k = \sum_{k=1}^{T} \left(\sum_{h=1}^{H} a_{h,q,k} W_h^{out} W_h^{val}\right) x_k$, and interpreting the inner summation in the RHS as the parameterization of the "value network". This comes from swapping the summation over heads $h$ with the summation over keys in the typical formulation of attention. The authors argue that this can give rise to some notion of "compositionality" and OOD generalization. While the value network here is parameterized as a linear combination of linear maps, the authors use this interpretation to propose a variant of linear attention where the value network is instead a two-layer MLP, with each linear map in the MLP being parameterized by the hypernetwork. The paper then explores two experimental settings: the first is based on ICL on "fuzzy logic functions" and the second is based on a symbolic version of Raven's progressive matrices. The authors show that networks with linear attention and softmax attention achieve some level of OOD generalization on these tasks, and that their proposed HYLA model achieves further improved performance.

### Strengths
- The question of how compositional generalization arises in neural networks, and Transformers specifically, is of broad interest to the ML research community. I believe that the interpretation proposed in this paper is a meaningful contribution to this question.
- The paper is well-written, well-structured, and clear.
- To my knowledge, the interpretation of combinations of heads as a hypernetwork is novel.
- The proposal of the HYLA attention mechanism is well-motivated and makes sense given the presented hypernetwork interpretation of MHA.
- The experimental results provide some empirical support for the general claims made in the paper, with HYLA performing favorably in terms of compositional generalization on the tasks considered.
- I found the figures showing the correlation between the latent codes and the target label (i.e., 2B and 5A) compelling, making the argument that some notion of compositionality is made possible by having multiple heads and

### Weaknesses
- The interpretation of MHA as a hypernetwork through re-ordering the summation over heads and keys makes sense in the case of linear attention, but seems less so in the case of softmax attention. In the case of softmax attention, the value network is a linear combination over heads, but with the normalization being over keys rather than heads. This doesn't fit into the claim that the value network is merely query-key dependent; rather it is a function of the entire sequence. Can the authors comment on this? You focus more on linear attention in the paper, and perhaps this is the reason. But, can you share more of your thinking about this?
- The idea of a value network being input-dependent seems closely related to Mixtures-of-Experts. However this is not directly addressed in the paper. Can you comment on this? In particular, is there a type of function that HYLA can represent that standard MHA followed by a MoE-MLP could not? Experimentally, how would the latter perform compared to HYLA? Would it be better or worse at compositional generalization?
- Although the experiments are clear, the scenarios where compositionality and the advantage of HYLA are demonstrated are somewhat restricted in scope. On complex real-world tasks such as language modeling, HYLA performs worse compared to Softmax attention (this is again related to HYLA being a variant of linear attention).

See also the questions below.

Overall, I think this is a nice research paper, with meaningful contributions, and I look forward to discussion with the authors.

### Questions
- How does this interpretation of MHA as a hypernetwork relate to the "residual stream" view, and related concepts on attentional circuits in previous mechanistic interpretability work? Does it have any implications on simple circuits like, say, induction heads?
- How do MoE models compare in these compositional generalization experiments?
- Typically, the value network in MHA is thought of as merely projecting out a certain component of the context token to be retrieved and incorporated into the query token. The value network is not intended to perform any complex processing, that is the job of the MLP that immediately follows. However, the interpretation of MHA as a hypernetwork seems to hinge on the hypothesis that the value network ought to be doing some more complex processing. Can you comment on this distinction? Why would we want the value network to perform complex processing rather than having that be performed by the MLP?
- Under the hypernetwork view, the "value network" has a pretty specific parameterization as a query/key-dependent linear combination of linear maps. Can you comment on why this type of parameterization would make sense? For example, is there some simple construction of a network where there is an interpretable view of the class of functions this could capture?
- In lines 370-374, the authors discuss the difference between standard RPM problems with visual inputs and their simplified version with symbolic input, arguing that the main source of difficulty in RPM tasks is the problem of finding correspondences of rules to feature dimensions. I found this unconvincing, however, and I hope the authors can expand. In particular, it seems to me that this argument relies on the assumption that the visual representation learning step of discovering the right symbolic abstractions is easy, citing work in cognitive science. But what might be easy for humans (who have in-built abilities through millions of years of evolution) may not be easy for machines. Learning the full visual RPM problem end-to-end is surely significantly more difficult for a machine learning system, no? Overall, I think it's unnecessary to try to make this argument. Yes, the symbolic RPM problem is easier, but it is still interesting; we don't need to argue that it captures all the interesting aspects of the full visual RPM problem.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper investigates how Transformers generalize to new, unseen task compositions by reformulating multi-head attention as a hypernetwork—a model that reconfigures another model's computation based on data. This study reveals that a compact, low-dimensional latent code within the attention mechanism specifies key-query operations, which can generalize compositional tasks even if those specific combinations weren’t encountered during training.

### Strengths
1. The paper provides an innovative perspective by reformulating multi-head attention as a hypernetwork, a novel conceptual framework that opens up new avenues for understanding attention mechanisms in neural networks.

2. The paper is well-structured, with clear explanations of its objectives, methods, and findings.

### Weaknesses
1. The paper makes a strong case for viewing multi-head attention as a hypernetwork, but it lacks a formal theoretical analysis showing why this approach specifically enhances compositional generalization over traditional attention mechanisms. Although the empirical results support the hypernetwork framing, a deeper theoretical exploration would add rigor and clarity.

2. The paper primarily contrasts the proposed approach with standard softmax and linear attention but lacks comparisons with other recent methods that enhance compositional generalization in Transformers.

3. While SRAVEN offers a modular benchmark for evaluating compositional generalization, it may not fully reflect the demands of compositional reasoning in language or perception tasks, which involve a broader range of dependencies and symbolic variations. 

4. Although the proposed Hypernetwork Linear Attention (HYLA) and hypernetwork framing show promising results, there is limited discussion on scalability and practical implementation challenges, especially as the model size increases. For instance, it remains unclear whether the approach would scale effectively in large language models (LLMs) or in high-dimensional data contexts.

5. The authors posit that the structured latent code in multi-head attention serves as a reusable resource for different compositional tasks. While the paper provides clustering analyses, there is limited interpretability regarding how the latent codes correlate with specific task elements or subtasks.

### Questions
1. Could you provide a more formal explanation or theoretical basis for why framing multi-head attention as a hypernetwork enhances compositional generalization? Specifically, what is the role of the latent code in enabling the network to handle unseen task compositions?

2. Given the computational demands of Transformers in large-scale applications, would HYLA introduce significant memory or efficiency constraints, and if so, how might these be addressed?

3. Have you considered conducting deeper analyses, such as probing or interpretability methods, better to understand the latent codes’ alignment with specific subtasks?

### Soundness
3

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
2

### Summary
The paper aims at understanding the ability for compositional generalization for the transformer networks. The idea is to view the multihead attention module as a hypernetwork, where a composable, low-dimensional latent code specifies key-query specific operations. Empirically, the paper finds that using non-linear value networks for the hypernetwork (i.e., the multihead attention module) leads to improved generalization ability. The paper further introduces a symbolic RAVEN test for fine-grained evaluations of compositional generalization ability.

### Strengths
- The idea of viewing multi-head attention as a hypernetwork is interesting.
- The experiments on a controlled domain such as symbolic RAVEN test are clear and thorough.
- The research problem is of great interest to the ML community.
- The paper is well-written and well-organized.

### Weaknesses
- I may be missing something. But the modification of the linear attention layer seems a little bit restricted. Will other non-linearity functions or $\sigma(\cdot)$ functions other than `RMSHead` also lead to improved compositional generalization ability? It would be great if the authors can comment on this and explain why this modification would work.
- I am admittedly not an expert of this domain, and would like to hear my colleague reviewers' thoughts on this.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

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
Transforms have been shown to compositionally generalize in some situations and not in others. This paper presents a novel perspective of multi-head self-attention, a core component of transformers, which better exposes the module's capacity for representing and reusing compositional structure. 

Under this perspective:
- The multi-head self-attention mechanism comprises of $\\text{num\\_queries}\\times\\text{num\\_keys}$ value networks that are generated by a hypernetwork (hypernet) parameterized by the value and output projection weights
- The inputs of the hypernet are $\\text{num\\_heads}$-dimensional "latent codes," each corresponding to a query-key pair. Relating to standard multi-head self-attention, these latent codes are obtained by indexing the attention matrix stacked for all heads at a specific key-value index.

With this perspective:
- The authors show that the latent codes exhibit structure that reflects the learned compositionality for the task, which can explain why transformers compositionally generalize when they do. For instance, the latent codes are empirically predictive of the subtasks the network performs on unseen compositions.
- The paper introduces the hypernetwork linear attention (HYLA), a drop-in replacement of standard multi-head self-attention, that modifies the hypernet-generated value network from linear to non-linear to make it more expressive. This modification also replaces the expensive global softmax normalizations per head with local key-query-specific normalizations across heads.

The authors introduce SRAVEN, a symbolic version of Raven Progressive Matrices human intelligence test, which gives them more precise control over the compositions during training and evaluation. Evaluation on SRAVEN shows clearer latent code structure and improved OOD accuracy of HYLA over softmax and linear attention.

Finally, the authors evaluate HYLA on a language modeling task, but it fails to outperform softmax attention here.

### Strengths
- The paper presents a novel perspective of multi-head attention as a hypernet. More specifically, the perspective that queries and keys dynamically create a function to apply to the values is a straightforward interpretation of the attention mechanism, but the proposed perspective further reveals where the learned compositional structure is potentially exhibited.
- The core technical contributions of the "multi-head self-attention as a hypernetwork" perspective and HYLA are presented very clearly.
- The proposed method HYLA is a novel drop-in replacement for standard multi-head self-attention.
- The authors clearly show emergent compositionality in the "latent codes" for the intrinsic hypernets of multi-head self-attention and HYLA.

### Weaknesses
The paper seems to consider the special case of self-attention:
- The proposed perspective of attention as hypernet is presented with the assumption of self-attention, which is a special case of attention (e.g., cross-attention is more general). While the assumption of self-attention can be seen in figures and equations, it is only explicitly mentioned in the text at line 114. Without clarifying that this perspective and the proposed modification (HYLA) consider the special case of self-attention, the paper can be seen as over-claiming its contribution. The paper's title, abstract, and introduction should clarify this constraint. Alternatively, the paper should clarify how its formulations work in more general cases of attention and if their experiments are limited to the particular case of self-attention.

Clarity and connection of multi-head to compositionality:
- The necessity and significance of the "multi-head" aspect for compositional generalization is underplayed. A relevant discussion is found in the first paragraph of Section 6, which could be discussed earlier in the paper to set up a clear connection to compositionality.
- The introduction of the paper is quite dense. After a clear flow of the first two paragraphs, it abruptly gets into too many technical specifics and jargon in the third paragraph that is difficult to parse without the context of the rest of the paper. Perhaps a less technical introduction of the paper's perspective and modification, with a motivation of how these relate to compositional generalization, can set up clearer expectations for what is to come and why it matters (maybe the Summary of this review can be helpful if accurate). Only after this, the details like how this paper's method differs from Schlag et al., 2021, or how the presented idea is evaluated become easier to place into context.

Correlation between compositionality and generalization:
- The paper claims that the compositional structure seen in the intrinsic hypernet of MHA can explain the compositional generalization of transformers. However, there is a missing connection here. An important experiment is to evaluate with confidence estimates how the compositionality of attention correlates with the OOD generalization of the model.

Weak performance on language modeling:
- Despite the positive results on synthetic tasks, the paper's proposed method, HYLA, fails to outperform softmax attention for language modeling with a clear performance gap (Figure 6). The authors hypothesize that softmax is essential here, but find that HYLA with softmax results in a drop in performance (Table A1). Please look at the last question in the Questions section for a possible actionable avenue.

RMSNorm:
- It looks like RMSNorm at lines 179-180 is the operation from [1], but the paper is not cited.
- The authors state that it ensures the parameters of the value network are appropriately initialized to prevent exploding/vanishing gradients. There needs to be an explanation of why this should be the case.

References:  
[1] Zhang, Biao and Sennrich, Rico. Root Mean Square Layer Normalization. NeurIPS 2019.

### Questions
HYLA with a single head:
- In the caption for Figure 5C, the authors state that the hypernet mechanism is absent for a single head. Why is this the case?

Scaling compositional difficulty:
- In Section 4.2, the authors state that 25% of possible rule combinations are held out for evaluation. How does performance change when this fraction is changed? Specifically, as the compositional difficulty (% of held-out rules) varies from 0% to 100% on the X-axis, how do the performances of softmax attention, linear attention, and HYLA on the Y-axis vary? This important experiment will help us understand how these methods compare in a fully in-distribution setting to much harder OOD settings.

Drop in performance of HYLA with softmax:
- For HYLA, in Eq (6), both the value network weights $\\mathbf{W}'\_{q,k}$ and $\\mathbf{W}\_{q,k}$ depend on $a\_{h,q,k}$, which makes sense from the hypernet perspective where $a$ is the latent code. In standard MHA, as shown in Eq (2), $\\mathbf{W}^\\text{out}$ operates after attention scaling has already been applied. However, from this perspective, Eq (6) indicates a re-application of attention scaling. Could this reapplication of attention scaling be responsible for the drop in performance when using softmax in Table A1, perhaps due to the reapplication of softmax being more prone to vanishing gradients?
- Do the language modeling results improve if a potential issue with using softmax with HYLA is identified and fixed?

### Soundness
4

### Presentation
3

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
This work reformulates linear multi-head attention as a hypernetwork to study compositional generalisation and introduces a simple non-linear variant that substantially increases generalisation both on abstract symbolic tasks as well as language modeling.

### Strengths
The paper is well-written and studies an important problem, namely compositional generalisation, from a new angle and with promising empirical results. The writing, structure and related work is outlined clearly, and (almost) all relevant details with regards to the theory and the empirical experiments are available in either the main text or the appendix. The perspective to see multi-head attention as a hypernetwork is novel and might open interesting future avenues for study.

### Weaknesses
One problem in the empirical evaluation is a certain lack of control over whether the suggested modification (HYLA) is actually increasing compositional generalization or whether it's just a way of increasing the capacity of the model. To this end, it would be interesting to look at the training loss of the models: does HYLA reach the same minimum as linear / softmax attention, or is it already performing much better on the training set (suggesting it is more the capacity increase of the model rather than a genuine way of increasing compositional generalisation abilities).

Similarly, it didn't get clear to me how the OOD sets were sampled? E.g., in the fuzzy logic dataset, when you say "hold out a fraction of the combinations for OOD" (line 206), does this refer to combinations of K terms, or are all K terms sampled and you just subsample each (meaning, in the OOD set all combinations of K are present)?

### Questions
- In HYLA, you not only add the nonlinearity but also change the normalisation. Did you perform ablations w.r.t. the normalization, meaning how well does HYLA perform with the standard normalization (or vice versa, if you add RMSHead in linear attention)?

- Please report the training performance of the models to shed light on the generalisation vs capacity question.

- Please provide more details on the OOD splitting of the datasets.

### Soundness
3

### Presentation
3

### Contribution
3
