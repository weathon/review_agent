# Attention to Mamba: A Recipe for Cross-Architecture Distillation

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge on how to train Transformers, and many pretrained Transformer models are readily available. 
To facilitate the adoption of SSMs while leveraging existing pretrained Transformers, we aim to identify an effective recipe to distill an Attention-based model into a Mamba-like architecture. In prior work on cross-architecture distillation, however, it has been shown that a naive distillation procedure from Transformers to Mamba fails to preserve the original teacher performance, a limitation often overcome with hybrid solutions combining Attention and SSM blocks.
The key argument from our work is that, by equipping Mamba with a principled initialization, we can recover an overall better recipe for cross-architectural distillation. To this end, we propose a principled two-stage approach: first, we distill knowledge from a traditional Transformer into a linearized version of Attention, using an adaptation of the _kernel trick_. Then, we distill the linearized version into an adapted Mamba model that does not use any Attention block.
Overall, the distilled Mamba model is able to preserve the original Pythia-1B Transformer performance in downstream tasks, maintaining a perplexity of 14.11 close to the teacher's 13.86. To show the efficacy of our recipe, we conduct thorough ablations at 1B scale with 10B tokens varying sequence mixer architecture, scaling analysis on model sizes and total distillation tokens, and a sensitivity analysis on tokens allocation between stages.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to present a better strategy for distilling Transformers to Mamba. The major concept of this paper is introducing an intermediate step that transfers knowledge from softmax based Transformers to the linear attention variant, which is basically proposed by Hedgehog. Then, reusing the QKV weights to their corresponding CBX components in Mamba, and distilling it again, the authors show that it leads to better performance than previous works.

### Strengths
The paper is very straightforward -- discovering the importance of the intermediate Hedgehog transfer. As the idea is very simple, it's easy to understand, probably also for people who are not familiar with distillation or SSM literature.

### Weaknesses
As the idea is very simple, the novelty is limited. Most of the contents in the paper are about preliminary papers; for example, Section 1.1, Section 2 are only about previous works, and about 80% of Section 3 are introducing the ideas from Hedgehog or Mamba2.

### Questions
One critical ablation is missing: how important are "Parameter initialization" and "Attention scores normalization" in Section 3.2? Also, I am not convinced with the necessity of attention scores normalization. Mamba architecture does not have a separate normalization of such scores, but why did the authors choose to use the trick?

The paper only compares to Hedgehog. How is it compared to MOHAWK?

In Table 1, Hedgehog's PPL is 14.89. But in Table 3, the 100/0 row shows very high PPL. Where does this significant difference come from?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the challenge of distilling a pretrained Transformer (Pythia) into Mamba to achieve faster inference and lower memory usage while retaining performance. Specifically, it proposes a two-stage distillation recipe: 1. Distilling the Transformer’s softmax Attention into a Linear Attention based on the Hedgehog approach. 2. Using this linearized Attention to initialize and fine-tune a Mamba-based architecture (HedgeMamba). The method is evaluated on the OpenWebText dataset, and the proposed HedgeMamba achieves superior performance compared to the simple direct distillation into Mamba and Hedgehog distillation into Linear Attention.

### Strengths
- The two-stage distillation process is novel and reasonable. It works much better than the simple direct distillation from Transformer to (Hedge)Mamba. Also, thanks to the superior expressiveness of Mamba over that of Linear Attention, the distilled HedgeMamba works better than Linear Attention models.  
- The ablation studies effectively show the importance of the two-stage recipe.  
- The paper is well written and provides enough details about their implementations, including the limitations.

### Weaknesses
#### Major Weaknesses  
- The important information about the efficiency of HedgeMamba architecture is missing. Because the motivation of the distillation in this paper is to create an efficient model, the author should compare the inference speed or FLOPs. I doubt the efficiency of the HedgeMamba because the hidden stage dimension is huge (2048) compared to Mamba (16 to 64), in addition to the newly introduced layers.  
- Why is the large SSM hidden state used? Although it is related to the previous weakness point, the inference speed should be much slower compared to the original Mamba (I guess 2~5 times slower) due to the large hidden state size. Did you use Mamba2 instead of Mamba? (Mamba2 should be relatively faster even if the hidden state size is large.)  
- A gap exists between the motivation and the evaluation. Although the motivation is to borrow the strong performance of large Transformers, the experiments are conducted with the 1B size model. The Pythia-1B is inferior to Mamba-790M, and, to this end, HedgeMamba is inferior to Mamba-790M. In addition, the distillation cost (12 days with 8xA100 GPUs) seems not so small compared with the scratch training cost of Mamba.  
- The lack of comparison against other methods proposing the Transformer to Mamba distillation [1, 2]. Although the authors mentioned that the experimental setups are different, comparisons can be possible by trying previous methods with this paper’s setup. Different architectures can be compared with accuracy-efficiency trade-off.  

#### Minor Weakness
- Typo  
    - L037) tokens representations -> token representations  

[1] Wang, J., Paliotta, D., May, A., Rush, A., & Dao, T. (2024). The mamba in the llama: Distilling and accelerating hybrid models. Advances in Neural Information Processing Systems, 37, 62432-62457.  
[2] Bick, Aviv, et al. "Transformers to ssms: Distilling quadratic knowledge to subquadratic models." Advances in Neural Information Processing Systems 37 (2024): 31788-31812.

### Questions
Please see major weakness. As to the third point, if it is difficult to evaluate with large models, it can be interesting if you can show some evaluations that the distillation improves the weak points of the vanilla Mamba described in some papers such as [3, 4]. In addition, if HedgeMamba is efficient, I want to raise the rating.

[3] Park, J., Park, J., Xiong, Z., Lee, N., Cho, J., Oymak, S., ... & Papailiopoulos, D. (2024). Can mamba learn how to learn? a comparative study on in-context learning tasks. arXiv preprint arXiv:2402.04248. 
[4] You, W., Tang, Z., Li, J., Yao, L., & Zhang, M. (2024). Revealing and Mitigating the Local Pattern Shortcuts of Mamba. arXiv preprint arXiv:2410.15678.

### Soundness
3

### Presentation
4

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
The paper tries to fine-tune full attention into linear attention, first by using the Hedgehog distillation, and then fine-tune into a Mamba variant with parameters reused in the wake of state space duality. Empirical experiments and ablation studies verify the effectiveness and necessity of the two-stage recipe.

### Strengths
The idea of conversion to Mamba by reusing weights based on state-space duality is interesting.  
Empirical results are good.

### Weaknesses
Stage 1 is identical to Hedgehog, which reduces the novelty of the approach.

### Questions
From my understanding, Stage 1 is identical to the Hedgehog work. Please clarify if there is any difference with their work. The identical part should be ideally presented in a more concise way, and details (e.g., the reference of kernelization/Mercer's theorem) should be put into the appendix, leaving more space for the novel part.

The difference between HedgeMamba and Mamba is important. It would be easier to follow if the author could prompt the reader earlier (e.g. in Fig 2) to check Fig 4 for it.

Have the authors considered the applicability to other linear attention approaches, e.g., DeltaNet?

Please also report the average accuracy in Table 1-3 for easier comparison.

The method becomes suspicious when the largest improvement comes from gated attention; Is the Mamba part really necessary? How about the case if you add gating only?

L69,350, ...: Should use \citep  
L138,353,354, ...: Should use \citet 
L249-253, ...: Should use "by \citet{...}" instead of "in \citet{...}"  
Some numbers are out-of-margin in Table 1

### Soundness
3

### Presentation
3

### Contribution
2
