# MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 2, 8, 6

## Abstract
We present MetaTT, a Tensor Train (TT) adapter framework for fine-tuning of
pre-trained transformers. MetaTT enables flexible and parameter-efficient model
adaptation by using a single shared TT to factorize transformer sub-modules. This
factorization indexes key structural dimensions, including layer and matrix type,
and can optionally incorporate heads and tasks. This design allows MetaTT’s pa-
rameter count to scale with the sum, rather than the product, of the modes, resulting
in a substantially more compact adapter. Our benchmarks compare MetaTT with
LoRA along with recent state-of-the-art matrix and tensor decomposition based
fine-tuning methods. We observe that when tested on single-task standard language
modeling benchmarks, MetaTT achieves competitive parameter efficiency to accu-
racy tradeoff. We further demonstrate that MetaTT performs competitively when
compared to state-of-the-art methods on multi-task learning. Finally, we leverage
the TT-ansatz to design a rank-adaptive optimizer inspired by the DMRG method
from many-body physics. Our results demonstrate that integrating this approach
with AdamW enhances optimization performance for a specified target rank.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MetaTT, a Tensor Train (TT)-based adapter framework for parameter-efficient fine-tuning of pre-trained transformers. MetaTT leverages a shared TT decomposition to factorize various transformer sub-modules across structural dimensions such as layer index, matrix type, and optionally heads and tasks. Additionally, they propose a rank-adaptive optimizer inspired by the DMRG method from quantum physics, demonstrating improved optimization when integrated with AdamW for fixed target ranks.

### Strengths
The presentation of this paper is clear and easy to follow. The topic of using a tensor-train adapter is quite novel, and the proposed rank-adaptive optimizer appears to be an important and reasonable enhancement that can be effectively integrated with existing tensorized methods.

### Weaknesses
- My main concern with this paper is the soundness of the proposed method. I understand that global compression can reduce the number of trainable parameters. However, I’m not convinced why this should lead to better performance. Intuitively, reweighting the entire transformer block should perform worse than reweighting individual linear layers within the block, since adjusting single layers allows more flexibility—especially when combined with in-block non-linearity. I didn’t see any discussion or justification from the authors on why the proposed method works in this regard.
- The experimental results further reinforce my concern. For example, in Table 1, the proposed method shows improvements over baselines like LoRA, but the gain is usually less than 1%, which is not substantial enough to confidently claim a real improvement. A similar trend is observed across other tables.
- I’m also wondering whether the DMRG optimizer is included in Tables 1–3. If not, why was it excluded? Based on the results and discussion, it seems that the DMRG-inspired method itself contributes significantly to the performance gains. In contrast, the improvements from the tensorized adapter alone appear to be limited.
- BERT-based models feel somewhat out-of-date to me. I highly suggest that the authors focus more on the tasks in Table 1 for the ablation studies instead.

### Questions
See the weakness

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
This paper introduces MetaTT, a parameter-efficient fine-tuning framework based on tensor train decomposition. MetaTT parameterizes all transformer sub-modules using a single shared TT, achieving global compression. The paper proposes two variants, MetaTT-4D and MetaTT-5D, for single-task fine-tuning and extends the architecture to joint multi-task learning with an additional tensor core. Furthermore, the paper leverages a rank-adaptive optimizer inspired by the DMRG method from quantum many-body physics to enhance optimization. Experimental results compare MetaTT against several PEFT baselines, including LoRA, VeRA, and LoTR, for both single-task and multi-task settings on standard benchmarks like GLUE and commonsense reasoning datasets.

### Strengths
1. The proposed use of a single shared TT for compressing all transformer layers and sub-modules is novel and shows promise in reducing the parameter count while maintaining competitive performance.
2. The rank-adaptive training inspired by DMRG is an interesting integration of techniques from quantum physics into machine learning, showcasing interdisciplinarity and potential for further exploration.
3. The paper compares MetaTT with several state-of-the-art PEFT methods, including LoRA and LoTR, across multiple tasks, and reports detailed results on parameter efficiency and accuracy.

### Weaknesses
1. Despite the novelty of the approach, the reported performance improvements are marginal or absent in most cases compared to simpler baselines like LoRA, especially given the significant computational complexity added by TT decomposition and rank-adaptive training.
2. While the DMRG-inspired optimizer is presented as a key contribution, its practical benefits over standard optimizers like AdamW are not convincingly demonstrated. The rank-adaptive approach introduces additional training complexity without a clear payoff in terms of accuracy or efficiency.
3. The paper overlooks some recent works on tensor-based adapters and PEFT methods, particularly those focusing on computational trade-offs and scalability, such as AdaLoRA and other adaptive rank methods.
4. While the authors provide detailed pseudocode, the implementation lacks clarity in critical areas like initialization strategies and hyperparameter choices, which are shown to influence MetaTT's performance heavily. This raises concerns about the reproducibility of results.

### Questions
1. The results show that MetaTT performs similarly to LoRA and LoTR in many benchmarks, with only marginal improvements in some cases. Can the authors clarify the practical advantages of MetaTT over these simpler methods, particularly in real-world scenarios? How do the authors justify the significant computational overhead introduced by TT decomposition and rank-adaptive training in light of these modest gains?
2. While the paper provides pseudocode and hyperparameter grids, the results seem highly dependent on initialization strategies and specific rank settings. Could the authors share more details about the exact initialization methods, hyperparameter tuning process, and any challenges encountered during experimentation? Are there plans to release the full implementation and training pipelines?
3. The paper notes that MetaTT-5D is more sensitive to initialization and training instability than MetaTT-4D. Can the authors elaborate on why this is the case? Are there specific guidelines or heuristics for initialization and hyperparameter selection that can make MetaTT-5D more robust? How does this sensitivity impact the usability of MetaTT in practice?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The submission introduces a Tensor Train adapter model that parameterizes LLM weight updates as a fourth or fifth order tensor. Moreover, they introduce an optimization scheme that starts with a larger rank and iteratively fits an adapter and applies (approximated) truncated SVD to reduce the rank.

### Strengths
The paper is well written, easy to follow, the contribution and references to prior work are clear, the approach is sound and experiments not only include the standard benchmarks but illustrate particularities of their proposed methods.

Both TT parameterizations (e.g. LoTR) and fifth order tensor adapter models that use layers input output dimensions and heads  (e.g. LoRTA) have been proposed, but not their conjunction. Secondly, treating tasks as an additional dimension is, to the best of my knowledge, novel. 

The iterative DMRG inspired training algorithm is novel and relevant contribution that motivates further research into designing optimization algorithms for low rank tensor adapters that dynamically adjust rank throughout optimization. Although truncated SVD has been proposed to initialise LoRA adapters, and some optimizers have been proposed specifically for low rank matrix adapters (e.g. GaLoRE), the low rank tensor literature primarily relies on standard (adamw) optimization tools and, more importantly, the rank is usually treated as a fixed hyper-parameter. This is relevant because when the rank is low - regardless of the adapter model - optimization dynamics usually becomes challenging and starting with a larger rank can mitigate this.

### Weaknesses
I think that the rank adaptive optimization scheme is a strong contribution. The experiments that showcase its benefits are centered in the standard NLU setting with roberta, but I think it would be useful to extend the empirical analysis to (at least one, ideally all) of other benchmarks/tasks/models in order to further substantiate the empirical gains from this scheme in settings that are regarded as more challenging.

### Questions
Why do most experiments show only meta-TT 4D (except for roberta in nlu tasks)?

Can you comment on initialization and sensitivity - perhaps provide an ablation ?

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
4

### Summary
This paper proposes to use tensor-train factorisation to compress the weight matrices of PEFT models.  They show how this elegant framework can be used to share implicit structure in the weights across parameter types, layers, attention heads, and multiple tasks.  Strong reductions in the number of parameters are achieved while getting similar or better accuracies across a number of tasks.

### Strengths
This is an elegant unified approach to compressing PEFT matrices to reduce parameter counts.  It also elegantly extends to multi-task PEFT, allowing shared structure across tasks.

Empirical evaluations are done on a good variety of tasks, using three versions of the model, each an extension of the previous model.  Results are generally good or comparable to previous methods, but with greatly reduced parameter counts.

### Weaknesses
The novelty is not high.  There has been a lot of work on PEFT already, and this work does not add much conceptual or theoretical novelty.  The contribution is in identifying a general-purpose mathematical framework which addresses PEFT in a consistent way, rather than a collection of ad-hoc methods.

The empirical results do not demonstrate any breakthroughs with respect to previous work.

### Questions
There is earlier work than the papers you cite which seems directly relevant to your approach of factorising parameter matrices and multi-task PEFT, respectively:
 Mahabadi, Henderson, and Ruder. Compacter: Efficient Low-Rank Hypercomplex Adapter Layers.  NeurIPS 2021.
 Mahabadi, Ruder, Dehghani, and Henderson.  Parameter-efficient Multi-task Fintuning for Transformers via Shared Hypernetworks.  ACL 2021.

### Soundness
4

### Presentation
4

### Contribution
3
