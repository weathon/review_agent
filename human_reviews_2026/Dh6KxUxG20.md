# Knowledge Distillation for Large Language Models through Residual Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
Knowledge distillation has become a crucial technique to transfer the capacities of large language models (LLMs) to smaller, more efficient models for practical deployment. While recent work exploits rich information from intermediate states of the teacher model for more effective knowledge transfer, imperfect knowledge from the teacher can also mislead student learning, restricting the student’s generalization capacity. In this work, we propose a two-stage distillation framework that is effective for diverse knowledge distillation scenarios. In the first stage, we pretrain projectors to extract and compress teacher knowledge into a low-dimensional vector space via self-reconstruction. In the second stage, we perform distillation with a hybrid objective that combines learning from the compressed teacher representations with standard supervised fine-tuning on ground-truth data. Our key innovation is residual learning for LLM distillation, where the student learns to make predictions based on the differential between its representations and projected states from the teacher. This approach encourages the student to further improve its representations beyond potentially erroneous teacher knowledge. For Mixture-of-Experts (MoE) teacher models, we further fuse the experts’ outputs using a self-attention mechanism for better utilizing the teacher knowledge. Moreover, to support the cross-tokenizer distillation setting, where the teacher and student models have different vocabularies, we adopt a cross-model attention mechanism that eliminates the need for explicit token alignment rules. Experimental results show the superior performance of our proposed framework under both same- and cross-tokenizer settings, demonstrating the effectiveness in preserving teacher knowledge and improving student generalization capability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a two-stage white-box knowledge distillation framework. In the first stage, the teacher’s hidden states are compressed into a low-dimensional latent space via self-reconstruction projectors. In the second stage, the student is trained through a residual learning mechanism, where it refines its representations by subtracting the teacher’s hidden states from its own. The paper claims that this design encourages the student to learn complementary knowledge and mitigate the propagation of teacher errors. Experimental results on instruction-following benchmarks demonstrate consistent performance gain.

### Strengths
- The introduction of a feature distillation method where the student learns from the difference between its and the teacher’s hidden states.

- The paper is well written and structured.

- KD for LLM is a very important topic.

### Weaknesses
- Projecting the teacher’s hidden states into the student space is an established paradigm widely used in feature-based distillation to handle mismatched hidden dimensions [1,2,3].

- The evaluation is restricted to instruction-following tasks and a limited set of model architectures. Experiments on more diverse tasks such as reasoning, summarization, etc are important.

- The paper does not include comparisons with several recent and relevant knowledge distillation methods (e.g., DistillM, DistillM2, ABKD, CKA-based distillation), limiting the strength of its empirical claims.

- The paper motivates residual learning as a means mitigate teacher errors, yet the link between this motivation and the proposed residual mechanism is not clear.



[1] Jiao X, Yin Y, Shang L, Jiang X, Chen X, Li L, Wang F, Liu Q. Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351. 2019 Sep 23.

[2] Miles, Roy, and Krystian Mikolajczyk. "Understanding the role of the projector in knowledge distillation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 5. 2024.

[3] Chen, Yudong, et al. "Improved feature distillation via projector ensemble." Advances in Neural Information Processing Systems 35 (2022): 12084-12095.

### Questions
See weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Traditional knowledge distillation may forces the student model to inherit the errors of an imperfect teacher. This paper proposes a novel framework called "Residual Learning" to address the core issue. The key idea is to guide the student to learn only the "difference"(residual) between its own understanding and its teacher, but only when the teacher makes a mistake.Thereby avoiding blind imitation and holding the potential to surpass the teacher. 
Combined with specialized modules designed for Mixture-of-Experts (MoE) models and cross-tokenizer scenarios, experiments show that this framework significantly outperforms existing methods across various distillation tasks and effectively enhances the generalization ability of the student model.

### Strengths
● Clear Problem Definition and Motivation:This paper identifies a issue in white-box knowledge distillation: the teacher model is not perfect. Forcing the student model to align with the output distribution of an imperfect teacher essentially sets an upper bound on its capability and may propagate biases. The proposed idea of "critical learning instead of blind imitation" is highly persuasive and innovative.
● Novel, Intuitive, and Easy-to-Understand Core Method:The method uses an indicator function 1[teacher is wrong] to determine when to activate "residual learning", shifting the focus from imitation to error correction. The idea is both intuitive and easy to understand.
● Comprehensive Experiments and Strong Results:
● Broad Range of Scenarios Covered:The experiments span various realistic and challenging distillation scenarios, including same/cross-tokenizer settings and distilling from Mixture-of-Experts (MoE) models to dense models. 
● Solid Ablation Studies:The ablation study is well-designed and thoroughly evaluates the importance of each component in residual learning: the accuracy mask, scaling factor β, pre-trained projector, and MoE expert fusion. This significantly strengthens the credibility of the proposed method.

### Weaknesses
1. In scenarios where there are no definitive correct answers, such as text generation or chain-of-thought (CoT) reasoning, the indicator function for determining whether the teacher is wrong may be limited in its applicability.
2. It would greatly enhance the persuasiveness of the paper if there are concrete examples that demonstrate the core claim of this paper — for instance, showing the process where "the teacher makes a mistake → the baseline student model follows the error → the proposed student model successfully corrects it"

### Questions
● Regarding the dependence on ground-truth labels:Do you think it is possible to extend the idea of "residual learning" to scenarios where no ground-truth is available? For example, could one use a discriminator model or leverage the consistency among multiple teacher models as a proxy signal for whether the teacher made a mistake?
● Regarding the breadth of evaluation:Have you considered evaluating your method on tasks that emphasize logic rather than text matching, such as reasoning, math problems, or code generation? We are particularly interested in whether residual learning over token-level prediction errors can effectively transfer and enhance complex reasoning capabilities.
● Regarding hyperparameter sensitivity:The final loss is a weighted combination of lambda​. In the experiments, both set to 0.5. How sensitive is the model performance to the ratio between these two coefficients? Have you conducted any sensitivity analysis regarding this?
● Regarding the MoE fusion mechanism:In the MoE knowledge fusion, you employ self-attention to aggregate outputs from all experts. Compared to simpler alternatives like averaging all expert outputs (average pooling) or activating all experts (dense activation), how much improvement does the self-attention mechanism actually bring? This comparison does not seem to be explicitly included in the ablation study.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a method for knowledge distillation including several novel innovations, notably "residual learning", a method to guide the student in order to avoid replicating mistakes from the teacher, among other innovations around effective combination of learning signal from MoE teachers, and handling teacher-student combinations with different tokenization methods.

Overall the paper was really interesting, the goals lofty, and the approach very engaging. My one gripe is I'm not sure I understood why residual learning works, and how the overall training objective can benefit positively from the student as it appears only to include negative supervision when the teacher makes mistakes. Hopefully the authors can clarify in the discussion phase.

### Strengths
This paper goes beyond the norm in presenting more algorithmic innovations than most. The residual part is the core contribution -- and ablations showed this to work very effectively -- but the MoE and cross-tokenizer methods were also good additions, slightly simpler in formulation, but very sensible and shown to have utility.

The experimental evaluation is solid, showing the utility of the method and its component parts on standard evals with a range of strong models (Mistral, LLama etc) at reasonable sizes 7B, and against competitive baselines. In many cases the student outperforms the teacher, despite using many fewer parameters, which is a very impressive result. I would expect these results to be interesting to the wider community.

### Weaknesses
The presentation is a bit murky at times. I was left puzzled about why the method works, and what the core motivation is around the various steps in the residual learning method. Let me elaborate on my understanding. Please correct my misconceptions in your rebuttal.

Stage 1: pretraining projectors, this appears to be done to as a form of regularization to smooth out meaningless variation in the teacher hidden states. It also allows for situations where the hidden dimension is different in the student, or the representations are the same size and have similar information but indices are permuted (e.g., identifiability). I'm not sure why it's trained with a softmax/categorical distribution to predict the gold next token - wouldn't matching teacher predictions, e.g., using KL, be more suitable?

Stage 2: during KD the model learns to project the smoothed teacher state to match the student state, and uses this only when there's a teacher mistake, and it fails to correctly predict the gold next token. In these settings, the student's hidden state is offset to subtract the effect of the teacher. Can this be understood as correcting a likely mistake in the student? And given this 'correction', how does learning via backpropagation of loss from the prediction of the gold next token affect the parameters? I see W^S may be better estimated, but I wonder about h^S_i and h^{(T -> A) -> S}_i and how they will be updated. Given this offsetting of hidden states won't be applied during inference, it's hard to see how this change in KD training will make an impact on the final model.

### Questions
Presentation clarification questions, see Weaknesses.

Why does unfreezing teacher projectors has such a negative impact on accuracy?

Why is there no KD loss using the teacher in a positive manner, e.g., KL against teacher outputs, or alignment to the teacher hidden states when the teacher is correct?

In 3.3, eq 8 how is the h_i^T used? Is this a drop in replacement for a dense h_i^T in section 3.2 eqs 4-5?

### Soundness
3

### Presentation
3

### Contribution
4
