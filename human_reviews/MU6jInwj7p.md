# LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices

- Avg Score: 5.25
- Decision: Reject
- Scores: 3, 6, 6, 6

## Abstract
With the commercialization of large language models (LLMs), weight-activation quantization has emerged to compress and accelerate LLMs, achieving high throughput while reducing inference costs. However, existing post-training quantization (PTQ) techniques for quantizing both weights and activations of LLMs still suffer from non-negligible performance drops, especially on massive multitask language understanding. To address this issue, we propose Low-Rank Quantization (LRQ) - a simple yet effective post-training weight quantization method for LLMs that reconstructs the outputs of an intermediate Transformer block by leveraging low-rank weight-scaling matrices, replacing the conventional full weight-scaling matrices that entail as many learnable scales as their associated weights. Thanks to parameter sharing via low-rank structure, LRQ only need to learn significantly fewer parameters while enabling the individual scaling of weights, thus boosting the generalization capability of quantized LLMs. Through extensive experiments, we demonstrate the superiority of LRQ to prior LLM INT8 PTQ works. Remarkably, we confirm for the first time the possibility of 4-bit weight and 8-bit activation quantization for LLMs with minimal accuracy loss among LLM PTQ studies.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a new weight rounding scheme via a small modification on top of FlexRound technique: whereas in the FlexRound the full matrix of elementwise reciprocals are being learned, the proposed method changes it to a low-rank matrix. In other words, if the FlexRound's quantization (rounding) function is given as $Q(\mathbf{W}; \alpha, \mathbf{S}) = \alpha \times \lfloor \frac{\mathbf{W}}{\mathbf S} \rceil$ where $\alpha$ is the shared re-scaling parameter, $\mathbf{S}$ is the matrix of learned reciprocals, and $\lfloor \cdot \rceil$ is the rounding-to-nearest function, then the proposed modification is to change the $\mathbf{S}$ to be of a low rank (with $r \in \{1024, 2048\}$ ) . The  parameters of this quantization function are learned in a block-wise reconstruction manner to minimize the $\ell_2$ difference between quantized and non-quantized outputs. Gradients for non-differentiable operations are computed via straight-through-estimation.
The motivation for the proposed change is rooted in the fact that learning the entire matrix $\mathbf{S}$ on the calibration dataset might lead to overfitting, and thus low-rank $\mathbf{S}$ will counteract it. The paper is supplemented with many experiments showcasing the effectiveness of the proposed method and comparisons to other techniques.

### Strengths
The proposed change requires minimal modification for any existing quantization pipelines and codebases, and thus can be quickly absorbed into actual real-world usage. 
Despite simplicity, the experimental evaluation suggests that the method provides measurable, albeit marginal, improvement over existing techniques.

### Weaknesses
Despite the aforementioned strengths, I am not very convinced on why these changes are helping the quantization. The original motivation to introduce low-rank $\mathbf{S}$ was to combat overfitting via "decreasing the number of learnable parameters". However, in the provided ablation study between the rank of the matrix $\mathbf{S}$ and the achieved zere-shot accuracy (figure 5a) on a task, it seems the rank has minimal effect on the accuracy (red curve is almost flat). Then, the question would be what is contributing to the improved accuracy if not lower rank? 

Improvements: whereas the presented results show indeed better accuracies, I am somewhat unsure to which degree these results showcase the best of the other methdos. For instance, it seems authors used same scale of 0.8 for SmoothQuant, whereas the scale should ideally be validated on a calibration data. Also, the same comparison to SmoothQuant seems to be unfair: smoothquant is pure no-learning post-training quantization technique, but FlexQuant and the suggested methods are actually introduces a learning objective that is optimized.  In many cases the difference between the suggested method and FlexQuant is such a minuscule number, that it can be attributed to a sampling noise (see table 1).

### Questions
1. Can you please address weaknesses raised above?
2. The FlexRound method as presented in the original paper does not have exponentiation on $mathbf{S}$, but the proposed method does have it. Is it a typo or the additional reparametrization on top of FlexRound?
3. What is the effect of $r$ and $c$ (eq.2)? Some ablation studies on those would be helpful
4. Some general recommendations on choosing rank, learning rates, and training regime would be of great help to all practitioners. Also, if you can tabularize the actual experiment details presented in appendix C; it would help to reproducibility as well
5. There are some minor typos in the paper: in section 2.1, we concentrate**s** => we concentrate in section 4, simeple => simple,

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces LRQ, a post-training quantization technique for LLM (Large Language Models), which improves upon the FlexRound methods by reducing the size of weight-scaling matrices to enhance generalization performance. Through a comprehensive set of experiments conducted on Llama and Llama2, LRQ demonstrates improvements for 4-bit and 8-bit quantization in both Common Sense Reasoning (CSR) and Massive Multitask Language Understanding (MMLU) tasks.

### Strengths
1. The paper is well-written, and the low-rank weight-scaling method is simple and efficient, making it easy to comprehend.
2. This paper has conducted an extensive ablation study on the size of parameters and calibration data, providing valuable empirical insights into the optimal cost of the weight-scaling method.

### Weaknesses
1. LRQ builds upon FlexRound by introducing a low-rank matrix into the full weight-scaling method. As this is my first time as an ICLR reviewer, I am uncertain whether the novelty meets ICLR's standards.
2. This paper extensively explores experiments on both CSR and MMLU tasks, using various quantization methods. However, it consistently shows only marginal improvements compared to FlexRound, with one exception being MMLU when employing per-tensor asymmetric static activation quantization. Yet, the performance gain with per-token asymmetric activation quantization, which offers finer granularity, remains limited.

### Questions
1. It appears that LRQ applies low-rank quantization to weights. When it comes to activation quantization, is RTN used?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work propose LRQ (Low-Rank Quantization), a simple yet effective PTQ method that leverages low-rank weight scaling matrices. This enables the method to learn with fewer parameters while still able to scale weights individually. Results show that the proposed method achieved better performance than existing works on both INT8 and INT4/8 cases.

### Strengths
(1) A thorough, detailed, easy-to-follow description of the method in section 2.

(2) A thorough evaluation of the proposed and related methods on various models and tasks.

### Weaknesses
(1) It would be helpful to add a comparison of the computation cost of the proposed and related quantization methods.

(2) Overall this is a solid work, but some of the results are not that exciting. For example, the authors made a big claim about INT4/8 in the abstract “Remarkably, we confirm for the first time the possibility of 4-bit weight and 8-bit activation quantization for LLMs with minimal accuracy loss among LLM PTQ studies.”. However, results in table 3 and 4 show that existing methods can do INT4/8 as well, sometimes even better than the proposed method.

### Questions
Please see my concerns in weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Thanks to the authors for a well written paper.

The paper proposes a weight-activation quantization method based on low-rank weight scaling matrices. It does a nice job contrasting 2 preceding works (smoothquant & flexround) via the number of scaling parameters they each have. The authors take a middle of the ground approach which they show to be empirically beneficial, and motivate it as well using a generalization argument.

### Strengths
- good job explaining background work in smoothquant and flex round, and how they differ via number of scaling parameters
- nice motivation from flex round, ie using fig2 to show that increasing calibration samples is not viable, so therefore decrease learnable parameters
- pretty thorough set of experimental results

### Weaknesses
- how meaningful is the improvement of LRQ over FlexRound? is it possible to share std dev for the metrics you reported? For example in Table 1 Llama2 70B, the difference between LRQ has an average accuracy of 65.93 vs FlexRound 65.86. I see that sometimes the gap between LRQ and FlexRound is bigger (table 1: Llama33B), but also sometimes FlexRound is better (table 1: llama2 7b). I see that the gaps are much more sizable in Table2 (MMLU), but then smaller again for Tables 3 and 4.
- going back to the improved generalization motivation, if this interpretation were correct then wouldn’t we expect to see better performance on the common sense reasoning and MMLU tasks? I agree that LRQ’s performance is slightly better, but based on my previous comment I’m not sure how significant the gap is.

### Questions
- How many calibration samples were used to generate Fig 3? Also how large was the held out set? I understand the generalization argument, I’d just like to know how sensitive (or not) it is to the number of samples.
- It took me a couple minutes to realize the difference between Tables 1 and 2 was the evaluation task (common sense reasoning vs mmlu). I was confused because “per-tensor” and “static” were italicized. Just a small point, to somehow make it easier to digest.
- could you clarify your comment that LRQ is the first to confirm the possibility or 4Wa8 quantization? Because in Tables 3,4 LRQ does not do that much better than FlexRound.

Overall, I think the motivation and presentation of the method are very nice. The proposed low-rank solution makes a lot of sense, though methodologically there is nothing novel in the low-rank formulation. My main concern is in having a more careful interpretation of the empirical results (If I missed any, please point me to the paper). To me it seems like the gap between LRQ and FlexRound may only be significant in Table 2, and somewhat small elsewhere. Also, the claim that this methods makes 4W8A possible for the first time confuses me, as FlexRound is not that much worse (Tables 3&4). Is this claim really true? Please let me know if I am misunderstanding something. My reading of the results is that LRQ non-trivially increases the performance on MMLU tasks when using the setting of Table 2, and has minor improvements in other cases.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
