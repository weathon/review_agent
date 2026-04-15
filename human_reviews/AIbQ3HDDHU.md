# Training and inference of large language models using 8-bit floating point

- Decision: Reject
- Scores: 6, 3, 5

## Abstract
FP8 formats are gaining popularity to boost the computational efficiency for training and inference of large deep learning models. Their main challenge is that a careful choice of scaling is needed to prevent degradation due to the reduced dynamic range compared to higher-precision formats. Although there exists ample literature about selecting such scalings for INT formats, this critical aspect has yet to be addressed for FP8. This paper presents a methodology to select the scalings for FP8 linear layers, based on dynamically updating per-tensor scales for the weights, gradients and activations. We apply this methodology to train and validate large language models of the type of GPT and Llama 2 using FP8, for model sizes ranging from 111M to 70B. To facilitate the understanding of the FP8 dynamics, our results are accompanied by plots of the per-tensor scale distribution for weights, activations and gradients during both training and inference.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper contains thorough details inference/fine-tuning with FP8 quantized linear layers in the context of language models. I believe it will be useful to the community. The main part of the method is how to choose the correct per-tensor scaling bias.

### Strengths
This paper is a key piece missing from the large scale FP8 literature. Figure 1 in particular is presented clearly and contains important details for successful FP8 inference/fine-tuning.

### Weaknesses
The largest weakness in this paper is in transparency -- it is claimed throughout (e.g., the title) that FP8 training will be demonstrated. However, results are only provided for fine-tuning. I would suggest changing the title/intro to be more clear.

### Questions
- Do the authors think that their results will hold for FP8 training from scratch?
- Do the authors believe it's possible to use any of their methodology for other layers in the network such as attention?

### Soundness
3 good

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
* The paper conducts experiments for FP8 inference and finetuning in the context of LLMs.
* It also provides various implementation details such as scaling factor calculations.

### Strengths
* The paper goes into great detail on how exactly quantization is carried out and how scaling factors are computed.
* Additional general discussion and statistics studies are presented in the Appendix.

### Weaknesses
* The paper repeatedly emphasizes "training" in FP8 (e.g., in the title, in the abstract, etc.), yet I could not find any actual training experiments, that is training a large LLM from scratch, in the paper. The paper only performs some finetuning on GLUE tasks, which is significantly less interesting given that it is comparatively cheap and FP8 speedups thus not so crucial while, in many cases, even more affordable finetuning techniques like QLoRA also work well. I think significant presentation changes are required to clarify that the paper focuses on inference and finetuning.
* Most of the methodology appears to me like standard low quantized training techniques, e.g., using additional scales that are determined dynamically, adapted directly to FP8. Could you explain more precisely what exactly is new? I also did not find a Related Work section discussing this in more detail.
* FP8 inference has been studied extensively by e.g. [3, 4] and also in the context of LLMs [1, 2]. Further, [5] finetunes (and even trains from scratch) large Transformers in FP8. Hence, the overall novelty of the work appears very low.

Unfortunately, as the paper overall does not seem to contain significant novelty, neither in methodology nor in results, I cannot recommend acceptance at this point.

[1] ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats, Wu et al.

[2] Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models, Zhang et al.

[3] FP8 Quantization: The Power of the Exponent, Kuzmin et al.

[4] FP8 versus INT8 for efficient deep learning inference, Baalen etl a.

[5] FP8 Formats for Deep Learning, Micikevicius et al.

### Questions
* What real-world inference speedups do you observe when finetuning and inferencing in FP8 using your setup?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a new methodology for selecting a scaling factor value (Exponent Bias) when representing numbers with an 8-bit floating point in deep learning training and inference. This methodology is based on roughly matching the dynamic range between the parameters and the 8-bit floating point numerical format. In particular, the exponent bias is either selected dynamically for each parameter ( FP8-AMAX ) or selected uniformly for all parameters (FP8-CSCALE). The paper explores the training of two types of large language models, namely GPT and LIama 2, using FP8 representation for model sizes ranging from 111M to 70B. The results indicate that the performance is on par with the FP16 representation.

### Strengths
1- The paper is well-written and organized.
2- The new methodology for FP8 has been evaluated on various large language models, demonstrating that this approach is generalizable.

### Weaknesses
1- The paper's contributions and novelty are not immediately clear. The methodology for calculating the Exponent Bias resembles the asymmetric quantization process for INT8, where the scaling factor is determined using a max operation. Furthermore, even within the scope of 8-bit floating point representation, determining the exponent bias based on the max operation has been explored in prior research. The author is recommended to clarify the paper's unique contributions, especially in comparison to the following studies:

[1] Tambe, Thierry, et al. "Algorithm-hardware co-design of adaptive floating-point encodings for resilient deep learning inference." 2020 57th ACM/IEEE Design Automation Conference (DAC). IEEE, 2020.

[2] Sun, Xiao, et al. "Hybrid 8-bit floating point (HFP8) training and inference for deep neural networks." Advances in Neural Information Processing Systems 32 (2019).

[3] Kuzmin, Andrey, et al. "Fp8 quantization: The power of the exponent." Advances in Neural Information Processing Systems 35 (2022): 14651-14662.

[4] Lee, Janghwan, and Jungwook Choi. "Optimizing Exponent Bias for Sub-8bit Floating-Point Inference of Fine-tuned Transformers." 2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS). IEEE, 2022.

2- Comparisons with other numerical formats, such as INT8, block floating point, logarithmic number systems, and posit, are not discussed. For instance, the results in [5,6] indicate that INT8 performance is superior for inference, even for models like the Transformer.

[5] van Baalen, Mart, et al. "FP8 versus INT8 for efficient deep learning inference." arXiv preprint arXiv:2303.17951 (2023).
[6] Zhang, Yijia, et al. "Integer or Floating Point? New Outlooks for Low-Bit Quantization on Large Language Models." arXiv preprint arXiv:2305.12356 (2023).

### Questions
What is the reason for using the max operation to compute the exponent bias? Why didn't the author consider other statistical metrics such as mean or mode? The max operation typically works best when the distribution is symmetric. However, the distribution in deep learning is often asymmetric.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
