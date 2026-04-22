# Norm-Bounded Low-Rank Adaptation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
In this work, we propose norm-bounded low-rank adaptation (NB-LoRA) for parameter-efficient fine tuning. NB-LoRA is a novel parameterization of low-rank weight adaptations that admits explicit bounds on each singular value of the adaptation matrix, which can thereby satisfy any prescribed unitarily invariant norm bound, including the Schatten norms (e.g., nuclear, Frobenius, spectral norm). The proposed parameterization is  unconstrained, smooth, and complete, i.e. it covers all matrices satisfying the prescribed rank and singular-value bounds. Natural language generation experiments show that NB-LoRA matches or surpasses performance of competing LoRA methods, while exhibiting stronger hyper-parameter robustness.  Vision fine-tuning experiments show that NB-LoRA can avoid model catastrophic forgetting with minor cost on adaptation performance, and compared to existing approaches it is substantially more robust to a hyper-parameters such as including learning rate, adaptation rank and number of training epochs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes NB-LoRA (Norm-Bounded Low-Rank Adaptation), a parameterization for parameter-efficient fine-tuning (PEFT) methods that enables explicit and smooth control over both the rank and unitarily invariant matrix norms of low-rank weight adaptations. The authors show that their construction can control the singular value spectrum of the adaptation matrix while maintaining completeness, and provide theoretical insights linking the method to improved initialization, gradient flow, and model forgetting mitigation.

### Strengths
1.	The NB-LoRA parameterization allows arbitrarily prescribable bounds on all singular values, enabling control over a variety of unitarily invariant matrix norms (not just Frobenius). Moreover, the authors clearly show the potential benefits of such a method with its insights provided in Section 3. 

2.	The methodology is tested across multiple backbone models, including LLaMA-2/3 (up to 70B params), Mistral-7B, and ViT-B/16, and also on diverse datasets (math, code, vision), providing a robust suite of experimental results.

3.	In terms of clarity and presentation, the paper is well-written and easy to follow. The motivation, methodology, and experimental setup are generally explained in a clear and organized manner, making the overall contribution accessible to readers.

### Weaknesses
1. Novelty: The core idea of controlling the norm of updated LoRA blocks has been extensively explored in prior works, including DeLoRA, DoRA, and QR-LoRA, among others. Various formulations of norm or magnitude regularization have already been proposed to stabilize or constrain LoRA updates. Although the authors make an effort to highlight distinctions between their approach and some of these existing methods, the conceptual novelty appears incremental rather than fundamental.

2. Similar to previous works in this area, the practical application of selecting an appropriate norm bound remains challenging. In particular, it is often unclear how to determine suitable values for the target norm in practice. Moreover, the reviewer has concerns about whether enforcing a norm constraint is necessary or beneficial in low-rank settings (see my Question 2).

3. Incomplete Experimental Comparison: Given the close similarity of the proposed algorithm to existing norm-control methods, a more comprehensive experimental comparison is necessary. In particular, evaluations against representative baselines such as DeLoRA and QR-LoRA would provide a clearer assessment of the proposed method’s advantages. However, the current experiments primarily compare only with standard LoRA and DoRA, which limits the strength of the empirical evidence supporting the claimed improvements.

### Questions
1. Norm Choice in Practice: Could the authors elaborate on situations where nuclear, Frobenius, or spectral norm control produces meaningfully different outcomes, and provide further intuition regarding the practical selection of the $p$-norm in real setups? Moreover, how to set a proper norm value in these cases?

2. [Key Issue] The reviewer is concerned about whether controlling the norm of LoRA weights is necessary or appropriate in low-rank settings. In many cases, such as when the pre-trained model is suboptimal and certain weights behave randomly for downstream tasks, larger LoRA updates may be required to compensate for these deficiencies. Moreover, based on the reviewer’s hands-on experience, for very low ranks (e.g., rank-1), the learned LoRA matrices naturally exhibit large spectral norms, and constraining them could hinder expressiveness and degrade performance. This has also been observed in numerous recent studies, where singular values often tend to be extremely large after the LoRA training. For example, In figure 1 of [1], the authors show the singular values are amplified significanlty in order to learn some new features and dominate the original weight matrix. 

3. Generality and Limitations of the Cayley-based Parameterization: Are there cases where the Cayley transformation could become numerically unstable, or where the mapping’s coverage is insufficient for adaptation, particularly for very large ranks or nearly non-square adaptation blocks?

4. The reviewer has concerns about the rationale presented around Line 123. Although the weights of the LoRA matrices may increase during training, the term $\partial L / \partial y$ typically decreases as the loss diminishes. As a result, the overall gradient magnitude may not necessarily grow excessively. Therefore, the explanation provided in this part does not seem fully convincing.

[1] Weight Spectra Induced Efficient Model Adaptation.

### Soundness
3

### Presentation
3

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
This paper proposes a reparametrization of LoRA via Cayleigh transform that enables norm-bounded fine-tuning, i.e. the learned weight delta is bounded in its distance to the pretrained weights as measured by Schatten-Norm. The paper then shows that this method has similar or better fine-tuning performance compared to baseline methods (Tab. 2 and Fig. 6), while being more robust to learning rate (Fig. 3) and mitigating catastrophic forgetting (Fig. 5). Finally, it analyzes the computational overhead of the proposed method, finding somewhat increased computational demands over baselines, which can be mitigated by using a computational setup for the backward pass through the Cayley transform especially designed for this setup (Appendix C).

### Strengths
**(S1)** The main insight (Thm. 4.2) is original and non-obvious. I find this an elegant solution.

**(S2)** Discussion of related work and positioning relative to it is excellent. The main paper in detail discusses similarities and differences to DeLoRA and PiSSA, including detailed experiments on the attainable norm bounds (Fig. 4b) and which matrices can be learned (Fig. 2).

**(S3)** The experiments are on point, to me the main evaluation of robust peft methods is not that they surpass the performance of existing methods, but roughly match their performance while retaining strengths of the original model (i.e. no catastrophic forgetting) and being less sensitive to hyperparameters, especially the learning rate. Both domains are covered in this paper and the results convincingly demonstrate the merits of the proposed method.

**(S4)** Describing the custom backward pass for the Cayleigh transform in Appendix C is a great addition to the paper and significantly increases its practical value.

**(S5)** Computational overhead incurred by the proposed method is sufficiently discussed.

**(S6)** Design decisions are generally well justified, for example, line 197 (unfortunately this equation does not have label)

**(S7)** The motivation in Sec. 3 is helpful and interesting to understand the merits of norm-bounded peft approaches.

### Weaknesses
**(W1)** Experiments only consider the high-data regime (MetaMathQA=395K samples, CodeFeedback=66.5K samples). However, to test the robustness of models, it would also be interesting to see if the proposed method makes models more robust to overfitting in low-data regime, for example, through the Dreambooth task used in DeLoRA or similar.

**(W2)** Appendices E and F mention that experiments use lr schedulers (cosine for LLMs, one-cycle for ViTs). This is unfortunate because it obscures the effect of scheduler vs. norm-bound on learning-rate robustness. Consider adding an ablation that demonstrates robustness with a constant learning rate, and ideally different schedulers, if compute budget allows.

**(W3)** I couldn't find the results for prolonged training mentioned in Appendix E (line 847). Perhaps they are in the paper, but the description is not clear.

**(W4)**  The paper convincingly shows _that_ the method works. However, is there any intuition as to _why_ the reparametrization in Eq. 3 works? I think this could help readers understand the method better and also help enable further research building on top of the proposed method.

### Minor Weaknesses:
  *  Line 18: "avoid model catastrophic forgetting without minor cost on adaptation performance": This sentence is (semantically) unclear to me.
   * References: Overall, please check carefully in which cases `\citep` is appropriate, and when `\citet`. I see many instances where this is not intuitive.
   * There are some typos and grammatical errors, for example, line 166.
   * Some equations don't have labels (mainly on page 4), but it would be convenient to have them to be able to refer to them, e.g. from other papers 
   * I find it confusing that captions appear below figures, but above tables. Consider placing captions below tables as well

### Questions
* Does the proposed method help prevent overfitting in low-data regime finetuning?
* What are the separate effects of learning rate scheduler and NB-LoRA on learning rate robustness?
* Is there any intuition that can be compactly explained in natural language, why the proposed reparameterization works?

I am already leaning toward the Rating "8 (accept)". I hope these points and other mentioned weaknesses can be addressed in the rebuttal, and I am looking forward to a constructive discussion with the authors.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a norm-bounded reparameterization for LoRA that directly controls the magnitude of low-rank updates during fine-tuning, aiming to improve training stability and reduce forgetting. The key idea is to turn a constrained optimization (on the Schatten/Frobenius/spectral norm of $\Delta W$) into an unconstrained learning problem via a smooth reparameterization, backed by theoretical guarantees. Experiments indicate solid downstream performance and less forgetting across settings.

### Strengths
- Turning norm-constrained low-rank adaptation into a smooth, complete reparameterization is novel and comes with clear theoretical underpinnings.
- The paper presents experiments in both language and vision models suggesting both performance gains and reduced forgetting, accompanied by useful analyses of training dynamics

### Weaknesses
- Gradient-dynamics mismatch in analyses/plots: The method trains *free* parameters $\tilde A,\tilde B$ that map to $A,B$ through a Cayley transformation. Under a given learning rate, the *effective* update on $A,B$ is not simply $-\eta\nabla_{A,B}L$; it is mediated by the Jacobian of the reparameterization. Therefore, comparing raw gradient norms (or update magnitudes) across methods in different parameterizations can be misleading. A fair comparison should report the induced per-step update on $A,B$ under the same learning rate (e.g., by mapping the optimizer step in $\tilde A,\tilde B$ to the resulting $\Delta A,\Delta B$. This would align the empirical evidence with the paper’s intuition that the reparameterization accelerates/steadies optimization.
- Forgetting claim not causally tied to “smaller updates”: The motivation argues that bounding the parameter update magnitude should mitigate forgetting, but the experiments do not directly validate this causal link. The paper should (i) quantify the post-training update magnitude of $\Delta W$ across methods under matched budgets, (ii) correlate these magnitudes with the measured forgetting, and (iii) show that the proposed norm bound achieves the claimed trade-off because of smaller/effectively shaped updates rather than unrelated side effects.

### Questions
- Could the authors explain more on how to pick the norm bound $\delta$ in practice? In the ViT experimental results, it seems that the performance is insensitivity within a range. However, we can imagin that extremely small $\delta$ should over-constrain adaptation while very large $\delta$ could negate the benefits claimed in the paper. Could the authors provide actionable guidance on estimating a suitabel range for $\delta$?

### Soundness
2

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
The paper proposes Norm-Bounded Low-Rank Adaptation (NB-LoRA), a PEFT method that enforces explicit norm constraints on low-rank weight updates via a smooth, complete reparameterization based on the Cayley transform. This formulation ensures stability during optimization by bounding singular values under any unitarily invariant norm. Experiments on large language models and vision transformers show that NB-LoRA achieves comparable or superior performance to existing PEFT methods while maintaining strong robustness to hyperparameters. The approach introduces minimal computational overhead and offers a principled framework for norm-controlled model adaptation.

### Strengths
1. **Clear motivation.** The paper targets a well-documented limitation of standard LoRA—small initial gradients and sensitivity to learning rate—and proposes a theoretically grounded reparameterization that tightly controls singular values and associated norms. The mapping is smooth and complete, covering all matrices within the prescribed rank and singular-value bounds
2. **Thorough empirical study.** The evaluation spans both LLMs and ViTs, includes comparisons against multiple baselines, ablations, and analyses of training dynamics, norm saturation, and hyperparameter robustness. The large-model case (LLaMA-3-70B) is also reported with memory and time.

### Weaknesses
1. **Additional Hyperparameters**: NB-LoRA introduces additional control (e.g., the norm bound $\delta$). Although the paper argues for improved robustness, this still expands the tuning surface in practice. And how to balance the performance and robustness by tuning $\delta$ is also a trade-off.
2. **Extra training time / GPU overhead.** The Cayley reparameterization (and its backward pass) adds measurable overhead relative to vanilla LoRA. But it's minor according to the paper.

### Questions
1. **Important**: Using the code from the Supplementary Material, I observe a higher training time when the micro-batch size is 1. The measured per-step times and memory are:

|         | Micro Batch Size | Time (per step) | GPU memory |
| ------- | ---------------- | --------------- | ---------- |
| LoRA    | 1                | 4.80            | 18643MiB   |
| NB-LoRA | 1                | 8.13            | 18859MiB   |
| LoRA    | 8                | 3.81            | 55309MiB   |
| NB-LoRA | 8                | 3.95            | 55601MiB   |
 I just modified my codebase according to the provided code. Specifically, in vanilla LoRA implementation, the core forward pass is:
   ```python
   result = result + lora_B(lora_A(dropout(x))) * scaling
   ```
I copied  ``Cayley(torch.autograd.Function)`` in ``code/llm/peft/tuners/lora/layer.py`` to my codebase and changed the forward pass to :
```python
   # result = result + lora_B(lora_A(dropout(x))) * scaling
   GH = Cayley.apply(torch.cat([lora_A.weight.T, lora_B.weight], dim=0))
   A, B = GH[:self.in_features, :], GH[self.in_features:, :].T
   result = result + (((dropout(x) @ A) * 1) @ B) * 2.0
```
Could the authors explain why the **time per step nearly doubles** at micro-batch size 1 while it becomes comparable at micro-batch size 8? Is this due to the cost of constructing G,HG,HG,H (including the r×rr\times rr×r inverse and custom backward), the additional autograd graph, or the loss of kernel fusion compared to standard LoRA?

**Experiment setup:** Model = LLaMA-2-7B-HF; Dataset = MetaMathQA; Global batch size = 32; Micro-batch size = 1.

2. In Table 1 (LLaMA-3-70B on GSM8K), why PiSSA appears to underperform LoRA at the smallest learning rate?
3. What is the difference of Figure 4a and Figure 3a for NB-LoRA？I notice that in Figure 4a NB-LoRA has a higher test accuracy.
4. The Cayley transform in Eq4 yields a semi-orthogonal matrix $G\in R^{(m+n) \times r }$. Splitting $G$ into blocks gives $A^T \in  R^{n \times r}$ and $B^T \in  R^{m \times r }$. However, this procedure may produce A and B that are not themselves orthogonal and may also introduce an imbalance between A and B, which appears to conflict with the discussion in the Motivating Analysis section.”

**Additional notes to the authors**: Your Hugging Face API key appears to be exposed in the uploaded supplementary material.

### Soundness
3

### Presentation
3

### Contribution
3
