# Unveiling the Basin-Like Loss Landscape in Large Language Models

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 2, 6, 8

## Abstract
We discover the emergence of basins in the loss landscape of large language models. As model scale increases, LLMs become progressively more resilient to random perturbations in the parameter space, giving rise to expansive stability regions where models exhibit nearly identical performance, but outside of which their capabilities collapse. We observe that pre-training creates a basic capability basin, and subsequent alignment fine-tuning forms specific capability basins (e.g., safety, math, coding). Thus, we argue that benign fine-tuning confined to the basin should preserve prior capabilities. Besides, we also analyze the loss landscape for worst-case directions, which is consistently sharp and detrimental. We find that adversarial fine-tuning moves along the nearly worst-case directions, thus rapidly degrading model capabilities. Finally, we provide a theoretical analysis demonstrating that the basin size bounds the performance degradation of any fine-tuning, including the adversarial ones, while also guaranteeing the model robustness w.r.t. input perturbations, suggesting the benefit of enlarging basins.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper analyzes the geometry of the loss landscape of LLMs and, by doing so, tries to explain why adversarial fine-tuning often causes large performance drops in language models. The key idea is that during pretraining, models settle into flat regions of the loss landscape, which they call basins, where many parameter configurations lead to nearly identical performance. Fine-tuning on specific tasks expands or refines these basins in a task-dependent way. However, adversarial fine-tuning moves the model in sharp, high-curvature directions that push it out of these stable regions, leading to steep increases in loss and a degradation in performance. The authors analyze average-case versus worst-case directions and find that while the average is smooth and forgiving, worst-case directions are abrupt and risky. They substantiate these claims with a theoretical investigation, showing that fine-tuning degradation can be bounded using basin width. Narrow basins are easier to escape, making them more vulnerable to adversarial updates. Preliminary experiments suggest that the geometry of the loss landscape plays a central role in explaining when and why adversarial fine-tuning fails.

### Strengths
- The results are interesting and novel. The paper opens a new perspective on how fine-tuning affects models and which models are more amenable to fine-tuning. In particular, it could open a different way on how to judge if the finetuning stage was succefull beyond simple accuracy scores.
- The authors clearly discuss and address potential caveats of their analysis and also propose solutions.
- The choice of models and benchmarks strikes a good balance between feasibility in academic settings and making generalizable claims about larger models.

### Weaknesses
- The theoretical analysis is formally correct, yet it seems a mere application of existing theorems  in a new setting, limiting the novelty and contribution.
- If I understand Theorem 4.5, it is clear that when the first layer is sufficiently smooth, a slight perturbation will yield the same result. The main problem is that LLMs operate in token space, which means a slight perturbation in the text input can yield very different tokenization, which is the reason simple whitespace or capitalization attacks work so well. This is something you yourself acknowledge three paragraphs later, making the point of this analysis unclear. I would suggest clarifying this.
The presentation is the major weakness. The paper is hard to follow, and the main contributions or takeaways are not clearly stated. 
- The "average" and "worst-case" landscape framing is simply a renaming of standard average- and worst-case robustness, as discussed in Andriushchenko et al., which you cite. It would be better to use the established terminology.
- The following statement is unclear: "as long as subsequent benign fine-tuning remains within the basin of a specific capability,
the parameters will remain within this basin and thus will not compromise those capabilities." Since fine-tuning induces a different loss landscape, the basins may not be comparable. As this is one of the main points of the paper, it should be made crisp and formal.
- The experiment section 5.2. is too thin. Although it's understandable that large models cannot be trained from scratch, the paper would benefit from a more detailed analysis of hyperparameters and the fine-tuning procedure. A more controlled study on how basins change during fine-tuning is required. One option would be to use LoRA for models such as Qwen.
- It's unclear how your proposed optimization scheme differs from SAM (Foret et. al.) or continuous dropout (Srivastava et al.).

Minor:

The related work section would benefit from older work on flatness and adversarial robustness in earlier architectures. Relevant work includes Stutz et al., Wu et al., and Wei et al., which also explore sharpness/basins in an adversarial setting.
- In this setting, using J to indicate the evaluation functional is suboptimal. The letter J is typically reserved for the Jacobian, in this 
- Sentence incomplete: e.g., MMLU scores range from at least 0.25 and rarely exceed 0.8


Note: To manage expectations, the paper seems unfinished and rushed. In my opinion, it requires major revisions to be accepted to this conference—likely beyond what is possible in the rebuttal phase. That said, the underlying ideas are promising and interesting. Yet, I am happy to be convinced otherwise. 
I would encourage the authors to focus less on the formal analysis, as it seems too similar to existing works to be a main contribution. Putting these insights into action to improve the fine-tuning pipeline could be a highly influential contribution. 

References:

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research.

Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2020). Sharpness-aware minimization for efficiently improving generalization. arXiv preprint.

Dongxian Wu, Shu-Tao Xia, and Yisen Wang (2020). Adversarial weight perturbation helps robust gener-
alization. Advances in neural information processing systems.

David Stutz, Matthias Hein, and Bernt Schiele. Relating Adversarially Robust Generalization to Flat Minima (2021).  IEEE/CVF International Conference on Computer Vision (ICCV)

Zeming Wei, Jingyu Zhu, and Yihao Zhang (2023). Sharpness-aware minimization alone can improve adversarial robustness. arXiv preprint.

Walter, N. P., Adilova, L., Vreeken, J., & Kamp, M. (2025). When Flatness Does (Not) Guarantee Adversarial Robustness. arXiv preprint.

### Questions
- Where did you demonstrate the following " some basins are sufficiently large to match
the size of the basic capability basin (e.g., safety in Llama and Qwen), while others are smaller (e.g.,
coding in Llama and Qwen)."? I suppose it is Table 1; if so, please add a line to this paragraph, where the Table can be found.

- What is the main message of Theorem 4.5? And how does it relate to your main message.

- Your analysis shows that bigger models are more stable, which is interesting when considering that bigger networks
have theoretically at least a larger Lipschitz constant.
Can you expand on why you believe that this is the case?

- In CNNs, Walter et al. found that the main cause of these basins is the connection between geometry and confidence. I am aware that this paper was released after the submission deadline. Hence, I do not expect you to know it, and it will not influence my decision. Yet it would be interesting if you could comment on how this relates to your setting and if a similar connection can be shown here.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study the loss landscape of large language models through and how this relates to adversarial fine tuning. They report the finding that landscape exhibits 'basins' in which models perform equivalently and muse about the reasons and implications of this, but forego a rigorous analysis of this. The significance of the paper therewith escapes me; the fact that stochastic gradient descent, i.e. with small stochastic weight updates, works, trivially suggests that networks trained like this will exhibit 'basins'.

### Strengths
- I appreciate the perspective that the authors take, flatness seems a powerful tool for the analysis of learning behaviour of LLMs
- I appreciate the introduction of the Gaussian-augmented optimizer.

### Weaknesses
- The paper remains very high level, primarily reporting the finding of 'basins', but lacks sufficiently convincing and rigorous formal and empirical analyses.
- The authors report that different models (eg. Llama, Qwen, Mistral) have different basin sizes and conjecture this could mean certain of these models are more prone to comprimising safety when fine-tuned, without providing clear solid reasoning or actual evidence. 
- The authors report that substituting some tokens preserves performance -- which is a known fact.
- The authors report that current LLMs are generally sensitive to input changes -- which is a known fact.
- The authors conjecture but do not convincingly show that basins 'may have sufficient expressive power in the future'. 
- The empirical evaluation in particular is very weak. By proposing GO, the authors show it is possible to widen basins, but do not evaluate the implications of this. 
- I am not convinced that without a rigorous analysis GO is a useful contribution by itself; methods for sharpness-aware learning (e.g. SAM) already exist, and intuitively should also result in wider basins. The authors do not compare to these at all.
- I do not understand the value of experiment 5c. What does showing that compared to AdamW, GO improves learning speed for one model, and harms for another model, but not connecting this to basin-bevahiour, tell us?

### Questions
- What are the novel insights that Theorem 4.5 provides? 

- How does GO relate to sharpness-aware optimization and what are its benefits?

- Do you have empirical proof that wider basins relate to stronger adversarial robustness?

- What are the key insights that experiment 5c brings?

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
2

### Summary
In this work the loss landscape is studied. In particular, the bason-like landscape is observed, where the value of loss function did not change (almost) with permutation to the model parameters, from different most-case directions like safety, math and coding. When it comes to adversarial finetuning, the model params were moving towards the "worst-case" directions and made the model collapse. Finally a theoretical analysis is provided to demonstrate that the basin size bounds the performance degradation of any fine-tuning.

### Strengths
- It's an interesting finding about the different loss landscape patterns and the relation to catastrophic forgetting
- Theoretical analysis showed the basin size bounds the performance degradation of any fine-tuning.

### Weaknesses
- The concept of landscape is intuitive but lack some rigorous definitions. For example, "most-case landscape" and "worst-case landscape" we only have qualitative definition but no quantitative definition.
- The connection between this loss landscape and other research topic is unclear, e.g. how do we put the "saddle point" concept into this framework? 
- In high dimension parameter space, the possible "direction" is actually infinite. In this work, only a few finetune direction is tested.

### Questions
- The concept of landscape is intuitive but lack some rigorous definitions. For example, "most-case landscape" and "worst-case landscape" we only have qualitative definition but no quantitative definition. - is it possible to give a more quantitative definition? 
- The connection between this loss landscape and other research topic is unclear, e.g. how do we put the "saddle point" concept into this framework? 
- In high dimension parameter space, the possible "direction" is actually infinite. In this work, only a few finetune direction is tested. Besides the bound proving? Can we test more finetune directions, e.g there are more tasks in vision-language model.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper analyses the local loss geometry of LLMs showing a basin-like structure for 0-1-loss, and links this geometry to capabilities of the network. For that, the paper differentiates between the typical geometry in most directions of parameter space and the geometry in directions with strong curvature (worst-case). This geometry ensures robustness to perturbations in _most_ directions, but still explains vulnerability to adversarial perturbations. Based on these findings, the paper proposes a Gaussian-augmented optimizer.

### Strengths
- The analysis of the local loss geometry of LLMs is sound. It might lead to a deeper theoretical understanding of LLM training through flatness [cf 6, 7].
- The partitioning in most-case and worst-case loss surface is interesting and novel. 
- The empirical analysis is comprehensive. The normalization across heterogeneous experiments enables consistent comparison across diverse generative tasks and is a nontrivial engineering effort.
- The GA-optimizer is a tangible output of the paper.
- The paper is very well written and structured.

### Weaknesses
- The fact that basins only occur for the 0-1-loss and not for likelihoods. This could hint at basins being a byproduct of thresholding, rather than a genuine property of the loss surface. The authors are open about this limitation, though, so I do not see this as a reason for rejection.
- Averaging over many samples of 1D slices is reasonable to obtain a big picture, but it would be interesting to look at deviations, e.g., by displaying variance of the basin. It could be, after all, that the basin shape is an artifact of averaging individual geometries, rather than an actual basin. 
- The GA-optimizer is a sound idea, but it is unclear whether it performs well in practice, in particular in comparison to simple techniques like weight-noise, or SWA.
- While the empirical evaluation of the basin is rigorous, the findings are not compared to standard measures of flatness, such as the Fisher-Rao-Norm [5] or Relative Flatness [7].

### Questions
- The GA-Optimizer essentially improves flatness. How does it compare to methods that directly improve flatness, like FAM [1], or SAM [3] and its variants (although for the latter it has been questioned whether it truly leads to flatter solutions [2,10]). 
- While the paper frames its contribution around the “most-case vs. worst-case” geometry, several of the observed phenomena,. i.e., anisotropic flatness, sharp directions governing adversarial vulnerability, and general robustness to random perturbations, are consistent with findings by Walter et al. [8,9]. They study a different problem, analyzing sample-wise Hessians to understand local curvature and adversarial robustness, but their results seem to provide a natural mechanistic explanation for much of the behavior reported here. Would the authors agree?
- The link between basins and capabilities is very interesting and is indicated through correlation, but the causal link remains unclear. What would happen to capabilities if you would regularize against flatness as in Han et al. [4]?

References: 

[1] Adilova, Linara, et al. "FAM: Relative Flatness Aware Minimization." Topological, Algebraic and Geometric Learning Workshops 2023. PMLR, 2023.

[2] Andriushchenko, Maksym, and Nicolas Flammarion. "Towards understanding sharpness-aware minimization." International conference on machine learning. PMLR, 2022.

[3] Foret, Pierre, et al. "Sharpness-aware Minimization for Efficiently Improving Generalization." International Conference on Learning Representations, 2021 

[4] Han, Ting, et al. "Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking." Advances in Neural Information Processing Systems, 2025.

[5] Liang, Tengyuan, et al. "Fisher-rao metric, geometry, and complexity of neural networks." The 22nd international conference on artificial intelligence and statistics. PMLR, 2019.

[6] Neyshabur, Behnam, et al. "Exploring generalization in deep learning." Advances in neural information processing systems 30 (2017).

[7] Petzka, Henning, et al. "Relative flatness and generalization." Advances in neural information processing systems 34 (2021): 18420-18432.

[8] Walter, Nils Philipp, et al. "The uncanny valley: Exploring adversarial robustness from a flatness perspective." arXiv preprint arXiv:2405.16918 (2024).

[9] Walter, Nils Philipp, et al. "When Flatness Does (Not) Guarantee Adversarial Robustness." arXiv preprint arXiv:2510.14231 (2025).

[10] Wen, Kaiyue, Tengyu Ma, and Zhiyuan Li. "How Does Sharpness-Aware Minimization Minimizes Sharpness?." OPT 2022: Optimization for Machine Learning (NeurIPS 2022 Workshop).

### Soundness
3

### Presentation
4

### Contribution
3
