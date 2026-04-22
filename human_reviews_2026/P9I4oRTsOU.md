# CoFrGeNet: Continued Fraction Architectures for Language Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Transformers are arguably the preferred architecture for language generation. In this paper, inspired by continued fractions, we introduce a new function class for generative modeling. The architecture family implementing this function class is named CoFrGeNets - Continued Fraction Generative Networks. We design novel architectural components based on this function class that can replace Multi-head Attention and Feed-Forward Networks in Transformer blocks while requiring much fewer parameters. We derive custom gradient formulations to optimize the proposed components more accurately and efficiently than using standard PyTorch-based gradients. Our components are a plug-in replacement requiring little change in training or inference procedures that have already been put in place for Transformer-based models thus making our approach easy to incorporate in large industrial workflows. We pre-train our models on two public text datasets - OpenWebText and GneissWeb. Results with our models show that the perplexity and performance on downstream GLUE tasks are superior or competitive with Transformer-based architectures, with two thirds to half the parameters and shorter pre-training time. We believe that future implementations customized to hardware will further bring out the true potential of our architectures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CoFrGeNet, a new architecture for language generation based on continued fractions. The authors replace Transformer attention and FFN layers with CoFrNet modules, using a continuant-based gradient formulation to reduce division operations and improve efficiency. Models are pretrained on OpenWebText and GneissWeb and show comparable or better results than GPT-2-xl on GLUE and language modeling tasks, while using fewer parameters and less training time.

### Strengths
- Novel and mathematically grounded idea — continued fractions as neural components.
- Clear derivations and theory with interpretable structure.
- Plug-and-play replacement for Transformer layers; minimal training changes needed.
- Custom gradient and training schedule improve efficiency and stability.

### Weaknesses
- No analysis of compute cost or FLOPs; efficiency claims incomplete.
- Attention complexity (Figure 2) not analyzed with respect to sequence length.
- Pseudocode or in-paper algorithmic overview would improve clarity, even though code is provided separately.
- Limited scaling experiments; unclear behavior for larger models.
- No long-context or “needle-in-a-haystack” tests to verify causal mixing.

### Questions
- Can the authors include FLOP or runtime comparisons beyond parameter count?
- What is the exact computational complexity of CoFr attention?
- Would including concise pseudocode or an algorithm outline improve readability?
- How does the model handle long-context reasoning?
- Are there training curves showing convergence and stability trends?

### Soundness
3

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
3

### Summary
This work proposes replacing the transformer attention and feed-forward (FFN) layers with faster components that are inspired by continued fractions. Since their formulations utilizes division-operations which are slower - they design a custom gradient formulation that reduces divisions during training (from d divisions to 1). They compare their models trained with the ladder with GPT-2-XL on perplexity and GLUE tasks and show competitive performance using fewer parameters.

### Strengths
The paper does seem to show that the ladder-fused models get competitive performance with fewer parameters. The ladder-setup also seems easy to plug into existing architectures.

### Weaknesses
- Latency argument needs more support - Since the final architecture for the proposed method retains the projection step and uses affine transformations - it seems like a large amount of compute is still allocated to matmul operations. Only the internal non-linearity is replaced by the optimized division introduced by the authors; It’s unclear if this is enough to claim the latency improvements (which aren’t empirically very clear as well as Table 4 shows that the best CoFreGeNet is only about 20 microseconds better than the gpt-2xl inference time ?). 

- Accuracy-Parameter efficiency trends are irregular - For Table-3, the biggest variants show the most competitive performance where the parameter efficiency isn’t as substantial. A few tasks show competitive performance at best (when Standard deviation is taken into consideration) so it’s not very evident that the method gives consistent accuracy improvements with greater efficiency. 

- Some ablations would make the stability of the method more evident: Some decisions in the paper are not very well justified right now - for instance, the effect of the training schedule adopted by the authors. Including that will make the claims more stable. 

Comment (not a weakness): Consider placing text around the wrap tables as Table 2 and Table 3 hinder natural reading significantly.

### Questions
- Can you provide any other metrics that improve while using your custom gradient change the optimization landscape (stability) ?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a new architecture that replaces attention and feed-forward layers in Transformers with continued fraction-based modules. This design reduces parameter count and computational cost while achieving competitive performance to GPT2-xl and efficient-attention baselines on GLUE and perplexity evaluations. The authors also demonstrate training and inference speed improvements, particularly when leveraging their “continuants” formulation for gradient computation.

### Strengths
1. Introduces an original mathematical formulation (continued fractions) to model dependencies, offering a new theoretical direction beyond attention.

2. Demonstrates reasonable empirical results, outperforming baselines while using fewer parameters.

3. Provides clear efficiency gains in both training and inference, showing practical relevance for language modeling.

### Weaknesses
1. My main concern lies in the evaluation scope of the proposed model. While the continued fraction-based architecture might be new and shows encouraging results, the experiments are limited to GPT2-scale models and GLUE benchmarks. This restricted evaluation leaves open questions about the model’s scalability, robustness, and generalization to more complex or diverse settings such as larger models or broader benchmarks such as reasoning and long-context tasks. A more comprehensive evaluation would strengthen the paper’s claims about efficiency and performance across different architectures and data regimes.

2. The method’s implementation complexity and generalization to other architectures (e.g., MoE or diffusion-based) remain unexplored.

### Questions
How does the model perform on larger-scale models and broader benchmarks?

### Soundness
3

### Presentation
3

### Contribution
2
