# Revisiting Random Generation Order: Ordinal-Biased Random Training for Efficient Visual Autoregressive Models

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
We observe an interesting Ordinal Asymmetry phenomenon when training visual autoregressive (AR) generators with randomized generation paths: *early tokens*, due to limited context, suffer higher losses and primarily capture *global structure*, while *later tokens*, with richer context, incur lower losses and refine *local detail*.
This suggests that conventional randomized training must optimize two qualitatively different ordinal subproblems at once.
From a curriculum perspective, training can focus on one of the two ordinal subproblems instead of optimizing both simultaneously.
Therefore, we propose Ordinal-biased Random Training (ORT), a simple strategy that first biases loss weights toward later tokens or early tokens and then gradually anneals to uniform weighting, ensuring that both global structure and local detail are learned.
Specifically, we implement ORT with an *ordinal focal loss* that assigns position-dependent weights; the schedule is controllable and can emphasize either early or late tokens.
In practice, ORT shows a striking *sudden convergence*: gradient norms collapse sharply during the middle of randomized-path training, providing clear evidence that late-biased weighting accelerates early-stage optimization.
Experiments on ImageNet-256 with RAR validate our analysis:  ORT halves the randomized training phase (200→100 epochs, $2\times$ faster) while maintaining FID comparable to the 400-epoch RAR-XL baseline.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper contributes both a conceptual insight that ordinal token positions play distinct roles in AR visual generation and a practical training strategy that leverages this insight to significantly improve training efficiency without altering the model architecture or inference process.

### Strengths
The paper presents an incremental contribution by proposing Ordinal-biased Random Training (ORT) to improve the efficiency of visual autoregressive models that originality lies mainly in reinterpreting token order as a curriculum-learning problem rather than introducing a fundamentally new model architecture. Its observations regarding early and late tokens may offer some insightful implications.

### Weaknesses
1.Insufficient effectiveness of the ORT mechanism: In Table 2, the RAR model with the ORT mechanism shows decreased metrics (e.g., FID, IS), indicating that ORT does not improve generation quality. More experiments are needed to clearly demonstrate its effectiveness.

2.The reliability of the metrics is questionable. The metrics for the baseline model (RAR-B) in Table 6 (FID=1.95, IS=290.5) are clearly inconsistent with those for the RAR model in Table 2 (FID=2.32, IS=277.31). Please provide a reasonable explanation.

3.Need for more quantitative experiments: Table 2 only shows the results of applying ORT to RAR model. Please test the ORT mechanism on more AR-based comparison methods to better validate its effectiveness.

4.Need for more qualitative comparisons: Figure 5 includes only one comparison. Please add qualitative results comparing more methods, such as those based on diffusion and VAR models.

### Questions
How general is the observed Ordinal Asymmetry phenomenon? Is it consistent across datasets or model architectures beyond ImageNet-256 and RAR? Understanding its generality is crucial before concluding that early tokens are inherently harder.

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
4

### Summary
This paper observes that in autoregressive (AR) model training, the prediction difficulty varies by token position. Based on this, it proposes a multi-stage training strategy that adjusts loss weights for tokens at different sequence positions. The method reportedly achieves performance in 300 epochs that is comparable to a 400-epoch baseline, suggesting an acceleration in convergence.

### Strengths
The paper presents clear experiments demonstrating its core premise: early tokens, due to limited context, suffer higher losses and primarily capture global structure, while later tokens, with richer context, incur lower losses and refine local detail.

### Weaknesses
1. **Limited Novelty:** The proposed method is simple, primarily involving a curriculum where the loss weight for tokens changes based on sequence position. The novelty of this re-weighting scheme appears limited.
2. **Questionable Practicality:** The practical significance of this acceleration is questionable. The training convergence of discrete AR models is already relatively fast (e.g., compared to diffusion models). A modest acceleration in *early* convergence, achieved by simple loss re-weighting, is not a significant contribution, especially as the paper does not show that this method improves the *final* generation quality.
3. **High Tuning Cost:** The method requires careful, model-specific tuning of its hyperparameters (alpha, beta) and lacks a general or adaptive approach. The cost of this hyperparameter search (requiring multiple training runs) could easily exceed the cost of simply training the baseline for more epochs, which undermines the method's entire premise of "acceleration".
4. **Incomplete Comparisons:** The experimental comparison in Table 2 is incomplete. To properly validate the claims, the table **must** include: (a) the baseline's 300-epoch result (to fairly quantify acceleration) and (b) the proposed method's 400-epoch result (to demonstrate if it ever surpasses the baseline, or just converges to the same point faster).
5. **Contradiction Between Claims and Results:** There is a direct contradiction between the paper's claims and its own results. Table 1 shows that in the first 50 epochs, the best FID comes from (alpha, beta)=(1, 0), which *emphasizes early tokens*. However, the text (lines 88-90) explicitly claims the strategy is to "emphasize later tokens at the beginning of training." This is a critical inconsistency that must be resolved.
6. **Narrow Experimental Scope:** The validation is too narrow. The claims of acceleration should be tested on a wider variety of models and tasks (e.g., Text-to-Image) to prove that the core observation and the proposed method are generally applicable.

### Questions
see weaknesses.

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
4

### Summary
* This paper focuses on optimizing the training efficiency of visual autoregressive (AR) generative models, with its core research centered on the "Ordinal Asymmetry" phenomenon under randomized generation orders. During training with randomized generation paths, early tokens face higher losses and greater optimization difficulty due to limited context, while also taking on the key task of capturing the global structure of images. In contrast, late tokens benefit from rich context, resulting in lower losses, easier optimization, and a primary focus on refining local details. This phenomenon reveals that tokens in different positions of the generation order play distinct roles in model training, providing a critical basis for optimizing training strategies.


* Building on this insight, the paper proposes the Ordinal-biased Random Training (ORT) strategy. Drawing on the idea of curriculum learning, ORT initially biases loss weights toward late, easily optimized tokens through an "Ordinal Focal Loss" to accelerate the model's initial convergence. Subsequently, it gradually transitions the weights to a uniform distribution to ensure that early, hard-to-optimize tokens are fully learned. This loss function controls weight allocation via linear interpolation parameters (α, β), allowing flexible adjustment of the emphasis on early/late tokens. Notably, ORT requires no modifications to the model architecture and can be seamlessly integrated into existing randomized-order AR frameworks


* Experimental validation on the ImageNet-256 dataset using the RAR model as a baseline demonstrates the effectiveness of ORT. Compared to the RAR-XL baseline, which requires 400 training epochs (including 200 epochs in the randomized training phase), ORT reduces the randomized training phase to 100 epochs (lowering the total training epochs to 300), doubling training efficiency while maintaining generation quality metrics such as FID on par with the baseline.

### Strengths
1. This paper is quite well-written and relatively easy to understand, with a clear motivation and an abundance of analytical experiments.  

2. I find the research focus of this paper very interesting. The issue of sequence in visual generation has long attracted the attention of researchers, including raster-scan order (as in Llamagen), next-scale order (as in VAR), and random order (as in RAR and RandAR). However, the question of which sequence is optimal remains. Starting from this context, this paper investigates the training efficiency of random generation order.  

3. I believe the authors’ analytical experiments on the impact of ordinal weighting strategies on generation quality are quite valuable. From these experiments, we can see that even with random order, the generation process still proceeds from global tokens to local tokens. Is my understanding correct here?

### Weaknesses
While I find the findings of this paper quite interesting, the improvement in practical performance is relatively marginal. For instance, with the RAR-XL model, the baseline achieves a 1.50 gFID after 400 epochs, whereas the RAR-XL-ORT only reduces the training epochs by 100 (reaching 1.53 gFID in 300 epochs). In my view, this improvement is rather limited.

Additionally, I notice that this training scheduler may be highly dependent on the parameter selection of α and β. This also implies the issue of over-tuning caused by an excessive number of parameters.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents an interesting approach to improving the training and inference of models using random paths and loss weighting. However, the lack of clarity in the implementation details and the absence of evaluation on text-to-image generation tasks make it difficult to fully assess the practical impact of the proposed method. Addressing these issues will significantly enhance the paper's quality and credibility.

### Strengths
see Summary

### Weaknesses
1. I have a question about the implementation of the "Random path" in Figure 1. In the training code, is the "Random path" implemented by truly shuffling the order of tokens in the sequence and using a traditional causal attention mask, or is it implemented by keeping the original order of tokens but setting a very complex attention mask?

2. Based on my understanding, the main contribution of ORT seems to be adding a loss weight to the "Random path." Is this correct? Other contributions appear to be relatively minor details.

3. The related work section could benefit from a more comprehensive review of VAR (Variational Autoencoder) methods, especially those that use bidirectional attention at the same scale.

If the authors address the above concerns effectively, particularly by clarifying the implementation details and expanding the evaluation to include text-to-image generation tasks, I would be willing to reconsider my assessment and potentially give a more positive score.

### Questions
no

### Soundness
3

### Presentation
3

### Contribution
3
