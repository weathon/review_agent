# HyperSyn: Synthesizing Instance-wise Model by Fusing Blackbox Expert via Hypernetwork

- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Pretrained experts are now ubiquitous, encouraging their ensembling to achieve improved performance. However, in many scenarios, they are exposed only through prediction APIs, creating black-box settings where weights and internal representations are unavailable. Existing black-box ensembling methods often perform poorly under out-of-domain conditions, since they rely solely on expert outputs. To highlight this limitation, we identify three types of data regions and show how current methods fail in certain cases. To address these challenges, we propose HyperSyn, a deep learning framework that synthesizes an instance-specific model for each data point using expert outputs as input to a hypernetwork. HyperSyn naturally fits the black-box setting and provides greater expressiveness, particularly when existing experts fail on unseen test domains. Extensive experiments on both synthetic and real-world datasets demonstrate that HyperSyn outperforms commonly used ensemble techniques and achieves state-of-the-art performance when the data exhibit complex and unknown domain structure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of *black-box expert synthesis*, where multiple pretrained models (experts) are only accessible through their output logits or probabilities. The authors identify that existing output-level fusion strategies (e.g., expert selection, mixture-of-experts, nonlinear aggregation) cannot handle cases where all experts fail or disagree, because the necessary information is absent from their outputs.

To formalize this, the paper introduces a three-region taxonomy (expert-covered, fusion-covered, and residual) and shows that output-only fusion has an intrinsic lower bound on achievable risk in the residual region.
To overcome this, the authors propose HyperSyn, a hypernetwork-based synthesis framework that takes expert outputs (U(x)) to generate sample-specific encoder parameters, which are then used to re-encode the original input (x). This design allows the model to revisit input information conditioned on expert predictions, while maintaining an *oracle-dominance* constraint to ensure that the synthesized model does not perform worse than the best expert on average.

Overall, the paper provides an interesting conceptual bridge between “output fusion” and “input-conditioned representation synthesis,” offering theoretical arguments and empirical results on benchmark datasets.

### Strengths
* **Novel conceptual insight:** The paper presents a clear and compelling perspective on the *information boundary* of black-box fusion, formalizing when and why output-only aggregation fails. The proposed three-region taxonomy (expert/fusion/residual) is conceptually elegant and could influence future research on ensemble learning and model combination.
* **Methodological originality:** The use of a hypernetwork to generate per-instance encoder parameters conditioned on expert outputs is an interesting and underexplored direction for black-box expert synthesis.
* **Safety guarantee:** The oracle-regret constraint is a well-motivated design to ensure “safe improvement,” which is practically meaningful in safety-critical settings such as medical or industrial applications.
* **Theoretical framing:** Theoretical results provide intuition about representational limits of output-only models and motivate why revisiting the input can help.

### Weaknesses
* **Limited experimental coverage:** The experiments are mostly conducted on smaller-scale classification tasks with relatively simple encoders. The paper lacks evaluation on larger and more diverse architectures (e.g., ViT, small LLMs) that better reflect the computational and representational complexity of modern expert systems.
* **Scalability and cost:** Hypernetworks typically scale poorly with model size, since they must output full encoder parameters for each instance. The paper does not sufficiently discuss or measure the computational overhead or memory footprint, which raises doubts about practical scalability.
* **Marginal gains over baselines:** The empirical improvement over strong fusion baselines such as FoE and FoE+X appears modest and sometimes within standard deviation. The results do not yet convincingly demonstrate that the proposed method’s additional complexity yields consistent benefits.

### Questions
1. How does HyperSyn scale when the base encoder is large or complex?
2. The improvement over FoE and FoE+X baselines is modest. Can you analyze in which regimes (e.g., OOD degree, expert disagreement) HyperSyn provides the largest gains?
3. Could HyperSyn be applied to generative or multimodal experts (e.g., text-to-image, speech-to-text), and what would need to change for such settings?
4. What is the actual runtime and parameter overhead of training the hypernetwork compared to simple output fusion?

*(If the authors can provide convincing large-scale results or clear cost-performance analyses, my recommendation could increase.)*

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
4

### Summary
This submission proposes a novel fusion method for model predictions. The authors consider a regime where several blackbox api experts are called on the same data. Their method uses that signal on two paths, one where a learned fusion directly leads to a prediction. The other uses the expert outputs as conditioning for a hypernetwork that leads to a second prediction. In experimental evaluation, the method appears to outperform other model fusion and ensembling methods methods.

### Strengths
- The setting of fusing expert API signals beyond ensembling appears interesting and relevant. 
- The decision families are convincing, and I encourage the authors to evaluate based on these categories. Ideally, a system would match or outperform the strongest single actor on each of them.
- The idea of combining learned softmax fusion with a hypernetwork appears novel. There might be something there, to leverage the signal from experts for a hypernetwork.

### Weaknesses
- The writing could be improved to emphasize clarity, especially in section 3 and 4. There are entire paragraphs that are repetitions of previous paragraphs. Symbols get defined and re-defined, sometimes differently ($U(x)$ for example). The authors included a proof for a residual region that I think could have just been a definition. There is quite a lot of boilerplate math, e.g. for ERM CE, etc. At the same time or because of that, crucial aspects of the method remain unclear, most notably the meta policy to decide whether to use the hyper network or fusion signal, see below.
- Method remains unclear. The main method identifies decision families (ll 137-147), and proposes a meta policy (ll 156-160) to decide what part of the signal to use. While that seems quite involved, it also doesn't map onto the actually proposed method in section 4. Out of the 4 decision families, only two seem to be represented. What is more, the meta policy to decide between the two predictions that seems to be made for every input seems absent. It remains unclear, what prediction is used for the experiment, and how picking a signal was chosen. There is a chance that the authors use the hypernetwork prediction always - but then what's the point of the rest of the method?
- The experiment section shows a large number of experiments on which the proposed method appears to perform well compared to other fusion methods. Notably, in tables 2 and 3, their proposed method performs worse than the oracle expert - which was the expressed goal to match. Given these results, I strongly miss experiments to build understanding or test assumptions. First, the added regret and distillation loss don't appear to have much of an impact - which should trigger some careful follow up experiments or lead to the decision to not include these components for reduced complexity. Second, The authors introduce the dicision families, but never check whether a) that intuition holds up in practice, or b) whether their hypernetwork based system does indeed help fuse predictions in cases where the oracle predictions lack information.

### Questions
- How are predictions chosen between fusion and hypernetworks?
- Why the choice to only feed the expert outputs into the hypernetwork, especially given the classification as not enough signal in expert outputs?
- How did you intend to use the decision families, if not to test your system against them or use them as switch between the different signal paths?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles black-box expert fusion: given only the outputs of frozen experts and a small labeled reference set, learn a predictor that is at least as good as the best expert and ideally better under distribution shift. The authors introduce HyperSyn, where a hypernetwork maps the concatenated expert outputs to per-example parameters of an input encoder, and its latent is passed through a shared predictor. During Training, it uses (i) supervised losses on both branches, (ii) a hinge oracle-regret term to discourage doing worse than the best expert, and (iii) a reliability-weighted distillation aligning input-aware and outputs-only latents. They also present a three-region taxonomy and simple theorems that show the limits of MoE and output-only fusion. They demonstrate the performance of their model on experiments using synthetic datasets and nine small real-world tabular datasets.

### Strengths
* The idea and motivation seem relevant. 
* They address a practical black-box composition setting. HyperSyn uses only expert outputs at training and inference to condition the hypernetwork, and it demonstrably goes beyond output-only fusion on constructed residual regimes.

### Weaknesses
* Text quality and cohesion are poor. There are many unnecessary hyphens, emphases, and bold text (e.g., “WEMOE,” lines 79 and 364–366).
* The intro claims HyperSyn strictly contains fusion by injecting input-dependent computation, but the formal section only states that A_syn​ is a different and often richer class. There is no proof that A_fus⊆A_syn​.
* FoE+X explicitly uses the raw input x alongside expert outputs, contradicting the stated scope that baselines use only expert outputs.
* All experiments are on small datasets, and reported improvements over competitors are often on the order of 1e-3, yet the text suggests they are “significantly” better. Evidence for statistical significance is limited.

### Questions
* Please report the fraction of test points where HyperSyn’s loss exceeds the oracle loss, and the average regret with confidence intervals.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the black-box expert fusion scenario, aiming to address the problem of model construction when only expert outputs (rather than internal parameters) are accessible — specifically, to build a predictive model that both possesses "Oracle Dominance" (not inferior to the best expert) and outperforms experts on out-of-distribution data. To achieve this goal, the study proposes a three-region classification method (expert coverage area, fusion coverage area, and residual area), theoretically analyzes the limitations of existing methods, and designs an architecture encompassing a hypernetwork, target encoder, fusion encoder, and shared predictor. It balances security and generalization ability through multiple loss functions (supervised loss, Oracle Regret Loss, and Fusion Distillation Loss), and ultimately conducts extensive experiments on synthetic datasets (PXOR, MDMF) and nine real-world datasets to verify the effectiveness of the proposed method.

### Strengths
1. The paper features in-depth theoretical analysis: the proposed three-region classification method provides a theoretical tool for understanding the limitations of different fusion methods and offers rigorous theorem proofs.

2. The paper conducts sufficient experiments to verify the effectiveness of the proposed method.

3. The paper elaborates on the proposed method in detail.

### Weaknesses
1. No code is provided to reproduce the experimental results.

2. Visualization analysis results are lacking.

3. Although the three-region classification method is proposed, the paper fails to clearly explain how HyperSyn automatically recognizes and adapts to these three regions during training.

4. Each instance requires generating and applying independent encoder parameters, leading to significantly higher computational and memory overhead during inference compared to traditional fusion methods. The paper does not provide comparative data on inference speed or memory usage, nor does it discuss the feasibility of its practical deployment.

### Questions
1. Why is the prediction loss of the fusion encoder chosen as the weight?

2. The structures of the hypernetwork and target encoder are designed as a single-layer MLP. Can it be considered that the training paradigm itself can achieve performance improvement? If a more complex hypernetwork structure is adopted, will the performance change significantly?

### Soundness
2

### Presentation
2

### Contribution
2
