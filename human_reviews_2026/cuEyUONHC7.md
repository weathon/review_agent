# Efficacy of Data-Free Metrics: Robust and Critical Evidence From Robust and Critical Layers

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Data-free methods for analysing and understanding the layers of neural networks have offered many metrics for quantifying notions of ``strong" versus ``weak" layers, with the promise of increased interpretability. We examine the robustness and predictive power of data-free metrics under randomised control conditions across a wide range of models, datasets and architectures. Contrary to some of the literature, we find strong evidence \emph{against} the efficacy of data-free methods. We show that they are not reparametrisation-invariant even for \emph{robust} layers, that is to say layers that can be reparametrised by re-initialisation or re-randomisation without affecting the accuracy of the model. Moreover, we also show that data-free metrics cannot be used for the arguably simpler tasks of (i) distinguishing between robust layers and critical layers, i.e.\ layers that cannot be reparametrised without significantly degrading the accuracy of the model, or (ii) predicting if there will be a performance difference between re-initialisation and re-randomisation. Thus, we argue that to understand neural networks, and in particular the difference between `strong" versus ``weak" layers, we must adopt mechanistic and functional approaches, contrary to the traditional Random Matrix Theory perspective.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors empirically investigate the link between data-free metrics from Martin & Mahoney [1, 2] and the notion of “robust” and “critical” layers in deep neural networks. In particular, they investigate how such data-free metrics relate to reinitialisation or rerandomisation of a neural network layer’s weights. Indeed, depending on whether such a layer is “robust” or “critical”, such alterations of the parameters will respectively have little or large impact on model performance.

The authors start with a simple experimental setup using a 6-layers MLP trained on MNIST. They demonstrate that the correlation between Martin & Mahoney’s \$\alpha\$ and model performance under both reinitialisation and rerandomisation is inconsistent. Further, the range of the \$\alpha\$ does not allow distinguishing robust from critical layers. The author then shows that such findings generalise to other data-free metrics, as well as to larger scale models.

## References

[1] Charles H. Martin and Michael W. Mahoney. Implicit self-regularization in deep neural networks: Evidence from random matrix theory and implications for learning. Journal of Machine Learning Research, 22(165):1–73, 2021. URL http://jmlr.org/papers/v22/20-410.html.

[2] Charles H Martin, Tongsu Peng, and Michael W Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications, 12(1): 4122, 2021. URL https://www.nature.com/articles/s41467-021-24025-8.

### Strengths
I find this paper explores a very relevant topic, challenging limitations of well-established data-free metrics in determining whether the layer of a neural network is well-trained or not. Indeed, knowing the limitation of such metrics is crucial to not use them where they do not apply and draw erroneous conclusions from them.

S1: The extensiveness of the experiments provided by the authors is good. I appreciate that the paper starts with the simple example of \$\alpha\$ with a simple MLP, and then builds towards the other metrics as well as larger, common architectures such as ResNet, ViT and GPT. The additional results provided in the Appendix make the evaluation very complete.

S2: For a lot of the cases studied, the authors conclusively show that the value of the data-free metrics fails to indicate whether a layer is well- or poorly-trained after reinitialisation/rerandomisation.

S3: In addition to showing that data-free methods fail to identify the impact of reinitialisation and rerandomisation on the different layers of their models, the authors adequately demonstrate that data-driven methods can capture such changes in performance accurately. This provides a good  alternative to data-free metrics.

S4: The paper is clear, detailed and well-written. Experiments are well contextualised and related work appears to be relatively extensive.

### Weaknesses
Although I am convinced that this paper explores an important topic, there are some weaknesses in the experiment design which may compromise the results from the paper.

W1: I do not believe that using correlation or linear regression metrics is the right way to measure whether there is a link between \$\alpha\$ (or other metrics) and test accuracy. Referring to my question Q1, I understand from the introduction of the \$\alpha\$ metric that what matters is whether such \$\alpha\$ lies within the \$ \[ 2, 6 \] \$ range. Following-up on that, I believe the authors should evaluate the proportion of cases where \$ \alpha \in \[ 2, 6 \] \$ for layers before and after reinitialisation/rerandomisation. For example, to conclusively assess whether \$\alpha\$ has predictive power over which layer is robust or not to reinitialisation/rerandomisation, I think the authors should compare the proportion of alphas that fall within the \$ \[ 2, 6 \] \$ range before and after the operation, and also compare between reinitialisation and rerandomisation.

W2: Linked with my question Q2, I am unsure whether comparing *layer-wise* weight statistics (e.g. \$\alpha\$) to *model-wise* performance metrics (test accuracy) is a fair comparison. I believe it would be possible for a single layer to be poorly trained, but its representation to still be sufficiently aligned to conserve performance. This would allow for a bad value of \$\alpha\$ despite good overall performance.

W3: In Section 4.3, the authors use Activation Disagreement to measure differences in activations between original and modified layers. I suggest the Centred Kernel Alignment (CKA) be used instead. This metric is the state-of-the-art in measuring representational similarity between neural networks, and has the main advantage of being invariant to invertible linear transformations and as such is more robust to different initialisations.

W4: In Section 4.1, the authors mention that “an initialised, untrained layer of this network,
can fall, with a small but non-negligeable probability, within the optimal \$\alpha\$ value range of 2 and 6”. They do not study, however, if such layers lead to higher test accuracy without training. An interesting follow-up would be to assess whether such layers are more resistant to reinitialisation than layers with higher \$\alpha\$, since as per that metric, their initialisation already behaves well.

### Questions
Q1: In Section 2, line 91, the authors state that “a value of \$\alpha\$ between 2 and 6 as a property of a good, well-trained layer, whereas \$\alpha > 6\$ indicates that a layer is underfitted and \$\alpha < 2\$ indicates that it is overfitted”. Could the authors please clarify whether values of \$\alpha\$ matter if they fall outside the \$ \[ 2, 6 \] \$ range? For example, both a value of \$\alpha = 10\$ or \$\alpha = 1000\$ would indicate underfitting, but would that translate into strong differences in model performance? For example, for a layer which is randomly initialised, Figure 2a shows a power-law distribution of the values of \$\alpha\$ for layers whose performance should be similarly close to random guessing. 

Q2: As a follow-up to the previous question, the same definition states that the value of \$\alpha\$ describes whether some specific layer is well-trained or not. On the other hand, the performance metrics used by the authors are computed at the model-level, where other layers would still have good values of \$\alpha\$ after one has been reinitialised/rerandomised. As such, it could be argued that although one specific layer is poorly trained, the neural network itself still is well trained. A poor value of \$\alpha\$ for a specific layer would thus not be incompatible with the neural network on a whole performing well. Do the authors have any elements that could contradict this claim? What does the existing literature state about this?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
:
This paper studies data-free metrics, primarily those based on random matrix theory from Martin and Mahoney, 2021. The authors find that these metrics cannot reliably disambiguate between robust and critical layers which have been measured by model’s performance under re-initialization and re-randomization.  The paper questions causality though I believe that it is not the claim of the prior work.
In contrast they also look at data-based metrics (activation disagreement and Jensen-Shanon divergence of the softmax) and show that as more plausible metrics.

### Strengths
- The experimental framework is well-designed. Using re-initialization vs. re-randomization provides clear functional ground truth for layer importance.The paper is covering a wide range of architectures (ResNet, ViT, GPT-2).  
- The paper explicitly tests claims from Martin & Mahoney (2021), making it easy to assess.  
- Strong evidence where alpha (and other metrics) shows no correlation with test accuracy.

### Weaknesses
- The paper is purely evaluative, showing what doesn't work without offering novel alternatives beyond briefly looking at data-based metrics without further insights or extensive evaluation.  
- The paper demonstrates that correlations fail but doesn't explain why.   
- The authors briefly mention the flat minima literature (Dinh et al. 2017\) however the connection is not clear in the paper.

### Questions
- While I briefly checked Martin and Mahoney for this review, it is unclear to me if expecting alpha to stay in the same range after re-initialization or re-randomization is plausible. Given the density of Martin and Mahoney’s work the underlying assumptions and why the authors would expect alpha to be concentrated should be explained better in the paper.  
- I realize they are not directly comparable, but do the authors have insights connecting their work to the Lottery Ticket Hypothesis? The re-initialization results (in data-based methods mostly) seem related to findings about network pruning and subnetworks. This could help situate the findings in the broader interpretability literature.  
-  While the call to explore alternative methods is valuable, do the authors have concrete recommendations?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether data-free metrics can reliably distinguish between "robust" and "critical" layers in neural networks. Robust layers can be re-initialized or re-randomized without affecting model accuracy, while critical layers cannot. The authors conduct empirical experiments across multiple scales MNIST, ImageNet, and GPT2 to test various data-free metrics, including alpha (α), spectral norms, and entropy measures. Data-free metrics fail to distinguish robust from critical layers, and they cannot predict performance differences between re-initialization and re-randomization.

### Strengths
- Experiments appear thorough and sound, including ImageNet and GPT2

### Weaknesses
- It is difficult to understand exactly what problem this paper is addressing. Are there previous works that are claiming that data-free metrics should be able to distinguish robust layers from critical layers? Why would we expect to be able to solve that task without data? Why is that an important task?
- Martin & Mahoney (2021) focused on predicting model test performance using data-free metrics. It is not clear to me what motivates attempting to connect to the idea of critical layers from Zhang et al. (2022).

### Questions
- Most modern architectures use residual skip connections. Is the concept of robust and critical layers applicable to networks with residual connections?
- Why is it useful to be able to identify robust and critical layers with a data-free metric? Why would we expect it to be possible to predict without data in the first place?

### Soundness
2

### Presentation
1

### Contribution
1
