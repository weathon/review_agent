# Small molecule retrieval from tandem mass spectrometry: what are we optimizing for?

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 6, 4, 4, 6

## Abstract
One of the central challenges in the computational analysis of liquid chromatography-tandem mass spectrometry (LC-MS/MS) data is to identify the compounds underlying the output spectra.
In recent years, this problem is increasingly tackled using deep learning methods.
A common strategy involves predicting a molecular fingerprint vector from an input mass spectrum, which is then used to search for matches in a chemical compound database.
While various loss functions are employed in training these predictive models, their impact on model performance remains poorly understood. Facilitated by the recent establishment of standardized benchmarks, we investigate in this study commonly used loss functions through both an empirical and theoretical lens. Our results reveal a fundamental trade-off between the two objectives of (1) fingerprint similarity and (2) molecular retrieval. In conclusion, optimizing for more accurate fingerprint predictions typically translates into worse retrieval results, and vice versa.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study the task of retrieving a molecular structure from a target database given its tandem mass spectrum. Specifically, they provide empirical and theoretical analysis of commonly used loss functions for molecular fingerprint prediction from a tandem mass spectrum.

### Strengths
- Identification of a “paradoxical” trade-off between fingerprint prediction accuracy and downstream retrieval accuracy.  
- Novel theoretical results, including regret analyses for common loss functions.  
- The paper is generally well structured and clear.

### Weaknesses
### Major concerns

- The main limitation is the lack of practical implications. Can the authors demonstrate how their theoretical findings could, for example, guide the design of improved loss functions or inform fingerprint selection? Could these insights improve state-of-the-art results on MassSpecGym? In Table 3, the current Contrastive Emb-Cos results substantially underperform MIST on retrieval with mass-based candidate sets.  
- While the paper provides substantial new theory, it is not clear why the Bayes-optimal decision framework is the most appropriate lens here. An alternative path to understanding the paradox would be, for example, an empirical analysis of the per-bit distributions in the ground-truth fingerprints. Some bits may simply be more informative for retrieval, and bitwise or vectorwise losses may lack the capacity to reflect this by weighting every bit equally. Please better justify and explain the choice of the Bayes-optimal framework and significance of the associated theoretical findings.
- The scope of the paper is quite limited considering that fingerprint prediction is one of possible approaches to predict molecular structures from tandem mass spectra.

### Minor concerns

- Although not elaborated there, Goldman et al. previously observed a similar paradox (Nature Machine Intelligence, 2023, https://www.nature.com/articles/s42256-023-00708-3): MIST outperforms SIRIUS in fingerprint prediction accuracy but underperforms SIRIUS in retrieval accuracy (see Fig. 2e and Fig. 3b).  
- Line 406: Please clarify what exactly rule V denotes. Is it equivalent to a loss function?  
- Lines 367–368: Please justify this statement, for example by adding a reference.

### Questions
- Lines 864–866: Could the authors clarify the hypothesis that mass-based candidate sets are more challenging than formula-based candidates. Intuitively, I would expect the opposite, since molecules sharing the same formula should on average be more similar than those sharing only the same mass.

### Soundness
2

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
3

### Summary
This paper addresses the task of identifying chemical compounds from mass spectra. The approach is using fingerprint prediction,
which involves two steps: 1) predicting fingerprints from spectra, and 2) using the predicted fingerprints to search for candidate compounds in chemical databases.

Main contributions: the paper investigate various loss functions for training fingerprint prediction models and evaluate their performance in terms of both fingerprint prediction accuracy and retrieval performance (i.e. hit rate). Interestingly, the study  empirically shows that achieving higher fingerprint accuracy do not necessarily results in better retrieval performance. Furthermore, the paper provides a theoretical analysis based on Bayes-optimal decision and regret bound to explain these findings.

### Strengths
1) the paper is well written and structured.
2) the empirical findings are both interesting and quite surprising - while improving fingerprint accuracy is often considered beneficial, the results demonstrate that it does not necessarily lead to better retrieval performance, which is the ultimate goal in the compound identification task.
3) the theoretical analysis provides an insightful explanation for this observed phenomenon.

### Weaknesses
1) the regret analysis in Theorem 2 and 3 is stated in terms of expected loss for Bayes-optimal decision rules, while fingerprint predictors are trained by minimizing. the empirical loss. The theory would be significantly strengthened by connecting the two: introduce the generalization error of the fingerprint predictor into the regret bounds so that regret of the decision rules is expressed as the sum of (i) regret under the empirical losses, and (ii) a provable generalization term.
2) minor issues / typos: 
- Eq (1) (line 166): y_{k}-> y^{(i)}_{k}
- Line 1134: equation 10 -> equation (10) or Eq. (10).

### Questions
See the weaknesses.

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
4

### Summary
A key problem when analyzing massspec data is to map from an observed spectrum to a molecule. One way to do this is to map from the spectrum to a fixed-length representation for a molecule and then perform database search to find candidate molecules with similar representations. This paper studies an important detail of this approach: the loss function used when training the spectrum -> representation function. Different loss functions can provide very different performance on the downstream molecule retrieval task, even when the loss function rewards accurate prediction of the representation.

This tradeoff is demonstrated empirically on a recent massspec benchmarking task and then analyzed theoretically.

### Strengths
The paper points out some key flaws in the algorithmic approach taken by prior work and offers some interesting observations, both empirical and theoretical, for how to navigate them.

The massspec problem is challenging and has broad applications if new methods are accurate at it.

### Weaknesses
It was hard for me to understand what the central contribution of this paper was. It targets an interesting application, but does not provide a new modeling technique or achieve SOTA performance on the application. It provides an interesting observation about the impact of different loss functions on eval metrics. Then, it does some theoretical work to show why this dependence on the loss function may exist. If reviewing as an applied paper, the novelty seems limited because the task and models are established. If reviewing as a theory paper, I would need to see more exposition on the relationship between these theory results and other work (e.g., related regret bounds). The empirical observation and theory, that optimizing for a loss function that isn't a good surrogate for the downstream metric can lead to bad performance on the downstream metric, is not particularly surprising.

It was hard for me to understand the MassspecGym results, since the relationship to prior work was unclear. See question below.

### Questions
Based on the text in sec 2.2, I would have expected that there was a row in Table 1 in the 'this study' section that is the same setup as MIST (the Goldman paper). However, the MIST numbers seem to be qualitatively different from anything in the 'this study' section. What accounts for the difference?

Can you explain what is novel in the theoretical analysis (besides the particular application to massspec)? Are there results here that would be of general interest to the ICLR community that haven't appeared in prior work?

A central part of the message of the paper is that 'better' fingerprint prediction does not lead to better molecule identification. To me, it's unclear, what better prediction of a fingerprint is, given that this is just some surrogate representation for a molecule, and we don't even know what it would mean to be 'better' (but not exactly perfect) at predicting a molecule. To what extent are your results contingent on the choice of molecular fingerprints? What if you had used, for example, embeddings from a graph neural network encoder? I'm not suggesting that you run such an experiment (which would be complex to set up), but I'd like some thoughts on the robustness of your claims to the choice of that particular molecular representation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzes different loss functions for molecule retrieval from tandem mass spectra via fingerprint prediction. While the paper is well written and comprehensive, its practical significance is unclear, and the experimental setup requires revision.

### Strengths
- The paper is well written and easy to follow.
- Standard datasets are used for analysis.

### Weaknesses
- The experimental design appears to be flawed. The paper currently demonstrates (e.g., Fig. 1) that using supervised losses for fingerprint prediction (e.g., BCE) results in better fingerprint prediction, while using supervised losses for retrieval (e.g., contrastive loss) results in better retrieval performance. This is an expected outcome and does not adequately support the claim that there is a “fundamental trade-off between the two objectives.” This claim could be better supported by, for example, analyzing how the performance of both tasks evolves when training for one of the objectives. Such an analysis could reveal, for instance, that training for fingerprint prediction improves retrieval up to a certain point, after which it begins to decrease.
- The practical significance of the paper is not clear. It seems that the main message is that accurate fingerprint prediction and retrieval cannot be achieved simultaneously. However, it is unclear whether this limitation is meaningful in practice. The paper does not propose any measures to address this issue or suggest, for example, that retrieval via fingerprint prediction is suboptimal and that better retrieval performance might be achieved by bypassing fingerprint prediction.

### Questions
1. How does the retrieval performance evolve over training time when optimizing for fingerprint prediction (e.g., using BCE loss)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates how different loss functions used to train molecular fingerprint predictors influence the trade-off between fingerprint similarity and molecular retrieval performance in LC-MS/MS analysis. Through empirical evaluation and Bayes-risk-based theoretical analysis, the authors show that improving fingerprint accuracy typically worsens retrieval performance, and vice versa. The work identifies a Pareto trade-off and provides regret bounds explaining why no single loss function can optimize both objectives simultaneously.

### Strengths
1. Clearly identifies and explains a practically important but previously overlooked trade-off in molecular retrieval.

2. Provides a unified theoretical framework that aligns well with empirical results.

3. Uses a standardized benchmark with appropriate splitting to ensure fair evaluation.

4. Systematic comparison across a wide range of commonly used loss functions.

### Weaknesses
1. Choice of Fingerprint Representation May Limit Generality

The study uses only Morgan fingerprints (radius=2, 4096 bits) as the molecular representation throughout all experiments. While this is a widely used fingerprint type, I am not sure whether the observed trade-off between fingerprint similarity and retrieval performance would still hold for other representations, such as MACCS keys, ECFP with different radii, topological torsions, or even learned neural fingerprints. Since different fingerprint types encode structural information with different sparsity patterns and semantic biases, I am uncertain whether the conclusions would generalize. Therefore, the scope of applicability may be narrower than it initially appears.

2. Model Architecture Might Be Relatively Simple

The fingerprint prediction model is based on a relatively straightforward MLP over binned spectra. I am not fully familiar with spectral modeling practices, but recent molecular identification methods often employ more expressive architectures (e.g., Transformers over peak sequences). Because the representational capacity of the backbone influences the learned distributions over fingerprints, I am curious whether the same Pareto trade-off curve would appear if a more expressive model were used. My confidence here is low, but this might be an important consideration for judging how generally the findings apply.

3. Limited Exploration of Potential Hybrid or Multi-Objective Training Approaches

The paper presents the trade-off clearly, but I am not entirely sure whether this trade-off is inescapable in practice, since the study does not evaluate any hybrid losses that deliberately balance retrieval and fingerprint accuracy. For example, a weighted mixture of contrastive and vectorwise objectives, curriculum training, or multi-objective optimization strategies might produce models closer to the middle region of the Pareto frontier. It would be helpful to know whether such strategies were tested and found ineffective, or simply not explored.

### Questions
- Have the authors attempted hybrid / weighted / multi-objective losses to intentionally balance retrieval and fingerprint similarity?

If so, what behaviors were observed?

### Soundness
3

### Presentation
3

### Contribution
3
