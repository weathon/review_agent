# How Confident are Video Models? Empowering Video Models to Express their Uncertainty

- Decision: Reject
- Scores: 2, 8, 6, 6

## Abstract
Generative video models demonstrate impressive text-to-video capabilities,
spurring widespread adoption in many real-world applications. However, like
large language models (LLMs), video generation models tend to hallucinate, pro-
ducing plausible videos even when they are factually wrong. Although uncertainty
quantification (UQ) of LLMs has been extensively studied in prior work, no UQ
method for video models exists, raising critical safety concerns. To our knowl-
edge, this paper represents the first work towards quantifying the uncertainty of
video models. We present a framework for uncertainty quantification of generative
video models, consisting of: (i) a metric for evaluating the calibration of video
models based on robust rank correlation estimation with no stringent modeling
assumptions; (ii) a black-box UQ method for video models (termed S-QUBED),
which leverages latent modeling to rigorously decompose predictive uncertainty
into its aleatoric and epistemic components; and (iii) a UQ dataset to facilitate
benchmarking calibration in video models, which will be released after the review
process. By conditioning the generation task in the latent space, we disentangle
uncertainty arising due to vague task specifications from that arising from lack
of knowledge. Through extensive experiments on benchmark video datasets, we
demonstrate that S-QUBED computes calibrated total uncertainty estimates that are
negatively correlated with the task accuracy and effectively computes the aleatoric
and epistemic constituents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Text-to-video models are quickly improving and creating excitement both among researchers and users of AI.  However, like LLMs, these models are prone to hallucinate details of their output, especially when the input prompt is underspecified or underrepresented in training data.  To address this challenge, this work presents the first (to their knowledge) study of uncertainty quantification in text-to-video models.  They propose a black-box UQ method based on the epistemic/aleatoric decomposition to help identify when a text-to-video model is likely to hallucinate, and also plan to release a dataset of 40K videos for benchmarking UQ.

### Strengths
Effective uncertainty quantification is a central pillar in creating trustworthy AI systems.  While most focus on UQ in deep learning has been in image classification and more recently LLMs, it is important that these tools are extended to other fields and application areas, for example robotics or other generative media besides text.  This paper aims to take the first step towards developing a framework and tools for UQ in text-to-video systems.  This is a very solid motivation, and creates the potential for a significant contribution.

### Weaknesses
The main weakness I find is that this paper does not carefully treat the concepts of epistemic and aleatoric uncertainty, in particular by treating them primarily in terms of the input prompt rather than as properties that depend on the interaction between the model, its capacity, and the data distribution. Aleatoric uncertainty is described as randomness from prompt vagueness, while epistemic uncertainty is tied to a lack of model knowledge. This framing assumes these uncertainties are intrinsic to the prompt, but in practice, they are model- and data-dependent. For instance, if the entire training set consists of videos of cats napping on purple beds in the backs of pickup trucks, then the prompt “a cat napping on a purple bed in the back of a pickup truck” would still display high aleatoric uncertainty, not because the prompt lacks specificity, but because the data distribution itself is highly variable in that region. By focusing almost entirely on prompt semantics, the paper overlooks the fact that the distinction between epistemic and aleatoric uncertainty depends fundamentally on the model and the data it has seen.

This conceptual problem extends directly into the method. The decomposition in Equation (3) is presented as a principled separation between epistemic and aleatoric uncertainty, but in practice both quantities depend on the behavior and biases of the specific models used to estimate them. The authors estimate aleatoric uncertainty by prompting an LLM to generate refined textual variants and epistemic by sampling multiple videos from the same generative model. Both steps produce variability that arises from model architectures and training data of the various models, not from isolated intrinsic uncertainty types. What they call aleatoric uncertainty reflects the LLM’s own distribution, while their epistemic uncertainty reflects the video embedding model’s representation space, making the split depend on implementation choices rather than underlying epistemic principles. As a result, the decomposition is not theoretically or empirically meaningful.

Beyond these conceptual issues, the method relies on untested and implausible assumptions. The independence assumption discards dependence between the text prompt and the generated video, which is unlikely to hold in text-to-video generation. The estimation of entropy in embedding spaces further introduces arbitrary geometric distortions, since the embedding dimensions and projection have a major effect on the computed entropies. The authors provide no sensitivity analysis or justification for these choices, leaving the reported uncertainty values largely uninterpretable.

The experimental evaluation generally lacks rigor. The decision to use CLIPScore as the primary accuracy metric is based on a small 10 sample correlation study, an inadequate basis for methodological justification. The subsequent experiments that claim to disentangle aleatoric and epistemic uncertainty depend on opaque subsetting of data where one component is deemed zero according to the authors’ own estimators, introducing circular reasoning. These experimental protocols make the reported calibration and correlation results difficult to trust.

Overall, the proposed decomposition lacks solid conceptual grounding, the implementation does not meaningfully separate uncertainty types, and the empirical evaluation does not convincingly support the claims.

### Questions
See weaknesses.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
- The authors introduced a framework to measure the uncertainty of video generative models. 
- The framework consists of a metric for evaluating the calibration of video models based on robust rank correlation estimation.
- They also introduce S-QUBED, a black-box UQ method for video models. S-QUBED effectively distinguishes between uncertainty arising from ambiguous prompts and uncertainty stemming from the model's lack of knowledge.
- They will also release a dataset of 40K videos across diverse tasks to help benchmark calibration in video models.
- The authors used their method to disentangle and understand aleatoric and epistemic misunderstandings of the video generation models. For example, to assess epistemic misunderstanding, they generated multiple videos for the same prompt and embedded them. Then, they measured the embeddings' spread, with wider spread indicating higher epistemic uncertainty.
- For the main result of their work, they further study the correlation between accuracy and the different uncertainties. They find that when uncertainty is higher, accuracy tends to be lower. This holds for both overall uncertainty and aleatoric/epistemic misunderstanding.

### Strengths
- Uncertainty quantification of LLMs is well studied, but not studied at all for video generation models. This work was novel in that it studied uncertainty quantification of video generation models.
- The black box approach makes it accessible to evaluate any video generation model.
- The authors presented the material well, providing the necessary background to understand the motivation and importance of this work, which is especially important given its novelty.

### Weaknesses
- I would like to see empirical results and to validate S-QUBED on other open (non-API) video models, given that it is a black-box approach. The authors mentioned that different models were considered but not evaluated due to access and compute constraints. However, I believe there should be multiple open text-to-video models to evaluate S-QUBED on (e.g., OpenSora).
- Typical metrics (e.g., CLIP, PSNR) for evaluating text-to-image and text-to-video models often do not align with human judgment. Would like to see the correlation of uncertainty with human judgment metrics.

### Questions
No questions as the background, motivation, and results were presented well.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper is (to the authors’ knowledge) the **first study of uncertainty quantification (UQ) for text-to-video models**, proposing a three-part framework: (i) a **calibration metric** based on robust rank correlation between uncertainty and task accuracy, (ii) a black-box UQ method, **S-QUBED**, that uses a **latent-space factorization** to **decompose total predictive uncertainty** into **aleatoric** (prompt vagueness) and **epistemic** (model ignorance) components, and (iii) a ~**40K-video UQ dataset** for benchmarking. Experiments on VidGen-1M and Panda-70M show that S-QUBED’s total uncertainty is **significantly negatively correlated** with semantic accuracy (CLIP score), and its decomposition yields calibrated aleatoric/epistemic trends on subsets where the other source of uncertainty is minimal.

### Strengths
* Positions UQ for video generation as a first-class problem; formal **entropy decomposition** (h(V|\ell)=h(V|Z)+h(Z|\ell)) cleanly maps to epistemic vs. aleatoric sources. 
* **S-QUBED** operates without model internals, aligning with many **closed-source video models**. 
* Uses **Kendall’s τ** and demonstrates **significant negative correlation** between S-QUBED uncertainty and **CLIP accuracy**, with visuals that match the trend.  
* Empirical **disentangling** of aleatoric vs. epistemic uncertainty shows expected behavior on curated subsets. 
* Plans to release a **~40K-video UQ dataset** covering diverse tasks.

### Weaknesses
* Calibration hinges primarily on **CLIP similarity**; other perceptual metrics (SSIM/PSNR/LPIPS) show weak or insignificant correlations, raising concerns about **metric sensitivity** and potential semantic-evaluator bias. 
* Estimating **epistemic uncertainty** requires **multiple generations per latent prompt**, which the authors acknowledge as a limitation. 
* Main experiments use **Cosmos-Predict2** and two datasets; broader **model diversity** and real-world perturbations (codecs, length, audio conditions) are not deeply explored.

### Questions
1. Beyond CLIP, what **additional accuracy signals** (e.g., human semantic judgments, video-text retrieval scores, physics consistency probes) are necessary to **validate calibration** and mitigate evaluator bias? 
2. What **sampling schedules** (fewer latent prompts/videos, adaptive stopping) or **latent-space proxies** would you require to deem S-QUBED **computationally practical** without sacrificing epistemic resolution? 
3. Which **additional models/datasets** or **deployment artifacts** (compression, prompt styles, audio/no-audio) would most convincingly demonstrate that the **aleatoric/epistemic decomposition** remains **stable and calibrated** in the wild?

### Soundness
3

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
4

### Summary
The paper proposes a black-box framework that lets text-to-video models express uncertainty by decomposing predictive uncertainty into aleatoric (prompt vagueness) and epistemic (model ignorance) components. The framework is evaluated with a rank-correlation-based calibration metric, and a 40K-video UQ benchmark is released.

### Strengths
1. The paper is well-presented, well-written, and the motivation is justified.
2. The research topic of principled evaluation of synthetic videos is very timely and important.
3. The proposed dataset will be valuable.

### Weaknesses
1. The method’s evidence of **general** video-model UQ almost entirely depends on one text-to-image-to-video pipeline (Cosmos-Predict2). While I appreciate that authors state the API/compute constraints, it will be more convincing if the paper proposes potential solutions or fixes to overcome the challenge. That being said, the practicality and calibration of stronger video models shall be evaluated.
2. Please fix salient typos such as "video modes" (Page 3) and "peak signal-to-noise ration" (Page 13).

### Questions
While there are several weaknesses stated above, I believe this paper will be contributive and will provide new insights to the community. I therefore have the initial rating of 6 for this paper. Please note that my final rating will be conditioned on the soundness of the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
