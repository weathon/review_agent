# Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?

- Decision: Accept (Poster)
- Scores: 8, 8, 6

## Abstract
Stage lighting is a vital component in live music performances, shaping an engaging experience for both musicians and audiences. In recent years, Automatic Stage Lighting Control (ASLC) has attracted growing interest due to the high costs of hiring or training professional lighting engineers. However, most existing ASLC solutions only classify music into limited categories and map them to predefined light patterns, resulting in formulaic and monotonous outcomes that lack rationality. To address this gap, this paper presents Skip-BART, an end-to-end model that directly learns from experienced lighting engineers and predict vivid, human-like stage lighting. To the best of our knowledge, this is the first work to conceptualize ASLC as a generative task rather than merely a classification problem. Our method adapts the BART model to take audio music as input and produce light hue and value (intensity) as output, incorporating a novel skip connection mechanism to enhance the relationship between music and light within the frame grid. To address the lack of available datasets, we create the first stage lighting dataset, along with several pre-training and transfer learning techniques to improve model training with limited data. We validate our method through both quantitative analysis and an human evaluation, demonstrating that Skip-BART outperforms conventional rule-based methods across all evaluation metrics and shows only a limited gap compared to real lighting engineers. The self-collected dataset, code, and trained model parameters of this paper are provided at https://github.com/RS2002/Skip-BART .

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the task of Automatic Stage Lighting Control (ASLC) by framing it as an end-to-end generative problem for the first time. The authors propose a novel end-to-end ASLC model, Skip-BART, and introduce a dedicated stage lighting dataset.

### Strengths
- The paper addresses an important task of the stage lighting generation.
- The paper's contribution is significant by introducing the end-to-end ASLC task.
- The paper is well written and easy to follow.
- Thorough experiments showing the effectiveness of the proposed method.

### Weaknesses
- The description of the pre-training stage lacks sufficient detail (see the Questions section).
- Some claims are not adequately supported by experimental evidence:
    - Allowing the model to identify which tokens have been replaced by [MASK] assists the learning process.
    - The role and impact of incorporating the discriminator
- The method does not support real-time lighting generation. While not a critical limitation, real-time generation is a compelling use case for ASLC, particularly in live performances. The reliance on an encoder-decoder architecture restricts the model to offline processing, preventing it from handling streaming music input.

### Questions
- In the pre-training stage, is it the decoder or the regressor that is responsible for recovering the embeddings of the masked segments?
- What are the targets of the decoder and the regressor during pre-training?
- What is the architectural design of the regressor?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents an engaging and well-articulated exploration of automatic stage lighting control, framing the problem not as a fixed, rule-driven process but as a generative task. The authors propose a deep learning model, Skip-BART, that translates audio into light parameters and seeks to bridge the gap between traditional handcrafted lighting rules and creative, data-driven systems. Alongside the model, the paper introduces a dataset and outlines a methodology that combines pre-training, fine-tuning, and sampling procedures.

### Strengths
I find the work’s concept compelling. The authors do a good job explaining why rule-based and procedural systems struggle to capture the expressive and performative nature of lighting design. Framing lighting generation as a creative modeling problem rather than a deterministic optimization task opens promising ground between creative AI and live performance technology.

The interdisciplinary scope of the paper feels well balanced - it takes technical rigor seriously without losing sight of the artistic intent behind lighting. The proposed model and dataset are well described, which makes the work easy to follow and potentially reproducible. I also appreciate that the evaluation includes both quantitative metrics and human judgment, acknowledging that creative systems can’t be fully understood through quantitative metrics alone.

The results indicate that the model performs comparably to professional lighting engineers within the tested scenarios, providing convincing support for the paper’s overall argument.

### Weaknesses
Overall, the paper is strong, but I think the framing of its contribution could be clearer. The discussion of data limitations in prior work is well justified - the authors point out that existing methods rely on small, coarse-grained, or biased datasets, which constrain performance. However, this feels more like a practical limitation than a fundamental research gap. It would help if the paper made clearer whether its main contribution is technical (a new model that performs better) or conceptual (reframing stage lighting as a generative, creative process). It currently aims to do both, but the balance between these two dimensions isn’t always explicit.

The human evaluation adds a relevant perspective, and it’s great that the authors use both quantitative and perceptual measures. That said, readers would benefit from a brief summary of main participant details and evaluation criteria in the main text (the info is in Appendix D, but it would be better to have a glimpse of it earlier). The main section does report the sample size (n=38), number of pieces, and metrics used, but a concise overview of demographics and procedure would make the “human-level performance” claim easier to interpret at a glance.

Lastly, while the paper acknowledges some constraints in passing - such as occasional rhythm instability and the need for more long-term temporal coherence - these points are scattered across the main text and appendices. A short, dedicated Limitations and Future Work section would help consolidate them, making it clear where the approach currently struggles (e.g., dataset bias, dependence on specific musical genres, or limited cross-domain evaluation beyond music-driven contexts). At moments, the discussion leans heavily toward optimization details like sampling temperature and adaptive weighting. While technically relevant, this can “dilute” the creative framing that makes the generative perspective so compelling in the first place.


Overall, I would recommend this work for acceptance once the points raised are addressed. The paper makes a meaningful contribution to the emerging conversation on generative modeling for creative control systems, presenting both a technical framework and an insightful reframing of stage lighting as a creative, data-driven process. Skip-BART is well-motivated and empirically supported and consistently surpasses rule-based baselines. With a few refinements (especially in clarifying the balance between the technical and conceptual contributions, summarizing key details of the human evaluation in the main text, and consolidating the discussion of limitations) I believe this work could serve as a strong reference point for future research on AI-assisted performance design and the broader intersection of generative models and creative practice.

### Questions
- Could the authors elaborate on whether Skip-BART is capable of generating new or unexpected lighting behaviors, or whether it primarily reproduces stylistic patterns learned from the training data? The paper strongly supports the generative framing, but most evidence focuses on similarity to ground truth and outperforming rule-based methods. Clarifying how “generative” this process truly is (e.g. in terms of diversity or novelty) would strengthen the conceptual contribution.

- The human study is clearly designed and statistically analyzed, with 38 participants evaluating lighting across six perceptual dimensions. Still, a brief summary of participant demographics and evaluation setup in the main text (currently detailed in Appendix D) would make the study easier to follow and lend more transparency to the “human-level performance” claim.

- ​The cross-domain evaluation (using Suno-generated folk, R&B, and jazz samples) is a nice addition, but it could be discussed more explicitly in terms of limitations. For instance, how robust is the model to genres or performance settings that differ substantially from those represented in the RPMC-L2 dataset? This reflection would clarify where the method generalizes well and where domain adaptation might be needed (paving the way and presenting a gap for further work).

- ​While some constraints are mentioned in the discussion and appendices (e.g., rhythm
stability and dataset scope), consolidating them in a concise section would help readers understand the boundaries of the approach. This could include points such as dataset bias toward rock/metal performances, limited exploration of multi-light setups, and future directions for broader aesthetic or cross-genre applications.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Skip-BART, the first end-to-end deep learning framework for Automatic Stage Lighting Control (ASLC), reframing the task from a rule-driven classification problem into a generative modelling problem.
The authors adapt BART to generate lighting hue and intensity directly from audio input, introducing a skip-connection mechanism to align music and light on a frame-wise level. They also introduce transfer learning (leveraging PianoBART), pre-training via a masked language modelling objective, and a novel Restricted Stochastic Temperature-Controlled (RSTC) sampling for inference.
A new dataset, RPMC-L2, containing 699 stage lighting samples from live performances, is constructed for training and evaluation. Quantitative metrics (RMSE, MAE, corr(|∆|)) and a human evaluation with 38 participants demonstrate that Skip-BART significantly outperforms rule-based baselines and achieves performance statistically indistinguishable from professional lighting engineers

### Strengths
• The introduction of Skip-BART with skip connections for frame-level alignment is technically sound and novel.

• The method is well-motivated and supported by solid engineering: effective dataset construction, comprehensive ablations, and careful pre-training/fine-tuning strategies.

• The paper is clearly structured, with diagrams and methodological details that enhance understanding.

• This work opens a new research direction in artistic multimodal generation, bridging music information retrieval and stage performance automation.

### Weaknesses
• The dataset is domain-specific (rock/punk/metal), limiting generalization to genres such as pop, jazz, or classical. Cross-domain evaluation shows promise but remains narrow in scope.

• The pre-training details (e.g., MLM masking ratios, discriminator architecture) could be clarified further for full reproducibility.

• Real-time or multi-light control is not addressed, which would be crucial for production-grade systems.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
4
