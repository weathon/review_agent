# Steering Risk Preferences in Large Language Models by Aligning Behavioral and Neural Representations

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Changing the behavior of large language models (LLMs) can be as straightforward as editing the Transformer’s residual streams using appropriately constructed "steering vectors." These modifications to internal neural activations, a form of representation engineering, offer an effective and targeted means of influencing model behavior without retraining or fine-tuning the model. But how can such steering vectors be systematically identified? We propose a principled approach, which we call self-alignment, that uncovers steering vectors by aligning latent representations elicited through behavioral methods (specifically, Markov chain Monte Carlo with LLMs) with their neural counterparts. To evaluate this approach, we focus on extracting latent risk preferences from LLMs and steering their risk-related outputs using the aligned representations as steering vectors. We show that the resulting steering vectors successfully and reliably modulate LLM outputs in line with the targeted behavior.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a novel “self-aligned” method for constructing steering vectors to modulate risk-related behavior in LLMs, drawing inspiration from cognitive psychology. It first elicits the model’s risk preferences using an MCMC-based choice procedure to obtain a behavioral signal, then aligns this signal with the model’s internal activations to locate the corresponding “risk” direction in residual-stream space—the steering vector. Injecting this vector at inference time shifts outputs toward risk-seeking with a positive multiplier and toward risk-aversion with a negative one.

### Strengths
- **Novel and more interpretable steering-vector construction:** The paper introduces a **self-aligned** steering-vector method that regresses behaviorally elicited risk preferences onto residual-stream activations, yielding a layer-specific direction. This offers clearer interpretability than mean-difference/Contrastive Activation vectors by directly tying the steering direction to the model’s revealed preferences rather than hand-picked word contrasts.
- **Principled behavioral-to-neural alignment:** The method first builds a probabilistic behavioral model of the LLM’s risk preferences via an MCMC choice process over the Marschak–Machina triangle, then aligns that distribution with neural activations using Lasso (L1=10) to extract the “risk” direction. This offers a rigorous route to extract behavior-related signals for activation engineering and avoids reliance on ad hoc prompt pairs or labels
- **Principled layer selection with a quantitative criterion:** The paper defines a steerability metric (average change between positive and negative steering) and systematically sweeps layers to pick the optimal intervention point. It also reveals a meaningful cognitive hierarchy: risky decisions steer best in later layers (e.g., layers ~39–41), while risk perception steers more effectively earlier (e.g., layers ~2–28), supporting both effective practice and interpretability about where different risk processes reside in the network

### Weaknesses
- In line 282~284, authors using a very large multipliers (±900) to probe most impactable layers. I am concerned that if the steered model could output plausible ansswers in this huge steering strength? since it is common that such magnitudes often risk semantic drift or degeneration in activation engineering.
- The proposed MCMC-based alignment does not clearly dominate the alternative self-aligned CE method on risky-choice control; in Table 2, CE slightly exceeds or matches MCMC on all four classic gambles.
- Only two Gamma model is evualuated in the experimental results and other commonly used model families, like Llama , Qwen, may provide more evidences on the effectiveness of the proposed method.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a steering methodology termed "self-alignment" where steering vectors are determined by aligning model's latent behavioral representations (gamble scores etc.) and neural representations (Transformer residual stream activations). Authors specifically focus on controlling risk-seeking behavior of LLMs.

### Strengths
1. **Novelty:** Adaptation of the MCMC procedure from Noguchi et. al. (2013) by replacing people with an LLM is an interesting touch with its application in LLM steering.
2. **Originality:** Estimating the steering vector from models own preference without external datasets of opposing prompts is not very common (to my knowledge).
3. **Significance:** The techniques (in Step 1 and 2), even though individually not brand new, adopted for LLMs can be useful contribution to the ICLR community working on AI alignment.

### Weaknesses
1. **Writing:** Although the paper is well-written and presents an easy-to-follow narrative, Section 3 reads with some friction, as most mathematical objects are described verbally rather than symbolically. Explicitly casting the output samples from Step 1 into mathematical variables, passing them into Step 2, and formally expressing the lasso regression problem would lower the cognitive load required from readers to understand the method.
2. **Generality:** The prompt set construction method (random sequence of gambles) is directly inherited from Noguchi et al (2013). This makes the method inherently tailored for risk preference modeling only. While this aligns with the paper's stated goal, it also limits the method's applicability beyond that specific domain (at least not as easily extendible to most type of preferences as Contrastive Activation Addition). This might restrict the independent contribution of the paper to just replacing the human feedback with LLM responses.

### Questions
1. **Generality:** Could authors comment on potential principles to extend the methodology for steering different types of preferences?
2. **Transferability:** Although it is the easiest question to ask, could authors provide any clue if the proposed self-alignment steering method works well with different model families (e.g. Qwen, Llama)?

___

Overall, I believe the paper delivers on what it promises and the adaptation of a historical methodology seems interesting and useful for advancing LLM behavior control research.

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
This paper proposes a self-alignment method to achieve precise control over risk-related behaviors in LLMs. Through the proposed MCMCb-based method, the latent risk preference representations of the LLM are inferred from its behavioral choices. Then Lasso regression is employed to compute a steering vector that aligns behavioral and neural representations. By injecting the steering vector into the model’s residual stream at each token position, the risk-related decision-making of the model can be effectively controlled.

### Strengths
1. The paper proposes a new self-alignment method that uniquely combines the probability triangle, MCMC method, and lasso regression to manipulate risk preferences in large language models.

2. The experimental evaluation is extensive, covering three different but related tasks: risky decision-making, risk perception, and text generation. 

3. The method uses the model itself to identify steering vectors associated with risk preferences, showing great potential for practical applications.

### Weaknesses
1. The experiments are conducted only on limited Gemma-models. It is unclear whether the same performance improvement can be achieved on other large-scale models.

2. Lack of the key details. During the construction of the steering vector, the paper does not explain how the Lasso regularization coefficient, MCMC sampling steps, or injection layer selection were chosen.

3. High initialization cost. When switching to another model or modifying other attributes, the steering process must start from scratch—although inference-time cost is low, the initialization cost is high.

### Questions
1. Could the authors conduct additional experiments on other open-source large language models to further verify the effectiveness and generalizability of the proposed method?

2. Could the authors incorporate additional risky-choice tasks to examine whether the same “optimal layer” remains consistent across different categories of risk-related decision-making tasks?

3.  Could the authors clarify whether the construction of risk-related steering vectors can be made deterministic and fully guaranteed when derived from gamble-based task? If add more risk-related task can improve the final performance? 

4. It would strengthen the paper if the authors could include ablation experiments to quantify the contribution and importance of each component in the proposed framework.

5. The explanation of Step 2 in Section 3 could be expanded with a more detailed and rigorous description of how Lasso regression is used to align behavioral and neural representations.

### Soundness
2

### Presentation
2

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
The paper presents a novel approach to identifying steering vectors in an LLM via behavioral preference tests, specifically in the domain of risk preferences, and then successfully steers the model with those vectors on risk-related prompts. It first elicits granular behavioral preferences using methodology derived from the human psychology literature, then uses regression to identify activation-space predictors of those preferences, which are used as the steering vector.

### Strengths
The approach is novel, and a persuasive case is made that the MCMC method is the right way to capture the structure of risk preferences in an LLM.

### Weaknesses
The much simpler Contrastive Activation method is not offered a fair comparison. The paper's contrast of "risk" and "safety" related words would have induced a vector related to the abstract concept of risk, but the behavioral methods identify vectors related to quantitative risk preferences. Thus when steering on risk preference-related prompts (Figure 3), the former is ineffective. A more appropriate comparison would be to a vector formed by contrasting risky with safe choices. The paper's contrast vector is more appropriate for steering on risk perception-related prompts (Figure 4), and there it appears to be similarly effective to the more compute-intensive methods. The vector is also ill-suited to inducing "risk seeking" (Figure 5); again, a vector elicited by contrasts over preferences would be more appropriate.

It's very difficult to draw conclusions of any sort from the word clouds in Figure 5.

It's not clear how well this methodology would extend beyond risk preferences, nor whether it could even be applied in non-preference-related domains.

### Questions
Steering vectors of magnitude 900 on unit-normed vectors are shockingly high; I don't recall ever seeing that in the literature. Why was that magnitude chosen? How does it compare to the activation magnitudes pre-steering? Were outputs not degraded with such extreme magnitudes?

Where might this approach be applied in practice? Contrastive activations are simple and general; does this approach offer any benefit beyond steering risk preferences?

### Soundness
2

### Presentation
3

### Contribution
2
