# Memory Retrieval and Consolidation in Large Language Models through Function Tokens

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
The remarkable success of large language models (LLMs) stems from their ability to consolidate vast amounts of knowledge into the memory during pre-training and to retrieve it from the memory during inference, enabling advanced capabilities such as knowledge memorization, instruction-following and reasoning. However, the mechanisms of memory retrieval and consolidation in LLMs remain poorly understood. In this paper, we propose the function token hypothesis to explain the workings of LLMs: During inference, function tokens activate the most predictive features from context and govern next token prediction (memory retrieval).
During pre-training, predicting the next tokens (usually content tokens) that follow function tokens increases the number of learned features of LLMs and updates the model parameters (memory consolidation). Function tokens here roughly correspond to function words in linguistics, including punctuation marks, articles, prepositions, and conjunctions, in contrast to content tokens. We provide extensive experimental evidence supporting this hypothesis. Using bipartite graph analysis, we show that a small number of function tokens activate the majority of features. Case studies further reveal how function tokens activate the most predictive features from context to direct next token prediction. We also find that during pre-training, the training loss is dominated by predicting the next content tokens following function tokens, which forces the function tokens to select the most predictive features from context.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to explain the memory retrieval and consolidation mechanisms of LLMs through a novel "Function Token Hypothesis." The authors first classify tokens based on frequency into "function tokens" (high-frequency, e.g., prepositions, articles, punctuation) and "content tokens" (low-frequency).
The hypothesis contains two core claims:
1. Memory Retrieval (Inference): Function tokens are responsible for activating the most relevant predictive features from the context to govern the generation of subsequent (usually content) tokens.
2. Memory Consolidation (Training): Predicting content tokens that follow function tokens (i.e., function -> content) is a high-difficulty task. The authors posit that the high loss from this task is the primary driver for updating model parameters and learning new features (i.e., consolidating memory).

To support this hypothesis, the authors provide three lines of empirical evidence:

- Bipartite Graph Analysis: Using Sparse Autoencoders (SAEs) to extract features, the authors build a "token-feature" bipartite graph, showing that a few function tokens are connected to (and activate) a vast majority of the model's features (high "degree").
- Activation Steering Case Studies: By causally intervening on the activations of specific features at function token positions (notably \n), they can change the model's output (e.g., factual content or language).
- Pre-training Dynamics Analysis: By training models from scratch and decomposing the loss, they show that the function -> content prediction loss is consistently the highest throughout training.

### Strengths
1. Novel Analytical Perspective: The paper's primary strength is its originality in proposing a new analytical framework. By classifying tokens into "function tokens" and "content tokens" based on frequency, the authors provide a novel and intuitive lens to investigate the long-standing question of LLM memory mechanisms. This perspective itself is a valuable contribution, opening a new direction for interpretability research.

2. Comprehensive Analysis with Novel Methods: The paper systematically analyzes the role of these function tokens across both key phases of a model's lifecycle:
	- During Inference (Retrieval): The authors introduce a novel method using SAE-extracted features to build a token-feature bipartite graph. This approach provides a quantitative measure (token degree) to support their claim that function tokens act as "hubs" for feature activation.
	- During Pre-training (Consolidation): The paper also investigates the training process by tracking the loss dynamics of different token-type transitions (e.g., function -> content). This is a compelling method for analyzing how such mechanisms might be learned.

### Weaknesses
The paper’s inferential leap from its insightful observations to its central hypothesis lacks rigor at several key points, primarily by conflating correlation with causation.

1. Linguistic Priors as an Alternative Explanation for High Token Degree: The paper argues that the high 'token degree' of function tokens (Fig 5) proves they are "responsible" for retrieval. However, this observation can be more simply explained as a direct artifact of a fundamental linguistic prior:

	- From a linguistic and informational standpoint, function tokens (e.g., "in") naturally precede a much wider and more uncertain range of subsequent tokens than content tokens (e.g., "Rowling").

	- This high contextual diversity mechanically requires function tokens to be associated with a more diverse set of features during training.

	- Therefore, the high 'degree' is a statistical correlation that confirms this linguistic property, but it does not, by itself, prove that function tokens are causally responsible for activating memory in any specific instance.

2. Correlation vs. Causation in the Training Loss Argument: This same linguistic prior also undermines the "memory consolidation" claim.

	- The paper observes that the function -> content loss is highest (Fig 9) and concludes this drives optimization, which in turn "pushes function tokens to develop the capability to reactivate..."

	- This is a logical leap. The high loss is an expected consequence of the task's inherent difficulty (high uncertainty/entropy). While high loss certainly contributes to the optimization gradient, the paper fails to prove a causal link—that this pressure specifically results in the function token becoming a "hub." It confuses the cause (inherent task difficulty) with the proposed, unproven effect (a specific hub-like capability developing in the function token).

3. Confounding Variables in the Activation Steering Experiment: The causal evidence provided (Section 3.2, Fig 6-7) is undermined by a critical flaw in its experimental design.

	- The experiment confounds two variables: "token type" (function) and "token position" (closest to the answer).

	- In a Transformer, the next-token prediction is naturally most influenced by the final hidden state of the last token. The authors intervene on the last token (\n), which happens to also be a "function token."

	- This makes it impossible to know if the effect is due to the token's functional property (as hypothesized) or its positional property (the standard mechanism). A rigorous control experiment, such as showing that a distant function token has more influence than a closer content token, is missing.

4. Lack of Practical Utility and Exploitability: The hypothesis is presented as a pure observation, but the paper fails to demonstrate its practical utility.

	- A stronger contribution would leverage this insight to propose tangible improvements. For instance, the paper does not explore:

		- (a) If this hypothesis can be used to improve training (e.g., by re-weighting the function -> content loss)?

		- (b) If it provides a new method for error detection (e.g., finding faulty activations at function tokens)?

		- (c) If it enables more robust inference-time control?

	- Without this, the hypothesis remains an "interesting phenomenon" rather than a "useful theory."

### Questions
1. The paper defines function tokens using frequency. This definition leads to the inclusion of different types of tokens, such as (a) punctuation/delimiters, (b) grammatical words (prepositions/articles), and (c) others (e.g., numbers). Is the behavior of these different token types expected to be uniform under the paper's hypothesis? Or might there be differences? Is it possible that punctuation/delimiters, rather than all tokens in the list, might be dominating the observations?

2. Regarding the confounding variables in the steering experiment (Weakness 3), it's unclear whether the observed effect is due to token type (function) or position (last token). It would be insightful to see these variables disentangled. For example, exploring the effect of steering on a distant function token versus a closer content token could be a valuable direction. The hypothesis in this paper would predict the former to be more effective, and further investigation in this direction could significantly strengthen the claim.

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
The authors find that some common tokens (that they call "function tokens") activate SAE features. They conduct experiments to examine how SAE feature activations change at function tokens based on context, and they examine the behavior of function tokens vs context tokens during pre-training.

### Strengths
This paper attempts to formalize some folk wisdom that models will aggregate information on common tokens (e.g. prepositions, punctuation). 

A particular strength is the examination of model behavior during pre-training. My (incomplete) understanding is that models learn n-gram statistics fairly early during training, then start to generalize beyond that. It would be interesting to see if models start aggregating information on "function" tokens as a way to transition past the n-gram regime.

### Weaknesses
- I think it's fairly well known that language models will use common tokens (e.g. punctuation, prepositions) to aggregate information from previous parts of the context, including some previous mechanistic work on understanding how models do this (e.g. Dissecting Recall of Factual Associations in Auto-Regressive Language Models). 
- Throughout the paper, there seems to be a confusion about what is going on; the "function tokens" are almost certainly not activating the features, but rather aggregating features from previous entities. So, while the SAE feature activation is happening in the residual stream associated with the "function token," it seems kind of misleading to say that the function tokens are activating these features.
- Classifying function tokens simply based on frequency in the training data doesn't seem like a terribly principled way to decide what's a function token.
- The comments on the loss of the function tokens during training seem kind of obvious? The model is better at predicting the most common tokens?

A few directions that could be interesting:
- Is there a principled way to figure out which tokens tend to aggregate information?
- I think the training dynamics and transitions in model behavior during pretraining (see comment above) related to function tokens is particularly interesting.

Nitpick:
- line 93: I've seen papers where people are loose with the term "neuron," but this generally refers to a weighted linear combination plus activation function (e.g. input into the FFN) — not just an axis in the residual stream

### Questions
Overall, I found the bipartite graph section to be fairly confusing. I think I get it, but just to be sure: is the idea that we are collecting all the SAE features that activate on the residual stream for a specific token in the SlimPajama dataset?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes the Function Token Hypothesis to explain memory mechanisms in LLMs. The authors define function tokens as the 122 most frequent tokens (covering 40% of corpus occurrences), primarily punctuation, articles, and prepositions, and contrast them with content tokens. The hypothesis claims that during inference, function tokens activate predictive features from context to guide next-token prediction (memory retrieval), while during pre-training, predicting content tokens after function tokens drives parameter updates and feature expansion (memory consolidation).The evidence comes from three experiments: (1) bipartite graph analysis on Gemma2-9B showing the top 10 function tokens activate 76% of SAE-decomposed features, (2) steering experiments showing that modifying feature activations at function token positions controls model outputs, and (3) pre-training experiments showing function -> content predictions have the highest loss and dominate optimization.

### Strengths
The bipartite graph analysis is creative and novel - mapping 128K tokens to 965K SAE features at scale provides a unique lens for understanding token-feature relationships. Additionally, the steering experiments are convincing as they combine multiple features (“Speak Chinese” + “UK” producing Chinese text about the UK). Overall, the experimental approach from the large-scale statistical analysis to pre-training analysis seemed sound and convincing.

### Weaknesses
The core definition conflates frequency with linguistic function, making it impossible to determine whether or not effects arise from frequency effects. No experiments match tokens by frequency or justify the arbitrary 40% threshold. Both the choice of 40% threshold and 122 tokens seems arbitrary, and probably contains tokens that would be classified as content tokens in linguistics but are included in their list. 

Only three features are analyzed in detail in section 3.2 with no statistics across all features.

The studies are limited to Gemma family probably due to availability of SAEs? 

It would be nice to see similar analysis for different layers of the models in appendix to see generalizability.

### Questions
Can experiments matching function and content tokens by frequency but showing different patterns be provided? This would require following the linguistic definition of function versus content words.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The goal of this paper is to elucidate how memory retrieval and consolidation emerge in large language models. The authors propose the *function token hypothesis*, which posits that during inference, function tokens activate the most predictive features from the context, while during pretraining, predicting content tokens conditioned on function tokens drives the model’s optimization. Here, *function tokens* refer to the most frequent tokens in the vocabulary.

To demonstrate the effectiveness of function tokens, the authors show that these tokens consistently activate the sparsest features in a sparse autoencoder. They further find that the features associated with function tokens can be leveraged for model steering. Finally, by training transformers from scratch, they observe that early stages of learning emphasize the prediction of function tokens, suggesting that such tokens play a key role in shaping the model’s representations.

### Strengths
- The paper is well-written and easy to follow.
- I find the authors’ experimental design innovative, as they employ multiple approaches and consider diverse forms of evidence (e.g., SAE features, loss curves).

### Weaknesses
- Ultimately, many results in this paper does not provide enough additional insight to the community. For example, in section 4, the authors claim that early training prioritizes predicting function tokens and subsequently, the optimization process becomes dominated by learning to predict content tokens. This seems obvious because function tokens are chosen to be the most frequent tokens.
- Experiments are limited. And I think some claims are not well supported by the experimental evidence. See question section for more.

### Questions
- On line 226, you mention that “A token is linked to a feature if it activates the feature in a context”. What exactly does it mean for a feature to be activate? When the feature is nonzero?
- The paper claim that “steering only the activations on the final function token in the prompt (‘\n’) changes the response”. But is it possible that it is due to it being the last position? Overall, I found evidence around this to be extremely weak. The paper only presents case studies without disapproving any alternative hypothesis.
- Also, is the claim that function token just combine features? It is also possible since function tokens activate most tokens. Their features are just mostly noise. So every feature got activated and might not be the effect of any composition.
- Also, in your Chinese example, it seems like the word “Chinese” help steer the answer more than any function tokens because it provides the “speak Chinese features”. Such ideas have been explored in previous papers [1].
- In figure 9, I don’t understand how you can claim that function→content drives the optimization. All curves have the same trend. In fact, green line decreases the least, which could suggest that the model is not learning this well at all.

[1] Jiang, Y., Rajendran, G., Ravikumar, P., & Aragam, B. (2024). Do llms dream of elephants (when told not to)? latent concept association and associative memory in transformers. *Advances in Neural Information Processing Systems*, *37*, 67712-67757.

### Soundness
2

### Presentation
3

### Contribution
1
