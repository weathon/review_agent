

{0}------------------------------------------------

# ADAPTIVE CIRCUIT BEHAVIOR AND GENERALIZATION IN MECHANISTIC INTERPRETABILITY

Anonymous authors

Paper under double-blind review

## ABSTRACT

Mechanistic interpretability aims to understand the inner workings of large neural networks by identifying *circuits*, or minimal subgraphs within the model that implement algorithms responsible for performing specific tasks. These circuits are typically discovered and analyzed using a narrowly defined prompt format. However, given the abilities of large language models (LLMs) to generalize across various prompt formats for the same task, it remains unclear how well these circuits generalize. For instance, it is unclear whether the model’s generalization results from reusing the same circuit components, the components behaving differently, or the use of entirely different components. In this paper, we investigate the generality of the indirect object identification (IOI) circuit in GPT-2 small, which is well-studied and believed to implement a simple, interpretable algorithm. We evaluate its performance on prompt variants that challenge the assumptions of this algorithm. Our findings reveal that the circuit generalizes surprisingly well, *reusing* all of its components and mechanisms while only adding additional input edges. Notably, the circuit generalizes even to prompt variants where the original algorithm should fail; we discover a mechanism that explains this which we term *S2 Hacking*. Our findings indicate that circuits within LLMs may be more flexible and general than previously recognized, underscoring the importance of studying circuit generalization to better understand the broader capabilities of these models.

## 1 INTRODUCTION

Mechanistic interpretability (Elhage et al., 2021) is an increasingly prominent approach to understanding the inner workings of large neural networks. Much work in this area is dedicated to discovering *circuits*, which are subgraphs within the large model that faithfully represent how the model solves a particular task of interest (Olah et al., 2020), by identifying attention heads and paths with significant causal effects on the model output (Vig et al., 2020; Stolfo et al., 2023; Hanna et al., 2023; Prakash et al., 2024). By studying these circuits, researchers aim to uncover simple, human-interpretable algorithms that explain how the model solves the task.

These circuits are typically analyzed only within the specific prompt format used to extract them. However, modern LLMs can often solve the same task across various prompt formats. This raises important questions about how a circuit generalizes when the prompt format is varied, especially when the full model generalizes. For instance, it is unclear whether the model’s generalization results from the reuse of the same circuit components, the components behaving differently, or the use of entirely different components. Evaluating the generality of a circuit can provide a deeper understanding of the circuit’s behavior and the range of scenarios in which the circuit serves as a valid explanation for the full model’s performance.

The generality of circuits has significant implications for mechanistic interpretability. Since circuits are typically extracted and evaluated using a specific set of prompts, there is no prior expectation for them to generalize. In the worst case, a different prompt format might require a completely different circuit. Ideally, however, the same circuit would solve the task across all prompt variants, providing a general and reliable explanation for how the model performs the task. Figure 1 presents several hypotheses for how a circuit could generalize as the prompt format changes. Given that the goal of

{1}------------------------------------------------

![Figure 1: Left: A line graph showing Circuit Overlap (Y-axis, 0.0 to 1.0) versus Change in Task Format (X-axis). The graph shows five scenarios: Perfect Generalization (blue line, constant at 1.0), Strong Generalization (green line, slightly decreasing), Weak Generalization (purple line, drops to ~0.4), Bounded Generalization (yellow line, drops to ~0.3), and No Generalization (red line, drops to 0.0). Right: Two diagrams illustrating the IOI algorithm. The top diagram shows the standard IOI algorithm on the prompt 'When John and Mary went to the store, John gave a drink to Mary', identifying 'Mary' as the answer. The bottom diagram shows the algorithm failing on the DoubleIO prompt variant 'When John and Mary went to the store, Mary was happy. John gave a drink to John', where the algorithm incorrectly identifies 'John' as the answer because it is duplicated.](49ad3a646d84bcfeac02bdf2b3792a3e_img.jpg)

Figure 1: Left: A line graph showing Circuit Overlap (Y-axis, 0.0 to 1.0) versus Change in Task Format (X-axis). The graph shows five scenarios: Perfect Generalization (blue line, constant at 1.0), Strong Generalization (green line, slightly decreasing), Weak Generalization (purple line, drops to ~0.4), Bounded Generalization (yellow line, drops to ~0.3), and No Generalization (red line, drops to 0.0). Right: Two diagrams illustrating the IOI algorithm. The top diagram shows the standard IOI algorithm on the prompt 'When John and Mary went to the store, John gave a drink to Mary', identifying 'Mary' as the answer. The bottom diagram shows the algorithm failing on the DoubleIO prompt variant 'When John and Mary went to the store, Mary was happy. John gave a drink to John', where the algorithm incorrectly identifies 'John' as the answer because it is duplicated.

Figure 1: **Left:** Different scenarios for the degree to which a circuit could change as the task format changes. **Right:** The IOI algorithm (top) and the result of applying that algorithm to the DoubleIO prompt variant (bottom) where the subject and indirect object tokens are both duplicated.

mechanistic interpretability is to explain the behavior of the full model, limiting the explanation to a narrow set of prompt formats restricts the insights we can gain about the full model.

*Indirect Object Identification (IOI)* is one of the most well-studied tasks in the mechanistic interpretability literature. The model is given a prompt such as “When John and Mary went to the store, John gave a drink to \_\_\_\_”, with the expected answer being “Mary”. Wang et al. (2023) discovered a circuit in GPT-2 small that is said to implement a simple three-step algorithm: (1) identify all previous names in the sentence (“Mary”, “John”, “John”); (2) remove all names that are duplicated (“John”); and (3) output the remaining name (“Mary”). We refer to this as the *IOI algorithm*.

This algorithm appears to be largely agnostic to prompt structure aside from the names, suggesting the circuit may generalize broadly. However, it should clearly fail on prompts where both names are duplicated, since removing all duplicated names leaves no obviously correct answer. For example, consider the prompt “When Mary and John went to the store, Mary was happy. John gave a drink to \_\_\_\_”, which breaks the IOI algorithm by adding three words. Yet, GPT-2 small returns the correct answer with  $\sim 89\%$  confidence. If the model can be explained by a circuit that implements the IOI algorithm, its ability to generalize to inputs where the algorithm itself predicts failure reveals a clear discrepancy and raises critical questions about the generality of the IOI circuit.

In this paper, we investigate the generality of the IOI circuit. We study its performance and behavior on two prompt variants, *DoubleIO* and *TripleIO*, which are designed to challenge the assumptions of the IOI algorithm. Our key findings are:

- The IOI circuit vastly outperforms the full model on prompt variants where the IOI algorithm would completely fail. Despite this, most of the attention heads in the circuit still *retain their functionalities* as specified in Wang et al. (2023).
- We discover a mechanism in the IOI circuit, which we call *S2 Hacking*, that explains how the circuit is able to outperform the model on the prompt variants. However, *S2 Hacking* only appears in these variants and does not arise in the base IOI prompt format.
- On discovering new circuits for the prompt variants, we find that they *reuse* all components of the base IOI circuit, only adding edges from additional input tokens. The *DoubleIO* and *TripleIO* circuits show 92% and 85% edge overlap respectively, and 100% node overlap in both cases.

These findings reveal that the base IOI circuit generalizes surprisingly well beyond its original task design, reusing many of its heads and paths to handle prompt variants effectively. They also highlight the value of studying circuit generalization, to assess the external validity of the algorithms they are claimed to implement and unveil deeper underlying complexities. Overall, these results represent a significant step towards understanding the more general capabilities of large neural networks and further demonstrate the promise of mechanistic interpretability in achieving this goal.

{2}------------------------------------------------

## 2 BACKGROUND AND RELATED WORK

In this work, we evaluate the circuit for indirect object identification (IOI) in GPT-2 small (Wang et al., 2023) using variants of the prompt format that was originally used to discover the circuit. We refer to the original prompt format as *base IOI*. In the example from Figure 1, “John” and “Mary” are referred to as the *subject* (S) and *indirect object* (IO) tokens respectively, and the IO token is the expected answer. Each base IOI prompt contains one instance of the indirect object token (IO) and two instances of the subject token (S1, S2). We later introduce variants of this prompt that can include multiple instances of the indirect object token (IO1, IO2).

As in Wang et al. (2023), we measure the performance of circuits and the full model using the *logit difference* metric, which is defined as the difference between the log probabilities of the IO token and the S token. A larger positive logit difference indicates that the model predicts the correct token (IO) with higher probability than the incorrect token (S).

The base IOI circuit contains a set of distinct attention head types, each of which performs a specific function in the overall mechanism of the circuit. We include the base IOI circuit diagram in Figure 7, and further details can be found in Wang et al. (2023). We focus on the following head types:

- **Name Mover heads** are responsible for copying the correct token (IO) to the output. These heads are active at the END token and attend to previous names in the sentence, copy the name they attend to, and send it to the output.
- **S-Inhibition heads** ensure that the Name Mover heads focus on the IO token by modifying the queries of the Name Mover heads and suppressing attention to the duplicated tokens (S1, S2). They are active at the END token and attend to S2.
- **Duplicate Token heads** identify tokens that have appeared earlier in the sequence. These heads are active at the duplicate instance of a token (S2), attend to its first occurrence (S1), and output the position of the repeated token.
- **Previous Token heads** copy information from token S1 to the next token (S1+1), the token facilitating the transfer of sequential token information.
- **Induction heads** serve a similar function to Duplicate Token Heads. They are active at the position of the duplicated token (S2) and attend to the token that follows its previous instance (S1+1). Their output is used as a pointer to S1, and to signal that S was duplicated.

A small amount of prior work on circuit discovery has evaluated circuit generalization performance using prompts outside of the format used for their discovery. However, these evaluations are often limited and do not fully investigate how the circuit components and mechanisms behave under these prompt changes. Wang et al. (2023) evaluated the base IOI circuit on adversarial examples similar to the DoubleIO variant we use in this paper, and observed a drop in model performance. We build on this finding by evaluating the circuit itself and explaining the mechanism behind the model’s lower yet nontrivial performance. The Greater-Than (Hanna et al., 2023) and Arithmetic (Stolfo et al., 2023) circuits were evaluated on prompt variants and included circuit overlap comparisons, though these studies are mostly qualitative and not explored in further detail. The EAP-IG circuit discovery method (Hanna et al., 2024) was also evaluated using circuit overlap and cross-task faithfulness.

## 3 TESTING GENERALITY OF THE IOI CIRCUIT USING PROMPT VARIANTS

Inferring a circuit from model behavior is a critical step, but it is not sufficient on its own—such a circuit acts as a formal hypothesis about the problem-solving mechanisms of the full model, and this hypothesis should be probed in various ways to determine the circumstances in which it holds. We aim to assess the generality of the base IOI circuit and the range of circumstances in which its core mechanisms persist. To this end, we introduce two *variants* of the base IOI prompt that, according to the original description of the IOI algorithm, should be unsolvable by the base IOI circuit.

In this section, we present three key findings. First, we demonstrate that the unmodified base IOI circuit is unexpectedly capable of sustaining its performance as we systematically vary the nature of the task. Secondly, we demonstrate that the circuit significantly outperforms the model on the variants, showing consistently high logit difference scores while the model performance drops. Finally, we show that most of the attention heads in the circuit behave nearly identically to how they would

{3}------------------------------------------------

on base IOI inputs. These findings suggest that the circuit effectively solves the task exactly as it would on base IOI prompts, even though the necessary conditions for its success, as explained by the IOI algorithm, are not met by the new prompts.

### 3.1 IOI PROMPT VARIANTS

The IOI algorithm assumes that the subject (S) token is duplicated while the indirect object (IO) token is not, and accordingly functions by detecting and suppressing the duplicated token. We can test how much the performance of the base IOI circuit relies on this assumption by manipulating the number of instances of the IO token. Our study includes the following range of prompts:

**Base IOI:** When John and Mary went to the store, John gave a drink to \_\_\_\_.

**DoubleIO:** When John and Mary went to the store, Mary was happy. John gave a drink to \_\_\_\_.

**TripleIO:** When John and Mary went to the store, Mary was happy. Mary sat on a bench. John gave a drink to \_\_\_\_.

In the DoubleIO variant, both the S and IO tokens are duplicated, making it unclear which one the circuit should detect and suppress. In the TripleIO variant, the IO token appears three times and the S token appears twice. Given the stated logic of the IOI algorithm, where the most frequently duplicated token is suppressed, the circuit should return the S token since it now appears most frequently in the prompt.

### 3.2 BASE IOI CIRCUIT PERFORMANCE ON VARIANTS

We first evaluate the performance of the full model as well as the base IOI circuit on datasets of 200 prompts generated for each of the variants. Our data generation strategy follows from Wang et al. (2023); we create a set of template sentences for each variant, along with lists of names, places, and objects, and sample from these to construct full sentences.

Table 1 shows the logit difference scores for the full model and the base IOI circuit on each of the variants. Faithfulness is defined as the ratio of the circuit’s logit difference to the model’s logit difference, which is a measure of how closely the circuit’s performance aligns with that of the model.

| Task | Model Logit Difference | Circuit Logit Difference | Faithfulness |
|-|-|-|-|
| Base IOI | 3.484 | 3.119 | 0.895 |
| DoubleIO | 2.118 | 2.722 | 1.285 |
| TripleIO | 1.227 | 3.174 | 2.586 |

Table 1: Base IOI circuit performance on DoubleIO and TripleIO inputs. The circuit maintains high performance while model performance drops, and faithfulness is far from the ideal value of 1.

While we anticipated that the full model might generalize to these new tasks, the base circuit consistently outperforms the model on both DoubleIO and TripleIO variants with near-perfect accuracy. This is particularly surprising because these variants were designed with the explicit purpose of being unsolvable by the circuit if it were to execute its hypothesized algorithm exactly. We also see that the faithfulness scores for the base IOI circuit on the DoubleIO and TripleIO variants is far above 1, since the model performance is lower on the variants while the circuit performance remains consistently high. This indicates that the performance of the base IOI circuit on the prompt variants is not faithful to the full model, and inconsistent with the hypothesized explanation of the circuit.

Wang et al. (2023) also suggested using adversarial examples that are very similar to DoubleIO, though they only evaluate the performance of the full model and do not explore how the circuit performs on the same task. The performance of the full model on the DoubleIO task is still non-trivial; a logit difference of 2.138 indicates that the model predicts IO over S  $\sim$  89% of the time.

### 3.3 HEAD-LEVEL ATTENTION PATTERNS ON BASE IOI AND VARIANTS

The sharp deviation in performance between the base IOI circuit and the full model raises an important question: *Are the elements of the circuit maintaining the same functions as they had on the base*

{4}------------------------------------------------

216  
217  
218  
219  
220  
221  
222  
223  
224  
225  
226  
227  
228  
229  
230  
231  
232  
233  
234  
235  
236  
237  
238  
239  
240  
241  
242  
243  
244  
245  
246  
247  
248  
249  
250  
251  
252  
253  
254  
255  
256  
257  
258  
259  
260  
261  
262  
263  
264  
265  
266  
267  
268  
269

![Figure 2: Two bar charts showing the difference of attention at relevant positions between base IOI and DoubleIO (left) and TripleIO (right) inputs. The y-axis is 'Difference of Attention' ranging from -0.3 to 0.4. The x-axis is '(Layer, Head)' with categories: S-Inhibition, Duplicate, Induction, Previous, and Name Movers. For each category, there are two bars: Model (blue) and Circuit (orange). Red arrows point to specific bars with the label 'Deviation from model'.](7a3561af571faf036baa93f5f4b1bdb9_img.jpg)

The figure consists of two bar charts. The left chart is titled 'Difference of Attention at Relevant Position between Base IOI and DoubleIO' and the right chart is titled 'Difference of Attention at Relevant Position between Base IOI and TripleIO'. Both charts have a y-axis labeled 'Difference of Attention' ranging from -0.3 to 0.4. The x-axis is labeled '(Layer, Head)' and has five categories: S-Inhibition, Duplicate, Induction, Previous, and Name Movers. Each category has two bars: a blue bar for 'Model' and an orange bar for 'Circuit'. Red arrows point to specific bars with the label 'Deviation from model'. In the left chart, the S-Inhibition head (8.6) shows a significant deviation for the circuit. In the right chart, the Induction heads (5.5, 5.8) and the Name Mover heads show significant deviations for the circuit.

Figure 2: Two bar charts showing the difference of attention at relevant positions between base IOI and DoubleIO (left) and TripleIO (right) inputs. The y-axis is 'Difference of Attention' ranging from -0.3 to 0.4. The x-axis is '(Layer, Head)' with categories: S-Inhibition, Duplicate, Induction, Previous, and Name Movers. For each category, there are two bars: Model (blue) and Circuit (orange). Red arrows point to specific bars with the label 'Deviation from model'.

Figure 2: Deviation in attention scores from base IOI inputs to DoubleIO (left) and TripleIO (right) inputs for the base IOI circuit and full model. Nonzero values indicate deviation in behavior due to the change in prompt format. For the circuit, most heads show low deviation ( $< 0.1$ ), particularly the Name Mover heads which are responsible for returning the output. Significant differences between the circuit and model indicate that the base IOI circuit is less faithful on the prompt variants.

*IOI task, or have their functions changed in response to changes in the prompt?* To examine this further, we calculate the average difference between the attention scores of each head in the base IOI circuit for its most relevant token position (specified in Wang et al. (2023)) on the base IOI and prompt variant datasets. Since the full model may respond differently to the prompt variants, we do the same calculations for these heads in the model as well. The results are shown in Figure 2.

We observe minimal deviation in attention scores for most heads, typically within 0.05, with only S-Inhibition Head 8.6 deviating significantly for both variants. We also see deviation in Induction heads 5.5 and 5.8 for DoubleIO, though head 5.8 shows similar deviation in the full model as well. In contrast, the full model exhibits greater deviation in attention scores between base IOI inputs and the variants for several other heads, particularly the Name Mover heads. These heads are responsible for returning the output, which highlights the disparity in performance between the circuit and model.

Overall, these results suggest that most components of the base IOI circuit show lower deviations in attention patterns on base IOI inputs and the prompt variants compared to the full model. We later show in Section 5 that these heads retain their original functionalities from the base IOI circuit.

## 4 S2 HACKING: PERFORMANCE WITHOUT FAITHFULNESS

We observed in the previous section that the base IOI circuit significantly outperforms the full model on the IOI prompt variants, with most of its components maintaining similar attention patterns to base IOI inputs. In this section, we identify the source of the deviations in behavior and performance between the base IOI circuit and the full model.

The basis of the discrepancy between the circuit and the model performance is a mechanism in the base IOI circuit which we term *S2 Hacking*. In the base IOI circuit, the Induction and Duplicate Token heads are primarily active at the S2 token, which is always the incorrect answer in each of the IOI prompt variants. The outputs of these heads are then used by the S-Inhibition heads to suppress the attention on the S1 and S2 tokens. This mechanism is a key factor in how the circuit is able to return the correct (IO) token with high probability.

However, evaluating this circuit requires *knocking out* all paths that are not part of the circuit using *mean ablation* (Wang et al., 2023). In the base IOI circuit, the S2 token is the only input token that has a path from the input tokens to the END token that passes through the Duplicate Token and Induction heads. This is because the paths from all other input tokens were shown to have a low causal effect on the model output during the circuit discovery process. As a result, the paths from all other input tokens to these heads are knocked out.

{5}------------------------------------------------

![Figure 3: S2 Hacking in S-Inhibition head 8.6. Left: Attention Pattern for S2 Inhibition L8H6 at the END position. Right: Knockout procedure for evaluating circuits.](c54b3ca7603d65d4589151bc3a49d054_img.jpg)

The figure consists of two parts. The left part is a heatmap titled 'Attention Pattern for S2 Inhibition L8H6 at the END position'. It shows attention weights for a 'Model' and a 'Circuit' across tokens in the sentence 'When john and Mary went to the store , Mary was happy , john gave a drink to END'. The 'Model' row shows attention weights for tokens 'S1', 'IO1', 'IO2', 'S2', and 'END'. The 'Circuit' row shows attention weights for the same tokens. A color bar on the right indicates attention weights from 0.1 to 0.5. The right part is a diagram showing the 'Knockout procedure for evaluating circuits'. It lists tokens 'IO2 Mary', 'was happy', 'S2 john', and 'gave'. Arrows indicate the 'Original Circuit Component' (red) and 'Mean Ablated' (blue) paths. The legend indicates 'Unablated' (red arrow) and 'Mean Ablated' (blue arrow).

Figure 3: S2 Hacking in S-Inhibition head 8.6. Left: Attention Pattern for S2 Inhibition L8H6 at the END position. Right: Knockout procedure for evaluating circuits.

Figure 3: S2 Hacking in S-Inhibition head 8.6. **Left:** Attention pattern at the END position for a DoubleIO prompt. Placing all attention on the S2 token would lead to near-perfect accuracy on the task. Head 8.6 splits attention between IO2 and S2 in the full model, but in the base IOI circuit it focuses primarily on S2. **Right:** Knockout procedure for evaluating circuits, where paths that are not part of the circuit (marked in blue) are mean-ablated out. For head 8.6, the paths from all input tokens other than S2 are knocked out, leading to S2 Hacking.

This knockout procedure effectively points the circuit toward the correct answer every time, since the S2 token is the only input to the Duplicate Token heads that is not knocked out. These heads feed into the Induction and S-Inhibition heads, which are directed to only attend to subject-related tokens, S+1 and S2 respectively. As a result, the S-Inhibition heads always suppress attention on the subject tokens (S1, S2), which pushes the Name Mover heads towards returning an IO token. This mechanism of consistently suppressing the subject tokens and returning the IO token enables the base IOI circuit to solve the task across all of the prompt variants.

We refer to this phenomenon as S2 Hacking. Note that this phenomenon only occurs in the base IOI circuit, as it is a byproduct of the knockout procedure for evaluating the circuit and not actually how the full model solves the task. This is evident from how much the circuit performance deviates from the model performance. Additionally, S2 Hacking is not observed on base IOI inputs since they only have one duplicated name (S), and was discovered by generalizing the prompt format. For the rest of this section, we focus on the DoubleIO prompt variant and demonstrate how S2 Hacking occurs within the base IOI circuit. Additional details on all metrics and experiments are in Appendix A.

### 4.1 METRICS

Based on the head types and their functions specified by Wang et al. (2023), each head is typically characterized by its focus on a specific token or set of tokens. We define the following metrics to compare the attention scores of these specific tokens for each of these head types, to understand the deviations in their behavior between the base IOI circuit and the full model.

$$\text{Confidence ratio} = \frac{\text{Attn}(\text{correct})}{\text{Attn}(\text{incorrect})} \quad \text{Functional faithfulness} = \frac{\text{Attn}(\text{token}) \text{ in circuit}}{\text{Attn}(\text{token}) \text{ in model}}$$

The *circuit confidence ratio* for a given head is considered high ( $> 1$ ) if its attention score in the circuit is higher for the correct token than the incorrect token, and the *model confidence ratio* is similarly defined for the head in the full model. If the circuit confidence ratio exceeds that of the model, it suggests that the circuit is more likely to attend to the correct token than the model is.

*Functional faithfulness* scores are used to compare the attention scores between the model and the circuit for each relevant token; we focus particularly on the S2 and IO2 tokens. The ratio is close to 1 if the circuit attends to the token as much as the model does, while a value greater than 1 indicates that the circuit attends to the token more than the model does, demonstrating a difference in behavior for the head. Confidence ratios and functional faithfulness scores for all heads in the circuit are given in Figure 4, where all metrics are plotted with confidence intervals based on 50 samples.

### 4.2 TRACING THE S2 HACKING MECHANISM

Our results demonstrate that the S2 Hacking mechanism is primarily carried out through S-Inhibition head 8.6, Induction heads 5.9 and 5.5, and Duplicate head 3.0. Through this mechanism, we find that all of the Name Mover heads show higher confidence in predicting an IO token over an S token. The remaining heads in the circuit show similar behavior to their performance on base IOI prompts.

{6}------------------------------------------------

![Figure 4: Two bar charts comparing model and circuit performance. Left: Confidence Ratio for model and circuit across heads. Right: Functional Faithfulness across Heads for Subject and IO tokens.](73c3e4508cae529acf4e6c7fa70b361a_img.jpg)

Figure 4 consists of two bar charts. The left chart, titled 'Confidence Ratio for model and circuit across Heads', shows the confidence ratio for 'Model' (blue) and 'Circuit' (orange) across four categories: S-Inhibition, Induction (Layer, Head), Duplicate, and Previous. The y-axis ranges from 0 to 16. The right chart, titled 'Functional Faithfulness across Heads for Subject and IO tokens', shows functional faithfulness scores for 'S' (blue) and 'IO' (orange) tokens across the same categories. The y-axis ranges from 0.0 to 2.5. Both charts include red annotations for 'S2 Hacking'.

| Category | Model | Circuit |
|-|-|-|
| S-Inhibition | ~7.3 | ~6.0 |
| Induction (Layer, Head) | ~8.6 | ~14.5 |
| Induction (Layer, Head) | ~5.5 | ~3.0 |
| Induction (Layer, Head) | ~5.9 | ~2.0 |
| Induction (Layer, Head) | ~3.0 | ~1.0 |
| Duplicate | ~8.1 | ~4.0 |
| Duplicate | ~3.0 | ~1.0 |
| Duplicate | ~2.2 | ~1.0 |
| Previous | ~1.1 | ~1.0 |

| Category | S | IO |
|-|-|-|
| S-Inhibition | ~1.1 | ~0.8 |
| Induction (Layer, Head) | ~2.0 | ~1.3 |
| Induction (Layer, Head) | ~0.8 | ~0.6 |
| Induction (Layer, Head) | ~1.3 | ~0.8 |
| Induction (Layer, Head) | ~1.2 | ~1.3 |
| Duplicate | ~1.0 | ~1.0 |
| Duplicate | ~1.0 | ~1.0 |
| Duplicate | ~0.5 | ~1.0 |
| Previous | ~0.8 | ~1.0 |

Figure 4: Two bar charts comparing model and circuit performance. Left: Confidence Ratio for model and circuit across heads. Right: Functional Faithfulness across Heads for Subject and IO tokens.

Figure 4: **Left:** Confidence ratios for model and base IOI circuit. S2 Hacking can be seen in heads 8.6, 5.5, 5.9, and 3.0, where confidence ratio is close to 1 for the model but greater than 1 for the circuit. **Right:** Functional faithfulness scores for the S and IO tokens. The output is more likely to be correct if these heads predict S, so high values for the subject token (blue) indicate that the circuit is more confident than the model at predicting the correct answer.

The S2 Hacking mechanism starts from Duplicate head 3.0, as shown by its confidence ratio being close to 1 in the full model but significantly higher than 1 in the base IOI circuit. This indicates that the model places roughly equal attention on the S2 and IO2 tokens, but the circuit places more attention on the S2 token. The functional faithfulness score for the IO2 token is also low, indicating that the attention on the IO2 token is significantly lower in the circuit compared to the full model.

The effect of Duplicate head 3.0 cascades down to the Induction heads 5.9 and 5.5. While their confidence ratios are close to 1 in the model, indicating roughly equal attention to both the S2 and IO2 tokens, the circuit confidence ratios are significantly higher, suggesting a stronger preference for the S2 token. Additionally, the functional faithfulness scores show a greater deviation from 1, with significantly lower values for the IO2 token, indicating that attention to IO2 is significantly reduced in the circuit compared to the model.

The effects of S2 Hacking on the above heads flow into S-Inhibition head 8.6, which receives input from the Duplicate heads. In this head, the circuit confidence ratio is greater than 2 while the model confidence ratio remains close to 1. Moreover, the functional faithfulness score for the S2 token is around 2 in the circuit but closer to 1 in the model. These results indicate that the head attends roughly equally to the S2 and IO2 tokens in the full model, but in the circuit this head significantly favors the S2 token. This enables it to effectively direct the Name Mover heads to focus much less on the S2 token and thereby avoid the incorrect answer.

All the other heads in the circuit do not appear to benefit significantly from S2 Hacking. In particular, the Previous Token heads appear largely unaffected and show very little deviation in behavior between the circuit and the model. While the S-Inhibition heads 7.3 and 7.9 exhibit significantly higher confidence ratios in the circuit compared to the model, their model confidence ratios are already very high. This indicates that they predominantly attend to the S2 token in both the circuit and the model. These heads suggest how the model can still achieve nontrivial performance on the DoubleIO prompt variant, even if it is not as high as its performance on base IOI prompts.

## 5 HOW DOES GPT-2 SMALL ACTUALLY SOLVE DOUBLEIO AND TRIPLEIO?

Having established that S2 Hacking explains how the base IOI circuit is able to outperform the full model on the DoubleIO and TripleIO variants, we now investigate how the full model actually solves these variants. To do this, we discover a new circuit for each variant using the same patch patching methodology and experimental framework that was used to discover the base IOI circuit (Wang et al., 2023). In addition to explaining how the model solves the DoubleIO and TripleIO variants, these circuits reveal that all components of the base IOI circuit are reused for these variants.

{7}------------------------------------------------

### 5.1 ADDING PATHS FROM INPUT TOKENS

The S2 Hacking mechanism suggests a starting point for discovering DoubleIO and TripleIO circuits. Recall that in the base IOI circuit, S2 is the only input token with outgoing paths to the Duplicate heads, with all paths from the remaining input tokens being ablated out. For the variants, we start with the base IOI circuit and restore some of these paths from other input tokens that were originally ablated out, and see if any of them have a causal effect on the output of the model. The results are shown in Figure 5.

For DoubleIO, we observe that adding paths from the IO2 token brings the circuit’s performance closest to the full model, with a normalized faithfulness of 0.77. This is done by adding 10 edges to the base IOI circuit: two for each of the three Duplicate heads and the two Previous Token heads they depend on. In contrast, adding paths from just the IO1 token has little impact on the circuit’s performance, while adding paths from other input tokens further degrades its performance. Hence for every path from the S2 token to a Duplicate Token head, the corresponding path from the IO2 token to the same head also has a causal effect on the model output.

A similar pattern emerges in TripleIO, where adding paths from the IO2 and IO3 tokens brings the circuit’s performance closest to that of the full model, achieving a normalized faithfulness of 0.79. This requires adding 20 edges to the base IOI circuit: the same 10 edges corresponding to the IO2 token that were added to the DoubleIO circuit, plus 10 edges corresponding to the IO3 token.

![Figure 5: Two bar charts showing Logit Difference and Normalized Faithfulness for DoubleIO (left) and TripleIO (right) across different circuit edge configurations. The configurations are: Full Model, S2 only, IO2 only, S2 + IO2, and All tokens. The y-axis represents values, with a dashed line at 1.0 indicating ideal faithfulness. The legend indicates that blue bars represent Logit Difference and green bars represent Normalized Faithfulness.](7ff005f9556dc6518981bb92091d36ab_img.jpg)

**DoubleIO Data (Left Chart):**

| Configuration | Logit Difference (Blue) | Normalized Faithfulness (Green) |
|-|-|-|
| Full Model | 2.12 | 1.00 |
| S2 only | 2.72 | 1.28 |
| IO2 only | 0.33 | 0.16 |
| S2 + IO2 | 1.62 | 0.77 |
| All tokens | 1.59 | 0.75 |

**TripleIO Data (Right Chart):**

| Configuration | Logit Difference (Blue) | Normalized Faithfulness (Green) |
|-|-|-|
| Full Model | 1.23 | 1.00 |
| S2 only | 3.17 | 2.59 |
| IO2 only | 0.04 | 0.03 |
| IO3 only | 0.38 | 0.33 |
| S2 + IO2 | 1.94 | 1.58 |
| S2 + IO2 + IO3 | 0.97 | 0.79 |
| All tokens | 0.97 | 0.79 |

Figure 5: Two bar charts showing Logit Difference and Normalized Faithfulness for DoubleIO (left) and TripleIO (right) across different circuit edge configurations. The configurations are: Full Model, S2 only, IO2 only, S2 + IO2, and All tokens. The y-axis represents values, with a dashed line at 1.0 indicating ideal faithfulness. The legend indicates that blue bars represent Logit Difference and green bars represent Normalized Faithfulness.

Figure 5: Logit difference and normalized faithfulness for DoubleIO (left) and TripleIO (right) after adding paths to the Duplicate and Previous Token heads from different input tokens. For both variants, the faithfulness is closest to 1 (ideal) when including paths from the input tokens corresponding to duplicated names: S2 and IO2 for DoubleIO, and S2, IO2, and IO3 for TripleIO.

### 5.2 DISCOVERING CIRCUIT REUSE THROUGH PATH PATCHING

We use the methodology of Wang et al. (2023) to discover a circuit for the DoubleIO and TripleIO variants, starting with the identification of Name Mover heads. For each attention head and relevant input token, we compute the direct causal effect of the path that starts from the token and proceeds through the head to the final logit at the END position. This type of token-level causal effect estimation is important for variants like DoubleIO and TripleIO where multiple tokens are duplicated, since each of these duplicates can have paths to different heads.

Surprisingly, we do not observe any significant deviation in results compared to the base IOI circuit. The Name Mover heads from the base IOI circuit are just as causally relevant for the DoubleIO and TripleIO variants, and none of the other heads in the model have a substantial enough direct causal effect score to suggest the addition of a new Name Mover head to the circuit. These results indicate that the Name Mover heads from the base IOI circuit are being reused for both variants.

We observe the same pattern in the S-Inhibition heads, which we refer to more generally as Inhibition heads since they could inhibit either of the duplicated names in the prompt. To identify these heads, we compute the direct causal effects of every head in the model on the queries of the Name Mover heads. We find that all of the S-Inhibition heads from the base IOI circuit have the most significant causal effects, indicating that the DoubleIO and TripleIO circuits are also reusing these heads.

{8}------------------------------------------------

![Figure 6: Direct effect on S-inhibition heads' values in % Logit Difference for Base IOI and DoubleIO. The figure contains three heatmaps: 'BaseIOI - S2 Token', 'DoubleIO - S2 Token', and 'DoubleIO - IO2 Token'. Each heatmap shows the effect of 12 heads (x-axis) across 11 layers (y-axis). A color scale on the right indicates the % Logit Difference, ranging from -5 (red) to 5 (blue).](b93cbfb52e37619e688175a6aad9edd9_img.jpg)

Figure 6: Direct effect on S-inhibition heads' values in % Logit Difference for Base IOI and DoubleIO. The figure contains three heatmaps: 'BaseIOI - S2 Token', 'DoubleIO - S2 Token', and 'DoubleIO - IO2 Token'. Each heatmap shows the effect of 12 heads (x-axis) across 11 layers (y-axis). A color scale on the right indicates the % Logit Difference, ranging from -5 (red) to 5 (blue).

Figure 6: Direct causal effect of all heads in the model on the values of Inhibition heads from the S2 token for base IOI inputs (left) and DoubleIO inputs (middle); the heads with the most significant causal effect are similar in both cases. For DoubleIO, these same heads have a significant causal effect in the opposite direction when measured from the IO2 token (right).

Furthermore, the DoubleIO and TripleIO circuits also reuse the Induction, Duplicate, and Previous Token heads from the base IOI circuit, as well as the paths to these heads from the S2 token. To demonstrate this, we compute the causal effect of patching the output of each head in the model from every possible input token to the values of the Inhibition heads.

However, we also find that the paths from the input tokens corresponding to duplicated instances of the IO token also have significant positive causal effect: IO2 for DoubleIO, and IO2 and IO3 for TripleIO. This is consistent with the results in Section 5.1. Since these paths provide input to Inhibition heads, a positive causal effect indicates a negative impact on the overall performance, as increasing the attention on IO tokens leads the Inhibition head to suppress it and reduces the probability of the circuit returning the correct answer. Since the full model has a lower logit difference for these variants, adding these paths from IO tokens increases the circuits' faithfulness.

Overall, the DoubleIO reuses all of the heads and paths from the base IOI circuit while adding 10 more edges corresponding to the IO2 token, and the TripleIO circuit reuses all of the heads and paths from the DoubleIO circuit while adding 10 more edges corresponding to the IO3 token. Table 2 shows the faithfulness of the base IOI, DoubleIO, and TripleIO circuits, and the extent of their overlap. These results align with the *strong generalization hypothesis* from Figure 1 and, to the best of our knowledge, serve as the first demonstration of circuit generalization through circuit reuse.

![Figure 7: Circuit diagrams for DoubleIO and Base IOI. The left diagram shows the full circuit for DoubleIO, including nodes for 'Then', 'IO1 Mary', 'IO3 John', 'S1 John', 'S1 went', 'to the station', 'IO2 Mary', 'was happy', 'S2 Mary', 'gave a drink', and 'END IO'. It includes sub-circuits for 'IO Duplicate Handling', 'S Duplicate Handling', 'Inhibition Heads', 'Negative Name Mover', 'Name Mover', and 'Backup Name Mover'. The top right diagram shows the 'Base IOI Circuit' with nodes for 'Then', 'IO1 Mary', 'S1 John', 'S1 went', 'to the station', 'S2 John', 'gave a drink', and 'END IO', including 'S Duplicate Handling', 'Inhibition Heads', 'Negative Name Mover', 'Name Mover', and 'Backup Name Mover'. The bottom right diagram shows the 'Duplicate Handling Sub-circuit With Decision Point' with nodes for 'Tok_1', 'Tok_1 + 1', 'Tok_2', 'Previous Token Heads', 'Duplicate Token Heads', and 'Induction Heads'.](dbe553cf16dd14073b89a8263a428664_img.jpg)

Figure 7: Circuit diagrams for DoubleIO and Base IOI. The left diagram shows the full circuit for DoubleIO, including nodes for 'Then', 'IO1 Mary', 'IO3 John', 'S1 John', 'S1 went', 'to the station', 'IO2 Mary', 'was happy', 'S2 Mary', 'gave a drink', and 'END IO'. It includes sub-circuits for 'IO Duplicate Handling', 'S Duplicate Handling', 'Inhibition Heads', 'Negative Name Mover', 'Name Mover', and 'Backup Name Mover'. The top right diagram shows the 'Base IOI Circuit' with nodes for 'Then', 'IO1 Mary', 'S1 John', 'S1 went', 'to the station', 'S2 John', 'gave a drink', and 'END IO', including 'S Duplicate Handling', 'Inhibition Heads', 'Negative Name Mover', 'Name Mover', and 'Backup Name Mover'. The bottom right diagram shows the 'Duplicate Handling Sub-circuit With Decision Point' with nodes for 'Tok\_1', 'Tok\_1 + 1', 'Tok\_2', 'Previous Token Heads', 'Duplicate Token Heads', and 'Induction Heads'.

Figure 7: **Left:** Circuit discovered for DoubleIO. All nodes and edges from the base IOI circuit are reused, with additional edges to the IO Duplicate Handling Sub-circuit. **Bottom Right:** Duplicate Handling Sub-circuit, which appears twice in the circuit to deal with the two duplicated tokens (S, IO) in the DoubleIO prompt. **Upper Right:** Base IOI circuit, reproduced from Wang et al. (2023).

{9}------------------------------------------------

| Metric | Base IOI | DoubleIO | TripleIO |
|-|-|-|-|
| # Nodes in circuit | 26 | 26 | 26 |
| # Edges in circuit | 110 | 120 | 130 |
| Node overlap w/ base IOI circuit | 100% | 100% | 100% |
| Edge overlap w/ base IOI circuit | 100% | 91.66% | 84.61% |
| Model Logit Difference | 3.484 | 2.118 | 1.227 |
| Circuit Logit Difference | 3.119 | 1.621 | 0.974 |
| Normalized Faithfulness | 0.895 | 0.765 | 0.778 |

Table 2: Circuit discovery results for DoubleIO and TripleIO. Both circuits reuse all nodes and edges from the base IOI circuit, with DoubleIO adding 10 edges and TripleIO adding 20 edges.

### 5.3 CHOOSING BETWEEN THE DUPLICATED NAMES

The circuit in Figure 7 demonstrates that Inhibition heads now receive information from both the IO2 and S2 duplicate tokens in the DoubleIO prompt. This raises a natural question: *How does the model decide which duplicate to suppress to produce the correct answer?* To answer this, we sought to find *decision points*: heads that are most responsible for choosing between the duplicates.

Surprisingly, we find that the order in which names appear in the prompt significantly affects the performance of both the full model and the DoubleIO circuit, with higher performance when the IO token comes first. Figure 8 (left) shows the logit differences stratified by whether S or IO comes first, and the overall logit difference appears to be an average of the two. This suggests that one or more attention heads respond differently depending on the order of the names in the prompt.

We find this behavior in head 2.2, a Previous Token head. This head appears to implement a “*first come, first serve*” mechanism, where it primarily attends to the name that appears first in the prompt, thereby serving as a key decision point in the circuit. As shown in Figure 8 (right), the head frequently assigns high attention to one of the name tokens (S, IO), based on which appeared first. In prompts where IO appeared first, the head attended far more to the IO+1 token than it did its S+1 counterpart, and vice versa when S appeared first. The full set of experiments is in Appendix B, and a more detailed study of the duplicate suppression mechanism is left to future work.

![Figure 8: Performance of the DoubleIO circuit and full model based on the order of appearance of S and IO in the prompt. Left: Bar chart showing Logit Difference for DoubleIO Circuit and Model, comparing IO first and Subject first. Right: Bar chart showing Average Normalized Attention Score for head 2.2, comparing IO+1 -> IO and S+1 -> S for Subject First and IO First.](7c2f0efb2c5d10a52ce19ba33d9d3cec_img.jpg)

**Figure 8 Left: Comparison of Circuit and Model by Name Sequence**

| Model | IO first | Subject first |
|-|-|-|
| DoubleIO Circuit | 2.485 | 0.927 |
| Model | 2.829 | 1.199 |

**Figure 8 Right: Average Attention Score of L2H2 by Sequence Order**

| Sequence Order | IO+1 -> IO | S+1 -> S |
|-|-|-|
| Subject First | 0.26 | 0.57 |
| IO First | 0.56 | 0.27 |

Figure 8: Performance of the DoubleIO circuit and full model based on the order of appearance of S and IO in the prompt. Left: Bar chart showing Logit Difference for DoubleIO Circuit and Model, comparing IO first and Subject first. Right: Bar chart showing Average Normalized Attention Score for head 2.2, comparing IO+1 -> IO and S+1 -> S for Subject First and IO First.

Figure 8: **Left:** Performance of the DoubleIO circuit and full model based on the order of appearance of S and IO in the prompt. Both perform better when IO appears first. **Right:** Average attention scores for head 2.2, which places much more attention on the first name that appears in the prompt.

## 6 CONCLUSION

Our investigation reveals that the IOI circuit in GPT-2 small generalizes more effectively than previously understood, reusing its core components and mechanisms while adapting through minimal structural changes, such as adding input edges. Although the base IOI circuit behaved unexpectedly on the DoubleIO and TripleIO prompt variants, as demonstrated by S2 Hacking, the functionality of its attention heads remained intact. This study of circuit generalization and evaluation on prompt variants ultimately deepened our understanding of how GPT-2 small solves the IOI task, while offering critical insights into the broader capabilities of large neural networks.

 Rest of paper (reference and Appendix) is removed.