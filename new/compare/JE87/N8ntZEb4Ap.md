# Review

## Summary
The paper proposes a method for feature selection using a neural network that combines a feature masking network with a task network. The masking network uses Gumbel-Sigmoid sampling to output a continuous mask that is then applied to the input features. The task network is a standard neural network that takes the masked input and produces the output. The loss function is the sum of the task loss and a selection loss, which is the mean of the mask, to encourage sparse features. The method is evaluated on a variety of datasets and compared with several feature selection methods.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well written and easy to follow. 
- The proposed method is simple and easy to implement. 
- The method is evaluated on a variety of datasets and compared with several feature selection methods.

## Weaknesses
- The proposed method is a straightforward combination of existing techniques. 
- The selection loss encourages sparsity but does not provide any guidance on which features to select. As a result, the method tends to select features randomly, as shown in Figure 5. 
- The method is not compared with state-of-the-art feature selection methods such as LASSO, Boruta, and RELIEFF.

## Questions
- How does the method compare with LASSO, Boruta, and RELIEFF?
- Can the method be modified to select features that are most relevant for the task?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4