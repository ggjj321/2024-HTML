# Logistic regression experiment

For binary classification, we first conducted experiments using logistic regression. Logistic regression computes the weighted sum of features through the linear function \( z = \mathbf{w}^T\mathbf{x} + b \), and then passes this result through the sigmoid function to convert it into a probability value. We predicted the probability of the home team winning; if the probability exceeded 50\%, the model predicted a win for the home team; otherwise, it predicted a loss. Initially, we proposed a hypothesis that there exist mutual dominance relationships between teams. Subsequently, we gradually added features to the model. To prevent overfitting, logistic regression was implemented with \( L_2 \) regularization. We started by selecting 5 features and incrementally increased the number of features by 5, up to 120 features. For each set of \( x \) features, we tested different constraints on \( L_2 \) regularization within the range of 0.00001 to 0.02. Finally, we used 5-fold cross-validation to identify the optimal combination of the number of features and the constraint value.

## Investigate the mutual dominance relationships between teams

To investigate the mutual dominance relationships between teams, we applied one-hot encoding to the **home team abbr** and **away team abbr** features while removing all other features. In this experiment, there were no combinations of feature selection, and we solely tested different constraints for \( L_2 \) regularization. The best validation accuracy achieved in this setup was 0.54, with Stage 1 reaching 0.56843 and Stage 2 achieving 0.54069. From the results shown in the figure, it can be observed that as the constraint decreases (i.e., \( \lambda \) increases), the model transitions from underfitting (on the left side of the plot) to overfitting (on the right side). This observation supports the hypothesis that the chosen approach can effectively assess whether the model is overfitting.