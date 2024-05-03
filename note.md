# Note for the project

## LSTM

- RNN: Able to retain previous information, but because hidden state morphs all the time, long-term dependencies are hard to be recalled. 
- Cell state: It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

![cell](./images/cell.png)

- Gates: look at $h_{t−1}$ and $x_t$ , and outputs a number between 0 and 1 for each number in the cell state $C_{t−1}$.

- Forget Gate: decides what information we’re going to throw away from the cell state. $f_t = \sigma(W_f[h_{t−1},x_t] + b_f)$
  - E.g. When seeing a new subject, it might want to throw away previous "gender" information in the cell state

- Output Gate: decide what to output. Since we don't necessarily need the whole history, we have to filter the cell state to make a relevant hidden state. $o_t = \sigma(W_o[h_{t−1},x_t] + b_o)$
  - E.g. (Auto regression) When wanting to predict a verb, it might want to output the "tense" information in the cell state

- Input Gate: decide what new information to store in the cell state. $i_t = \sigma(W_i[h_{t−1},x_t] + b_i)$
  - E.g. When seeing a blank, it might not want to store any information in the cell state

## CRF

- A kind of undirected graphical model(Markov Random Field) that only models the hidden states, while the observed data is always given. 

![crf](./images/crf.png)

- As a Markov Random Field, it models the joint probability of the hidden states $p(y|x)$, by calculating potential functions over cliques in the graph and adding them together. It is defined as:
  $$ p(y|x) = \frac{1}{Z(x)} \prod_{c \in C} \theta_c\psi_c(y_c, x) $$
  where $Z(x)$ is the normalization term, and $f_k$ are feature functions that take the previous and current hidden states, the observed data, and the position in the sequence as input.

- The potential function $\psi_c$ is defined over subsequent hidden states. 
  $$ \psi_c(y_c, x) = exp(f_k(y_c, y_{c+1}, x)) $$

- Hard Code CRF transition matrix.
