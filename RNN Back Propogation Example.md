# RNN Back Propogation Through Time  (BPTT)

## Example using 4 time steps:

$E_k$ is the loss associationed with time step $k$.

### Forward Propogation

$h_t=\tanh(W_xx_t+W_hh_{t-1})$ and $y_t=softmax(W_0h_t)$

### Back Propogation

We sum up the gradients at each time step for one training example:

--------------------------------------------------------

Time step 1:

$\frac{dE_1}{dW_0}=\frac{dE_1}{dy_1}\frac{dy_1}{dW_0}$

$\frac{dE_1}{dW_h}=\frac{dE_1}{dy_1}\frac{dy_1}{dh_1}\frac{dh_1}{dW_h}$

--------------------------------------------------------
Time step 2:

$\frac{dE_2}{dW_0}=\frac{dE_2}{dy_2}\frac{dy_2}{dW_0}$

$\frac{dE_2}{dW_h}=\frac{dE_2}{dy_2}\frac{dy_2}{dh_2}\frac{dh_2}{dh_1}\frac{dh_1}{dW_h}+\frac{dE_2}{dy_2}\frac{dy_2}{dh_2}\frac{dh_2}{dW_h}$

--------------------------------------------------------
Time step 3:

$\frac{dE_3}{dW_0}=\frac{dE_3}{dy_3}\frac{dy_3}{dW_0}$

$\frac{dE_3}{dW_h}=\frac{dE_3}{dy_3}\frac{dy_3}{dh_3}\frac{dh_3}{dh_2}\frac{dh_2}{dh_1}\frac{dh_1}{dW_h}+\frac{dE_3}{dy_3}\frac{dy_3}{dh_3}\frac{dh_3}{dh_2}\frac{dh_2}{dW_h}+\frac{dE_3}{dy_3}\frac{dy_3}{dh_3}\frac{dh_3}{dW_h}$

--------------------------------------------------------
Time step 4:

$\frac{dE_4}{dW_0}=\frac{dE_4}{dy_4}\frac{dy_4}{dW_0}$

$\frac{dE_4}{dW_h}=\frac{dE_4}{dy_4}\frac{dy_4}{dh_4}\frac{dh_4}{dh_3}\frac{dh_3}{dh_2}\frac{dh_2}{dh_1}\frac{dh_1}{dW_h}+\frac{dE_4}{dy_4}\frac{dy_4}{dh_4}\frac{dh_4}{dh_3}\frac{dh_3}{dh_2}\frac{dh_2}{dW_h}+\frac{dE_4}{dy_4}\frac{dy_4}{dh_4}\frac{dh_4}{dh_3}\frac{dh_3}{dW_h}+\frac{dE_4}{dy_4}\frac{dy_4}{dh_4}\frac{dh_4}{dW_h}$





