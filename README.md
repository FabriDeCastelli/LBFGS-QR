# CMLDA-23-24

Computational Mathematics for Learning and Data Analysis, A.Y. 2023/2024.

## Project 19 (Non-ML)

(P) is the linear least squares problem
$$\displaystyle \min_{w} \lVert \hat{X}w-\hat{y} \rVert$$
where

$$\hat{X}= \begin{bmatrix} X^T \newline \lambda I_m \end{bmatrix},\ \ \hat{y} = \begin{bmatrix} y \newline 0 \end{bmatrix},$$

with $X$ the (tall thin) matrix from the ML-cup dataset by prof. Micheli, $\lambda > 0$ and $y$ is a random vector.

- (A1) is an algorithm of the class of **limited-memory quasi-Newton methods**.
- (A2) is **thin QR factorization with Householder reflectors**, in the variant where one does not form the matrix $Q$, but stores the Householder vectors $u_k$ and uses them to perform (implicitly) products with $Q$ and $Q^T$.
