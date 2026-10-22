## =====================================================================
## Identify the distribution of FPC scores with a normalizing flow
##
##  1. Simulate dense curves  X_i(t) = mu(t) + sum_k xi_ik phi_k(t) + eps
##     with KNOWN, non-Gaussian score distribution (choose a scenario)
##  2. Estimate mu, phi_k and the scores xi_ik by classical FPCA
##  3. Fit (a) a bivariate Gaussian and (b) a RealNVP normalizing flow
##     to the estimated scores
##  4. Compare each fit with the TRUE score distribution:
##       KL(p_true || q)  estimated by Monte Carlo
##       held-out log-likelihood
##       energy distance between fitted and true samples
##
## Packages: torch (CRAN). First time only:
##   install.packages("torch"); torch::install_torch()
## =====================================================================

library(torch)
set.seed(27695); torch_manual_seed(27695)

scenario <- "mixture"     # "gaussian", "mixture", "banana", "skew_heavy"

## ---------------- 1. True score distributions ----------------
## Each scenario: r(N) draws an N x 2 matrix, logd(X) is the true log density.
ldnorm2 <- function(X, m, S) {              # bivariate normal log density
  D  <- sweep(X, 2, m)
  Si <- solve(S)
  -log(2 * pi) - 0.5 * log(det(S)) - 0.5 * rowSums((D %*% Si) * D)
}
mvn <- function(N, m, S) sweep(matrix(rnorm(2 * N), N) %*% chol(S), 2, m, "+")

## FPC scores must be uncorrelated with decreasing variances (that is what
## defines eigenfunctions). The mixture is therefore rotated by the
## eigenvectors of its population covariance.
w_mix <- c(0.6, 0.4)
m_mix <- list(c(-1.6, 0.9), c(2.4, -1.35))               # weighted mean = 0
S_mix <- list(matrix(c(0.30, 0.15, 0.15, 0.20), 2),
              matrix(c(0.25, -0.10, -0.10, 0.30), 2))
V_mix <- eigen(w_mix[1] * (S_mix[[1]] + tcrossprod(m_mix[[1]])) +
               w_mix[2] * (S_mix[[2]] + tcrossprod(m_mix[[2]])))$vectors
m_mix <- lapply(m_mix, function(m) drop(crossprod(V_mix, m)))
S_mix <- lapply(S_mix, function(S) crossprod(V_mix, S %*% V_mix))

scen <- list(
  gaussian = list(
    r    = function(N) mvn(N, c(0, 0), diag(c(2, 0.5))),
    logd = function(X) ldnorm2(X, c(0, 0), diag(c(2, 0.5)))),
  mixture = list(                              # two clusters, mean zero, uncorrelated
    r = function(N) {
      z <- runif(N) < w_mix[1]
      X <- mvn(N, m_mix[[2]], S_mix[[2]])
      X[z, ] <- mvn(sum(z), m_mix[[1]], S_mix[[1]])
      X },
    logd = function(X) log(
      w_mix[1] * exp(ldnorm2(X, m_mix[[1]], S_mix[[1]])) +
      w_mix[2] * exp(ldnorm2(X, m_mix[[2]], S_mix[[2]])))),
  banana = list(                               # curved dependence
    r = function(N) {
      z1 <- rnorm(N); z2 <- rnorm(N)
      cbind(1.5 * z1, 0.5 * z2 + 0.6 * (z1^2 - 1)) },
    logd = function(X) {
      z1 <- X[, 1] / 1.5
      z2 <- (X[, 2] - 0.6 * (z1^2 - 1)) / 0.5
      dnorm(z1, log = TRUE) + dnorm(z2, log = TRUE) - log(1.5) - log(0.5) }),
  skew_heavy = list(                           # skewed xi_1, heavy-tailed xi_2
    r = function(N) cbind(rgamma(N, 2, 1) - 2, 0.6 * rt(N, 4)),
    logd = function(X) dgamma(X[, 1] + 2, 2, 1, log = TRUE) +
                       dt(X[, 2] / 0.6, 4, log = TRUE) - log(0.6))
)
truth <- scen[[scenario]]

## ---------------- 2. Simulate dense curves ----------------
n      <- 400                                   # curves
n_grid <- 501
gp     <- seq(0, 1, length.out = n_grid)
h      <- gp[2] - gp[1]
sigma  <- 0.2

mu_true  <- gp + sin(2 * pi * gp)
phi_true <- cbind(sqrt(2) * sin(2 * pi * gp),   # orthonormal on [0,1]
                  sqrt(2) * cos(2 * pi * gp))

xi_true <- truth$r(n)                           # n x 2, saved for evaluation
X <- matrix(mu_true, n, n_grid, byrow = TRUE) +
     xi_true %*% t(phi_true) +
     matrix(rnorm(n * n_grid, 0, sigma), n, n_grid)

## ---------------- 3. Classical FPCA on the grid ----------------
## Riemann approximation: int f(t) g(t) dt ~ h * sum f(t_j) g(t_j)
mu_hat <- colMeans(X)
Xc     <- sweep(X, 2, mu_hat)
G      <- crossprod(Xc) / (n - 1)               # covariance on the grid
eg     <- eigen(G * h, symmetric = TRUE)        # eigen-problem of the operator
K      <- 2
lambda_hat <- eg$values[1:K]
phi_hat    <- eg$vectors[, 1:K] / sqrt(h)       # so that h * sum phi^2 = 1

## Align signs with the true eigenfunctions (sign is not identifiable)
for (k in 1:K)
  if (sum(phi_hat[, k] * phi_true[, k]) < 0) phi_hat[, k] <- -phi_hat[, k]

xi_hat <- h * Xc %*% phi_hat                    # projection scores, n x K
cat(sprintf("Score estimation MSE: %.5f\n", mean((xi_hat - xi_true)^2)))
cat("Estimated eigenvalues:", round(lambda_hat, 3),
    "  sample variances of true scores:", round(apply(xi_true, 2, var), 3), "\n")

## ---------------- 4. Fit a Gaussian to the scores ----------------
m_g <- colMeans(xi_hat)
S_g <- cov(xi_hat)
gauss_logd <- function(Z) ldnorm2(Z, m_g, S_g)
gauss_draw <- function(N) mvn(N, m_g, S_g)

## ---------------- 5. RealNVP normalizing flow (torch) ----------------
coupling <- nn_module(
  initialize = function(mask, hidden = 64) {
    self$mask <- nn_buffer(torch_tensor(mask))
    self$net  <- nn_sequential(
      nn_linear(2, hidden), nn_tanh(),
      nn_linear(hidden, hidden), nn_tanh(),
      nn_linear(hidden, 4))
  },
  st = function(x_masked) {
    out <- self$net(x_masked)
    list(s = torch_tanh(out[, 1:2]) * (1 - self$mask),
         t = out[, 3:4] * (1 - self$mask))
  },
  to_z = function(x) {
    p <- self$st(x * self$mask)
    z <- x * self$mask + (1 - self$mask) * (x * torch_exp(p$s) + p$t)
    list(z = z, logdet = torch_sum(p$s, dim = 2))
  },
  to_x = function(z) {
    p <- self$st(z * self$mask)
    z * self$mask + (1 - self$mask) * ((z - p$t) * torch_exp(-p$s))
  }
)

realnvp <- nn_module(
  initialize = function(n_layers = 8, hidden = 64) {
    self$layers <- nn_module_list(lapply(seq_len(n_layers), function(l)
      coupling(if (l %% 2 == 1) c(1, 0) else c(0, 1), hidden)))
  },
  log_prob = function(x) {
    z <- x; ld <- torch_zeros(x$size(1))
    for (l in seq_len(length(self$layers))) {
      r <- self$layers[[l]]$to_z(z); z <- r$z; ld <- ld + r$logdet
    }
    -0.5 * torch_sum(z^2, dim = 2) - log(2 * pi) + ld
  },
  draw = function(N) {
    z <- torch_randn(N, 2)
    for (l in rev(seq_len(length(self$layers)))) z <- self$layers[[l]]$to_x(z)
    z
  }
)

## Train on standardized scores; keep the epoch with the best validation NLL
train_flow <- function(Xtr, Xva, epochs = 3000, lr = 2e-3, wd = 1e-5) {
  ctr <- colMeans(Xtr); scl <- apply(Xtr, 2, sd)
  std <- function(A) torch_tensor(sweep(sweep(A, 2, ctr), 2, scl, "/"),
                                  dtype = torch_float())
  tr <- std(Xtr); va <- std(Xva)
  model <- realnvp()
  opt   <- optim_adam(model$parameters, lr = lr, weight_decay = wd)
  best  <- Inf; best_state <- NULL
  for (e in seq_len(epochs)) {
    model$train(); opt$zero_grad()
    loss <- -model$log_prob(tr)$mean()
    loss$backward(); opt$step()
    if (e %% 25 == 0) {
      model$eval()
      v <- with_no_grad(-model$log_prob(va)$mean()$item())
      if (v < best) {
        best <- v
        best_state <- lapply(model$state_dict(), function(p) p$clone())
      }
      if (e %% 500 == 0)
        cat(sprintf("  epoch %4d  train NLL %.3f  valid NLL %.3f\n", e, loss$item(), v))
    }
  }
  model$load_state_dict(best_state); model$eval()
  list(model = model, ctr = ctr, scl = scl)
}

## log density on the ORIGINAL score scale: log p(u) - sum log(scl)
flow_logd <- function(fit, Z) {
  U <- torch_tensor(sweep(sweep(Z, 2, fit$ctr), 2, fit$scl, "/"),
                    dtype = torch_float())
  with_no_grad(as_array(fit$model$log_prob(U))) - sum(log(fit$scl))
}
flow_draw <- function(fit, N) {
  U <- with_no_grad(as_array(fit$model$draw(N)))
  sweep(sweep(U, 2, fit$scl, "*"), 2, fit$ctr, "+")
}

idx <- sample(n, round(0.8 * n))                # 80/20 train/validation split
cat("Training flow:\n")
flow <- train_flow(xi_hat[idx, ], xi_hat[-idx, ])

## ---------------- 6. Compare fits with the truth ----------------
energy_dist <- function(A, B) {                 # 2E|A-B| - E|A-A'| - E|B-B'|
  d <- function(P, Q) mean(sqrt(outer(rowSums(P^2), rowSums(Q^2), "+") - 2 * P %*% t(Q) + 1e-12))
  2 * d(A, B) - d(A, A) - d(B, B)
}

N_eval <- 20000
X_eval <- truth$r(N_eval)                       # fresh draws from the truth
lp_true <- truth$logd(X_eval)
lp_g    <- gauss_logd(X_eval)
lp_f    <- flow_logd(flow, X_eval)

N_ed  <- 2000
X_ed  <- truth$r(N_ed)
results <- data.frame(
  fit         = c("Gaussian", "Normalizing flow"),
  KL_true_fit = c(mean(lp_true - lp_g), mean(lp_true - lp_f)),
  heldout_LL  = c(mean(lp_g), mean(lp_f)),
  energy_dist = c(energy_dist(gauss_draw(N_ed), X_ed),
                  energy_dist(flow_draw(flow, N_ed), X_ed)))
cat(sprintf("\nScenario: %s   (best possible held-out LL = %.3f)\n",
            scenario, mean(lp_true)))
print(results, row.names = FALSE, digits = 3)

## ---------------- 7. Plots ----------------
pdf(sprintf("score_flow_%s.pdf", scenario), width = 11, height = 6.4)
par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
lim1 <- range(xi_true[, 1], xi_hat[, 1]) + c(-1, 1)
lim2 <- range(xi_true[, 2], xi_hat[, 2]) + c(-1, 1)
g1 <- seq(lim1[1], lim1[2], length.out = 120)
g2 <- seq(lim2[1], lim2[2], length.out = 120)
grid <- as.matrix(expand.grid(g1, g2))
pts <- function(Z, main) plot(Z, pch = 16, cex = 0.4, col = "#00000060",
  xlim = lim1, ylim = lim2, main = main,
  xlab = expression(xi[1]), ylab = expression(xi[2]))
dens <- function(ld, main) {
  contour(g1, g2, matrix(exp(ld), 120), nlevels = 10, drawlabels = FALSE,
          main = main, xlab = expression(xi[1]), ylab = expression(xi[2]))
  points(xi_hat, pch = 16, cex = 0.3, col = "#00000040")
}
pts(xi_true, "True scores")
pts(xi_hat,  "Estimated scores (FPCA)")
pts(flow_draw(flow, n), "Draws from fitted flow")
dens(truth$logd(grid),    "True density")
dens(gauss_logd(grid),    "Gaussian fit")
dens(flow_logd(flow, grid), "Flow fit")
dev.off()

## Estimated vs true eigenfunctions (sanity check)
pdf("eigenfunctions_check.pdf", width = 8, height = 3.5)
par(mfrow = c(1, 2), mar = c(4, 4, 2.5, 1))
for (k in 1:K) {
  plot(gp, phi_true[, k], type = "l", lwd = 2, main = bquote(phi[.(k)]),
       xlab = "t", ylab = "")
  lines(gp, phi_hat[, k], col = "red", lty = 2, lwd = 2)
}
dev.off()
