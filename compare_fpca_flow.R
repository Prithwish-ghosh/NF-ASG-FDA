## =====================================================================
## Does a normalizing-flow score prior beat standard FPCA (PACE)?
##
## Sparse, noisy curves:  Y_i(t_ij) = X_i(t_ij) + eps_ij,
##   X_i(t) = mu(t) + sum_k xi_ik phi_k(t),   3-6 observations per curve.
##
## Step 1 (shared by all methods, fully data-driven, no truth used):
##   estimate mu, covariance, phi_k, lambda_k and sigma^2 from the pooled data.
## Step 2: estimate each curve's scores under three priors on xi:
##   PACE   : Gaussian N(0, diag(lambda))           [standard FPCA]
##   Flow   : normalizing flow fitted by stochastic EM
##   Oracle : the true score distribution            [best achievable]
## Step 3: compare, over replications,
##   score MSE, curve MSE vs the true X_i(t),
##   coverage and width of 95% pointwise bands for X_i(t),
##   KL(true score density || prior).
##
## Set use_flow <- FALSE to run only PACE and Oracle (no torch needed).
## torch, first time only: install.packages("torch"); torch::install_torch()
## =====================================================================

use_flow <- TRUE
scenario <- "mixture"     # "gaussian", "mixture", "banana", "skew_heavy"
n_rep    <- 10            # replications
n        <- 500           # curves per replication
sigma    <- 0.5           # noise sd
m_range  <- 3:6           # observations per curve
K        <- 2             # number of components
S        <- 20000         # prior draws for importance sampling
n_draw   <- 1000          # posterior draws per curve for bands

if (use_flow) library(torch)
library(splines)

## ---------------- True score distributions ----------------
ldnorm2 <- function(X, m, S) {
  D <- sweep(X, 2, m); Si <- solve(S)
  -log(2 * pi) - 0.5 * log(det(S)) - 0.5 * rowSums((D %*% Si) * D)
}
mvn <- function(N, m, S) sweep(matrix(rnorm(2 * N), N) %*% chol(S), 2, m, "+")

## FPC scores are uncorrelated with decreasing variance, so the mixture is
## rotated by the eigenvectors of its population covariance.
w_mix <- c(0.6, 0.4)
m_mix <- list(c(-1.6, 0.9), c(2.4, -1.35))
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
  mixture = list(
    r = function(N) {
      z <- runif(N) < w_mix[1]
      X <- mvn(N, m_mix[[2]], S_mix[[2]])
      X[z, ] <- mvn(sum(z), m_mix[[1]], S_mix[[1]])
      X },
    logd = function(X) log(w_mix[1] * exp(ldnorm2(X, m_mix[[1]], S_mix[[1]])) +
                           w_mix[2] * exp(ldnorm2(X, m_mix[[2]], S_mix[[2]])))),
  banana = list(
    r = function(N) { z1 <- rnorm(N); z2 <- rnorm(N)
                      cbind(1.5 * z1, 0.5 * z2 + 0.6 * (z1^2 - 1)) },
    logd = function(X) { z1 <- X[, 1] / 1.5
                         z2 <- (X[, 2] - 0.6 * (z1^2 - 1)) / 0.5
                         dnorm(z1, log = TRUE) + dnorm(z2, log = TRUE) - log(0.75) }),
  skew_heavy = list(
    r = function(N) cbind(rgamma(N, 2, 1) - 2, 0.6 * rt(N, 4)),
    logd = function(X) dgamma(X[, 1] + 2, 2, 1, log = TRUE) +
                       dt(X[, 2] / 0.6, 4, log = TRUE) - log(0.6))
)
truth <- scen[[scenario]]

tg  <- seq(0, 1, length.out = 101)                # evaluation grid
h   <- tg[2] - tg[1]
mu_true  <- function(t) t + sin(2 * pi * t)
phi_true <- function(t) cbind(sqrt(2) * sin(2 * pi * t), sqrt(2) * cos(2 * pi * t))

## ---------------- Step 1: sparse FPCA estimation ----------------
## Mean: least squares on a cubic B-spline basis B(t), pooled data.
## Covariance: C(s,t) = B(s)' A B(t), A fitted by least squares to the raw
##   products r_ij r_il (j != l), where r_ij = Y_ij - mu_hat(t_ij).
## lambda_k, sigma^2: Gaussian marginal likelihood given phi_hat
##   (consistent for these variances even when scores are non-Gaussian).
fpca_sparse <- function(curves, K) {
  tt <- unlist(lapply(curves, `[[`, "t")); yy <- unlist(lapply(curves, `[[`, "y"))
  kn <- seq(0, 1, length.out = 6)[2:5]
  Bf <- function(t) bs(t, knots = kn, degree = 3, intercept = TRUE,
                       Boundary.knots = c(0, 1))
  b_mu   <- qr.solve(Bf(tt), yy)
  mu_hat <- function(t) drop(Bf(t) %*% b_mu)
  p  <- ncol(Bf(0.5))
  rows <- list(); resp <- list(); dg_t <- list(); dg_r2 <- list()
  for (cv in curves) {
    r <- cv$y - mu_hat(cv$t); B <- Bf(cv$t); m <- length(r)
    dg_t[[length(dg_t) + 1]] <- cv$t; dg_r2[[length(dg_r2) + 1]] <- r^2
    if (m < 2) next
    pr <- which(upper.tri(diag(m)) | lower.tri(diag(m)), arr.ind = TRUE)
    rows[[length(rows) + 1]] <- t(sapply(seq_len(nrow(pr)), function(q)
      kronecker(B[pr[q, 2], ], B[pr[q, 1], ])))
    resp[[length(resp) + 1]] <- r[pr[, 1]] * r[pr[, 2]]
  }
  D <- do.call(rbind, rows); z <- unlist(resp)
  A <- matrix(solve(crossprod(D) + 1e-6 * diag(p^2), crossprod(D, z)), p)
  A <- (A + t(A)) / 2

  Bg <- Bf(tg)
  C  <- Bg %*% A %*% t(Bg)
  eg <- eigen(C * h, symmetric = TRUE)
  lambda <- pmax(eg$values[1:K], 1e-6)
  phi_g  <- eg$vectors[, 1:K, drop = FALSE] / sqrt(h)

  ## Given phi_hat, (lambda, sigma^2) maximize the marginal likelihood of
  ## r_i ~ N(0, Phi_i diag(lambda) Phi_i' + sigma^2 I), started at the
  ## moment estimates.
  phi_at <- function(t) apply(phi_g, 2, function(f) approx(tg, f, t, rule = 2)$y)
  Pd   <- matrix(phi_at(unlist(dg_t)), ncol = K)
  sig2 <- max(mean(unlist(dg_r2) - drop(Pd^2 %*% lambda)), 1e-2)
  Bl <- lapply(curves, function(cv) matrix(phi_at(cv$t), ncol = K))
  Rl <- lapply(curves, function(cv) cv$y - mu_hat(cv$t))
  negll <- function(par) {
    L <- diag(exp(par[1:K]), K); s2 <- exp(par[K + 1]); v <- 0
    for (i in seq_along(Bl)) {
      Sy <- Bl[[i]] %*% L %*% t(Bl[[i]]) + diag(s2, length(Rl[[i]]))
      ch <- tryCatch(chol(Sy), error = function(e) NULL)
      if (is.null(ch)) return(1e10)
      z  <- backsolve(ch, Rl[[i]], transpose = TRUE)
      v  <- v + 2 * sum(log(diag(ch))) + sum(z^2)
    }
    v / 2
  }
  opt    <- optim(log(c(lambda, sig2)), negll, method = "L-BFGS-B",
                  lower = rep(log(1e-4), K + 1), upper = rep(log(1e3), K + 1))
  lambda <- exp(opt$par[1:K]); sig2 <- exp(opt$par[K + 1])

  list(mu_hat = mu_hat, mu_g = mu_hat(tg), phi_g = phi_g, lambda = lambda, sig2 = sig2,
       phi_at = phi_at)
}

## ---------------- Step 2a: PACE (Gaussian prior) ----------------
pace <- function(curves, fp) {
  L <- diag(fp$lambda, length(fp$lambda))
  out <- lapply(curves, function(cv) {
    B  <- matrix(fp$phi_at(cv$t), ncol = K)
    Sy <- B %*% L %*% t(B) + diag(fp$sig2, length(cv$t))
    G  <- L %*% t(B) %*% solve(Sy)
    list(mean = drop(G %*% (cv$y - fp$mu_hat(cv$t))), cov = L - G %*% B %*% L)
  })
  M  <- t(sapply(out, `[[`, "mean"))
  Xh <- sweep(M %*% t(fp$phi_g), 2, fp$mu_g, "+")
  sd <- t(sapply(out, function(o) sqrt(pmax(rowSums((fp$phi_g %*% o$cov) * fp$phi_g), 0))))
  list(scores = M, X = Xh, lo = Xh - 1.96 * sd, hi = Xh + 1.96 * sd)
}

## ---------------- Step 2b: posterior under a sampled prior ----------------
## Self-normalized importance sampling with prior draws as proposal;
## returns posterior means, curve estimates and 95% pointwise bands.
post_prior <- function(curves, fp, prior_draws, bands = TRUE, n_rs = n_draw) {
  res <- lapply(curves, function(cv) {
    B  <- matrix(fp$phi_at(cv$t), ncol = K)
    ll <- -colSums((cv$y - fp$mu_hat(cv$t) - B %*% t(prior_draws))^2) / (2 * fp$sig2)
    w  <- exp(ll - max(ll)); w <- w / sum(w)
    rs <- prior_draws[sample.int(nrow(prior_draws), n_rs, TRUE, prob = w), , drop = FALSE]
    out <- list(mean = colSums(prior_draws * w), rs = rs)
    if (bands) {
      Xd <- fp$mu_g + fp$phi_g %*% t(rs)                 # grid x draws
      out$X  <- fp$mu_g + drop(fp$phi_g %*% out$mean)
      q <- apply(Xd, 1, quantile, c(0.025, 0.975))
      out$lo <- q[1, ]; out$hi <- q[2, ]
    }
    out
  })
  o <- list(scores = t(sapply(res, `[[`, "mean")),
            rs = lapply(res, `[[`, "rs"))
  if (bands) { o$X  <- t(sapply(res, `[[`, "X"))
               o$lo <- t(sapply(res, `[[`, "lo"))
               o$hi <- t(sapply(res, `[[`, "hi")) }
  o
}

## ---------------- Step 2c: normalizing flow fitted by stochastic EM --------
if (use_flow) {
  coupling <- nn_module(
    initialize = function(mask, hidden = 64) {
      self$mask <- nn_buffer(torch_tensor(mask))
      self$net  <- nn_sequential(nn_linear(2, hidden), nn_tanh(),
                                 nn_linear(hidden, hidden), nn_tanh(),
                                 nn_linear(hidden, 4))
    },
    st = function(xm) { out <- self$net(xm)
      list(s = torch_tanh(out[, 1:2]) * (1 - self$mask), t = out[, 3:4] * (1 - self$mask)) },
    to_z = function(x) { p <- self$st(x * self$mask)
      list(z = x * self$mask + (1 - self$mask) * (x * torch_exp(p$s) + p$t),
           logdet = torch_sum(p$s, dim = 2)) },
    to_x = function(z) { p <- self$st(z * self$mask)
      z * self$mask + (1 - self$mask) * ((z - p$t) * torch_exp(-p$s)) }
  )
  realnvp <- nn_module(
    initialize = function(n_layers = 6, hidden = 64) {
      self$layers <- nn_module_list(lapply(seq_len(n_layers), function(l)
        coupling(if (l %% 2 == 1) c(1, 0) else c(0, 1), hidden))) },
    log_prob = function(x) { z <- x; ld <- torch_zeros(x$size(1))
      for (l in seq_len(length(self$layers))) {
        r <- self$layers[[l]]$to_z(z); z <- r$z; ld <- ld + r$logdet }
      -0.5 * torch_sum(z^2, dim = 2) - log(2 * pi) + ld },
    draw = function(N) { z <- torch_randn(N, 2)
      for (l in rev(seq_len(length(self$layers)))) z <- self$layers[[l]]$to_x(z)
      z }
  )
  train_flow <- function(X, fit = NULL, epochs = 1500, lr = 2e-3) {
    if (is.null(fit)) fit <- list(model = realnvp(), ctr = colMeans(X), scl = apply(X, 2, sd))
    Xs  <- torch_tensor(sweep(sweep(X, 2, fit$ctr), 2, fit$scl, "/"), dtype = torch_float())
    opt <- optim_adam(fit$model$parameters, lr = lr)
    for (e in seq_len(epochs)) {
      opt$zero_grad(); loss <- -fit$model$log_prob(Xs)$mean(); loss$backward(); opt$step()
    }
    fit
  }
  flow_draw <- function(fit, N) {
    U <- with_no_grad(as_array(fit$model$draw(N)))
    sweep(sweep(U, 2, fit$scl, "*"), 2, fit$ctr, "+")
  }
  flow_logd <- function(fit, Z) {
    U <- torch_tensor(sweep(sweep(Z, 2, fit$ctr), 2, fit$scl, "/"), dtype = torch_float())
    with_no_grad(as_array(fit$model$log_prob(U))) - sum(log(fit$scl))
  }
  ## Start from draws of the PACE posteriors, then alternate:
  ## E-step: draw from each curve's posterior under the current flow;
  ## M-step: refit the flow to those draws.
  fit_flow_em <- function(curves, fp, pc, n_em = 6, n_per = 5) {
    L <- diag(fp$lambda)
    init <- do.call(rbind, lapply(seq_along(curves), function(i) {
      B  <- matrix(fp$phi_at(curves[[i]]$t), ncol = K)
      Sy <- B %*% L %*% t(B) + diag(fp$sig2, length(curves[[i]]$t))
      Cv <- L - L %*% t(B) %*% solve(Sy, B %*% L)
      mvn(n_per, pc$scores[i, ], Cv + 1e-8 * diag(K))
    }))
    fit <- train_flow(init)
    for (it in seq_len(n_em)) {
      pd  <- post_prior(curves, fp, flow_draw(fit, S), bands = FALSE, n_rs = n_per)$rs
      fit <- train_flow(do.call(rbind, pd), fit, epochs = 400, lr = 1e-3)
    }
    fit
  }
}

## ---------------- Step 3: one replication ----------------
one_rep <- function(seed) {
  set.seed(seed); if (use_flow) torch_manual_seed(seed)
  xi <- truth$r(n)
  curves <- lapply(seq_len(n), function(i) {
    m <- sample(m_range, 1); t <- sort(runif(m))
    list(t = t, y = mu_true(t) + drop(phi_true(t) %*% xi[i, ]) + rnorm(m, 0, sigma))
  })
  X_true <- sweep(xi %*% t(phi_true(tg)), 2, mu_true(tg), "+")

  fp <- fpca_sparse(curves, K)
  ## Signs of eigenfunctions are arbitrary: align with the truth so that
  ## scores and score densities are comparable (curve estimates unaffected).
  sg <- sign(colSums(fp$phi_g * phi_true(tg)))
  fp$phi_g <- sweep(fp$phi_g, 2, sg, "*")
  fp$phi_at <- function(t) apply(fp$phi_g, 2, function(f) approx(tg, f, t, rule = 2)$y)

  fits <- list(PACE = pace(curves, fp))
  if (use_flow) {
    fl <- fit_flow_em(curves, fp, fits$PACE)
    fits$Flow <- post_prior(curves, fp, flow_draw(fl, S))
  }
  fits$Oracle <- post_prior(curves, fp, truth$r(S))

  ## KL(true || prior) by Monte Carlo
  Xe <- truth$r(20000); lt <- truth$logd(Xe)
  kl <- c(PACE = mean(lt - ldnorm2(Xe, c(0, 0), diag(fp$lambda))),
          Flow = if (use_flow) mean(lt - flow_logd(fl, Xe)) else NA,
          Oracle = 0)

  res <- do.call(rbind, lapply(names(fits), function(nm) { f <- fits[[nm]]
    data.frame(method = nm,
               score_MSE = mean((f$scores - xi)^2),
               curve_MSE = mean((f$X - X_true)^2),
               coverage  = mean(X_true >= f$lo & X_true <= f$hi),
               band_width = mean(f$hi - f$lo),
               KL_prior  = unname(kl[nm])) }))
  list(res = res, fits = fits, curves = curves, X_true = X_true, xi = xi, fp = fp)
}

## ---------------- Run replications ----------------
all <- vector("list", n_rep)
for (r in seq_len(n_rep)) {
  cat(sprintf("Replication %d / %d\n", r, n_rep))
  out <- one_rep(1000 + r)
  all[[r]] <- out$res
}
tab <- do.call(rbind, all)
metrics <- c("score_MSE", "curve_MSE", "coverage", "band_width", "KL_prior")
summ <- do.call(rbind, lapply(split(tab, tab$method), function(d)
  data.frame(method = d$method[1],
             t(sapply(metrics, function(v)
               sprintf("%.4f (%.4f)", mean(d[[v]]), sd(d[[v]]) / sqrt(nrow(d))))))))
summ <- summ[match(intersect(c("PACE", "Flow", "Oracle"), summ$method), summ$method), ]
cat(sprintf("\nScenario: %s | n = %d | %d-%d points per curve | sigma = %.2f | %d replications\n",
            scenario, n, min(m_range), max(m_range), sigma, n_rep))
cat("Mean (standard error) over replications; coverage target is 0.95\n")
print(summ, row.names = FALSE)

## ---------------- Plots from the last replication ----------------
pdf(sprintf("compare_fpca_flow_%s.pdf", scenario), width = 11, height = 7)
par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
meths <- names(out$fits); cols <- c(PACE = "#1f77b4", Flow = "#d62728", Oracle = "#2ca02c")
lim1 <- range(out$xi[, 1]) + c(-0.5, 0.5); lim2 <- range(out$xi[, 2]) + c(-0.5, 0.5)
plot(out$xi, pch = 16, cex = 0.4, col = "#00000070", xlim = lim1, ylim = lim2,
     main = "True scores", xlab = expression(xi[1]), ylab = expression(xi[2]))
for (nm in meths[meths != "Oracle"])
  plot(out$fits[[nm]]$scores, pch = 16, cex = 0.4, col = cols[nm], xlim = lim1, ylim = lim2,
       main = paste(nm, "score estimates"), xlab = expression(xi[1]), ylab = expression(xi[2]))
for (i in 1:(6 - 1 - length(meths[meths != "Oracle"]))) {
  cv <- out$curves[[i]]
  plot(tg, out$X_true[i, ], type = "l", lwd = 2,
       ylim = range(out$X_true[i, ], cv$y, sapply(out$fits, function(f) c(f$lo[i, ], f$hi[i, ]))),
       main = sprintf("Curve %d (%d points)", i, length(cv$t)), xlab = "t", ylab = "X(t)")
  for (nm in meths[meths != "Oracle"]) {
    lines(tg, out$fits[[nm]]$X[i, ], col = cols[nm], lwd = 2)
    lines(tg, out$fits[[nm]]$lo[i, ], col = cols[nm], lty = 2)
    lines(tg, out$fits[[nm]]$hi[i, ], col = cols[nm], lty = 2)
  }
  points(cv$t, cv$y, pch = 19)
  if (i == 1) legend("topright", c("truth", meths[meths != "Oracle"]), lwd = 2, cex = 0.8,
                     col = c("black", cols[meths[meths != "Oracle"]]))
}
invisible(dev.off())
