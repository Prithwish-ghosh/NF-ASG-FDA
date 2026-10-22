## =====================================================================
## Normalizing-flow score prior + flip-ASG posterior sampling for sparse FPCA
##
## Sparse, noisy curves:  Y_i(t_ij) = X_i(t_ij) + eps_ij,
##   X_i(t) = mu(t) + sum_k xi_ik phi_k(t),   3-6 observations per curve.
##
## Step 1 (shared, data-driven): estimate mu, phi_k, lambda_k, sigma^2.
## Step 2: score posteriors  p(xi_i | Y_i) ∝ L_i(xi) q(xi)  under
##   PACE        Gaussian prior, closed form                 [standard FPCA]
##   Flow-IS     flow prior fitted by stochastic EM, E-step and final
##               posterior by importance sampling             [previous method]
##   Flow-ASG    flow prior fitted by stochastic EM, E-step and final
##               posterior by flip-ASG (asg_flip_fast)        [new]
##   Oracle-IS   true prior, importance sampling              [benchmark]
##   Oracle-ASG  true prior, flip-ASG                         [benchmark]
## Step 3: score MSE, curve MSE, coverage / width of 95% bands for X_i(t),
##   KL(true || prior), effective sample size per curve, run time.
##
## Needs asg_flip_fast.R and effective_support_cached.R in the working
## directory. Set use_flow <- FALSE to run without torch (PACE + Oracle only).
## torch, first time only: install.packages("torch"); torch::install_torch()
## =====================================================================

use_flow <- TRUE
scenario <- "mixture"     # "gaussian", "mixture", "banana", "skew_heavy"
n_rep    <- 5             # replications
n        <- 500           # curves per replication
sigma    <- 0.5           # noise sd
m_range  <- 3:6           # observations per curve
K        <- 2
S        <- 20000         # prior draws for importance sampling
n_draw   <- 500           # posterior draws per curve (IS resamples / ASG samples)
asg_burn <- 100           # ASG burn-in per curve (final posterior)
n_em     <- 6             # stochastic-EM iterations
n_per    <- 5             # posterior draws per curve per E-step

if (use_flow) library(torch)
library(splines)
source("asg_flip_fast.R")          # also sources effective_support_cached.R

## ---------------- True score distributions ----------------
ldnorm2 <- function(X, m, S) {
  D <- sweep(X, 2, m); Si <- solve(S)
  -log(2 * pi) - 0.5 * log(det(S)) - 0.5 * rowSums((D %*% Si) * D)
}
mvn <- function(N, m, S) sweep(matrix(rnorm(2 * N), N) %*% chol(S), 2, m, "+")

## FPC scores are uncorrelated with decreasing variance: rotate the mixture
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

tg  <- seq(0, 1, length.out = 101)
h   <- tg[2] - tg[1]
mu_true  <- function(t) t + sin(2 * pi * t)
phi_true <- function(t) cbind(sqrt(2) * sin(2 * pi * t), sqrt(2) * cos(2 * pi * t))

## ---------------- Step 1: sparse FPCA estimation ----------------
## Mean: least squares on a cubic B-spline basis B(t), pooled data.
## Covariance: C(s,t) = B(s)' A B(t), A by least squares on r_ij r_il (j != l).
## lambda_k, sigma^2: Gaussian marginal likelihood given phi_hat.
fpca_sparse <- function(curves, K) {
  tt <- unlist(lapply(curves, `[[`, "t")); yy <- unlist(lapply(curves, `[[`, "y"))
  kn <- seq(0, 1, length.out = 6)[2:5]
  Bf <- function(t) bs(t, knots = kn, degree = 3, intercept = TRUE, Boundary.knots = c(0, 1))
  b_mu   <- qr.solve(Bf(tt), yy)
  mu_hat <- function(t) drop(Bf(t) %*% b_mu)
  p <- ncol(Bf(0.5))
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
  Bg <- Bf(tg); C <- Bg %*% A %*% t(Bg)
  eg <- eigen(C * h, symmetric = TRUE)
  lambda <- pmax(eg$values[1:K], 1e-6)
  phi_g  <- eg$vectors[, 1:K, drop = FALSE] / sqrt(h)
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
      zz <- backsolve(ch, Rl[[i]], transpose = TRUE)
      v  <- v + 2 * sum(log(diag(ch))) + sum(zz^2)
    }
    v / 2
  }
  opt <- optim(log(c(lambda, sig2)), negll, method = "L-BFGS-B",
               lower = rep(log(1e-4), K + 1), upper = rep(log(1e3), K + 1))
  list(mu_hat = mu_hat, mu_g = mu_hat(tg), phi_g = phi_g,
       lambda = exp(opt$par[1:K]), sig2 = exp(opt$par[K + 1]), phi_at = phi_at)
}

## Per-curve pieces used by every method: B_i = phi_hat(t_i), r_i = Y_i - mu_hat(t_i)
curve_parts <- function(curves, fp)
  lapply(curves, function(cv) list(B = matrix(fp$phi_at(cv$t), ncol = K),
                                   r = cv$y - fp$mu_hat(cv$t)))

## Summaries of posterior draws (n_draw x K) for one curve
summ_draws <- function(dr, fp) {
  Xd <- fp$mu_g + fp$phi_g %*% t(dr)
  q  <- apply(Xd, 1, quantile, c(0.025, 0.975))
  list(mean = colMeans(dr), X = rowMeans(Xd), lo = q[1, ], hi = q[2, ])
}
collect <- function(res, extra = list()) {
  c(list(scores = t(sapply(res, `[[`, "mean")),
         X  = t(sapply(res, `[[`, "X")),
         lo = t(sapply(res, `[[`, "lo")),
         hi = t(sapply(res, `[[`, "hi")),
         ess = sapply(res, `[[`, "ess")), extra)
}

## ---------------- PACE (Gaussian prior, closed form) ----------------
pace <- function(parts, fp) {
  L <- diag(fp$lambda, K)
  res <- lapply(parts, function(pt) {
    Sy <- pt$B %*% L %*% t(pt$B) + diag(fp$sig2, length(pt$r))
    G  <- L %*% t(pt$B) %*% solve(Sy)
    mn <- drop(G %*% pt$r); Cv <- L - G %*% pt$B %*% L
    X  <- fp$mu_g + drop(fp$phi_g %*% mn)
    sd <- sqrt(pmax(rowSums((fp$phi_g %*% Cv) * fp$phi_g), 0))
    list(mean = mn, cov = Cv, X = X, lo = X - 1.96 * sd, hi = X + 1.96 * sd, ess = NA)
  })
  collect(res, list(cov = lapply(res, `[[`, "cov")))
}

## ---------------- Importance sampling under a sampled prior ----------------
is_post <- function(parts, fp, prior_draws, n_rs, summarise = TRUE) {
  res <- lapply(parts, function(pt) {
    ll <- -colSums((pt$r - pt$B %*% t(prior_draws))^2) / (2 * fp$sig2)
    w  <- exp(ll - max(ll)); w <- w / sum(w)
    dr <- prior_draws[sample.int(nrow(prior_draws), n_rs, TRUE, prob = w), , drop = FALSE]
    out <- if (summarise) summ_draws(dr, fp) else list()
    out$draws <- dr; out$ess <- 1 / sum(w^2)
    out
  })
  if (!summarise) return(lapply(res, `[[`, "draws"))
  collect(res)
}

## ---------------- flip-ASG under a prior given by log density ----------------
## The prior log density is tabulated once on a fine grid and bilinearly
## interpolated, so each kernel evaluation is cheap (no torch call).
## Outside the grid the kernel is 0 (nlk = Inf), i.e. the prior is truncated
## to a box that is wide enough to hold all of its mass.
grid_logd <- function(logd_fun, lim1, lim2, ng = 300) {
  g1 <- seq(lim1[1], lim1[2], length.out = ng)
  g2 <- seq(lim2[1], lim2[2], length.out = ng)
  Z  <- matrix(logd_fun(as.matrix(expand.grid(g1, g2))), ng)
  function(Th) {                               # Th: N x 2 -> log q(Th)
    x <- Th[, 1]; y <- Th[, 2]
    out <- rep(-Inf, length(x))
    ok  <- x >= g1[1] & x <= g1[ng] & y >= g2[1] & y <= g2[ng]
    if (!any(ok)) return(out)
    i <- pmin(findInterval(x[ok], g1), ng - 1); j <- pmin(findInterval(y[ok], g2), ng - 1)
    fx <- (x[ok] - g1[i]) / (g1[i + 1] - g1[i]); fy <- (y[ok] - g2[j]) / (g2[j + 1] - g2[j])
    out[ok] <- (1 - fx) * (1 - fy) * Z[cbind(i, j)] + fx * (1 - fy) * Z[cbind(i + 1, j)] +
               (1 - fx) * fy * Z[cbind(i, j + 1)] + fx * fy * Z[cbind(i + 1, j + 1)]
    out
  }
}

## Negative log kernel of curve i's score posterior:
##   nlk(xi) = ||r_i - B_i xi||^2 / (2 sigma^2) - log q(xi)
make_nlk <- function(pt, sig2, lq) {
  nlk_vec <- function(Th) {
    v <- colSums((pt$r - pt$B %*% t(Th))^2) / (2 * sig2) - lq(Th)
    v[!is.finite(v)] <- Inf
    v
  }
  list(nlk = function(th) nlk_vec(matrix(th, 1)), nlk_vec = nlk_vec)
}

## Effective sample size of an MCMC chain (Geyer initial positive sequence)
ess_mcmc <- function(x) {
  N <- length(x); if (var(x) == 0) return(1)
  ac <- acf(x, lag.max = min(N - 1, 200), plot = FALSE)$acf[-1]
  s <- 0
  for (k in seq(1, length(ac) - 1, by = 2)) {
    pr <- ac[k] + ac[k + 1]; if (pr <= 0) break; s <- s + pr
  }
  N / (1 + 2 * s)
}

asg_post <- function(parts, fp, lq, inits, n_samp, burn, thin = 1, summarise = TRUE) {
  res <- lapply(seq_along(parts), function(i) {
    f  <- make_nlk(parts[[i]], fp$sig2, lq)
    th <- inits[i, ]
    if (!is.finite(f$nlk(th))) th <- c(0, 0)
    fit <- asg_flip_fast(f$nlk, th, n_samples = n_samp, burn_in = burn, thin = thin,
                         nlk_vec = f$nlk_vec, bounds_mode = "freeze_burnin")
    dr  <- fit$samples
    out <- if (summarise) summ_draws(dr, fp) else list()
    out$draws <- dr
    out$ess   <- if (summarise) min(apply(dr, 2, ess_mcmc)) else NA
    out$flip  <- fit$flip_accept_rate
    out
  })
  if (!summarise) return(lapply(res, `[[`, "draws"))
  collect(res, list(flip = mean(sapply(res, `[[`, "flip"))))
}

## ---------------- Normalizing flow (RealNVP) ----------------
if (use_flow) {
  coupling <- nn_module(
    initialize = function(mask, hidden = 64) {
      self$mask <- nn_buffer(torch_tensor(mask))
      self$net  <- nn_sequential(nn_linear(2, hidden), nn_tanh(),
                                 nn_linear(hidden, hidden), nn_tanh(), nn_linear(hidden, 4))
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

  ## Stochastic EM. Initial flow: fit to draws from the PACE posteriors.
  ## E-step: n_per posterior draws per curve under the current flow, by
  ##   "IS"  : importance sampling with S flow draws, or
  ##   "ASG" : a short flip-ASG chain per curve, warm-started at the
  ##           previous E-step's last draw.
  ## M-step: 400 more Adam steps of the flow on all E-step draws.
  fit_flow_em <- function(parts, fp, pc, estep = c("IS", "ASG"), lim1, lim2) {
    estep <- match.arg(estep)
    init <- do.call(rbind, lapply(seq_along(parts), function(i)
      mvn(n_per, pc$scores[i, ], pc$cov[[i]] + 1e-8 * diag(K))))
    fit <- train_flow(init)
    state <- pc$scores
    for (it in seq_len(n_em)) {
      if (estep == "IS") {
        pd <- is_post(parts, fp, flow_draw(fit, S), n_rs = n_per, summarise = FALSE)
      } else {
        lq <- grid_logd(function(Z) flow_logd(fit, Z), lim1, lim2)
        pd <- asg_post(parts, fp, lq, state, n_samp = n_per, burn = 20, thin = 4,
                       summarise = FALSE)
        state <- t(sapply(pd, function(d) d[nrow(d), ]))
      }
      fit <- train_flow(do.call(rbind, pd), fit, epochs = 400, lr = 1e-3)
    }
    fit
  }
}

## ---------------- One replication ----------------
one_rep <- function(seed) {
  set.seed(seed); if (use_flow) torch_manual_seed(seed)
  xi <- truth$r(n)
  curves <- lapply(seq_len(n), function(i) {
    m <- sample(m_range, 1); t <- sort(runif(m))
    list(t = t, y = mu_true(t) + drop(phi_true(t) %*% xi[i, ]) + rnorm(m, 0, sigma))
  })
  X_true <- sweep(xi %*% t(phi_true(tg)), 2, mu_true(tg), "+")

  fp <- fpca_sparse(curves, K)
  sg <- sign(colSums(fp$phi_g * phi_true(tg)))        # align eigenfunction signs
  fp$phi_g  <- sweep(fp$phi_g, 2, sg, "*")
  fp$phi_at <- function(t) apply(fp$phi_g, 2, function(f) approx(tg, f, t, rule = 2)$y)
  parts <- curve_parts(curves, fp)

  ## box for the gridded prior: +/- 5 sd of each estimated component
  lim1 <- c(-1, 1) * 5 * sqrt(fp$lambda[1]); lim2 <- c(-1, 1) * 5 * sqrt(fp$lambda[2])

  fits <- list(); tm <- c()
  tm["PACE"] <- system.time(fits$PACE <- pace(parts, fp))[["elapsed"]]
  if (use_flow) {
    tm["Flow-IS"] <- system.time({
      fl_is <- fit_flow_em(parts, fp, fits$PACE, "IS", lim1, lim2)
      fits$`Flow-IS` <- is_post(parts, fp, flow_draw(fl_is, S), n_rs = n_draw)
    })[["elapsed"]]
    tm["Flow-ASG"] <- system.time({
      fl_asg <- fit_flow_em(parts, fp, fits$PACE, "ASG", lim1, lim2)
      lq <- grid_logd(function(Z) flow_logd(fl_asg, Z), lim1, lim2)
      fits$`Flow-ASG` <- asg_post(parts, fp, lq, fits$PACE$scores, n_draw, asg_burn)
    })[["elapsed"]]
  }
  tm["Oracle-IS"] <- system.time(
    fits$`Oracle-IS` <- is_post(parts, fp, truth$r(S), n_rs = n_draw))[["elapsed"]]
  tm["Oracle-ASG"] <- system.time({
    lq_true <- grid_logd(truth$logd, lim1, lim2)
    fits$`Oracle-ASG` <- asg_post(parts, fp, lq_true, fits$PACE$scores, n_draw, asg_burn)
  })[["elapsed"]]

  Xe <- truth$r(20000); lt <- truth$logd(Xe)
  kl <- c(PACE = mean(lt - ldnorm2(Xe, c(0, 0), diag(fp$lambda))),
          `Flow-IS`  = if (use_flow) mean(lt - flow_logd(fl_is, Xe)) else NA,
          `Flow-ASG` = if (use_flow) mean(lt - flow_logd(fl_asg, Xe)) else NA,
          `Oracle-IS` = 0, `Oracle-ASG` = 0)

  res <- do.call(rbind, lapply(names(fits), function(nm) { f <- fits[[nm]]
    data.frame(method = nm,
               score_MSE  = mean((f$scores - xi)^2),
               curve_MSE  = mean((f$X - X_true)^2),
               coverage   = mean(X_true >= f$lo & X_true <= f$hi),
               band_width = mean(f$hi - f$lo),
               KL_prior   = unname(kl[nm]),
               med_ESS    = if (all(is.na(f$ess))) NA else median(f$ess),
               min_ESS    = if (all(is.na(f$ess))) NA else min(f$ess),
               flip_acc   = if (is.null(f$flip)) NA else f$flip,
               seconds    = unname(tm[nm])) }))
  list(res = res, fits = fits, curves = curves, X_true = X_true, xi = xi)
}

## ---------------- Run replications ----------------
all <- vector("list", n_rep)
for (r in seq_len(n_rep)) {
  cat(sprintf("Replication %d / %d\n", r, n_rep))
  out <- one_rep(1000 + r)
  all[[r]] <- out$res
}
tab <- do.call(rbind, all)
metrics <- c("score_MSE", "curve_MSE", "coverage", "band_width", "KL_prior",
             "med_ESS", "min_ESS", "flip_acc", "seconds")
ord <- intersect(c("PACE", "Flow-IS", "Flow-ASG", "Oracle-IS", "Oracle-ASG"), unique(tab$method))
fmt <- function(v) if (all(is.na(v))) "-" else
  sprintf(if (mean(v) >= 100) "%.0f (%.0f)" else "%.4f (%.4f)", mean(v), sd(v) / sqrt(length(v)))
summ <- do.call(rbind, lapply(ord, function(nm) { d <- tab[tab$method == nm, ]
  data.frame(method = nm, t(sapply(metrics, function(v) fmt(d[[v]])))) }))
cat(sprintf("\nScenario: %s | n = %d | %d-%d points per curve | sigma = %.2f | %d replications\n",
            scenario, n, min(m_range), max(m_range), sigma, n_rep))
cat("Mean (standard error) over replications. Coverage target 0.95.",
    "ESS: per curve, IS = 1/sum(w^2) over", S, "draws, ASG = autocorrelation ESS of",
    n_draw, "draws.\n")
print(summ[, c("method", "score_MSE", "curve_MSE", "coverage", "band_width")], row.names = FALSE)
print(summ[, c("method", "KL_prior", "med_ESS", "min_ESS", "flip_acc", "seconds")], row.names = FALSE)
write.csv(tab, sprintf("compare_fpca_flow_asg_%s.csv", scenario), row.names = FALSE)

## ---------------- Plots from the last replication ----------------
pdf(sprintf("compare_fpca_flow_asg_%s.pdf", scenario), width = 11, height = 7)
par(mfrow = c(2, 3), mar = c(4, 4, 2.5, 1))
show <- setdiff(names(out$fits), c("Oracle-IS"))
cols <- c(PACE = "#1f77b4", `Flow-IS` = "#ff7f0e", `Flow-ASG` = "#d62728",
          `Oracle-ASG` = "#2ca02c")
lim1 <- range(out$xi[, 1]) + c(-0.5, 0.5); lim2 <- range(out$xi[, 2]) + c(-0.5, 0.5)
plot(out$xi, pch = 16, cex = 0.4, col = "#00000070", xlim = lim1, ylim = lim2,
     main = "True scores", xlab = expression(xi[1]), ylab = expression(xi[2]))
for (nm in intersect(c("PACE", "Flow-ASG"), show))
  plot(out$fits[[nm]]$scores, pch = 16, cex = 0.4, col = cols[nm], xlim = lim1, ylim = lim2,
       main = paste(nm, "posterior means"), xlab = expression(xi[1]), ylab = expression(xi[2]))
band_m <- intersect(c("PACE", "Flow-ASG"), show)
## curves with the largest PACE error: where the prior matters most
worst <- order(rowMeans((out$fits$PACE$X - out$X_true)^2), decreasing = TRUE)
for (i in worst[seq_len(6 - 1 - length(band_m))]) {
  cv <- out$curves[[i]]
  plot(tg, out$X_true[i, ], type = "l", lwd = 2,
       ylim = range(out$X_true[i, ], cv$y, sapply(out$fits[band_m], function(f) c(f$lo[i, ], f$hi[i, ]))),
       main = sprintf("Curve %d (%d points)", i, length(cv$t)), xlab = "t", ylab = "X(t)")
  for (nm in band_m) {
    lines(tg, out$fits[[nm]]$X[i, ], col = cols[nm], lwd = 2)
    lines(tg, out$fits[[nm]]$lo[i, ], col = cols[nm], lty = 2)
    lines(tg, out$fits[[nm]]$hi[i, ], col = cols[nm], lty = 2)
  }
  points(cv$t, cv$y, pch = 19)
  if (i == worst[1]) legend("topright", c("truth", band_m), lwd = 2, cex = 0.8,
                            col = c("black", cols[band_m]))
}
invisible(dev.off())
