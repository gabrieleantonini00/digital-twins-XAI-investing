# Run from any directory: Rscript Analysis.R [data.csv] [output-directory]
# Primary: cumulative logit, continuous baseline literacy. Protocol: PROTOCOL.md.
args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(), value = TRUE)
root <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
.libPaths(c(file.path(root, ".Rlib"), .libPaths()))
required <- c("ordinal", "ARTool", "ggplot2")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Install R packages first: ", paste(missing, collapse = ", "))
input <- if (length(args) >= 1) args[1] else file.path(root, "Data_500.csv")
out <- if (length(args) >= 2) args[2] else file.path(root, "results")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
df <- read.csv(input, stringsAsFactors = FALSE, check.names = FALSE)
needed <- c("pid", "Market_Context", "Transparency", "Financial_Literacy", "Willingness_to_Invest")
stopifnot(all(needed %in% names(df)), nrow(df) == 500, !anyDuplicated(df$pid), !anyNA(df$pid))
for (v in c("Financial_Literacy", "Willingness_to_Invest")) {
  stopifnot(is.numeric(df[[v]]), !anyNA(df[[v]]), all(df[[v]] %in% 1:7))
}
stopifnot(length(unique(df$Financial_Literacy)) >= 2, length(unique(df$Willingness_to_Invest)) >= 2)
market_labels <- c("Bear Market (Severe Downturn)", "Bull Market (High Growth)")
transparency_labels <- c("Standard AI (Performance metrics only)", "Explainable AI (Detailed rationales)")
stopifnot(all(df$Market_Context %in% market_labels), all(df$Transparency %in% transparency_labels))
df$Market <- factor(df$Market_Context, levels = market_labels, labels = c("Bear", "Bull"))
df$Transparency_F <- factor(df$Transparency, levels = transparency_labels, labels = c("Standard", "XAI"))
stopifnot(all(table(df$Market, df$Transparency_F) == 125))
df$Literacy_c <- df$Financial_Literacy - 4
df$WTI <- ordered(df$Willingness_to_Invest, levels = sort(unique(df$Willingness_to_Invest)))
median_lit <- median(df$Financial_Literacy)
df$Literacy_Group <- factor(ifelse(df$Financial_Literacy > median_lit, "High", "Low"), levels = c("Low", "High"))

sink(file.path(out, "analysis.txt"), split = TRUE)
options(width = 160, digits = 10)
cat("Input dataset MD5:", unname(tools::md5sum(input)), "\n")
cat("N =", nrow(df), "unique personas; one investment observation each.\n")
cat("Financial literacy is an LLM-inferred pre-treatment proxy.\n")
cat("Median proxy score:", median_lit, "\n")
print(table(df$Market, df$Transparency_F))
print(table(df$Market, df$Transparency_F, df$Literacy_Group))

summarize_cells <- function(groups) {
  chunks <- split(df, interaction(df[groups], drop = TRUE))
  do.call(rbind, lapply(chunks, function(d) data.frame(
    d[1, groups, drop = FALSE], N = nrow(d), Mean = mean(d$Willingness_to_Invest),
    SD = sd(d$Willingness_to_Invest), Median = median(d$Willingness_to_Invest),
    Literacy_Min = min(d$Financial_Literacy), Literacy_Max = max(d$Financial_Literacy),
    row.names = NULL)))
}
desc <- summarize_cells(c("Market", "Transparency_F"))
desc_groups <- summarize_cells(c("Market", "Transparency_F", "Literacy_Group"))
cat("\nDESCRIPTIVE TABLES\n")
print(desc)
print(desc_groups)
cat("\nLiteracy distribution\n")
print(table(df$Financial_Literacy))
cat("\nOutcome counts by randomized condition\n")
print(table(df$Market, df$Transparency_F, df$Willingness_to_Invest))

cat("\nPRIMARY MODEL: cumulative logit\n")
full <- ordinal::clm(WTI ~ Market * Transparency_F * Literacy_c, data = df, link = "logit", Hess = TRUE)
reduced <- ordinal::clm(WTI ~ (Market + Transparency_F + Literacy_c)^2, data = df, link = "logit", Hess = TRUE)
print(summary(full))
convergence <- full$convergence
if (!is.null(convergence$code) && convergence$code != 0) stop("Ordinal model did not converge; inspect diagnostics")
if (anyNA(full$beta) || any(!is.finite(full$beta))) stop("Ordinal model not identifiable")
lrt <- anova(reduced, full)
print(lrt)
chi <- 2 * (as.numeric(logLik(full)) - as.numeric(logLik(reduced)))
delta_df <- attr(logLik(full), "df") - attr(logLik(reduced), "df")
primary_p <- pchisq(chi, delta_df, lower.tail = FALSE)
b <- full$beta
V <- vcov(full)[names(b), names(b), drop = FALSE]
se <- sqrt(diag(V))
coefs <- data.frame(Term = names(b), Estimate = unname(b), SE = se,
                    OR = exp(b), OR_Lower = exp(b - qnorm(.975)*se), OR_Upper = exp(b + qnorm(.975)*se),
                    p = 2*pnorm(abs(b/se), lower.tail = FALSE), row.names = NULL)
cat("\nOrdinal coefficients and pointwise odds-ratio intervals\n")
print(coefs)
po <- tryCatch(ordinal::nominal_test(full), error = function(e) list(error = conditionMessage(e)))
cat("\nProportional-odds diagnostics:\n")
print(po)

make_x <- function(market, transparency, literacy) {
  new <- data.frame(Market = factor(market, levels = levels(df$Market)),
                    Transparency_F = factor(transparency, levels = levels(df$Transparency_F)), Literacy_c = literacy - 4)
  model.matrix(~ Market * Transparency_F * Literacy_c, new)[, names(b), drop = FALSE]
}
contrasts <- do.call(rbind, lapply(c("Bear", "Bull"), function(market) {
  do.call(rbind, lapply(c(2,4,6), function(literacy) {
    difference <- make_x(market, "XAI", literacy) - make_x(market, "Standard", literacy)
    estimate <- as.numeric(difference %*% b)
    std <- sqrt(as.numeric(difference %*% V %*% t(difference)))
    support <- vapply(c("Standard", "XAI"), function(tr) {
      values <- df$Financial_Literacy[df$Market == market & df$Transparency_F == tr]
      literacy >= min(values) && literacy <= max(values)
    }, logical(1))
    data.frame(Market = market, Literacy = literacy, Log_OR = estimate, SE = std,
               OR = exp(estimate), OR_Lower = exp(estimate - qnorm(.975)*std),
               OR_Upper = exp(estimate + qnorm(.975)*std), p_raw = 2*pnorm(abs(estimate/std), lower.tail = FALSE),
               Extrapolation = !all(support))
  }))
}))
contrasts$p_Holm <- p.adjust(contrasts$p_raw, method = "holm")
print(contrasts)

cat("\nSENSITIVITY: OLS with HC3 standard errors\n")
ols <- lm(Willingness_to_Invest ~ Market * Transparency_F * Literacy_c, data = df)
X <- model.matrix(ols)
if (qr(X)$rank != ncol(X)) stop("OLS design is rank deficient")
bread <- solve(crossprod(X))
weighted <- X * as.numeric(residuals(ols) / (1 - hatvalues(ols)))
hc3 <- bread %*% crossprod(weighted) %*% bread
ols_se <- sqrt(diag(hc3))
ols_table <- data.frame(Term = names(coef(ols)), Estimate = coef(ols), SE_HC3 = ols_se,
                        p = 2*pt(abs(coef(ols)/ols_se), df = df.residual(ols), lower.tail = FALSE), row.names = NULL)
print(ols_table)

cat("\nSENSITIVITY: legacy median-split ART\n")
art_table <- NULL
if (length(unique(df$Literacy_Group)) == 2 && all(table(df$Market, df$Transparency_F, df$Literacy_Group) > 0)) {
  art <- ARTool::art(Willingness_to_Invest ~ Market * Transparency_F * Literacy_Group, data = df)
  art_table <- as.data.frame(anova(art))
  print(art_table)
  posthoc <- as.data.frame(ARTool::art.con(art, "Market:Transparency_F:Literacy_Group", adjust = "bonferroni"))
  cat("\nART-C sensitivity pairwise comparisons (Bonferroni)\n")
  print(posthoc)
} else cat("ART sensitivity unavailable: at least one median-split subgroup is empty.\n")

library(ggplot2)
plot <- ggplot(df, aes(x = factor(Willingness_to_Invest, levels = 1:7), fill = Transparency_F)) +
  geom_bar(position = "dodge") + facet_wrap(~Market) +
  scale_x_discrete(drop = FALSE) +
  scale_fill_manual(values = c(Standard = "#64748B", XAI = "#087F8C")) +
  labs(x = "Willingness to invest (1-7)", y = "Number of simulated responses", fill = "Transparency") +
  theme_minimal(base_size = 11) + theme(legend.position = "bottom", panel.grid.minor = element_blank())
ggsave(file.path(out, "outcome_distributions.png"), plot, width = 7.5, height = 3.8, dpi = 200)
interaction_plot <- ggplot(desc_groups, aes(x = Market, y = Mean, group = Transparency_F, color = Transparency_F)) +
  geom_line(linewidth = .7) + geom_point(size = 2.5) + facet_wrap(~Literacy_Group) +
  scale_color_manual(values = c(Standard = "#64748B", XAI = "#087F8C")) +
  scale_y_continuous(limits = c(1,7), breaks = 1:7) +
  labs(x = "Market context", y = "Mean willingness to invest", color = "Transparency",
       subtitle = "Descriptive median-split groups; primary analysis uses numerical literacy") +
  theme_minimal(base_size = 10) + theme(legend.position = "bottom", panel.grid.minor = element_blank())
ggsave(file.path(out, "interaction_descriptive.png"), interaction_plot, width = 7.5, height = 3.8, dpi = 200)

cat("\nSoftware versions\n")
print(sessionInfo())
sink()
cat("Analysis complete:", normalizePath(out, winslash = "/"), "\n")
