# Render the metafor plot gallery figures that metafopy reproduces, for the comparison in the user manual (docs/guide/gallery.md).
#
# The code of each figure is the code of the matching page of the metafor plot gallery,
# <https://www.metafor-project.org/doku.php/plots>, written by Wolfgang Viechtbauer. Only the graphics device calls differ.
# The metafopy counterparts are the Python blocks of docs/guide/gallery.md, rendered by bin/gallery.py.
#
# Usage (from the repository root):  LC_ALL=C.UTF-8 Rscript src/tests/r/gallery.R

suppressPackageStartupMessages(library(metafor))
stopifnot(as.character(packageVersion("metafor")) == "5.2.1")

out_dir <- "docs/images/gallery/r"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# open a png device of the given size (inches); the Python side uses the same sizes at the same resolution
device <- function(name, width, height) {
  png(file.path(out_dir, paste0(name, ".png")), width = width, height = height, units = "in", res = 100, type = "cairo")
}

### forest

device("forest", 8, 7)
dat <- dat.bcg
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat, slab=paste(author, year, sep=", "))
res <- rma(yi, vi, data=dat)
forest(res, atransf=exp, at=log(c(0.05, 0.25, 1, 4)), xlim=c(-16,6),
       ilab=cbind(tpos, tneg, cpos, cneg), ilab.lab=c("TB+","TB-","TB+","TB-"),
       ilab.xpos=c(-9.5,-8,-6,-4.5), cex=0.75, header="Author(s) and Year",
       mlab="", shade=TRUE)
text(c(-8.75,-5.25), res$k+2.8, c("Vaccinated", "Control"), cex=0.75, font=2)
text(-16, -1, pos=4, cex=0.75, bquote(paste(
      "RE Model (Q = ", .(fmtx(res$QE, digits=2)),
      ", df = ", .(res$k - res$p), ", ",
      .(fmtp2(res$QEp)), "; ",
      I^2, " = ", .(fmtx(res$I2, digits=1)), "%, ",
      tau^2, " = ", .(fmtx(res$tau2, digits=2)), ")")))
dev.off()

### forest_textpos

device("forest_textpos", 8, 7)
forest(res, atransf=exp, at=log(c(0.05, 0.25, 1, 4)), xlim=c(-20,2),
     ilab=cbind(tpos, tneg, cpos, cneg), ilab.lab=c("TB+","TB-","TB+","TB-"),
     ilab.xpos=c(-13.5,-12,-10,-8.5), cex=0.75, header="Author(s) and Year",
     mlab="", textpos=c(-20,-3.5), shade=TRUE)
text(c(-12.75,-9.25), res$k+2.8, c("Vaccinated", "Control"), cex=0.75, font=2)
text(-20, -1, pos=4, cex=0.75, bquote(paste(
      "RE Model (Q = ", .(fmtx(res$QE, digits=2)),
      ", df = ", .(res$k - res$p), ", ",
      .(fmtp2(res$QEp)), "; ",
      I^2, " = ", .(fmtx(res$I2, digits=1)), "%, ",
      tau^2, " = ", .(fmtx(res$tau2, digits=2)), ")")))
dev.off()

### forest_subgroups

device("forest_subgroups", 8, 7.5)
mlabfun <- function(text, x) {
   list(bquote(paste(.(text),
      " (Q = ", .(fmtx(x$QE, digits=2)),
      ", df = ", .(x$k - x$p), ", ",
      .(fmtp2(x$QEp)), "; ",
      I^2, " = ", .(fmtx(x$I2, digits=1)), "%, ",
      tau^2, " = ", .(fmtx(x$tau2, digits=2)), ")")))}
forest(res, xlim=c(-16, 4.6), at=log(c(0.05, 0.25, 1, 4)), atransf=exp,
       ilab=cbind(tpos, tneg, cpos, cneg), ilab.lab=c("TB+","TB-","TB+","TB-"),
       ilab.xpos=c(-9.5,-8,-6,-4.5), cex=0.75, ylim=c(-2,28), top=4, order=alloc,
       rows=c(3:4,9:15,20:23), mlab=mlabfun("RE Model for All Studies", res),
       psize=1, header="Author(s) and Year")
op <- par(cex=0.75)
text(c(-8.75,-5.25), 27, c("Vaccinated", "Control"), font=2)
text(-16, c(24,16,5), pos=4, c("Systematic Allocation",
                               "Random Allocation",
                               "Alternate Allocation"), font=4)
par(op)
res.s <- rma(yi, vi, subset=(alloc=="systematic"), data=dat)
res.r <- rma(yi, vi, subset=(alloc=="random"),     data=dat)
res.a <- rma(yi, vi, subset=(alloc=="alternate"),  data=dat)
addpoly(res.s, row=18.5, mlab=mlabfun("RE Model for Subgroup", res.s))
addpoly(res.r, row= 7.5, mlab=mlabfun("RE Model for Subgroup", res.r))
addpoly(res.a, row= 1.5, mlab=mlabfun("RE Model for Subgroup", res.a))
res.m <- rma(yi, vi, mods = ~ alloc, data=dat)
text(-16, -1.8, pos=4, cex=0.75, bquote(paste("Test for Subgroup Differences: ",
     Q[M], " = ", .(fmtx(res.m$QM, digits=2)),
     ", df = ", .(res.m$p - 1), ", ", .(fmtp2(res.m$QMp)))))
dev.off()

### forest_multiple_cis

device("forest_multiple_cis", 7, 7)
with(dat, forest(yi, vi, xlim=c(-8,6), ylim=c(-2,nrow(dat)+3),
                 psize=1, efac=0, shade=TRUE))
par(new=TRUE)
with(dat, forest(yi, vi, xlim=c(-8,6), ylim=c(-2,nrow(dat)+3), level=80, lwd=3,
                 psize=NA, efac=NA, annotate=FALSE, refline=NA, slab=NA,
                 xaxt="n", xlab="", lty=c("solid","blank")))
addpoly(res, row=-1, level=99, mlab="RE Model (with 99% CI)", annotate=TRUE)
abline(h=0)
dev.off()

### forest_grouped

device("forest_grouped", 7, 9)
dat.a <- dat.assink2016
V <- vcalc(vi, cluster=study, obs=esid, data=dat.a, rho=0.6)
res.a <- rma.mv(yi, V, random = ~ 1 | study/esid, data=dat.a, digits=3)
dat.a <- dat.a[order(dat.a$study, dat.a$yi, decreasing=TRUE), ]  # sort_by(dat.a, ~ study + yi, decreasing=TRUE) needs R >= 4.4
dat.a$row <- as.numeric(factor(dat.a$study))
dat.a$row.seq <- ave(dat.a$row, dat.a$row, FUN = function(x) {
   n <- length(x)
   x + if (n == 1) 0 else if (n == 2) c(-0.1, 0.1) else seq(-0.3, 0.3, length.out=n)
})
dat.a$slab <- paste("Study", dat.a$study)
dat.a$slab[duplicated(dat.a$slab)] <- ""
dat.a$colpt <- rep(rep(palette.colors(8, palette="Dark 2"),
                   length.out=length(unique(dat.a$study))), times=rle(dat.a$study)$lengths)
dat.a$colci <- adjustcolor(dat.a$colpt, 0.5, 1, 1, 1)
sav <- with(dat.a, forest(yi, vi, rows=row.seq, slab=NA, annotate=FALSE,
                          xlim=c(-4.4,4.2), ylim=c(-2.0,floor(max(dat.a$row))+2.5),
                          cex=1, pch=19, psize=0.8, efac=0, col=colpt, colci=colci))
text(sav$xlim[1], dat.a$row, dat.a$slab, pos=4)
abline(h=0:max(dat.a$row)+0.5, col="gray20", lwd=0.2)
addpoly(res.a, row=-0.25, efac=1, mlab="Pooled Estimate", predstyle="bar")
abline(h=0.5)
dev.off()

### forest_orchard

device("forest_orchard", 7, 5)
dat.m <- dat.moura2021$dat
dat.m <- escalc(measure="ZCOR", ri=ri, ni=ni, data=dat.m)
nlvls <- length(table(dat.m$subphylum))
res.m <- rma.mv(yi, vi, mods = ~ 0 + subphylum,
                random = list(~ subphylum | study.id, ~ subphylum | effect.size.id),
                struct="DIAG", sparse=TRUE, data=dat.m)
set.seed(1234)
dat.m$row <- as.numeric(factor(dat.m$subphylum))
d <- abs(resid(res.m))^(1/8)
d <- (d - min(d)) / (max(d) - min(d))
d <- (1 - d) * 0.5
dat.m$row.jitter <- dat.m$row + runif(nrow(dat.m), -d, d)
cols <- rep(palette.colors(8, palette="Dark 2"), length.out=nlvls)
cols.adj <- adjustcolor(cols, alpha.f=0.6)
dat.m$colpt <- cols.adj[dat.m$row]
dat.m$viadj <- pmax(dat.m$vi, quantile(dat.m$vi, 0.05))
sav <- with(dat.m, forest(yi, viadj, header="Subphylum", rows=row.jitter, slab=NA,
                          annotate=FALSE, xlim=c(-1.8,1.5), ylim=c(0.5,nlvls+2.5),
                          cex=1, pch=19, efac=0, col=colpt, colci=NA,
                          transf=transf.ztor, digits=1))
text(sav$xlim[1], 1:nlvls, names(table(dat.m$subphylum)), pos=4)
abline(h=0:nlvls+0.5, col="gray20", lwd=0.2)
pred <- predict(res.m, newmods=diag(res.m$p), tau2.levels=1:nlvls,
                gamma2.levels=1:nlvls, transf=transf.ztor)
segments(pred$pi.lb, 1:nlvls, pred$pi.ub, 1:nlvls, lwd=1)
segments(pred$ci.lb, 1:nlvls, pred$ci.ub, 1:nlvls, lwd=3)
points(pred$pred, 1:nlvls, pch=21, bg=cols, cex=1.5)
text(sav$xlim[2], sav$ylim[2]-1, "k", pos=2, font=2)
text(sav$xlim[2], 1:nlvls, table(dat.m$subphylum), pos=2, cex=0.75)
dev.off()

### caterpillar

device("caterpillar", 6, 8)
set.seed(5132)
k <- 250
vi <- rchisq(k, df=1) * .03
yi <- rnorm(k, rnorm(k, 0.5, 0.4), sqrt(vi))
res.c <- rma(yi, vi)
forest(yi, vi,
       header=FALSE,
       xlim=c(-2.5,3.5),
       ylim=c(-8,254),
       order="est",
       slab=NA, annotate=FALSE,
       efac=c(0,1),
       pch=19,
       colci="gray40",
       psize=2.5,
       cex.lab=1, cex.axis=1,
       lty=c("solid","blank"))
addpoly(res.c, mlab="", cex=1)
text(-2, -2, "RE Model", pos=4, offset=0, cex=1)
dev.off()

### funnel_variations

device("funnel_variations", 9, 9)
res.h <- rma(yi, vi, data=dat.hackshaw1998, measure="OR", method="EE")
par(mfrow=c(2,2))
funnel(res.h, main="Standard Error")
funnel(res.h, yaxis="vi", main="Sampling Variance")
funnel(res.h, yaxis="seinv", main="Inverse Standard Error")
funnel(res.h, yaxis="vinv", main="Inverse Sampling Variance")
dev.off()

### funnel_contour

device("funnel_contour", 7, 7)
res.b <- rma(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
funnel(res.b, level=c(90, 95, 99), refline=0, legend=TRUE)
dev.off()

### funnel_contour2

device("funnel_contour2", 7, 7)
dat.g <- escalc(measure="OR", ai=ai, n1i=n1i, ci=ci, n2i=n2i, data=dat.graves2010)
funnel(dat.g$yi, dat.g$vi, yaxis="seinv", ylab="Precision (1/se)",
       xlim=c(-3,2), ylim=c(.00001,8), xaxs="i", yaxs="i", las=1,
       level=c(.10, .05, .01), legend=TRUE, back="gray90", lty=0)
dev.off()

### funnel_limit

device("funnel_limit", 7, 7)
res.l <- rma(yi, vi, data=dat.hackshaw1998)
funnel(res.l, refline=0)
reg <- regtest(res.l)
se <- seq(0,0.8,length=100)
lines(coef(reg$fit)[1] + coef(reg$fit)[2]*se, se)
reg <- regtest(res.l, predictor="vi")
lines(coef(reg$fit)[1] + coef(reg$fit)[2]*se^2, se, lwd=2, lty="dotted")
legend("topright", inset=.02, lty=c("solid","dotted"), lwd=2, cex=0.9, bg="white",
       legend=c("Standard Errors as Predictor",
                "Sampling Variances as Predictor"))
dev.off()

### funnel_trimfill

device("funnel_trimfill", 7, 7)
res.t <- rma(yi, vi, data=dat.hackshaw1998, measure="OR")
taf <- trimfill(res.t)
funnel(taf, legend=list(show="cis"))
dev.off()

### baujat

device("baujat", 7, 7)
dat.p <- dat.pignon2000
dat.p$yi <- with(dat.p, OmE/V)
dat.p$vi <- with(dat.p, 1/V)
res.p <- rma(yi, vi, data=dat.p, method="EE", slab=id)
baujat(res.p, xlim=c(0,20), ylim=c(0,0.2), bty="l", las=1)
dev.off()

### gosh

device("gosh", 7, 7)
res.e <- rma(measure="OR", ai=ai, n1i=n1i, ci=ci, n2i=n2i, data=dat.egger2001, method="EE")
sav <- gosh(res.e)
plot(sav, out=16, breaks=100)
dev.off()

### labbe

device("labbe", 9, 9)
dat.d <- dat.damico2009
res.d <- rma(measure="OR", ai=xt, n1i=nt, ci=xc, n2i=nc, data=dat.d)
par(mfrow=c(2,2))
par(cex=0.7)
labbe(res.d)
labbe(res.d, ci=TRUE, pi=TRUE, grid=TRUE, legend=TRUE, bty="l",
      xlab="Log Odds (Control Group)", ylab="Log Odds (Treatment Group)")
labbe(res.d, ci=TRUE, pi=TRUE, grid=TRUE, legend=TRUE, bty="l",
      transf=exp, xlab="Odds (Control Group)", ylab="Odds (Treatment Group)")
labbe(res.d, ci=TRUE, pi=TRUE, grid=TRUE, legend=TRUE, bty="l",
      transf=plogis, lim=c(0,1), xlab="Risk (Control Group)",
      ylab="Risk (Treatment Group)")
dev.off()

### qqnorm

device("qqnorm", 9, 9)
res1 <- rma(yi, vi, data=dat, method="EE")
res2 <- rma(yi, vi, data=dat)
res3 <- rma(yi, vi, mods=~ablat, data=dat, method="FE")
res4 <- rma(yi, vi, mods=~ablat, data=dat)
par(mfrow=c(2,2))
qqnorm(res1, main="Equal-Effects Model")
qqnorm(res2, main="Random-Effects Model")
qqnorm(res3, main="Fixed-Effects with Moderators Model")
qqnorm(res4, main="Mixed-Effects Model")
dev.off()

### radial

device("radial", 7, 7)
res.r <- rma(yi, vi, data=dat.hackshaw1998, method="EE")
par(mar=c(5,4,0,3))
radial(res.r)
dev.off()

### influence

device("influence", 7, 11)
res.i <- rma(ri=ri, ni=ni, measure="ZCOR", data=dat.mcdaniel1994)
inf <- influence(res.i)
par(mfrow=c(8,1))
plot(inf)
dev.off()

### forest_cumul

dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
device("forest_cumul", 8, 7)
res.c <- rma(yi, vi, data=dat, slab=paste(author, year, sep=", "))
tmp <- cumul(res.c, order=year)
forest(tmp, xlim=c(-4,2), at=log(c(0.125, 0.25, 0.5, 1, 2)), shade=TRUE, atransf=exp,
       digits=c(2L,3L), cex=0.85, header="Author(s) and Year")
dev.off()

### regplot

device("regplot", 7, 7)
res.m <- rma(yi, vi, mods = ~ ablat, data=dat)
regplot(res.m, xlim=c(10,60), predlim=c(10,60), xlab="Absolute Latitude", refline=0, atransf=exp,
        at=log(seq(0.2,1.6,by=0.2)), digits=1, las=1, bty="l", label=c(4,7,12,13), offset=c(1.6,0.8), labsize=0.9)
dev.off()

### cumul

device("cumul", 7, 7)
res.cu <- rma(yi, vi, data=dat)
tmp <- cumul(res.cu, order=year)
plot(tmp, transf=exp, xlim=c(0.25,0.5), lwd=3, cex=1.3)
dev.off()

### llplot

device("llplot", 7, 7)
llplot(measure="OR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg, lwd=1, refline=NA, xlim=c(-4,4))
dev.off()

### permutest

device("permutest", 7, 7)
res.p <- rma(yi, vi, mods = ~ ablat + year, data=dat)
set.seed(1234)
perm <- permutest(res.p, iter=1000)
plot(perm, QM=TRUE, lwd=c(2,1,1,0), legend=TRUE)
dev.off()

### vif

device("vif", 7, 7)
set.seed(1234)
plot(vif(res.p, btt=2, sim=TRUE, progbar=FALSE))
dev.off()

### forest_superscripts

device("forest_superscripts", 8, 7)
dat <- dat.bcg
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat)
dat$slab <- with(dat, paste(author, year, sep=", "))
dat$slab.ref <- mapply(function(x,y) as.expression(bquote(.(x)^.(y))), dat$slab, dat$trial)
res <- rma(yi, vi, data=dat)
mlabfun <- function(text, x) {
   list(bquote(paste(.(text),
      " (Q = ", .(fmtx(x$QE, digits=2)),
      ", df = ", .(x$k - x$p), ", ",
      .(fmtp2(x$QEp)), "; ",
      I^2, " = ", .(fmtx(x$I2, digits=1)), "%, ",
      tau^2, " = ", .(fmtx(x$tau2, digits=2)), ")")))}
forest(res, atransf=exp, at=log(c(0.05, 0.25, 1, 4)), xlim=c(-16,6),
       ilab=cbind(tpos, tneg, cpos, cneg), ilab.lab=c("TB+","TB-","TB+","TB-"),
       ilab.xpos=c(-9.5,-8,-6,-4.5), cex=0.75, header="Author(s) and Year",
       mlab=mlabfun("RE Model", res), slab=slab.ref, shade=TRUE)
text(c(-8.75,-5.25), res$k+2.8, c("Vaccinated", "Control"), cex=0.75, font=2)
dev.off()

### forest_revman

png(file.path(out_dir, "forest_revman.png"), width = 9.13, height = 4.71, units = "in", res = 100, type = "cairo")
dat <- data.frame(author = c("Amore-Coffea", "Deliciozza", "Kahve-Paradiso", "Mama-Kaffa", "Morrocona", "Norscafe", "Oohlahlazza", "Piazza-Allerta"),
                  year   = c(2000, 2004, 2002, 1999, 1998, 1998, 1998, 2003),
                  ai     = c(2, 10, 0, 12, 3, 19, 4, 8),
                  n1i    = c(31, 40, 0, 53, 15, 68, 35, 35),
                  ci     = c(10, 9, 0, 9, 1, 9, 2, 6),
                  n2i    = c(34, 40, 0, 61, 17, 64, 37, 37),
                  rb.a   = c("?", "+", "-", "-", "+", "?", "+", "?"),
                  rb.b   = c("?", "?", "-", "-", "?", "?", "+", "?"),
                  rb.c   = c("?", "?", "?", "?", "?", "+", "+", "?"),
                  rb.d   = c("?", "?", "+", "-", "+", "?", "+", "+"),
                  rb.e   = c("-", "-", "+", "-", "+", "-", "+", "+"),
                  rb.f   = c("+", "+", "+", "+", "+", "-", "+", "+"))
dat[7:12] <- lapply(dat[7:12], factor, levels=c("+", "-", "?"))
dat <- escalc(measure="OR", ai=ai, n1i=n1i, ci=ci, n2i=n2i, data=dat,
              slab=paste(author, year), drop00=TRUE)
res <- rma(yi, vi, data=dat, method="DL")
pred <- predict(res, transf=exp, digits=2)
pred <- fmtx(c(pred$pred, pred$ci.lb, pred$ci.ub), digits=2)
k <- nrow(dat)
options(na.action = "na.pass")
weights <- paste0(fmtx(weights(res), digits=1), "%")
weights[weights == "NA%"] <- ""
par(mar=c(10.9,0,1.8,1.3), mgp=c(3,0.2,0), tcl=-0.2)
sav <- forest(res, atransf=exp, at=log(c(0.01, 0.10, 1, 10, 100)), xlim=c(-30,11),
       xlab="", efac=c(0,4), textpos=c(-30,-4.7), lty=c(1,1,0), refline=NA,
       ilab=cbind(ai, n1i, ci, n2i, weights),
       ilab.xpos=c(-20.6,-18.6,-16.1,-14.1,-10.8), ilab.pos=2,
       cex=0.78, header=c("Study or Subgroup","IV, Random, 95% CI"), mlab="",
       colout="#0000cc", colci="black")
segments(sav$xlim[1]+0.5, k+1, sav$xlim[2], k+1, lwd=0.8)
segments(0, -2, 0, k+1, lwd=0.8)
par(xpd=NA)
par(cex=sav$cex, font=2)
text(sav$ilab.xpos, k+2, pos=2, c("Events","Total","Events","Total","Weight"))
text(c(mean(sav$ilab.xpos[1:2]),mean(sav$ilab.xpos[3:4])), k+3, pos=2, c("Caffeine","Decaf"))
text(sav$textpos[2], k+3, "Odds ratio", pos=2)
text(0, k+3, "Odds ratio")
text(sav$xlim[2]-0.6, k+3, "Risk of Bias", pos=2)
text(0, k+2, "IV, Random, 95% CI")
text(c(sav$xlim[1],sav$ilab.xpos[c(2,4,5)]), -1, pos=c(4,2,2,2,2),
     c("Total (95% CI)", sum(dat$n1i), sum(dat$n2i), "100.0%"))
text(sav$xlim[1], -7, pos=4, "Risk of bias legend")
rect(sav$textpos[2], -1.5, sav$ilab.xpos[5], -0.5, col="white", border=NA)
text(sav$textpos[2], -1, paste0(pred[1], " [", pred[2], ",  ", pred[3], "]"), pos=2)
par(cex=sav$cex, font=1)
text(log(c(0.01, 100)), -4, c("Favours caffeine","Favours decaf"), pos=c(4,2), offset=-0.5)
text(sav$textpos[2], k+1-which(is.na(dat$yi)), "Not estimable", pos=2)
text(sav$xlim[1], -2, pos=4, "Total events:")
text(sav$ilab.xpos[c(1,3)], -2, c(sum(dat$ai),sum(dat$ci)), pos=2)
text(sav$xlim[1], -3, pos=4, bquote(paste("Heterogeneity: ",
   "Tau"^2, " = ", .(fmtx(res$tau2, digits=2)), "; ",
   "Chi"^2, " = ", .(fmtx(res$QE, digits=2)),
   ", df = ", .(res$k - res$p),
   " (", .(fmtp(res$QEp, digits=2, pname="P", add0=TRUE, sep=TRUE, equal=TRUE)), "); ",
   I^2, " = ", .(round(res$I2)), "%")))
text(sav$xlim[1], -4, pos=4, bquote(paste("Test for overall effect: ",
   "Z = ", .(fmtx(res$zval, digits=2)),
   " (", .(fmtp(res$pval, digits=2, pname="P", add0=TRUE, sep=TRUE, equal=TRUE)), ")")))
text(sav$xlim[1], -5, pos=4, bquote(paste("Test for subgroup differences: Not applicable")))
cols <- c("#00cc00", "#cc0000", "#eeee00")
syms <- levels(dat$rb.a)
pos  <- seq(sav$xlim[2]-5.5,sav$xlim[2]-0.5,length=6)
for (i in 1:6) {
   points(rep(pos[i],k), k:1, pch=19, col=cols[dat[[6+i]]], cex=2.2)
   text(pos[i], k:1, syms[dat[[6+i]]], font=2)
}
text(pos, k+2, c("A","B","C","D","E","F"), font=2)
text(sav$xlim[1], -8:-13, pos=4, c(
   "(A) Random sequence generation (selection bias)",
   "(B) Allocation concealment (selection bias)",
   "(C) Blinding of participants and personnel (performance bias)",
   "(D) Incomplete outcome data (attrition bias)",
   "(E) Selective reporting (reporting bias)",
   "(F) Other bias"))
dev.off()
options(na.action = "na.omit")

### forest_bmj

png(file.path(out_dir, "forest_bmj.png"), width = 10.51, height = 3.9, units = "in", res = 100, type = "cairo")
dat <- data.frame(author = c("Dyson", "Jönsson", "Morris", "Saslow", "Saslow", "Sato", "Tay", "Yamada"),
                  year   = c(2010, 2009, 2019, 2014, 2017, 2017, 2014, 2014),
                  ai     = c(3, 6, 11, 8, 6, 4, 36, 2),
                  n1i    = c(6, 6, 21, 9, 11, 22, 46, 12),
                  ci     = c(1, 3, 0, 5, 0, 0, 30, 2),
                  n2i    = c(6, 6, 12, 13, 8, 27, 47, 12))
dat <- escalc(measure="RD", ai=ai, n1i=n1i, ci=ci, n2i=n2i, data=dat,
              slab=paste(" ", author, year), addyi=FALSE)
res <- rma(yi, vi, data=dat, method="DL")
colp <- "#5f4c9a"
coll <- "#a7a9ac"
k <- nrow(dat)
psize <- weights(res)
psize <- 1.2 + (psize - min(psize)) / (max(psize) - min(psize))
weights <- fmtx(weights(res), digits=1)
par(mar=c(2.7,3.2,2.5,1.3), mgp=c(3,0,0), tcl=0.15)
sav <- forest(dat$yi, dat$vi, xlim=c(-3.4,2.1), ylim=c(-1,k+3), alim=c(-1,1), cex=0.88,
              header=FALSE, pch=18, psize=psize, efac=0, refline=NA, lty=c(1,0), xlab="",
              ilab=cbind(paste(dat$ai, "/", dat$n1i), paste(dat$ci, "/", dat$n2i), weights),
              ilab.xpos=c(-1.9,-1.3,1.2), annosym=c(" (", " to ", ")"), rowadj=-.07)
segments(0, -1, 0, k+1.6, col=coll)
segments(coef(res), 0, coef(res), k, col=colp, lty="33", lwd=0.8)
segments(summary(dat)$ci.lb, k:1, summary(dat)$ci.ub, k:1, col=colp, lwd=1.5)
points(dat$yi, k:1, pch=18, cex=psize*1.15, col="white")
points(dat$yi, k:1, pch=18, cex=psize, col=colp)
addpoly(res, row=0, mlab="Total (95% CI)", efac=2, col=colp, border=colp)
abline(h=k+1.6, col=coll)
axis(side=1, at=seq(-1,1,by=0.5), col=coll, labels=FALSE)
par(xpd=NA)
par(cex=sav$cex, font=2)
text(sav$xlim[1], k+2.5, pos=4, "Study or\nsubgroup")
text(sav$ilab.xpos[1:2], k+2.3, c("Experimental","Control"))
text(mean(sav$ilab.xpos[1:2]), k+3.4, "No of events / total")
text(0, k+2.7, "Risk difference, IV,\nrandom (95% CI)")
segments(sav$ilab.xpos[1]-0.24, k+2.8, sav$ilab.xpos[2]+0.14, k+2.8)
text(c(sav$ilab.xpos[3],sav$xlim[2]-0.37), k+2.7, c("Weight\n(%)","Risk difference, IV,\nrandom (95% CI)"))
text(c(-1,1), -2.5, c("Favors control","Favors experimental"), pos=c(4,2), offset=-0.3)
par(cex=sav$cex, font=1)
text(sav$ilab.xpos[3], 0, "100.0")
text(sav$ilab.xpos[1:2], 0, c(paste(sum(dat$ai), "/", sum(dat$n1i)), paste(sum(dat$ci), "/", sum(dat$n2i))))
text(sav$xlim[1], -1, pos=4, bquote(paste("Test for heterogeneity: ",
   tau^2, "=", .(fmtx(res$tau2, digits=2)), "; ",
   chi^2, "=", .(fmtx(res$QE, digits=2)),
   ", df=", .(res$k - res$p), ", ",
   .(fmtp(res$QEp, digits=2, pname="P", add0=TRUE, equal=TRUE)), "; ",
   I^2, "=", .(round(res$I2)), "%")))
text(sav$xlim[1], -2, pos=4, bquote(paste("Test for overall effect: ",
   "Z=", .(fmtx(res$zval, digits=2)), ", ",
   .(fmtp(res$pval, digits=3, pname="P", add0=TRUE, equal=TRUE)))))
dev.off()
