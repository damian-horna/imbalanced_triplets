library(knitr)
library(scmamp)
library(readr)

reportFriedmana <- function(averages,  metricName, make_plot = F){
  ranks <- t(apply(-averages, 1, rank))
  ranks <- ranks[,order(colMeans(ranks, na.rm=TRUE))]
  
  cat("<hr><strong>Friedman rank sum test</strong><br />")
  testResult <- capture.output(print(friedman.test( data.matrix(averages))))
  cat(testResult[5])
  cat("\r\n")
  print(kable(t(colMeans(ranks, na.rm=TRUE)), digits = 2))
  cat("\r\n")
  
  if (make_plot){
    if (SAVE_EPS) {
      setEPS()
      postscript(paste0(metricName, "_",  "Friedman.eps"), width = 7, height = 3.4)
      plot = tryCatch({
        plotCD(results.matrix = averages, alpha = 0.05, cex = 1.1)
      }, error = function(e) {
        return(last_plot())
      })
      dev.off()
    }
    
    plot = tryCatch({
      plotCD(results.matrix = averages, alpha = 0.05)
    }, error = function(e) {
      return(last_plot())
    })
    
    plot
  }
  
  cat("<hr>")
}

data <- read_csv("/home/dhorna/dev/studies/mgr/mgr-repo/results_csv/f1_knn.csv")
#View(data)

# Remove dataset column from data
averages <- data[-1]
#averages
ranks <- t(apply(-averages, 1, rank))
ranks <- ranks[,order(colMeans(ranks, na.rm=TRUE))]
rowMeans(t(ranks))
sort(rowMeans(t(ranks)))
SAVE_EPS=T

# Removing cleveland_v2
averages = averages[-12,]

reportFriedmana(averages, "KNN_f", make_plot = F)

pvalues_vec <- c(wilcox.test(averages$`New Rep.`,averages$Baseline, paired=T)$p.value,
wilcox.test(averages$`New Rep.`,averages$`Global-CS`, paired=T)$p.value,
wilcox.test(averages$`New Rep.`,averages$`Static-SMOTE`, paired=T)$p.value,
wilcox.test(averages$`New Rep.`,averages$MDO, paired=T)$p.value
)
pvalues_vec