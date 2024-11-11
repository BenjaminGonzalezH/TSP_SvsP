################ Libraries.
library(ggplot2)

################ WorkSpaces.
setwd("C:/Users/benja/OneDrive/Escritorio/WorkSpace/TSP_SvsP/Results/Experimentals")

################
# Convergencia.
################

Converge_SA <- read.csv("tabu_search_sample_converge_194.csv")
Converge_GA <- read.csv("GA_converge_194.csv")

################
# Resultados.
################
Caso_GA_1 <- read.csv("genetic_algorithm_results_38_80000.txt")
Caso_GA_2 <- read.csv("genetic_algorithm_results_76_160000.txt")
Caso_GA_3 <- read.csv("genetic_algorithm_results_194_2000000.txt")
Caso_GA_4 <- read.csv("genetic_algorithm_results_318_4000000.txt")
Caso_TS_1 <- read.csv("tabu_search_results_38_80000.txt")
Caso_TS_2 <- read.csv("tabu_search_results_76_160000.txt")
Caso_TS_3 <- read.csv("tabu_search_results_194_2000000.txt")
Caso_TS_4 <- read.csv("tabu_search_results_318_4000000.txt")

################
# Gráfico convergencia TS.
################
plot(Converge_SA[["Mejor"]], type = "o", col = "blue", 
     xlab = "Iteracion", 
     ylab = "GAP", 
     main = "Convergencia Tabu Search")

################
# Gráfico convergencia GA.
################
# Lista.
Converge_GA_list <- split(Converge_GA, Converge_GA$Iteration)

# Elementos a mostrar.
min_iter_1 <- 0
max_iter_1 <- 50
min_iter_2 <- 100
max_iter_2 <- 150
min_iter_3 <- 500
max_iter_3 <- 550
min_iter_4 <- 3200
max_iter_4 <- 3250

# Filtrar el dataframe Converge_GA para incluir solo el rango de iteraciones deseado
Converge_GA_filtered_1 <- subset(Converge_GA, Iteration >= min_iter_1 & Iteration <= max_iter_1)
Converge_GA_filtered_2 <- subset(Converge_GA, Iteration >= min_iter_2 & Iteration <= max_iter_2)
Converge_GA_filtered_3 <- subset(Converge_GA, Iteration >= min_iter_3 & Iteration <= max_iter_3)
Converge_GA_filtered_4 <- subset(Converge_GA, Iteration >= min_iter_4 & Iteration <= max_iter_4)

# Dividir el dataframe filtrado en subconjuntos según la columna 'Iteration'
Converge_GA_list_1 <- split(Converge_GA_filtered_1, Converge_GA_filtered_1$Iteration)
Converge_GA_list_2 <- split(Converge_GA_filtered_2, Converge_GA_filtered_2$Iteration)
Converge_GA_list_3 <- split(Converge_GA_filtered_3, Converge_GA_filtered_3$Iteration)
Converge_GA_list_4 <- split(Converge_GA_filtered_4, Converge_GA_filtered_4$Iteration)

valores_1 <- unlist(lapply(Converge_GA_list_1, function(df) df[["Error"]]))
valores_2 <- unlist(lapply(Converge_GA_list_2, function(df) df[["Error"]]))
valores_3 <- unlist(lapply(Converge_GA_list_3, function(df) df[["Error"]]))
valores_4 <- unlist(lapply(Converge_GA_list_4, function(df) df[["Error"]]))

etiquetas_iteracion_1 <- rep(names(Converge_GA_list_1), sapply(Converge_GA_list_1, nrow))
etiquetas_iteracion_2 <- rep(names(Converge_GA_list_2), sapply(Converge_GA_list_2, nrow))
etiquetas_iteracion_3 <- rep(names(Converge_GA_list_3), sapply(Converge_GA_list_3, nrow))
etiquetas_iteracion_4 <- rep(names(Converge_GA_list_4), sapply(Converge_GA_list_4, nrow))

# Convergencia.
boxplot(valores_1 ~ etiquetas_iteracion_1, 
        xlab = "Iteración", 
        ylab = "Error", 
        main = "Convergencia GA - Boxplot por Iteración (0-50)",
        col = "lightblue")

boxplot(valores_2 ~ etiquetas_iteracion_2, 
        xlab = "Iteración", 
        ylab = "Error", 
        main = "Convergencia GA - Boxplot por Iteración (100-150)",
        col = "lightblue")

boxplot(valores_3 ~ etiquetas_iteracion_3, 
        xlab = "Iteración", 
        ylab = "Error", 
        main = "Convergencia GA - Boxplot por Iteración (500-550)",
        col = "lightblue")

boxplot(valores_4 ~ etiquetas_iteracion_4, 
        xlab = "Iteración", 
        ylab = "Error", 
        main = "Convergencia GA - Boxplot por Iteración (3200-3242)",
        col = "lightblue")

################
# Competencia.
################

Caso_TS_1$Origen <- "Tabu Search"
Caso_TS_2$Origen <- "Tabu Search"
Caso_TS_3$Origen <- "Tabu Search"
Caso_TS_4$Origen <- "Tabu Search"
Caso_GA_1$Origen <- "Genetic Algorithm"
Caso_GA_2$Origen <- "Genetic Algorithm"
Caso_GA_3$Origen <- "Genetic Algorithm"
Caso_GA_4$Origen <- "Genetic Algorithm"

# Combinar ambos dataframes
Caso_combined_1 <- rbind(Caso_TS_1, Caso_GA_1)
Caso_combined_2 <- rbind(Caso_TS_2, Caso_GA_2)
Caso_combined_3 <- rbind(Caso_TS_3, Caso_GA_3)
Caso_combined_4 <- rbind(Caso_TS_4, Caso_GA_4)

# Crear boxplot comparativo
boxplot(Error ~ Origen, data = Caso_combined_1,
        xlab = "Algoritmo",
        ylab = "Error",
        main = "Comparativo TS-GA (dj38) 80.000 llamadas",
        col = c("lightblue", "lightgreen"))

boxplot(Error ~ Origen, data = Caso_combined_2,
        xlab = "Algoritmo",
        ylab = "Error",
        main = "Comparativo TS-GA (pr76) 160.000 llamadas",
        col = c("lightblue", "lightgreen"))

boxplot(Error ~ Origen, data = Caso_combined_3,
        xlab = "Algoritmo",
        ylab = "Error",
        main = "Comparativo TS-GA (qa194) 2.000.000 llamadas",
        col = c("lightblue", "lightgreen"))

boxplot(Error ~ Origen, data = Caso_combined_4,
        xlab = "Algoritmo",
        ylab = "Error",
        main = "Comparativo TS-GA (lin318) 4.000.000 llamadas",
        col = c("lightblue", "lightgreen"))

################
# Prueba Normal.
################

resultado_shapiro <- shapiro.test(Caso_GA_1$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_2$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_3$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_4$Error)
print(resultado_shapiro)


resultado_shapiro <- shapiro.test(Caso_TS_1$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_TS_2$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_TS_3$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_TS_4$Error)
print(resultado_shapiro)

################
# Pruebas estadísticas.
################

wilcox.test(Caso_GA_1$Error, Caso_TS_1$Error, paired = TRUE, alternative = "less")
t.test(Caso_GA_2$Error, Caso_TS_2$Error, paired = TRUE)
t.test(Caso_GA_3$Error, Caso_TS_2$Error, paired = TRUE)
t.test(Caso_GA_4$Error, Caso_TS_2$Error, paired = TRUE)