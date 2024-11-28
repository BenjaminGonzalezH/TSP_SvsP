################ Libraries.
library(ggplot2)
library(dplyr)

################ WorkSpaces.
setwd("C:/Users/benja/OneDrive/Escritorio/WorkSpace/TSP_SvsP/Results/Experimentals")

################
# Convergencia.
################

Converge_SA <- read.csv("TS_converge_38.csv")
Converge_GA <- read.csv("GAePBXscr_converge_38.csv")

################
# Resultados.
################
Caso_GA_1 <- read.csv("")
Caso_GA_2 <- read.csv("")
Caso_GA_3 <- read.csv("")
Caso_TS_1 <- read.csv("TS_results_38_80000.txt")
Caso_TS_2 <- read.csv("TS_results_76_80000.txt")
Caso_TS_3 <- read.csv("TS_results_194_80000.txt")


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
inicio <- 1
fin <- 50

# Filtrar el dataframe para obtener las iteraciones dentro del rango especificado
df_filtrado <- Converge_GA %>%
  filter(Iteration >= inicio & Iteration <= fin) %>%
  arrange(Iteration)

# Graficar boxplot para 'Error' por cada 'iteration'
ggplot(df_filtrado, aes(x = factor(Iteration), y = Error)) +
  geom_boxplot(aes(fill = factor(Iteration)), alpha = 0.7) + 
  labs(
    title = "Gráfico de convergencia GA celular PBX-scramble(Shuffle)",
    x = "Iteración",
    y = "Error"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

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