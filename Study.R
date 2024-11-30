################ Libraries.
library(ggplot2)
library(dplyr)

################ WorkSpaces.
setwd("C:/Users/benja/OneDrive/Escritorio/WorkSpace/TSP_SvsP/Results/Experimentals")

################
# Convergencia.
################

Converge_SA <- read.csv("TS_converge_38.csv")
Converge_GA <- read.csv("GAcPBXscr_converge_38.csv")

################
# Resultados.
################
Caso_GA_1 <- read.csv("GAe_PMX_swp_results_76_80000.txt",header = FALSE)
Caso_GA_2 <- read.csv("GAe_OX_inv_results_76_80000.txt", header = FALSE)
Caso_GA_3 <- read.csv("GAe_PBX_scr_results_76_80000.txt", header = FALSE)
Caso_GA_4 <- read.csv("GAc_PMX_sw_results_76_80000.txt",header = FALSE)
Caso_GA_5 <- read.csv("GAc_OX_inv_results_76_80000.txt", header = FALSE)
Caso_GA_6 <- read.csv("GAc_PBX_scr_results_76_80000.txt", header = FALSE)
Caso_TS_1 <- read.csv("TS_results_76_80000.txt",header = FALSE)

df <- bind_rows(
  data.frame(Caso = "GAcPMX-swp", Error = Caso_GA_1$V2),
  data.frame(Caso = "GAcOX-inv", Error = Caso_GA_2$V2),
  data.frame(Caso = "GAcPBX-scr", Error = Caso_GA_3$V2),
  data.frame(Caso = "GAePMX-swap", Error = Caso_GA_4$V2),
  data.frame(Caso = "GAeOX-inv", Error = Caso_GA_5$V2),
  data.frame(Caso = "GAePBX-scr", Error = Caso_GA_6$V2),
  data.frame(Caso = "TS", Error = Caso_TS_1$V2)
)

ggplot(df, aes(x = Caso, y = Error, fill = Caso)) +
  geom_boxplot(alpha = 0.7) + 
  labs(
    title = "Boxplots de Error normalizado qa194 (80.000 llamadas)",
    x = "Caso",
    y = "Error"
  ) +
  theme_minimal() +
  theme(legend.position = "none", 
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

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
    title = "Gráfico de convergencia GA clásico PBX-scramble(Shuffle)",
    x = "Iteración",
    y = "Error"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

################
# Prueba Normal.
################

resultado_shapiro <- shapiro.test(Caso_GA_1$V2)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_2$V2)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_3$V2)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_4$V2)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_5$V2)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(Caso_GA_6$V2)
print(resultado_shapiro)


resultado_shapiro <- shapiro.test(Caso_TS_1$V2)
print(resultado_shapiro)

################
# Pruebas estadísticas.
################

wilcox.test(Caso_GA_1$Error, Caso_TS_1$Error, paired = TRUE, alternative = "less")
t.test(Caso_GA_2$Error, Caso_TS_2$Error, paired = TRUE)
t.test(Caso_GA_3$Error, Caso_TS_2$Error, paired = TRUE)
t.test(Caso_GA_4$Error, Caso_TS_2$Error, paired = TRUE)