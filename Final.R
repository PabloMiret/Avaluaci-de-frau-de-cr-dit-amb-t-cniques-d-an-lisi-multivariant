# Cargo les dades
datos <- read.csv("dataset.csv")

# Miro si tot està ok
head(datos)

# Llibreries necessaries
library(caret)
library(knitr)
library(MASS)
library(dplyr)
library(ROSE)

# Funció per a separar train/test
train_test <- function(datos, target_column = "TARGET", train_ratio = 0.8, seed = 42) {
  set.seed(seed)
  y <- datos[[target_column]]
  index <- createDataPartition(y, p = train_ratio, list = FALSE)
  
  datos_entrenamiento <- datos[index, ]
  datos_prueba <- datos[-index, ]
  
  y_train <- datos_entrenamiento[[target_column]]
  x_train <- datos_entrenamiento[, !colnames(datos_entrenamiento) %in% target_column]
  
  y_test <- datos_prueba[[target_column]]
  x_test <- datos_prueba[, !colnames(datos_prueba) %in% target_column]
  
  list(
    x_train = x_train,
    y_train = y_train,
    x_test = x_test,
    y_test = y_test
  )
}

# MODELS
models <- function(x_train, y_train, x_test, y_test) {
  ## Regresió Logística
  modelo_rl <- glm(y_train ~ ., data = cbind(x_train, y_train), family = binomial)
  predicciones_rl <- ifelse(predict(modelo_rl, x_test, type = "response") > 0.5, 1, 0)
  metricas_rl <- calcular_metriques(y_test, predicciones_rl)
  
  ## Anàlisi Discriminant Lineal
  modelo_d <- lda(y_train ~ ., data = cbind(x_train, y_train))
  predicciones_d <- predict(modelo_d, x_test)$class
  metricas_d <- calcular_metriques(y_test, predicciones_d)
  
  list(
    predicciones_rl = predicciones_rl,
    metricas_rl = metricas_rl,
    predicciones_d = predicciones_d,
    metricas_d = metricas_d
  )
}


# Funció per a extreure informació de les prediccions
calcular_metriques <- function(y_test, predicciones) {
  
  # Matriu de confució
  matriz_confusion <- table(y_test, predicciones)
  
  # Extreure els valors
  TP <- matriz_confusion[2, 2]
  FP <- matriz_confusion[1, 2]
  TN <- matriz_confusion[1, 1]
  FN <- matriz_confusion[2, 1]
  
  # Calcular els indicadors de precisió
  accuracy <- (TP + TN) / sum(matriz_confusion) * 100
  precision <- TP / (TP + FP) * 100
  recall <- TP / (TP + FN) * 100
  specificity <- TN / (TN + FP) * 100
  
  # Round i %
  accuracy <- paste0(round(accuracy, 2), "%")
  precision <- paste0(round(precision, 2), "%")
  recall <- paste0(round(recall, 2), "%")
  specificity <- paste0(round(specificity, 2), "%")
  
  # Crear data frame 
  resultados <- data.frame(
    Mètrica = c("Veritables positius (TP)", "Falsos positius (FP)", "Veritables negatius (TN)", "Falsos negatius (FN)",
                "Taxa d'encerts (Accuracy)", "Precisió (Precision)", "Sensibilitat (Recall)", "Especificitat (Specificity)"),
    Valor = c(TP, FP, TN, FN, accuracy, precision, recall, specificity)
  )
  
  # Mostrar els resultats
  kable(resultados, caption = "Resultats Matriu de Confusió i Indicadors de Precisió")
}


train_test_datos <- train_test(datos)

# Acceder a los datos de entrenamiento y prueba
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

resultados<- models(x_train, y_train, x_test, y_test) #dona error

#Mirar el motiu del error
# Regresión Logística
modelo_glm <- glm(y_train ~ ., data = x_train, family = binomial)
predicciones_glm <- predict(modelo_glm, newdata = x_test, type = "response")
predicciones_glm <- ifelse(predicciones_glm > 0.5, 1, 0) # Ajusta según tus clases
precision_glm <- mean(predicciones_glm == y_test)
print(precision_glm)

# Análisis Discriminante Lineal
modelo_lda <- lda(y_train ~ ., data = x_train)
predicciones_lda <- predict(modelo_lda, newdata = x_test)
precision_lda <- mean(predicciones_lda$class == y_test)
print(precision_lda)

matriz_confusion1 <- table(y_test,predicciones_glm)
matriz_confusion2 <- table(y_test,predicciones_lda$class)

print(matriz_confusion1)
print(matriz_confusion2)

# Mirar balanç entre classes
table(datos$TARGET) #Es veu com hi ha una gran diferencia entre valors target = 0 i target = 1 fet pel qual fa que classifiqui tot a 0.

# Igualar num d'obserbaciones
# Dividir el dataframe en dos subconjunts 
datos_0 <- datos %>% filter(TARGET == 0)
datos_1 <- datos %>% filter(TARGET == 1)

# Valor min
n_min <- min(nrow(datos_0), nrow(datos_1))

# Mostreig aleatori del mateix nombre de obserbaciones 
set.seed(42)  
datos_0_sample <- datos_0 %>% sample_n(n_min + (n_min*0.2)) 
datos_1_sample <- datos_1 %>% sample_n(n_min)

# Nou dataset amb files mezcladas
datos_balanceados <- bind_rows(datos_0_sample, datos_1_sample)
datos <- datos_balanceados %>% sample_frac(1)

table(datos$TARGET)


train_test_datos <- train_test(datos)

# Acceder a los datos de entrenamiento y prueba
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

resultados<- models(x_train, y_train, x_test, y_test)

# Mostrar las métricas de cada modelo
print(resultados$metricas_rl)
print(resultados$metricas_d)



###############################################################
###############################################################
#Millorar Resultats 1.0
datos <- read.csv("dataset_v2.csv") # Nou dataset amb variables amb una millor correlació amb target 2prov pitjor

set.seed(242)
y <- datos$TARGET
index <- createDataPartition(y, p = 0.8, list = FALSE)
datos_entrenamiento <- datos[index, ]
datos_prueba <- datos[-index, ]
table(datos_entrenamiento$TARGET)

datos <- ovun.sample(TARGET ~ ., data = datos_entrenamiento, method = "over", N = 2.5* sum(datos_entrenamiento$TARGET == 0))$data

table(datos$TARGET)


# Igualar num d'obserbaciones
# Dividir el dataframe en dos subconjunts 
datos_0 <- datos %>% filter(TARGET == 0)
datos_1 <- datos %>% filter(TARGET == 1)

# Valor min
n_min <- min(nrow(datos_0), nrow(datos_1))

# Mostreig aleatori del mateix nombre de obserbaciones 
set.seed(42)  
datos_0_sample <- datos_0 %>% sample_n(n_min) 
datos_1_sample <- datos_1 %>% sample_n(n_min -(n_min*0.2))

# Nou dataset amb files mezcladas
datos_balanceados <- bind_rows(datos_0_sample, datos_1_sample)
datos <- datos_balanceados %>% sample_frac(1)

table(datos$TARGET)

#train_test_datos <- train_test(datos)

# Acceder a los datos de entrenamiento y prueba
x_train <- datos[, !colnames(datos) %in% "TARGET"] 
y_train <- datos$TARGET
x_test <- datos_prueba[, !colnames(datos_prueba) %in% "TARGET"]
y_test <- datos_prueba$TARGET


resultados<- models(x_train, y_train, x_test, y_test)

# Mostrar las métricas de cada modelo
print(resultados$metricas_rl)
print(resultados$metricas_d)
###############################################################
###############################################################


###############################################################
###############################################################
#Millorar Resultats 2.0
datos <- read.csv("dataset_v3.csv") # Nou dataset amb variables amb una millor correlació amb target

# Mirar valance entre classes
table(datos$TARGET) #Es veu com hi ha una gran diferencia entre valors target = 0 i target = 1 fet pel qual fa que classifiqui tot a 0.

# Igualar num d'obserbaciones
# Dividir el dataframe en dos subconjunts 
datos_0 <- datos %>% filter(TARGET == 0)
datos_1 <- datos %>% filter(TARGET == 1)

# Valor min
n_min <- min(nrow(datos_0), nrow(datos_1))

# Mostreig aleatori del mateix nombre de obserbaciones 
set.seed(42)  
datos_0_sample <- datos_0 %>% sample_n(n_min + (n_min*0.2)) 
datos_1_sample <- datos_1 %>% sample_n(n_min)

# Nou dataset amb files mezcladas
datos_balanceados <- bind_rows(datos_0_sample, datos_1_sample)
datos <- datos_balanceados %>% sample_frac(1)

table(datos$TARGET)

train_test_datos <- train_test(datos)

# Acceder a los datos de entrenamiento y prueba
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

resultados<- models(x_train, y_train, x_test, y_test)

# Mostrar las métricas de cada modelo
print(resultados$metricas_rl)
print(resultados$metricas_d)

###############################################################
###############################################################


