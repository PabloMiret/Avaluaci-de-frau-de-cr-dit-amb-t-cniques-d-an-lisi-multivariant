# Cargo les dades
datos <- read.csv("dataset.csv")

# Miro si tot està ok
head(datos)

#grafiques d'alguna columna del dataset
ggplot(datos, aes(x = factor(FLAG_OWN_CAR))) +
  geom_bar(fill = "lightblue", color = "black") +
  labs(x = "Propietat de cotxe", y = "Freqüència", 
       title = "Distribució de Propietat de cotxe") +
  scale_x_discrete(labels = c("0" = "No", "1" = "Si")) +
  theme_minimal()

boxplot(datos$YEARS_BIRTH,
        main = "Boxplot de YEARS BIRTH",
        ylab = "EDAT",
        col = "lightgreen",   # Color del boxplot
        border = "darkgreen", # Color del borde
        notch = TRUE)         # Mostrar notch (muesca)



# Llibreries necessaries
library(caret)
library(knitr)
library(MASS)
library(dplyr)
library(ROSE)
library(randomForest)

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

# Funció per entrenar, treure prediccións i mètriques dels MODELS
models <- function(x_train, y_train, x_test, y_test) {
  ## Regressió Logística
  modelo_rl <- glm(y_train ~ ., data = cbind(x_train, y_train), family = binomial)
  predicciones_rl <- ifelse(predict(modelo_rl, x_test, type = "response") > 0.5, 1, 0)
  metricas_rl <- calcular_metriques(y_test, predicciones_rl)
  
  ## Anàlisi Discriminant Lineal
  modelo_d <- lda(y_train ~ ., data = cbind(x_train, y_train))
  predicciones_d <- predict(modelo_d, x_test)$class
  metricas_d <- calcular_metriques(y_test, predicciones_d)
  
  ## Regressió Lineal
  modelo_reg_lineal <- lm(y_train ~ ., data = cbind(x_train, y_train))
  predicciones_reg_lineal  <- ifelse(predict(modelo_reg_lineal, newdata = x_test)  > 0.5, 1, 0)
  metricas_reg_lineal <- calcular_metriques(y_test, predicciones_reg_lineal)
  
  #randomforest
  y_train <- factor(y_train)
  datos_train <- cbind(x_train, y_train)
  modelo_rf <- randomForest(y_train ~ ., data = datos_train,  ntree = 80, method = "class")
  predicciones_rf <- predict(modelo_rf, newdata = x_test, type = "class")
  metricas_rf <- calcular_metriques(y_test, predicciones_rf, pca_guardar = FALSE)
  
  list(
    predicciones_rl = predicciones_rl,
    metricas_rl = metricas_rl,
    predicciones_d = predicciones_d,
    metricas_d = metricas_d,
    predicciones_reg_lineal = predicciones_reg_lineal,
    metricas_reg_lineal = metricas_reg_lineal,
    predicciones_rf = predicciones_rf ,
    metricas_rf = metricas_rf
  )
}


# Funció per a extreure informació de les prediccions
calcular_metriques <- function(y_test, predicciones, pca_guardar = FALSE) {
  
  # Matriu de confusió
  matriz_confusion <- table(y_test, predicciones)
  print(matriz_confusion)
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
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Round i %
  accuracy <- paste0(round(accuracy, 2), "%")
  precision <- paste0(round(precision, 2), "%")
  recall <- paste0(round(recall, 2), "%")
  specificity <- paste0(round(specificity, 2), "%")
  f1_score <- paste0(round(f1_score, 2), "%")
  
  #si és pel pca vull guardar les dades i retornar-les iteració per iteració 
  if (pca_guardar) {
    resulta2 <- list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      specificity = specificity,
      f1_score = f1_score
    )
  }
  if (pca_guardar){
    return(resulta2) 
  }
  # Crear data frame 
  resultados <- data.frame(
    Mètrica = c("Veritables positius (TP)", "Falsos positius (FP)", "Veritables negatius (TN)", "Falsos negatius (FN)",
                "Taxa d'encerts (Accuracy)", "Precisió (Precision)", "Sensibilitat (Recall)", "Especificitat (Specificity)", "F1 Score"),
    Valor = c(TP, FP, TN, FN, accuracy, precision, recall, specificity, f1_score)
  )
  
  # Mostrar els resultats
  kable(resultados, caption = "Resultats Matriu de Confusió i Indicadors de Precisió")
  
}

#separo en train/test
train_test_datos <- train_test(datos)

x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

#entreno models
resultados<- models(x_train, y_train, x_test, y_test) #dona error

#Mirar el motiu del error
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

# Regressió Logística
modelo_glm <- glm(y_train ~ ., data = x_train, family = binomial)
predicciones_glm <- predict(modelo_glm, newdata = x_test, type = "response")
predicciones_glm <- ifelse(predicciones_glm > 0.5, 1, 0) # Ajusta según tus clases
precision_glm <- mean(predicciones_glm == y_test)
print(precision_glm)

# Anàlisi Discriminant 
modelo_lda <- lda(y_train ~ ., data = x_train)
predicciones_lda <- predict(modelo_lda, newdata = x_test)
precision_lda <- mean(predicciones_lda$class == y_test)
print(precision_lda)

# Regressió Lineal
modelo_reg_lineal <- lm(y_train ~ ., data = cbind(x_train, y_train))
predicciones_reg_lineal <- predict(modelo_reg_lineal, newdata = x_test)
predicciones_reg_lineal  <- ifelse(predicciones_reg_lineal  > 0.5, 1, 0)
precision_reg_lineal <- mean(predicciones_reg_lineal == y_test)
print(precision_reg_lineal)

#randomforest
y_train <- factor(y_train)
datos_train <- cbind(x_train, y_train)
modelo_rf <- randomForest(y_train ~ ., data = datos_train,  ntree = 80, method = "class")
predicciones_rf <- predict(modelo_rf, newdata = x_test, type = "class")
precision_rf <- mean(predicciones_rf == y_test)
print(precision_rf)


matriz_confusion1 <- table(y_test,predicciones_glm)
matriz_confusion2 <- table(y_test,predicciones_lda$class)
matriz_confusion3 <- table(y_test,predicciones_reg_lineal)
matriz_confusion4 <- table(y_test,predicciones_rf)

#miro les matrius per saber que passa
print(matriz_confusion1)
print(matriz_confusion2)
print(matriz_confusion3)
print(matriz_confusion4)

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

#miro si està balancejat
table(datos$TARGET)

#separo en train/test
train_test_datos <- train_test(datos)
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

y_train <- factor(y_train)
datos_train <- cbind(x_train, y_train)

error_rate <- data.frame(ntree = integer(), OOB_error = numeric())

# Avaluar el rendimient del model amb diferents números d'arbres
for (ntree in seq(10, 100, by = 10)) {
  set.seed(123)
  rf_model <- randomForest(x = x_train, y = y_train, ntree = ntree, method = 'class')
  oob_error <- rf_model$err.rate[ntree, "OOB"]
  error_rate <- rbind(error_rate, data.frame(ntree = ntree, OOB_error = oob_error))
  print(ntree)
}

# Generar la gráfica
ggplot(error_rate, aes(x = ntree, y = OOB_error)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Error OOB vs Número de Árboles en Random Forest",
       x = "Número de Árboles",
       y = "Error OOB (Out-of-Bag)") +
  theme_minimal()
#identifico que a partir de ntree=80 ni hi ha quasi millora


#separo en train/test
train_test_datos <- train_test(datos)

x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

#entreno models
resultados<- models(x_train, y_train, x_test, y_test)

# Mostrar las mètriques de cada model
print(resultados$metricas_rl)
print(resultados$metricas_d)
print(resultados$metricas_reg_lineal)
print(resultados$metricas_rf)

###############################################################
###############################################################
#Millorar Resultats 1.0
datos <- read.csv("dataset_v2.csv") # Nou dataset amb variables amb una millor correlació amb target 2prov pitjor

#faig el sobre mostreig de dades
set.seed(242)
y <- datos$TARGET
index <- createDataPartition(y, p = 0.8, list = FALSE)
datos_entrenamiento <- datos[index, ]
datos_prueba <- datos[-index, ]
table(datos_entrenamiento$TARGET)
datos <- ovun.sample(TARGET ~ ., data = datos_entrenamiento, method = "over", N = 2.5* sum(datos_entrenamiento$TARGET == 0))$data
#miro quantes dades queden i de quin target son
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

#divisió train/test
x_train <- datos[, !colnames(datos) %in% "TARGET"] 
y_train <- datos$TARGET
x_test <- datos_prueba[, !colnames(datos_prueba) %in% "TARGET"]
y_test <- datos_prueba$TARGET

y_train <- factor(y_train)
datos_train <- cbind(x_train, y_train)

error_rate <- data.frame(ntree = integer(), OOB_error = numeric())

# Avaluar el rendimient del model amb diferents números d'arbres
for (ntree in seq(10, 100, by = 10)) {
  set.seed(123)
  rf_model <- randomForest(x = x_train, y = y_train, ntree = ntree, method = 'class')
  oob_error <- rf_model$err.rate[ntree, "OOB"]
  error_rate <- rbind(error_rate, data.frame(ntree = ntree, OOB_error = oob_error))
  print(ntree)
}

# Generar la gráfica
ggplot(error_rate, aes(x = ntree, y = OOB_error)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Error OOB vs Número de Árboles en Random Forest",
       x = "Número de Árboles",
       y = "Error OOB (Out-of-Bag)") +
  theme_minimal()
#ntrees=80

#torno a fer la divisió perquè anteriorment he modificat el y_test
x_train <- datos[, !colnames(datos) %in% "TARGET"] 
y_train <- datos$TARGET
x_test <- datos_prueba[, !colnames(datos_prueba) %in% "TARGET"]
y_test <- datos_prueba$TARGET

#faig models
resultados<- models(x_train, y_train, x_test, y_test)

# Mostrar las mètriques dels models
print(resultados$metricas_rl)
print(resultados$metricas_d)
print(resultados$metricas_reg_lineal)
print(resultados$metricas_rf)
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

#divisió train/test
train_test_datos <- train_test(datos)
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

y_train <- factor(y_train)
datos_train <- cbind(x_train, y_train)

error_rate <- data.frame(ntree = integer(), OOB_error = numeric())

# Avaluar el rendimient del model amb diferents números d'arbres
for (ntree in seq(10, 100, by = 10)) {
  set.seed(123)
  rf_model <- randomForest(x = x_train, y = y_train, ntree = ntree, method = 'class')
  oob_error <- rf_model$err.rate[ntree, "OOB"]
  error_rate <- rbind(error_rate, data.frame(ntree = ntree, OOB_error = oob_error))
}

# Generar la gráfica
ggplot(error_rate, aes(x = ntree, y = OOB_error)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Error OOB vs Número de Árboles en Random Forest",
       x = "Número de Árboles",
       y = "Error OOB (Out-of-Bag)") +
  theme_minimal()
#ntrees=80

#divisió train/test
train_test_datos <- train_test(datos)

x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

#models
resultados<- models(x_train, y_train, x_test, y_test)

# Mostrar mètriques dels models
print(resultados$metricas_rl)
print(resultados$metricas_d)
print(resultados$metricas_reg_lineal)
print(resultados$metricas_rf)
###############################################################
###############################################################
# PCA
###############################################################
###############################################################

# Gràfica de la variància explicada acumulativa

pca_resultats <- prcomp(datos, center = TRUE, scale. = TRUE)


variancia_explicada <- pca_resultats$sdev^2 / sum(pca_resultats$sdev^2)
variancia_explicada_acumulativa <- cumsum(variancia_explicada)


pca_variancia_df <- data.frame(
  Components = 1:length(variancia_explicada_acumulativa),
  Variancia_Explicada_Acumulativa = variancia_explicada_acumulativa
)


ggplot(pca_variancia_df, aes(x = Components, y = Variancia_Explicada_Acumulativa)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  ggtitle("Variància Explicada Acumulativa per Components Principals") +
  xlab("Nombre de Components Principals") +
  ylab("Variància Explicada Acumulativa") +
  theme_minimal()


#preparació per mirar quin nombre de components és el millor pels models
rl<- list()
d <- list()
rln <- list()
rf <- list()
pca_rl <- list()
pca_d <- list()
pca_rln <- list()
pca_rf <- list()  

#Es mira quin nombre d'arbres és òptim 
train_test_datos <- train_test(pca_datos)
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

y_train <- factor(y_train)
datos_train <- cbind(x_train, y_train)

error_rate <- data.frame(ntree = integer(), OOB_error = numeric())

# Avaluar el rendimient del model amb diferents números d'arbres
for (ntree in seq(10, 100, by = 10)) {
  set.seed(123)
  rf_model <- randomForest(x = x_train, y = y_train, ntree = ntree, method = 'class')
  oob_error <- rf_model$err.rate[ntree, "OOB"]
  error_rate <- rbind(error_rate, data.frame(ntree = ntree, OOB_error = oob_error))
}

# Generar la gráfica
ggplot(error_rate, aes(x = ntree, y = OOB_error)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Error OOB vs Número de Árboles en Random Forest",
       x = "Número de Árboles",
       y = "Error OOB (Out-of-Bag)") +
  theme_minimal()
#ntrees=80


  
#es fa l'execució de 5 components a 35 components fet que ens marca la corba de variança(tot i que ja ens diu que el nostre resultat que voldrem estarà entre els 20 i 30)
for (num_pca in seq(5, 35, by = 1)) {
  
  pca_resultados <- prcomp(datos[, -ncol(datos)], center = TRUE, scale. = TRUE)
  
  varianza_explicada <- pca_resultados$sdev^2 / sum(pca_resultados$sdev^2)
  
  pca_datos <- as.data.frame(pca_resultados$x[, 1:num_pca])#a partir del num_pca es crea les dades 

  pca_datos$TARGET <- datos$TARGET#genera el nou dataset amb la variable target
  
  
  train_test_datos <- train_test(pca_datos) #divisió de dades en train/test
  
  x_train <- train_test_datos$x_train
  y_train <- train_test_datos$y_train
  x_test <- train_test_datos$x_test
  y_test <- train_test_datos$y_test
  

  #s'entrenen i s'avaluen els models i es guarden en les llistes anteriorment creades
  ## Regressió logística
  modelo_rl <- glm(y_train ~ ., data = cbind(x_train, y_train), family = binomial)
  predicciones_rl <- ifelse(predict(modelo_rl, x_test, type = "response") > 0.5, 1, 0)
  metricas_rl <- calcular_metriques(y_test, predicciones_rl, TRUE)
  
  rl <- list(
    n_pca = num_pca, 
    accuracy = metricas_rl$accuracy, 
    precision = metricas_rl$precision, 
    recall = metricas_rl$recall,
    specificity = metricas_rl$specificity,
    f1_score = metricas_rl$f1_score
  )
  pca_rl <- append(pca_rl, list(rl))
  
  ## Anàlisi Discriminant Lineal
  modelo_d <- lda(y_train ~ ., data = cbind(x_train, y_train))
  predicciones_d <- predict(modelo_d, x_test)$class
  metricas_d <- calcular_metriques(y_test, predicciones_d, TRUE)
  
  d <- list(
    n_pca = num_pca, 
    accuracy = metricas_d$accuracy, 
    precision = metricas_d$precision, 
    recall = metricas_d$recall,
    specificity = metricas_d$specificity,
    f1_score = metricas_rl$f1_score
  )
  pca_d <- append(pca_d, list(d))
  
  ## Regressió Lineal
  modelo_reg_lineal <- lm(y_train ~ ., data = cbind(x_train, y_train))
  predicciones_reg_lineal  <- ifelse(predict(modelo_reg_lineal, newdata = x_test)  > 0.5, 1, 0)
  metricas_reg_lineal <- calcular_metriques(y_test, predicciones_reg_lineal, TRUE)
  
  rln <- list(
    n_pca = num_pca, 
    accuracy = metricas_reg_lineal$accuracy, 
    precision = metricas_reg_lineal$precision, 
    recall = metricas_reg_lineal$recall,
    specificity = metricas_reg_lineal$specificity,
    f1_score = metricas_rl$f1_score
  )
  pca_rln <- append(pca_rln, list(rln))
  
  #random forest
  y_train <- factor(y_train)
  datos_train <- cbind(x_train, y_train)
  modelo_rf <- randomForest(y_train ~ ., data = datos_train,  ntree = 80, method = "class")
  predicciones_rf <- predict(modelo_rf, newdata = x_test, type = "class")
  metricas_rf <- calcular_metriques(y_test, predicciones_rf, TRUE)
  
  rf <- list(
    n_pca = num_pca, 
    accuracy = metricas_rf$accuracy, 
    precision = metricas_rf$precision, 
    recall = metricas_rf$recall,
    specificity = metricas_rf$specificity,
    f1_score = metricas_rf$f1_score
  )
  pca_rf <- append(pca_rf, list(rf))
  
  print('EJECUCIÓN')
}

#un cop tinc els resultats els passo a base de dades per poder graficar els resultats
resultats_pca_rl <- do.call(rbind, lapply(pca_rl, as.data.frame))
resultats_pca_d <- do.call(rbind, lapply(pca_d, as.data.frame))
resultats_pca_rln <- do.call(rbind, lapply(pca_rln, as.data.frame))
resultats_pca_rf <- do.call(rbind, lapply(pca_rf, as.data.frame))

df1 <- resultats_pca_rl
df2<- resultats_pca_d
df3 <- resultats_pca_rln
df4 <- resultats_pca_rf

#tracto les dades per poder-les graficar
df1$accuracy <- as.numeric(sub("%", "", df1$accuracy))
df1$precision <- as.numeric(sub("%", "", df1$precision))
df1$recall <- as.numeric(sub("%", "", df1$recall))
df1$specificity <- as.numeric(sub("%", "", df1$specificity))
df1$f1_score <- as.numeric(sub("%", "", df1$f1_score))

df2$accuracy <- as.numeric(sub("%", "", df2$accuracy))
df2$precision <- as.numeric(sub("%", "", df2$precision))
df2$recall <- as.numeric(sub("%", "", df2$recall))
df2$specificity <- as.numeric(sub("%", "", df2$specificity))
df2$f1_score <- as.numeric(sub("%", "", df2$f1_score))

df3$accuracy <- as.numeric(sub("%", "", df3$accuracy))
df3$precision <- as.numeric(sub("%", "", df3$precision))
df3$recall <- as.numeric(sub("%", "", df3$recall))
df3$specificity <- as.numeric(sub("%", "", df3$specificity))
df3$f1_score <- as.numeric(sub("%", "", df3$f1_score))

df4$accuracy <- as.numeric(sub("%", "", df4$accuracy))
df4$precision <- as.numeric(sub("%", "", df4$precision))
df4$recall <- as.numeric(sub("%", "", df4$recall))
df4$specificity <- as.numeric(sub("%", "", df4$specificity))
df4$f1_score <- as.numeric(sub("%", "", df4$f1_score))

#genero les diferents gràfiques
plot(df1$n_pca, df1$accuracy, type = "l", ylim = c(0, 100),xlab = "n_pca", ylab = "accuracy")
lines(df1$n_pca, df1$precision, col = "red")
lines(df1$n_pca, df1$recall, col = "blue")
lines(df1$n_pca, df1$specificity, col = "green")
lines(df1$n_pca, df1$f1_score, col = "yellow")
legend("bottomright", legend = c("Accuracy", "Precision", "Recall", "Specificity", "F1 score"), col = c("black", "red", "blue", "green", "yellow"), lty = 1)

plot(df2$n_pca, df2$accuracy, type = "l", ylim = c(0, 100), xlab = "n_pca", ylab = "accuracy")
lines(df2$n_pca, df2$precision, col = "red")
lines(df2$n_pca, df2$recall, col = "blue")
lines(df2$n_pca, df2$specificity, col = "green")
lines(df1$n_pca, df1$f1_score, col = "yellow")
legend("bottomright", legend = c("Accuracy", "Precision", "Recall", "Specificity","F1 score"), col = c("black", "red", "blue", "green", "yellow"), lty = 1)

plot(df3$n_pca, df3$accuracy, type = "l", ylim = c(0, 100), xlab = "n_pca", ylab = "accuracy")
lines(df3$n_pca, df3$precision, col = "red")
lines(df3$n_pca, df3$recall, col = "blue")
lines(df3$n_pca, df3$specificity, col = "green")
lines(df3$n_pca, df3$f1_score, col = "yellow")
legend("bottomright", legend = c("Accuracy", "Precision", "Recall", "Specificity", "F1 score"), col = c("black", "red", "blue", "green", "yellow"), lty = 1)

plot(df4$n_pca, df4$accuracy, type = "l", ylim = c(0, 100), xlab = "n_pca", ylab = "accuracy")
lines(df4$n_pca, df4$precision, col = "red")
lines(df4$n_pca, df4$recall, col = "blue")
lines(df4$n_pca, df4$specificity, col = "green")
lines(df4$n_pca, df4$f1_score, col = "yellow")
legend("bottomright", legend = c("Accuracy", "Precision", "Recall", "Specificity", "F1 score"), col = c("black", "red", "blue", "green", "yellow"), lty = 1)

#################################################
# Corba ROC
#################################################

#genero pca amb el nombre de components seleccionades =29
pca_resultados <- prcomp(datos[, -ncol(datos)], center = TRUE, scale. = TRUE)
pca_datos <- as.data.frame(pca_resultados$x[, 1:29])
pca_datos$TARGET <- datos$TARGET

#divisió train/test
train_test_datos <- train_test(pca_datos)
x_train <- train_test_datos$x_train
y_train <- train_test_datos$y_train
x_test <- train_test_datos$x_test
y_test <- train_test_datos$y_test

#models
#Regressió logística
modelo_rl <- glm(y_train ~ ., data = cbind(x_train, y_train), family = binomial)
predicciones_rl <- ifelse(predict(modelo_rl, x_test, type = "response") > 0.5, 1, 0)
metricas_rl <- calcular_metriques(y_test, predicciones_rl, TRUE)
print(metricas_rl)

## Anàlisi Discriminant Lineal
modelo_d <- lda(y_train ~ ., data = cbind(x_train, y_train))
predicciones_d <- predict(modelo_d, x_test)$class
metricas_d <- calcular_metriques(y_test, predicciones_d, TRUE)
print(metricas_d)

## Regresió Lineal
modelo_reg_lineal <- lm(y_train ~ ., data = cbind(x_train, y_train))
predicciones_reg_lineal  <- ifelse(predict(modelo_reg_lineal, newdata = x_test)  > 0.5, 1, 0)
metricas_reg_lineal <- calcular_metriques(y_test, predicciones_reg_lineal, TRUE)
print(metricas_reg_lineal)

##randomforest
y_train <- factor(y_train)
datos_train <- cbind(x_train, y_train)
modelo_rf <- randomForest(y_train ~ ., data = datos_train,  ntree = 80, method = "class")
predicciones_rf <- predict(modelo_rf, newdata = x_test, type = "class")
metricas_rf <- calcular_metriques(y_test, predicciones_rf, TRUE)
print(metricas_rf)

# Es fa la corba ROC 
library(pROC)
roc_rl <- roc(y_test, as.numeric(predicciones_rl))
roc_lda <- roc(y_test, as.numeric(predicciones_d))
roc_regl <- roc(y_test, as.numeric(predicciones_reg_lineal))
roc_rf <- roc(y_test, as.numeric(predicciones_rf))

# es grafiquen les corbes 
plot(roc_rl, col = "blue", main = "")
plot(roc_lda, col = "red", add = TRUE)
plot(roc_regl, col = "green",lwd=2,lty=2, add = TRUE)
plot(roc_rf, col = "yellow", add = TRUE)
legend("bottomright", legend = c("Regressió Logística", "LDA", "Regressió Lineal", "Random Forest"), col = c("blue", "red", "green", "yellow"), lwd = 2)

