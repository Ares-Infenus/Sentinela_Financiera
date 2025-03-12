# Cargar librerías
library(caret)
library(glmnet)

# Cargar los datos
X_train <- read.csv("C:/Users/spinz/OneDrive/Documentos/Portafolio oficial/Sentinela_financiera/data/processed/X_train_scaled.csv", header = FALSE)
y_train <- read.csv("C:/Users/spinz/OneDrive/Documentos/Portafolio oficial/Sentinela_financiera/data/processed/y_train.csv", header = FALSE)
X_test  <- read.csv("C:/Users/spinz/OneDrive/Documentos/Portafolio oficial/Sentinela_financiera/data/processed/X_test_scaled.csv", header = FALSE)
y_test  <- read.csv("C:/Users/spinz/OneDrive/Documentos/Portafolio oficial/Sentinela_financiera/data/processed/y_test.csv", header = FALSE)
X_val   <- read.csv("C:/Users/spinz/OneDrive/Documentos/Portafolio oficial/Sentinela_financiera/data/processed/X_val_scaled.csv", header = FALSE)
y_val   <- read.csv("C:/Users/spinz/OneDrive/Documentos/Portafolio oficial/Sentinela_financiera/data/processed/y_val.csv", header = FALSE)

# Asegurar que y es un factor con niveles consistentes (0 y 1)
y_train <- factor(y_train[, 1], levels = c(0, 1))
y_test  <- factor(y_test[, 1], levels = c(0, 1))
y_val   <- factor(y_val[, 1], levels = c(0, 1))

# Asignar nombres de columna a X_train y replicarlos en X_test y X_val
colnames(X_train) <- paste0("V", 1:ncol(X_train))
colnames(X_test)  <- colnames(X_train)
colnames(X_val)   <- colnames(X_train)

# Ajustar la regresión logística usando el conjunto de entrenamiento
df_train <- data.frame(X_train, y_train)
modelo_logistico <- glm(y_train ~ ., data = df_train, family = binomial)

# -------------------------------
# Evaluación en el conjunto de entrenamiento
# -------------------------------
prob_train <- predict(modelo_logistico, newdata = X_train, type = "response")
pred_train <- factor(ifelse(prob_train > 0.5, 1, 0), levels = c(0, 1))
conf_train <- confusionMatrix(pred_train, y_train)
cat("Resultados en el conjunto de entrenamiento:\n")
print(conf_train)

# -------------------------------
# Evaluación en el conjunto de testeo
# -------------------------------
prob_test <- predict(modelo_logistico, newdata = X_test, type = "response")
pred_test <- factor(ifelse(prob_test > 0.5, 1, 0), levels = c(0, 1))
conf_test <- confusionMatrix(pred_test, y_test)
cat("\nResultados en el conjunto de testeo:\n")
print(conf_test)

# -------------------------------
# Evaluación en el conjunto de validación
# -------------------------------
prob_val <- predict(modelo_logistico, newdata = X_val, type = "response")
pred_val <- factor(ifelse(prob_val > 0.5, 1, 0), levels = c(0, 1))
conf_val <- confusionMatrix(pred_val, y_val)
cat("\nResultados en el conjunto de validación:\n")
print(conf_val)

# Función para extraer los resultados del objeto confusionMatrix
extraer_resultados <- function(cm) {
  # Extraer métricas generales (overall)
  overall <- as.data.frame(t(cm$overall))
  
  # Extraer métricas por clase (byClass)
  byClass <- as.data.frame(t(cm$byClass))
  
  # Combinar ambas tablas de métricas en una sola fila
  res <- cbind(overall, byClass)
  return(res)
}

# Extraer resultados para cada conjunto
resultados_train <- extraer_resultados(conf_train)
resultados_test  <- extraer_resultados(conf_test)
resultados_val   <- extraer_resultados(conf_val)

# Agregar una columna que identifique el conjunto
resultados_train$Conjunto <- "Train"
resultados_test$Conjunto  <- "Test"
resultados_val$Conjunto   <- "Validation"

# Unir todos los resultados en un solo dataframe
df_resultados <- rbind(resultados_train, resultados_test, resultados_val)

# Opcional: Reordenar las columnas para que la columna "Conjunto" quede al inicio
df_resultados <- df_resultados[, c(ncol(df_resultados), 1:(ncol(df_resultados)-1))]

# Mostrar el dataframe con todos los detalles
print(df_resultados)

# Si deseas exportar el dataframe a un archivo CSV:
write.csv(df_resultados, file = "resultados_modelo.csv", row.names = FALSE)
