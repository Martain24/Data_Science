import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def barplot_category_counts_percentages(categorical_column: pd.Series, 
                                        dpi=150, figsize=(8, 4)):
    """
    Función para crear un gráfico de barras con dos subplots que muestra 
    el conteo y porcentaje de cada categoría en una variable categórica.

    Parámetros:
        - categorical_column: Pandas Series. La columna categórica que se desea visualizar.
        - dpi: int, opcional (default=150). Resolución del gráfico en puntos por pulgada.
        - figsize: tuple, opcional (default=(8, 4)). Tamaño de la figura del gráfico.

    La función crea dos subplots: el primero muestra el conteo de cada categoría 
    en la variable categórica,y el segundo muestra el porcentaje de cada categoría 
    respecto al total de observaciones.

    Ejemplo de uso:
        barplot_values_percent(data['columna_categorica'], dpi=150, figsize=(10, 6))
    """
    
    # Calcular el conteo de cada categoría y el total de observaciones
    category_counts = categorical_column.value_counts()
    total_observations = category_counts.sum()

    # Calcular porcentages de cada categoría
    porcentages = np.round(category_counts/total_observations * 100, 2)

    # Crear el gráfico con dos subplots
    fig,axes = plt.subplots(ncols=2, dpi=dpi, figsize=figsize)

    # Primer subplot: conteo de cada categoría
    ax_count = axes[0]
    bar_count = sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax_count)
    bar_count.set_title("Count de cada categoría.")
    bar_count.bar_label(bar_count.containers[0])
    bar_count.set_ylim(0, np.max(category_counts)+0.1*np.max(category_counts))

    # Segundo subplot: porcentajes de cada categoría
    ax_percentages = axes[1]
    bar_percentages = sns.barplot(x=porcentages.index, y=porcentages.values, ax=ax_percentages)
    bar_percentages.set_title("Porcentajes de cada categoría")
    bar_percentages.bar_label(bar_percentages.containers[0])
    bar_percentages.set_ylim(0, np.max(porcentages)+0.1*np.max(porcentages))

    # Ajustar espaciado entre subplots
    plt.tight_layout(w_pad=3, h_pad=3)


def calculate_difference_in_means(numeric_column, binary_column):
    """
    Calcula la diferencia en las medias de una variable numérica entre dos categorías
    de una variable binaria.

    Parámetros:
        - numeric_column: Pandas Series. La columna numérica que se desea analizar.
        - binary_column: Pandas Series. La columna binaria que define las dos categorías para comparar las medias.

    Retorna:
        - infor_means_transposed: Pandas DataFrame. Un DataFrame transpuesto que muestra las medias y diferencia.

    La función crea un DataFrame "infor_means" donde se almacenan las medias de la 
    variable numérica para cada categoría de la variable binaria. Luego, se calcula 
    la diferencia en las medias de ambas categorías y se agrega esta diferencia al 
    DataFrame. El resultado final muestra las medias y la diferencia en un formato 
    tabular transpuesto para facilitar su lectura.
    """

    # Crear DataFrame para almacenar los resultados
    infor_means = pd.DataFrame()

    # Obtener las categorías únicas de la variable binaria
    categories = binary_column.unique()

    # Calcular la media de la variable numérica para cada categoría de la variable binaria
    for cat in categories:
        infor_means[f"Mean {cat}"] = [numeric_column[binary_column == cat].mean()]

    # Calcular la diferencia en las medias y agregarla al DataFrame
    infor_means["Diferencia"] = np.abs(infor_means[f"Mean {categories[0]}"] 
                                       - infor_means[f"Mean {categories[1]}"])

    # Establecer el nombre de la columna numérica como índice del DataFrame
    infor_means.index = [numeric_column.name]

    # Transponer el DataFrame para mostrar las medias y diferencia en filas en lugar de columnas
    infor_means_transposed = infor_means.transpose()

    return infor_means_transposed


def plot_means_with_differences(numeric_columns, binary_column, figsize=(10, 20)):
    """
    Grafica las medias de variables numéricas para cada categoría de una variable binaria.

    Parámetros:
        - numeric_columns: Lista de strings. Nombres de columnas numéricas a analizar.
        - binary_column: String. Nombre de la columna binaria con dos categorías.
        - figsize: Tupla (ancho, alto). Tamaño de la figura para los subplots. (Opcional)

    Retorna:
        - None

    La función calcula las medias de las variables numéricas para cada categoría de
    la variable binaria usando "calculate_difference_in_means". Genera un gráfico de 
    barras para cada variable numérica, mostrando las medias para cada categoría y 
    la diferencia entre las medias. El gráfico se organiza en una matriz de subplots 
    con un máximo de 3 columnas, y el número de filas se ajusta según la cantidad de 
    columnas numéricas para visualizar. Se agrega información adicional a las barras, 
    como etiquetas con los valores de las medias y se ajusta el rango del eje y para 
    mejorar la visualización.
    """

    # Calcular número de filas para organizar subplots
    n_rows = len(numeric_columns) // 3 if len(numeric_columns) % 3 == 0 else len(numeric_columns) // 3 + 1

    # Crear la figura y los ejes para los subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=figsize)

    # Contadores para las filas y columnas de los subplots
    current_row = 0
    current_col = 0

    # Iterar sobre cada columna numérica
    for col in numeric_columns:
        # Obtener los datos de las medias para cada categoría de la variable binaria
        infor_means = calculate_difference_in_means(col, binary_column)

        # Seleccionar el eje correspondiente para el subplot actual
        ax = axes[current_row][current_col]

        # Crear el gráfico de barras con las medias para cada categoría
        bar = sns.barplot(x=infor_means.index, y=infor_means.values.reshape(3), ax=ax)

        # Asignar el título del subplot con el nombre de la columna numérica
        bar.set_title(f"{col.name}")

        # Agregar etiquetas a las barras con los valores de las medias
        bar.bar_label(bar.containers[0])

        # Ajustar el rango del eje y para mejorar la visualización
        bar.set_ylim(0, infor_means[col.name].max() + 0.1 * infor_means[col.name].max())

        # Actualizar los contadores de filas y columnas para el próximo subplot
        current_col += 1
        if current_col == 3:
            current_row += 1
            current_col = 0

    # Ajustar los espacios entre los subplots
    plt.tight_layout(w_pad=1.5, h_pad=1.5)

    # No es necesario retornar nada ya que los subplots se muestran directamente
    return None


def bootstrap_mean_difference_test(num_col, binary_col, plot_hist=False):
    """
    Calcula la diferencia real en las medias de una variable numérica entre
    dos categorías de una variable binaria y evalúa su significancia mediante
    el análisis de bootstrap.

    Parámetros:
        - num_col: Pandas Series. Columna numérica a analizar.
        - binary_col: Pandas Series. Columna binaria con las dos categorías para comparar las medias.
        - plot_hist: bool, opcional. Si es True, muestra histograma con las diferencias obtenidas en bootstrap.

    Retorna:
        - real_diff: float. Diferencia real de medias en la muestra original.
        - p_value: float. Valor p calculado a partir del análisis de bootstrap.

    La función calcula la diferencia real en las medias de la variable numérica para
    cada categoría de la variable binaria. Luego, realiza un análisis de bootstrap con
    500 iteraciones, obteniendo muestras bootstrap con reemplazo de ambas columnas.
    Se calcula la diferencia en las medias para cada muestra bootstrap y se almacenan
    en la lista "diferencias". A partir de las diferencias obtenidas, se calcula el valor p
    como la proporción de veces que las diferencias bootstrap son mayores o iguales a la
    diferencia real en la muestra original. Si "plot_hist" es True, muestra un histograma
    con las diferencias bootstrap junto con una línea vertical que representa la diferencia
    real en el gráfico.
    """
    # Diferencia real en medias.
    real_diff = calculate_difference_in_means(num_col, binary_col).loc["Diferencia"][0]  

    # Lista de diferencias bootstrap y número de iteraciones.
    diferencias, epochs = [], 350 

    for epochi in range(epochs):  # Realizar iteraciones bootstrap.

        # Muestras de  bootstraps de la variable numérica y de variable binaria
        sample_num = num_col.sample(len(num_col), replace=True).reset_index(drop=True)  
        sample_bin = binary_col.sample(len(binary_col), replace=True).reset_index(drop=True)

        # Diferencia de las medias en la muestra de bootstrap
        diff = calculate_difference_in_means(sample_num, sample_bin).loc["Diferencia"][0] 

        # Añadir diff a diferencias 
        diferencias.append(diff)

    # Calcular valor p.
    p_value = np.sum(diferencias >= real_diff) / epochs  

    # Mostrar histograma si plot_hist es True.
    if plot_hist:  

        # Histograma de diferencias bootstrap.
        sns.histplot(diferencias)  

        # Línea vertical con diferencia real.
        plt.axvline(real_diff, ls="--", c="r", label=f"Diferencia real = {real_diff:.2f}")  
        plt.legend() 

    return real_diff, p_value  # Retornar diferencia real y valor p.
