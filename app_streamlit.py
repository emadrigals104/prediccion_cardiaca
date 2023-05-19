# Importamos las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Hoja1')
    writer.book.save(output)
    output.seek(0)
    return output

# Configuramos la página de Streamlit
st.set_page_config(page_title="App de predicción",
                   page_icon='https://cdn-icons-png.flaticon.com/512/5935/5935638.png',  
                   layout='wide', 
                   initial_sidebar_state='expanded')



# Definimos el título y la descripción de la aplicación

st.title("App de predicción de enfermedades cardíacas")
st.markdown("""Esta aplicación predice si tienes una enfermedad cardíaca basándose en tus datos ingresados.""")
st.markdown("""---""")

# Cargamos y mostramos el logo en la barra lateral
logo = "imagen.png"
st.sidebar.image(logo, width=150)
# Añadimos un encabezado para la sección de datos del usuario en la barra lateral
st.sidebar.header('Datos ingresados por el usuario')

# Permitimos al usuario cargar un archivo CSV o ingresar datos manualmente
uploaded_file = st.sidebar.file_uploader("Cargue su archivo CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        # Creamos controles deslizantes y cuadros de selección para que el usuario ingrese datos
        sbp = st.sidebar.slider('Presión Arterial Sistólica', 101, 218, 150)
        Tabaco = st.sidebar.slider('Tabaco Acumulado (kg)', 0.00, 31.20, 2.00)
        ldl = st.sidebar.slider('Colesterol de Lipoproteínas de Baja Densidad', 0.98, 15.33, 4.34)
        Adiposidad = st.sidebar.slider('Adiposidad', 6.74, 42.49, 26.12)
        Familia = st.sidebar.selectbox('Antecedentes Familiares de Enfermedad Cardíaca', ('Presente', 'Ausente'))
        Tipo = st.sidebar.slider('Tipo', 13, 78, 53)
        Obesidad = st.sidebar.slider('Obesidad', 14.70, 46.58, 25.80)
        Alcohol = st.sidebar.slider('Consumo Actual de Alcohol ', 0.00, 147.19, 7.51)
        Edad = st.sidebar.slider('Edad', 15, 64, 45)
        
        # Creamos un diccionario con los datos ingresados por el usuario
        data = {'sbp': sbp,
                'Tabaco': Tabaco,
                'ldl': ldl,
                'Adiposidad': Adiposidad,
                'Familia': Familia,
                'Tipo': Tipo,
                'Obesidad': Obesidad,
                'Alcohol': Alcohol,
                'Edad': Edad
                }
        
        # Convertimos el diccionario en un DataFrame
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

st.subheader('Datos ingresados por el usuario')
#Mostramos los datos ingresados por el usuario en la página principal

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders_ = {col: LabelEncoder().fit(X[col]) for col in X.columns}
        return self

    def transform(self, X):
        return X.apply(lambda col: self.encoders_[col.name].transform(col))
# Separamos las características (X) y la variable objetivo (y)
X = input_df

#Cargamos el modelo de clasificación previamente entrenado
with open('modelo_regresion_logistica.pkl', 'rb') as f:
    modelo = pickle.load(f)

if uploaded_file is not None:
    if st.checkbox("Mostrar Datos Ingresados"):
                st.write(input_df)
    prediccion_modelo = modelo.predict(input_df)
    prediction_proba_modelo = modelo.predict_proba(input_df)
    # Crear un DataFrame con la predicción de chd y transformar los valores 0 y 1 a "No" y "Si"
    df_chd = pd.DataFrame(prediccion_modelo,columns=["chd"])
    df_chd = df_chd.applymap(lambda x: "No" if x == 0 else "Si")
            
            # Crear un DataFrame con la probabilidad máxima de las predicciones y transformar los valores 0 y 1 a "No" y "Si"    
    df_chd = pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["chd"])
    df_chd = df_chd.applymap(lambda x: "No" if x == 0 else "Si")
            
            # Calcular las probabilidades correspondientes a la predicción de chd
    probabilidades = np.where(df_chd["chd"] == "No", prediction_proba_modelo[:, 0], prediction_proba_modelo[:, 1])
            
            # Crear un DataFrame con las predicciones de chd y las probabilidades correspondientes
    df_resultado = pd.DataFrame({"chd": df_chd["chd"], "Probabilidad": probabilidades})
            
            # Unir el DataFrame de los datos ingresados con el DataFrame del resultado de las predicciones
    df_unido = pd.concat([input_df,df_resultado],axis=1)
            
            # Convertir el DataFrame unido a formato CSV
    csv_data = df_unido.to_csv(index=False)
    st.divider()
    st.markdown("### Datos con la Predicción")
    mostrar_prediccion = st.checkbox("Mostrar Predicción Final")
            
            # Mostrar los datos con la predicción
    if mostrar_prediccion:
        st.write(df_unido)
    st.divider()
    st.markdown("### Gráficos de la predicción")
    col_gra1, col_gra2 = st.columns((5,5))
    valores_categoricas = df_unido["chd"].value_counts()
    #Colores de los gráficos
    colorscale = px.colors.sequential.YlOrBr
    num_categorias = len(valores_categoricas.index)
    step_size = int(len(colorscale) / num_categorias)
    colores = colorscale[::step_size]
            
            
    with col_gra1:
                
        #gráfico de barras
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=valores_categoricas.index,
            y=valores_categoricas,
            text=valores_categoricas,
            textposition='auto',
            hovertemplate='%{x}: <br>valores_categoricas: %{y}',
            marker=dict(color=colores)

        ))

        fig.update_layout(
            title=f"Gráfico de Barras - Predicción",
            font=dict(size=12),
            width=500,
            height=500
        )

        st.plotly_chart(fig)

            
    with col_gra2:
        #gráfico de pastel
        fig = go.Figure()
                
        fig.add_trace(go.Pie(
            labels=valores_categoricas.index,
            values=valores_categoricas.values,
            textinfo='label+percent',
            insidetextorientation='radial',
            hovertemplate='%{label}: <br>valores_categoricas: %{value} <br>Porcentaje: %{percent}',
            showlegend=True,
            marker=dict(colors=colores)
                    
        ))

        fig.update_layout(
            title=f"Gráfico Circular - Predicción",
            font=dict(size=15),
            width=500,
            height=500
        )

        st.plotly_chart(fig)
            

            

    st.divider()
            
    # Descargar el archivo con las predicciones en formato Excel o CSV
    st.markdown("### Descargar el Archivo Predecido en Diferentes Formatos")        
    st.download_button(
        label=":file_folder: Descargar El Archivo Excel",
        data=to_excel(df_unido),
        file_name='Reporte.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    st.download_button(
        label=":file_folder: Descargar El Archivo CSV",
        data=csv_data,
        file_name="reporte.csv",
        mime="text/csv"
        )

else:
    st.write(input_df)
    #Aplicamos el modelo para realizar predicciones en base a los datos ingresados
    prediction = modelo.predict(input_df)
    prediction_proba = modelo.predict_proba(input_df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Predicción')
        st.write(prediction)

    with col2:
        st.subheader('Probabilidad de predicción')
        st.write(prediction_proba)
            
    if prediction ==0:
            st.subheader('La persona no tiene problemas Cardíacos')
    else:
            st.subheader('La persona tiene problemas Cardíacos')
    st.markdown("""---""")









            
