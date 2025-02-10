Descripción

Este proyecto tiene como objetivo cumplir con el primer trabajo practico individual en Henry, en la carrera Data Science. El mismo consiste en realizar un MVP( Minimum Viable Product) sosbre un sistema de recomendacion de peliculas en una start-up que provee servicios de agregación de plataformas de streaming.


Instalación y Requisitos


Todo presente en el archivo: requirements.txt. 


Pasos de instalación:

Clonar el repositorio: git clone https://github.com/pabloargprog2/Trabajo_Individual_1_Henry


Instalar las dependencias: pip install -r requirements.txt

Estructura del Proyecto


limpieza y transformacion datos/: esta carpeta contiene los datasets orginales de creditos y peliculas, y todos los cambios que se le aplicaron para llegar a los datasets utilizados.

dataset_actor_director.csv/: dataset utilizado para algunas funciones en FastApi.

dataset_funciones_basicas.csv/: dataset utilizado para algunas funciones en FastApi.

dataset_modelo.csv/: dataset utilizado para el modelo de ML del sistema de recomendacion de peliculas.

dataset_final.csv/: dataset que contiene los tres anteriores.

main.py/: archivo python que contiene todas las funciones, endpoints y el modelo de recomendacion de peliculas presente en la aplicacion desarollada en FastApi y Render.

README.md: Documentación.

Metodología
Se creo una matriz TF-IDF y luego se calculo la similitud del coseno para realizar el sistema de recomendacion. Se tuvo en en cuenta la descripcion general y los generos de la pelicula para calcular la matriz, y previamente para filtar el dataset y que no sea tan grande se buscaron peliculas que tengan un minimo de 100 votos y un percentil 90 de popularidad.
