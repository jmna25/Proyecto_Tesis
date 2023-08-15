
import pandas as pd
import numpy as np
from glob import glob
import os
import variables_nombres as vn
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
#import datawig
#from datawig.utils import random_split
#from datawig import SimpleImputer






####################################
# ÍNDICE
# 1. IMPORTACIÓN Y PREPROCESAMIENTO
# 2. FILTROS: SELECCIÓN DE VARIABLES
# 3. MÉTODOS DE IMPUTACIÓN
# 4. TRANSFORMACIONES
####################################
       

    
    
    
    
# 1. IMPORTACIÓN Y PREPROCESAMIENTO
###################################
    
def contar_variables( datasets ):
    '''
    Propósito:
        - Determinar cuántas variables pertenecen a SIAF y cuantas a RENAMU
    Inputs:
        - Lista de bases indexada
    Outputs:
        - Oraciones por cada una de las bases de la lista de bases,
          que especifican el número de variables de Renamu y SIAF
    Especificaciones:
        - Las bases deben estar nombradas previamente. Ejemplo:
          bases[0].name = 'base0'
    '''
    renamu_vars = []
    siaf_vars = []
    for dataset in datasets:
        renamu_vars = [ var for var in dataset.columns if var in vn.renamu_variables ]
        renamu_count = len( renamu_vars )
        siaf_vars = [ var for var in dataset.columns if var in vn.siaf_variables ]
        siaf_count = len( siaf_vars )
        politica_vars = [ var for var in dataset.columns if var in vn.politica_variables ]
        politica_count = len( politica_vars )
        
        print(f'{ dataset.name }: Variables de Renamu: { renamu_count }; Variables de SIAF: { siaf_count }; variables políticas { politica_count }')
        
        
def listar_variables( dataset, df = "siaf" ):
    '''
    Propósito:
        - Generar una lista de variables pertenecientes, o bien a SIAF o bien
          a Renamu, dada la base usada.
    Imputs:
        - dataset: dataset cuyas variables pertenecientes a SIAF o a Renamu quiere
          conocerse. A diferencia de otras funciones, se trata de una dataset, no
          de una lista de variables.
        - df: booleano. Si es siaf, se genera la lista de variables pertenecientes
          a SIAF. Si es renamu, se genera la lista de variables pertenecientes a
          Renamu. Si es politica, se genera la lista de variables pertenecientes a
          politica.
    Output:
        - Lista de variables de la dataset usada, que pertenecen a SIAF/Renamu, según
          lo espeficado en el segundo argumento de la función.
    '''    
    if df == "siaf": 
        variables = [ var for var in dataset.columns if var in vn.siaf_variables ]
    if df == "renamu":
        variables = [ var for var in dataset.columns if var in vn.renamu_variables ]
    if df == "politica":
        variables = [ var for var in dataset.columns if var in vn.politica_variables ]
        
    return variables    
    
    
def importar_bases( ruta ):
    '''
    Propósito:
        - Abre multiples datasets al mismo tiempo
    Inputs:
        - La ruta de acceso donde se encuentra las datasets
    Outputs:
        - La función crea una lista de bases indexada.
          Para acceder a acada una de las bases solo hay que 
          llamar a la lista con el índice correspondiente.
          Por ejemplo: lista_bases[0]
    Especificaciones:
        - Las datasets se encuentran en formato .dta
        - Si que quiere abrir todas las datasets de una determinada
          carpeta, se adiciona la siguiente expresión a la ruta de
          acceso de la carpeta "*.dta". 
    '''
    ruta_de_acceso = glob( ruta )
    lista_bases = []
    for i, base in enumerate( ruta_de_acceso ):
        dataframes = pd.read_stata( base,
                                    convert_categoricals = False,
                                    convert_dates = False,
                                    convert_missing = True )
        lista_bases.append( dataframes )
        
    return lista_bases



def determinar_dimensiones( dataframes ):
    '''
    Propósito:
        - Determinar las dimensiones de cada una de las bases
          de datos pertenecientes a una lista de bases
    '''
    for dataframe in dataframes:
        
        print( dataframe.name, dataframe.shape )
        
                 
def generar_tablas_descriptivas( bases, filename, ruta_output ):
    
    '''
    Propósito:
        - Generar tablas descriptivas para cada una de las bases de datos
          de un path brindado.
    '''
    
    export_path = os.path.join( ruta_output, f'{ filename }.xlsx' )
    
    writer = pd.ExcelWriter( export_path, engine =  'xlsxwriter' )
    
    for i, base in enumerate( bases ):
        n_obs = bases[ i ].shape[ 0 ]
        desc = bases[ i ].describe( include = 'all' ).T
        desc[ 'missing_values_count' ] = n_obs - desc[ 'count' ]
        desc[ 'missing_values_percentage' ] = round( desc[ 'missing_values_count' ] / n_obs, 2 )
        desc = desc.sort_values( by = "missing_values_percentage",
                                 ascending = False )

        fuente = []
        for index, row in desc.iterrows():
            if index in vn.siaf_variables: fuente.append( "SIAF" )
            if index in vn.renamu_variables: fuente.append( "Renamu" )
            if index not in vn.siaf_variables and index not in vn.renamu_variables: fuente.append( "Other" )
        desc[ 'fuente' ] = fuente
        
        desc = desc[ [ 'missing_values_percentage', 'missing_values_count', 'fuente', 
                       'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max' ] ]
        desc = desc.round( 2 )  
        
        desc.to_excel( writer, sheet_name = f'caso_{ i + 1 }' )  
        
    writer.save()  
        
        

# 2. FILTROS: SELECCIÓN DE VARIABLES
####################################
        
def filtro_missings( dataframe, umbral ):
    '''
    Propósito: 
        - Aplicar un filtro de variables basado en el porcentaje
          de valores perdidos de cada columna. Al aplicar el filtro,
          permanecen las columnas con un porcentaje de valores perdidos
          menor al umbral asignado.
    Inputs:
        - dataframe: bases de datos
        - umbral: porcentaje de valores perdidos
    Output:
        - Base de datos filtrada de variables con valores perdidos igual o mayor
          al umbral espeficiado.
    '''
    df_modificada = dataframe.copy()
    df_modificada = df_modificada.loc[ : , df_modificada.isnull().mean() < umbral ]
    
    return df_modificada
    
    
def filtro_variabilidad( dataset, umbral, deps ):
    '''
    Propósito:
        - Aplicar un filtro de columnas (variables) basado en la variabilidad
          de las columnas. Al aplicar el filtro, permanecen las columnas con
          un porcentaje de variabilidad mayor al umbral asignado.
    Inputs:
        - dataset: bases de datos
        - umbral: umbral de variabilidad
        - deps: variables que no serán filtradas
    Output:
        - Base de datos filtrada de columnas con una variabilidad mayor o igual al
          umbral asignado.
    Especificaciones:
        - Esta función utiliza el algoritmo: sklearn.feature_selection.VarianceThreshold
        - La implementación del algoritmo se basa en el siguiente texto: 
          https://medium.com/nerd-for-tech/removing-constant-variables-feature-selection-463e2d6a30d9
    '''
    df_modificada = dataset.copy()
    
    var_thr = VarianceThreshold( threshold = umbral )
    var_thr.fit( df_modificada )
    columnas_a_quedarse = var_thr.get_support()
    columnas_a_borrar = [ col for col in df_modificada.columns if col not in df_modificada.columns[ columnas_a_quedarse ] and col not in deps ]
    df_modificada = df_modificada.drop( columnas_a_borrar, axis = 1 )
    
    return df_modificada


def filtro_correlacion( dataset, dep, umbral ):
    '''
    Propósito:
        - Aplicar un filtro de columnas (variables) basado en la correlación
          de columnas. Si la correlación entre dos variables predictoras supera
          el umbral, se elimina la variable que tiene menor correlación con la 
          variable dependiente.
    Inputs:
        - dataset: base de datos
        - dep: variable dependiente. Usualmente se llama así: 
          dataset['dep'].reset_index()
          umbral: umbral de correlación
    Output:
        - Base de datos en la que se eliminaron las variables descritas en el
          Propósito.
    Especificaciones:
        - A diferencia de las anteriores funciones, en este caso la función
          trabaja sobre una base de datos. El filtro debe repetirse por cada
          base de datos y por cada variable dependiente.
        - La función "corr" (calcula correlaciones) ignora por default aquellas
          filas con valores perdidos. 
    Créditos:
        - Basado en la función "low_corr_vars" creada por el profesor A. Cozzubo
          en: https://github.com/acozzubo/ML_Public_Policy
    '''
    df_modificada = dataset.copy()
    
    candidatas = df_modificada.columns.values.tolist()
    for var1 in df_modificada.columns.values.tolist():
        for var2 in df_modificada.columns.values.tolist():
            if var1 != var2 and var1 in candidatas and var2 in candidatas:
                if abs( df_modificada[ var1 ].corr( df_modificada[ var2 ] ) ) > umbral:
                    if abs( df_modificada[ var1 ].corr( dep ) ) <= abs( df_modificada[ var2 ].corr( dep ) ):
                        candidatas.remove( var1 )
                    else:
                        candidatas.remove( var2 )
    df_modificada = df_modificada[ candidatas ]
    
    return df_modificada
        

# 3. MÉTODOS DE IMPUTACIÓN
##########################
        
def imputar_i( dataframe, vars, val, dummy = False ):
    '''  
    Propósito:
       -  Imputar con un valor asignado por el usuario una lista determinada
          de variables, con la posibilidad de generar variables dummies de
          control por cada variable imputada.
    Inputs:
        - dataframe: dataframe
        - vars: lista de variables a imputar
        - val: valor que se usará para imputar
        - dummy: booleano. True en caso se desee generar una variable dummy 
          de control por cada variable imputada. 
    Output:
        - Dataframe imputada en las columnas seleccionadas, con el valor especificado, 
          y con/sin variables dummy de control.
    Créditos:
        - Basado en la función "mv_treat" creada por el profesor A. Cozzubo
          en: https://github.com/acozzubo/ML_Public_Policy
    '''
    
    df_modificada = dataframe.copy()
    
    for var in df_modificada.columns:
        if var in vars: 
            if df_modificada[ var ].isnull().sum() > 0:
                df_modificada.loc[ df_modificada[ var ].isnull(), var ] = val
                if dummy == True:
                    df_modificada[ 'im_' + var ] = 0
                    df_modificada.loc[ df_modificada[ var ].isnull(), 'im_' + var ] = 1
                
    return df_modificada


def imputar_ii( dataframe, variables, num = True, dummy = True ):
    '''
    Propósito:
        - Imputar con media o moda las bases de datos de la lista
          de bases brindada, con la posibilidad de generar variables
          dummies de control por cada variable imputada.
    Inputs:
        - dataframe: bases de datos
        - variables: lista de variables sobre las que se realizará la imputación
        - num: booleano. Indica si la lista de variables a ser imputada es 
          numérica (se imputa con media) o categórica (se imputa con moda)
        - dummy: booleano. Si es True, se generan variables de control 
          por cada una de las variables imputadas.
        
    Output:
        - Lista de bases de datos imputada en el conjunto de variables 
          seleccionadas, sean provenientes de Renamu o SIAF, con el valor 
          especificado, y con/sin variables dummy de control.
    '''
    df_modificada = dataframe.copy()
    
    for var in df_modificada.columns:
        if df_modificada[ var ].isna().sum() > 0:
            if num:
                if var in variables:
                    media = df_modificada[ var ].mean()
                    df_modificada.loc[ :, var ] = df_modificada[ var ].fillna( media )                    
                    if dummy:
                        df_modificada[ 'im_' + var ] = 0
                        df_modificada.loc[ df_modificada[ var ].isnull(), 'im_' + var ] = 1
            else:
                if var in variables:
                    moda = df_modificada[ var ].mode()[ 0 ]
                    df_modificada.loc[ :, var ] = df_modificada[ var ].fillna( moda )
                    if dummy:
                        df_modificada[ 'im_' + var ] = 0
                        df_modificada.loc[ df_modificada[ var ].isnull(), 'im_' + var ] = 1
                        
    return df_modificada


# def imputar_iii(dataframe):
    # '''
    # Propósito:
        # - Imputar valores perdidos usando la libreria datawig
    # Inputs:
        # - dataframe: base de datos que se utilizará
    # Output:
        # - Base de datos con todas las columnas que tienen valores perdidos
        #    imputadas. Las columnas imputadas "x" ahora se llaman "x_imputed".
    # Especificaciones:
        # - La libreria datawig solo puede ser ejecutada con la versión de
          # Python 3.7. Por ello, se sugiere crear un nuevoenvironment como 
          # se explica en: https://towardsdatascience.com/imputation-of-missing-data-in-tables-with-datawig-2d7ab327ece2
        # - El algoritmo datawig funciona del siguiente modo: se divide los datos
          # en training y test set. Se determina una columna a ser imputada con base
          # a otras columnas escogidas por el investigador. Se genera un modelo para
          # la columna específica, y se predice sobre el test set. Si la columna a ser
          # imputada es "x", se genera una nueva columna denominada "x_imputed".
        # - La presente función opera del siguiente modo: se divide la data en
          # training y test set. Se genera un modelo imputador por cada columna que presente un número
          # de missings mayor a 1. Posteriormente se predice cada columna a 
          # ser imputada en el test set. Si hay n columnas a ser imputadas, se generan
          # n bases de datos. Luego, se hace una concatenación horizontal a
          # todas las bases creadas en el test set. Se repite el mismo procedimiento
          # sobre el training set. Posteriormente se realiza una concatenación vertical
          # para unificar las observaciones pertenecientes al training set y el test set.
          # Finalmente, se eliminan las variables duplicadas "x" 
          # (dado que las columnas "x_imputed" las reemplazan").
          # '''
    # df_train, df_test = random_split(dataframe, split_ratios=[ 0.9, 0.1 ])
    # input_cols = df_train.columns.values.tolist()
    # append_test = []
    # append_train = []
    # vars = [ var for var in df_train.columns if df_train[var].isnull().sum() > 0 ]
    # for var in vars:
        # imputer = SimpleImputer( input_columns = input_cols,
                                       # output_column = var,
                                       # output_path = 'imputar_iii' )
        # imputer.fit( train_df = df_train )
        # predicciones_test = imputer.predict( df_test )
        # predicciones_train = imputer.predict( df_train )
        # append_test.append( predicciones_test )
        # append_train.append( predicciones_train )
    # pred_test = pd.concat( append_test, axis = 1 )
    # pred_train = pd.concat( append_train, axis = 1 )
    # pred = pd.concat( [ pred_test, pred_train ], axis = 0 )
    # pred = pred.loc[ : , ~pred.columns.duplicated() ].copy()
    # pred = pred.drop( pred[ vars ], axis = 1 )
    # return pred


# 4. TRANSFORMACIONES
#####################
    
    
def imputar_outliers( dataframe, vars, percentil_superior ):
    '''
    Propósito:
        - Imputar los valores del "percentil_superior" asignado
          con 1 - "percentil_superior". Ejemplo: imputar los 
          valores del percentil superior 1% con el percentil 99%. 
    Inputs:
        - dataframes bases de datos
        - vars: variables que se busca imputar
        - percentil_superior: percentil asignado. Los valores que 
          sobrepasen este percentil serán imputados con 1 - "percentil_superior".
    Output:
        - Base de datos con aquellos valores del "percentil_superior" 
          asignado imputados con 1 - "percentil_superior". 
    Créditos:
        - Basado en la función "outliers_imputation" creada por el 
          profesor A. Cozzubo en: https://github.com/acozzubo/ML_Public_Policy
    '''

    df_modificada = dataframe.copy()
    
    for var in vars:
        if var in df_modificada.columns:
            perc = df_modificada[ var ].quantile( 1 - percentil_superior )
            df_modificada.loc[ df_modificada[ var ] > perc, var ] = perc
            
    return df_modificada



def dividir_variables_negativas( dataset, val, deps ):
    
    '''
    Propósito: 
        - Dividir todas aquellas variables negativas en la dataset
          por el valor asignado en la función
    Inputs:
        - dataset: base de datos
        - val: valor por el que las columnas negativas de la base de datos
          será reemplazado
        - deps: variables que no serán modificadas
    Output:
        - Base de datos con todas sus columnas negativas divididas entre
          el valor especificado
    Especificaciones:
        - Valor usado en la investigación: 1_000_000
    '''

    df_modificada = dataset.copy()
    
    variables_negativas = df_modificada.columns[ ( df_modificada < 0 ).any() ].tolist()   
    variables_negativas = [var for var in variables_negativas if var not in deps ]
    for var in variables_negativas:
        df_modificada[ var ] = df_modificada[ var ] / val  
        
    return df_modificada  



def transformacion_log( dataset, vars, deps ):
    
    '''
    Propósito:
        - Realizar una transformación logarítmica a las variables
          provenientes de SIAF y a las variables dependientes numéricas.
    Inputs:
        - dataset: base de datos
        - vars: variables a ser transformadas logaritmicamente
        - deps: variables que no serán modificadas
    Output:
        - Base de datos con valores pertenecientes a variables
          de SIAF transformadas logarítmicamente.
    Especificaciones:
        - Para evitar que los valores transformados logarítmicamente tomen
          valores negativos, se suma 1 a todos los valores de las variables SIAF.
    '''
    
    df_modificada = dataset.copy()    

    negative_vars = df_modificada.columns[ ( df_modificada < 0 ).any() ].tolist()
    log_vars      = [ var for var in vars if var not in negative_vars and var not in deps ]
    
    for var in log_vars:
        if var in df_modificada.columns: 
            df_modificada[ var ] = df_modificada[ var ].astype( int )
            logaritmo            = np.log( df_modificada[ var ] + 1 )
            df_modificada[ var ] = logaritmo
        
    return df_modificada               


def drop_missing_rows( dataset, missing_vars ):
    
    '''
    Propósito:
        - Eliminar las filas con valores perdidos de todas aquellas columnas con valores
          perdidos en una dataframe
    Input:
        - dataset: Base de datos
    Output:
        - Base de datos modificada con las filas con valores perdidos eliminadas
    '''
    
    df_modificada = dataset.copy()
    
    missing_cols = df_modificada[ missing_vars ].isnull().any( axis = 1 )
    data_final   = df_modificada[ ~missing_cols ]
    
    return data_final