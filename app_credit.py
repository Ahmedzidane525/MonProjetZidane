import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns 
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report, roc_curve
#from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Application de ML pour la detection de Fraude par carte de credit")
    st.subheader("Auteur : Ahmed Zidane")

    @st.cache_data
    def load_data():
        return pd.read_excel('creditcard.xlsx')
    
    df = load_data()
    df_sample = df.sample(6)

    if st.sidebar.checkbox("Afficher les données brutes",False):
        st.subheader("Jeu de données 'credit card' : echantillon de 06 obseravtions")
        st.write(df_sample)
    
    seed = 123
    
    # Train/test split
    def split(dataframe):
        y = dataframe['Class']
        X = dataframe.drop('Class', axis=1)
        return train_test_split(X,y, test_size=0.2,random_state=seed,stratify=y)
        # l'argument stratify permet d'equilibrer la classe à predire lors de la division de la base
         

    X_train, X_test, y_train, y_test = split(df)

    classifier = st.sidebar .selectbox("Choisir le classificateur : ", ("","Random Forest","SVM","Logistic Regression"))

    # Analyse de la perf des modeles
    def plot_perf(graphes):
        if 'Confusion Matrix' in graphes:
            st.subheader('Matrice de confusion')
            fig, ax = plt.subplots()
            confusion_matrix(y_test, model.predict(X_test))
            st.pyplot(fig)
            
        '''
        if 'ROC Curve' in graphes:
            st.subheader('Courbe ROC')
            roc_curve(model, X_test, y_test)
            st.pyplot()'''

        if 'Rapport de classification' in graphes:
            st.subheader('Rapport de classification')
            classification_report(y_test, model.predict(X_test))
            

    # RandomForest
    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle : ")
        nb_arbre = st.sidebar.number_input("Choisir le nombre d'arbres dans la forêt : ",
                                               100, max_value=1000, step=10, key='n_estimatos')
        profondeur = st.sidebar.number_input("Choisir la profondeur maximale d'un arbre : ",
                                               1, 20)
        bootstrap = st.sidebar.radio("Echantillion bootstrap lors de la creation d'arbre ? ",(True,False))

        graphe_perf = st.sidebar.multiselect("Choisir un graphique de perf : ", ("Confusion Matrix","ROC Curve","Rapport de classification"))

        if st.sidebar.button("Execution", key= "classifiy"):
            st.subheader("Random Forest Results")
            
            # Instanciation du modele randomForest
            model = RandomForestClassifier(n_estimators= nb_arbre, max_depth= profondeur, bootstrap= bootstrap)
            model.fit(X_train, y_train)
            
            # predictions
            y_pred = model.predict(X_test)
            
            # Metrique de perf du modele
            accuracy = model.score(X_test, y_test)
            precison = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)

            # Affiche les metriques
            st.write("Accuracy : ", round(accuracy,2))
            st.write("Précision : ", precison)
            st.write("Recall : ", recall)

            # Affiche les graphiques de perf
            plot_perf(graphe_perf)



        
        

if __name__ == '__main__':
    main()
