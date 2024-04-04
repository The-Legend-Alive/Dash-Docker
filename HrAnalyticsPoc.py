# visit http://127.0.0.1:8050/ in your web browser.


# #-------------Retrieving Data----------------

from tkinter.tix import ROW
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier




app = Dash(__name__)

colors = {
    'backgroundfig': '#111111',
    'background':'#EEE8AA',
    'text': '#7FDBFF'
}

# #-------------Retrieving Data----------------

df3 = pd.read_excel (r'New Excell leavers analytics.xlsx')
df=df3[['Business Unit','Left','Employee ID','Start Date','End Date','End Year','Seniority','Age','Starting Level','Current/End Level','Gross salary currently/when they left',
'Difference in Salary','Starting gross salary (2018)','Promotion? (since 2018)','# Trainings (since 2018)','# Personal objectives','# Team objectives','# Check ins','Gender','Current/last client'
,'# of clients since 2018','length of last mission (year)']]

# #-------------Data Quality Limiations----------------


df['Business Unit'] = df['Business Unit'].astype('category').cat.codes
df['Promotion? (since 2018)'] = df['Promotion? (since 2018)'].astype('category').cat.codes
df['Gross salary currently/when they left'].fillna('3000',inplace=True)
df['Starting gross salary (2018)'].fillna('3000',inplace=True)
df['Difference in Salary'].fillna('1',inplace=True)
df['# Trainings (since 2018)'].fillna('1',inplace=True)
df['# Personal objectives'].fillna('0',inplace=True)
df['# Team objectives'].fillna('0',inplace=True)
df['# Check ins'].fillna('0',inplace=True)
df['Current/last client'].fillna('Unkown',inplace=True)
df['# of clients since 2018'].fillna('1',inplace=True)
df['length of last mission (year)'].fillna('1',inplace=True)

# #-------------Data prep----------------
df['# Trainings (since 2018)'] = pd.to_numeric(df['# Trainings (since 2018)'], downcast='float')
df['# Personal objectives'] = pd.to_numeric(df['# Personal objectives'], downcast='float')
df['# Team objectives'] = pd.to_numeric(df['# Team objectives'], downcast='float')
df['# Check ins'] = pd.to_numeric(df['# Check ins'], downcast='float')
df['# of clients since 2018'].mask(df['# of clients since 2018'] == '1', 1, inplace=True)
df['# of clients since 2018'].mask(df['# of clients since 2018'] == '\xa0', 1, inplace=True)
df['length of last mission (year)'].mask(df['length of last mission (year)'] == '1', 1, inplace=True)
df['length of last mission (year)'].mask(df['length of last mission (year)'] == '\xa0', 1, inplace=True)
df['# of clients since 2018'] = pd.to_numeric(df['# of clients since 2018'], downcast='float')
df['length of last mission (year)'] = pd.to_numeric(df['length of last mission (year)'], downcast='float')


#--------------------- Data Analytics Exploration---------------------- 
#Age
fig3=plt.figure(figsize=(15,6))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Left'] == 0, 'Age'], label = 'Active Employee')
sns.kdeplot(df.loc[df['Left'] == 1, 'Age'], label = 'Ex-Employees')
plt.xlim(left=18, right=60)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Age Distribution in Percent by Attrition Status')
plt.legend()
fig3 = mpl_to_plotly(fig3)


#Seniority
fig4=plt.figure(figsize=(9,3))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Left'] == 0, 'Seniority'], label = 'Active Employee          ')
sns.kdeplot(df.loc[df['Left'] == 1, 'Seniority'], label = 'Ex-Employees             ')
plt.xlim(left=0, right=12)
plt.xlabel('Seniority(years)')
plt.ylabel('Density')
plt.title('Seniority Distribution by Attrition Status')
plt.legend()
fig4 = mpl_to_plotly(fig4)




#Level
fig5=plt.figure(figsize=(9,3))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Left'] == 0, 'Current/End Level'], label = 'Active Employee          ')
sns.kdeplot(df.loc[df['Left'] == 1, 'Current/End Level'], label = 'Ex-Employees             ')
plt.xlim(left=0, right=8)
plt.xlabel('Current/End Level')
plt.ylabel('Density')
plt.title('Current/End Level Distribution by Attrition Status')
plt.legend()
fig5 = mpl_to_plotly(fig5)

#info on Difference of Salary for Employees that left

fig7=plt.figure(figsize=(9,3))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Left'] == 0, '# of clients since 2018'], label = 'Active Employee          ')
sns.kdeplot(df.loc[df['Left'] == 1, '# of clients since 2018'], label = 'Ex-Employees             ')
plt.xlim(left=0, right=8)
plt.xlabel('# of clients since 2018')
plt.ylabel('Density')
plt.title('# of clients since 2018 by Attrition Status')
plt.legend()
fig7= mpl_to_plotly(fig7)

fig8=plt.figure(figsize=(9,3))
plt.style.use('seaborn-colorblind')
plt.grid(True, alpha=0.5)
sns.kdeplot(df.loc[df['Left'] == 0, 'length of last mission (year)'], label = 'Active Employee       ')
sns.kdeplot(df.loc[df['Left'] == 1, 'length of last mission (year)'], label = 'Ex-Employees          ')
plt.xlim(left=0, right=8)
plt.xlabel('length of last mission (year)')
plt.ylabel('Density')
plt.title('length of last mission (year) by Attrition Status')
plt.legend()
fig8= mpl_to_plotly(fig8)



#Violin Plot
fig10 = px.violin(df,x='Left', y='Seniority',color="Gender", box=True, points="all",
          hover_data=df.columns,title="Violin plot of Seniority Vs Gender by Attrition Status")

#Boxplot of #Check Ins by left
fig11 = px.box(df, y='# Check ins',x='Left',title="box plot of Check Ins by Attrition Status")


fig9 = go.Figure()
fig9.add_trace(go.Histogram(
    x=df.loc[df['Left'] == 0,'Current/last client'],
    histnorm='percent',
    name='Active Employee', # name used in legend and hover labels
    marker_color='#EB89B5',
    opacity=0.75
))
fig9.add_trace(go.Histogram(
    x=df.loc[df['Left'] == 1,'Current/last client'],
    histnorm='percent',
    name='Ex-Employees',
    marker_color='#330C73',
    opacity=0.75
))

fig9.update_layout(
    title_text='Current/last client of the consultants', # title of plot
    xaxis_title_text='Clients', # xaxis label
    yaxis_title_text='Count', # yaxis label
    width=1400,
    plot_bgcolor=colors['backgroundfig'],
    paper_bgcolor=colors['backgroundfig'],
    font_color=colors['text']
)

df['Gender'] = df['Gender'].astype('category').cat.codes
df['Left'] = df['Left'].astype('category')

# #-------------Machine Learning Modeling----------------
df['Current/last client'] = df['Current/last client'].astype('category').cat.codes
X = pd.get_dummies(df.drop(['Left','Employee ID','Start Date','End Date','End Year'], axis=1)).values
y = df['Left'].cat.codes.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#-------------Parameter Fine Tuning----------------

#-------------KNeighborsClassifier----------------
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier(n_neighbors=6)
param_grid = {'n_neighbors': np.arange(3, 15)}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_

#-------------Building the machine learning models----------------
knn = KNeighborsClassifier(n_neighbors=8)
logreg = LogisticRegression()
tree = DecisionTreeClassifier()
svm_linear= SVC(kernel = 'linear', random_state = 42, probability=True)
svm_nonlinear= SVC(kernel = 'rbf', random_state = 42, probability=True)
GaussianNB = GaussianNB()

classification_models = {
    'KNeighboursClassfier': knn,
    'DecisionTreeClassifier': tree,
    'svm_linear': svm_linear,
    'svm_nonlinear': svm_nonlinear,
    'GaussianNB': GaussianNB
}

regression_models = {
    'LogisticRegression': logreg
}




#------------- Model Scores----------------
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#Let's evaluate each model in turn and provide accuracy and standard deviation scores

acc_results = []
auc_results = []
names = []

col = ['Algorithm', 'Recall Mean', 'Recall STD', 
       'Accuracy Mean', 'Accuracy STD','Precision Mean', 'Precision STD']
df_results = pd.DataFrame(columns=col)
i = 1
#only done for debug graph issue
df_results.loc[0] =  ['Algorithm', 'Recall Mean', 'Recall STD', 
       'Accuracy Mean', 'Accuracy STD','Precision Mean', 'Precision STD']

for name, model in classification_models.items():
    model.fit(X_train, y_train)
    globals()[f"y_pred_{name}"] =model.predict(X_test)
    globals()[f"y_test_proba_{name}"] =model.predict_proba(np.array(X_test))[:,1]

    kfold = model_selection.KFold(
        n_splits=10)  # 10-fold cross-validation

    cv_acc_results = model_selection.cross_val_score(  # accuracy scoring
        model, X_train, y_train, cv=kfold, scoring='accuracy')

    cv_auc_results = model_selection.cross_val_score(  # recall scoring
       model, X_train, y_train, cv=kfold, scoring='recall')

    cv_prec_results = model_selection.cross_val_score(  # precision scoring
       model, X_train, y_train, cv=kfold, scoring='precision')


    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    auc_results.append(cv_prec_results)
    names.append(name)
    

    df_results.loc[i] = [name,
                         round(cv_auc_results.mean()*100, 2),
                         round(cv_auc_results.std()*100, 2),
                         round(cv_acc_results.mean()*100, 2),
                         round(cv_acc_results.std()*100, 2),
                         round(cv_prec_results.mean()*100, 2),
                         round(cv_prec_results.std()*100, 2)
                         ]
    i += 1


model = ExtraTreesClassifier()
model.fit(X, y)

features_scores=pd.DataFrame(model.feature_importances_,(df.drop(['Left','Employee ID','Start Date','End Date','End Year'], axis=1)).columns,columns=['Importance'])

def generate_table(dataframe, max_rows=50):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])



features_scores_sorted=features_scores.sort_values('Importance',ascending=False)
features_names=features_scores_sorted.index
features_scores_sorted['Importance']= pd.Series(["{0:.2f}%".format(val * 100) for val in features_scores_sorted['Importance']], index = features_scores_sorted.index)


fig = go.Figure(data=[go.Table(
    header=dict(values=['Feature','Importance'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[features_names,features_scores_sorted],
               fill_color='lavender',
               align='left'
               ) )    
])


result=svm_linear.predict(X)
EmpAtRisk=[]
i=0
for x in result:
     if x == 1:
        if df.loc[i,'Left'] == 0:
            EmpAtRisk.append(str(df.loc[i,'Employee ID']))
        
     i=i+1



fig2 = go.Figure(data=[go.Table(
    header=dict(values=['Employee # At Risk of Leaving Company'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[EmpAtRisk],
               fill_color='lavender',
               align='left')
              )              
])

colmunspivot=['KNeighboursClassfier','DecisionTreeClassifier','svm_linear','svm_nonlinear','GaussianNB']
new_row=['Algorithm', 'Recall Mean', 'Recall STD', 'Accuracy Mean', 'Accuracy STD','Precision Mean', 'Precision STD']


fig6 = go.Figure(data=[go.Table(
    cells=dict(values=df_results,
               fill_color='lavender',
               align='left'))              
])
fig6.update_layout(
    title_text='Machine Learning Model Results', # title of plot
   
)



#--------------Improving the ML Model using PCA -----------------------------
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


PCA_acc_results = []
PCA_auc_results = []
PCA_names = []

PCA_col = ['Algorithm', 'Recall Mean', 'Recall STD', 
       'Accuracy Mean', 'Accuracy STD','Precision Mean', 'Precision STD']
PCA_df_results = pd.DataFrame(columns=col)
i = 1
#only done for debug graph issue
PCA_df_results.loc[0] =  ['Algorithm', 'Recall Mean', 'Recall STD', 
       'Accuracy Mean', 'Accuracy STD','Precision Mean', 'Precision STD']

for name, model in classification_models.items():
    model.fit(X_train, y_train)
   
    PCA_kfold = model_selection.KFold(
        n_splits=10)  # 10-fold cross-validation

    PCA_cv_acc_results = model_selection.cross_val_score(  # accuracy scoring
        model, X_train, y_train, cv=kfold, scoring='accuracy')

    PCA_cv_auc_results = model_selection.cross_val_score(  # recall scoring
       model, X_train, y_train, cv=kfold, scoring='recall')

    PCA_cv_prec_results = model_selection.cross_val_score(  # precision scoring
       model, X_train, y_train, cv=kfold, scoring='precision')


    PCA_acc_results.append(PCA_cv_acc_results)
    PCA_auc_results.append(PCA_cv_auc_results)
    PCA_auc_results.append(PCA_cv_prec_results)
    names.append(name)
    

    PCA_df_results.loc[i] = [name,
                         round(PCA_cv_auc_results.mean()*100, 2),
                         round(PCA_cv_auc_results.std()*100, 2),
                         round(PCA_cv_acc_results.mean()*100, 2),
                         round(PCA_cv_acc_results.std()*100, 2),
                         round(PCA_cv_prec_results.mean()*100, 2),
                         round(PCA_cv_prec_results.std()*100, 2)
                         ]
    i += 1





fig12 = go.Figure(data=[go.Table(
    cells=dict(values=PCA_df_results,
               fill_color='lavender',
               align='left'))              
])
fig12.update_layout(
    title_text='Results of Applying PCA', # title of plot
    
   
)

#-------------------------MODEL PERFORMANCE SECTION-----------------

#-------------confusion matrix----------------
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred_GaussianNB)

#-------------dataframe from the confusion matrix array----------------
val = np.mat(conmat) 
classnames = list(set(y_train))
df_cm = pd.DataFrame(
        val, index=classnames, columns=classnames, 
    )
df_cm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]  


fig13 = px.imshow(df_cm)
fig13.update_layout(
    title_text='Attrition GaussianNB Model Results', # title of plot
    xaxis_title_text='True label', # xaxis label
    yaxis_title_text='Predicted label', # yaxis label
    plot_bgcolor=colors['backgroundfig'],
    paper_bgcolor=colors['backgroundfig'],
    font_color=colors['text']
)

import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, random_state=0)

model = svm_nonlinear
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_score)

fig14 = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve for SVM NonLinear (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig14.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig14.update_yaxes(scaleanchor="x", scaleratio=1)
fig14.update_xaxes(constrain='domain')

X, y = make_classification(n_samples=500, random_state=0)
model = knn
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]

fpr, tpr, thresholds = roc_curve(y, y_score)

fig15 = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve for GaussianNB (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig15.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig15.update_yaxes(scaleanchor="x", scaleratio=1)
fig15.update_xaxes(constrain='domain')


from dash.dependencies import Input, Output
app.layout = html.Div(className='row', style={'backgroundColor': colors['background']},children=[
    html.H1("-------------------------EMPLOYEE ATTRITION ---------------------(POC dashboard)------------------------------",style={'text-align':'center','font_size': '30px',"color": "green"}),
    html.Div(children=[
     html.H3("-------------------------MODEL SCORE SUMMARY ------------------------------------",style={'text-align':'center','font_size': '15px',"color": "white"}),
         dcc.Graph(id="graph1", figure=fig,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph2", figure=fig2,style={'width': '50%','display': 'inline-block'}),
      html.H3("-------------------------DATA ANYALTICS EXPLORATION ------------------------------------",style={'text-align':'center','font_size': '15px',"color": "white"}),
         dcc.Graph(id="graph3", figure=fig3,style={'width': '100%','display': 'inline-block'}),
         dcc.Graph(id="graph4", figure=fig4,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph5", figure=fig5,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph9", figure=fig9,style={'display': 'inline-block'}),
         dcc.Graph(id="graph7", figure=fig7,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph8", figure=fig8,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph10", figure=fig10,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph11", figure=fig11,style={'width': '50%','display': 'inline-block'}),
     html.H3("-------------------------MACHINE LEARNING MODEL SCORING------------------------------------",style={'text-align':'center','font_size': '15px',"color": "white"}),
         dcc.Graph(id="graph6", figure=fig6,style={'width': '100%','display': 'inline-block'}),
         dcc.Graph(id="graph12",figure=fig12,style={'width': '100%','display': 'inline-block'}),
     html.H3("-------------------------MODEL PERFORMANCE EVALUATION ------------------------------------",style={'text-align':'center','font_size': '15px',"color": "white"}),
         dcc.Graph(id="graph13", figure=fig13,style={'width': '100%','display': 'inline-block'}),
         dcc.Graph(id="graph14", figure=fig14,style={'width': '50%','display': 'inline-block'}),
         dcc.Graph(id="graph15", figure=fig15,style={'width': '50%','display': 'inline-block'}),
       
         
    ])
])



if __name__ == '__main__':
    app.run_server(debug=True)



#Annex further extensions

# #plot between Personal objectives and Trainings
