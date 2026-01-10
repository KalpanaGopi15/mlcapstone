from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
#from sklearn.tree import export_graphviz #plot tree
#from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
##from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = strpredForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()#Anxiety_Level, Self_esteem, Depression_level]
            data_anx= request.POST.get('Anxiety_Level')
            data_self=request.POST.get('Self_esteem')
            data_dep=request.POST.get('Depression_level')
            print (data)
            dataset1=pd.read_csv("StressLevelDataset",index_col=None)
            dicc={'Low':0,'Mediam':1,'High':2}
            filename = 'finalized_model.sav'
            loaded = pickle.load(open(filename, 'rb'))
            def validate_input(selected_features, max_value):
                value = int(input(f"Enter {selected_features} (1–{max_value-1}): "))
                while value <= 0 or value >= max_value:
                    print(f"❌ {selected_features} must be greater than 0 and less than {max_value}")
                    value = int(input(f"Re-enter {selected_features} (1–{max_value-1}): "))
                return value
            Anxiety_Level = validate_input("Anxiety Level", 20)     # 1–19
            Self_esteem = validate_input("Self Esteem", 30)       # 1–29
            Depression_level = validate_input("Depression Level", 30)  # 1–29
            user_input = np.array([[Anxiety_Level, Self_esteem, Depression_level]])
            prediction = loaded.predict(user_input)

            stress_map = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
            print("Predicted Stress Level:", stress_map[prediction[0]])
            data = np.array([data_anx,data_self,data_dep])
            #sc = StandardScaler()
            #data = sc.fit_transform(data.reshape(-1,1))
            out=loaded.predict(data.reshape(1,-1))
# providing an index
            #ser = pd.DataFrame(data, index =['bgr','bu','sc','pcv','wbc'])

            #ss=ser.T.squeeze()
#data_for_prediction = X_test1.iloc[0,:].astype(float)

#data_for_prediction =obj.pca(np.array(data_for_prediction),y_test)
            #obj=ckd()
            ##plt.savefig("static/force_plot.png",dpi=150, bbox_inches='tight')







            return render(request, "succ_msg.html", {'data_anx':data_anx,'data_self':data_self,'data_dep':data_dep,'out':out})


        else:
            return redirect(self.failure_url)
