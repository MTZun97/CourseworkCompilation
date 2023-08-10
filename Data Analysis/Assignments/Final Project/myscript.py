# Importing Libraries
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Initiating the class statement for the module
class hypothesis_ml:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.nlr = None
        self.no_corr = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    #Importing and cleaning data    
    def load(self):
        # Load CSV data using PandaS
        self.df      = pd.read_csv(self.filepath, delimiter=',', parse_dates = True , index_col = 0 )
        
        # Check null values
        for c in self.df.isna().sum():
            assert c == 0

        # drop random variables in the data sets
        self.df.drop(columns = ["rv1", "rv2"], inplace = True)
        
        # Calulate the total load (appliances + lights) and drop columns = ["Appliances", "lights"]
        self.df["total"] = self.df["Appliances"] + self.df["lights"]
        self.df.drop(columns = ["Appliances", "lights"], inplace = True)
        return self.df

        return self.df
    
    #Plotting the pivot table for average energy consumption for each day
    def consumption_plot(self):
        data = self.df.copy()
        
        # Prepare the datetime data into day of the week
        data['date'] = pd.to_datetime(data.index)
        data["day_of_week"] = pd.DatetimeIndex(data["date"]).dayofweek
        data['month'] = data['date'].dt.month
        day_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
        data['day_of_week'] = data['day_of_week'].map(day_map)
        
        # Create the pivot table
        pivot = pd.pivot_table(data, index="day_of_week", columns="month", values="total", aggfunc='mean')
        pivot = pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Change the column names to the names of the months
        pivot.columns = ['January', 'February', 'March', 'April', 'May']

        # Create heatmap to visualize the pivot table
        plt.figure(figsize = (12,12))
        sns.heatmap(pivot, cmap="YlGnBu", annot = True).set_title(label='Figure 1: Average Consumption of Each day', fontdict={'fontsize': 18})
        plt.show()
    
    # Resampling data to 1 hour
    def resample(self):
        self.df = self.df.resample("H").mean()
        
    # Distribution of each variables
    def distribution(self):
        fig = plt.figure(figsize = (16, 16))
        fig.suptitle("Figure 2:Distribution of the independent Variables", fontsize = 18)
        plt.subplots_adjust(wspace = 0.3, hspace= 0.5)
        for idx, col in zip(range(len(self.df.drop(columns = ["total"]).columns)), self.df.drop(columns = ["total"]).columns):
            plt.subplot(8,3, idx+1)
            plt.boxplot(self.df[col], vert = 0)
            plt.title(col)
        plt.show()

    # First hypothesis test: ANOVA Test
    def ANOVA(self):
        f_value_T, p_value_T = stats.f_oneway(self.df['T1'], self.df['T2'], self.df['T3'], self.df['T4'], self.df['T5'], self.df['T6'], self.df['T7'], self.df['T8'], self.df['T_out'])
        print("ANOVA Test for Temperature")
        if p_value_T < 0.05:
            print("P-value:", p_value_T)
            print("Reject the null hypothesis. There is a significant difference between the temperatures.")
        else:
            print("Accept the null hypothesis. There is no significant difference between the temperatures.")
        print()

        f_value_RH, p_value_RH = stats.f_oneway(self.df['RH_1'], self.df['RH_2'], self.df['RH_3'],  self.df['RH_4'],  self.df['RH_5'], self.df['RH_6'], self.df['RH_7'], self.df['RH_8'], self.df["RH_9"], self.df["RH_out"])
        print("ANOVA Test for Relative Humidity")
        if p_value_RH < 0.05:
            print("P-value:", p_value_RH)
            print("Reject the null hypothesis. There is a significant difference between relative humidity.")
        else:
            print("Accept the null hypothesis. There is no significant difference between relative humidity.")
    
    # Visiualization of correlation among variables
    def corr(self):
        corr = self.df.corr()
        fig, ax = plt.subplots(figsize=(15,12))
        sns.heatmap(corr, cmap="YlGnBu")
        plt.title("Figure 3: Correlation Heatmap among IVs and DV", fontsize = 18)
        plt.show()

    # Second Hyppthesis Test: Testing the linear relationship between all independent variable and dependent variables
    def linear_test(self):
        variables = self.df.drop(columns = ["total"])
        no_linear_relationship = []
        for column in variables:
            results = stats.linregress(self.df[column], self.df["total"])
            p = results.pvalue
            if p > 0.05:
                no_linear_relationship.append(column)
        self.nlr = no_linear_relationship
    
    #Testing to determine the strength of correlation between independent variables that have no linear relationship and dependent variables to drop
    def pearson_test(self):
        self.no_corr = []
        for var in self.nlr:
            r,p = pearsonr(self.df[var], self.df["total"])
            if p > 0.05:
                self.no_corr.append(var)
        print("The independent variable that do not have correlation with the dependent variable are")
        return self.no_corr
    
    #Dropping the independent variable columns with no correlation to dependent variable
    def clean(self):
        self.df.drop(columns = self.no_corr, inplace = True)
        
    # First Machine Learning Algorithm: Mulitple Linear Regression
    def linear_regression(self):
        #Prepare training and testing data
        X       = self.df.drop(columns = ["total"])
        y       = self.df["total"]
        cutoff  = int(len(self.df)*0.2)
        X_train = X.iloc[:cutoff]
        X_test  = X.iloc[cutoff:]
        y_train = y.iloc[:cutoff]
        y_test  = y.iloc[cutoff:]
        y_pred_baseline = [y_train.mean()] * len(y_train)
        mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
        
        # Initiate the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Compute the mean absolute error
        mae_training = mean_absolute_error(y_train, model.predict(X_train))
        mae_testing = mean_absolute_error(y_test, model.predict(X_test))
        mae_df = pd.DataFrame({"Baseline": round(mae_baseline, 3), "Training": round(mae_training, 3), "Testing":  round(mae_testing, 3)}, index=[0])
        mae_df = mae_df.T
        mae_df = mae_df.rename(columns = {0: "Mean Absolute Error"})
        
        # Plot the prediction and actual data for training set
        plt.figure(figsize = (15,6))
        plt.plot(y_train, "blue")
        plt.plot(pd.Series(model.predict(X_train), index = y_train.index), "red")
        plt.xlabel("Time")
        plt.ylabel("Total Load [Wh]")
        plt.title("Figure 4: Time Series Data of Energy of a Household [Train Data]", fontsize = 18)
        plt.grid(linestyle = "dashed", linewidth = 1)
        plt.show()
        
        # Plot the prediction and actual data for testing set
        plt.figure(figsize = (15,6))
        plt.plot(y_test, "blue")
        plt.plot(pd.Series(model.predict(X_test), index = y_test.index), "red")
        plt.xlabel("Time")
        plt.ylabel("Total Load [Wh]")
        plt.title("Figure 5: Time Series Data of Energy of a Household [Test Data]", fontsize = 18)
        plt.grid(linestyle = "dashed", linewidth = 1)
        plt.show()
        
        #Plot the feature importance of the model
        coefficients = model.coef_
        feat_imp = pd.Series(coefficients, index = X_train.columns)
        plt.figure()
        feat_imp.sort_values(key = abs).tail(7).plot(kind = "barh")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Figure 6: Top 7 Feature Importance for Energy Consumptions", fontsize = 18)
        plt.grid(linestyle = "dashed", linewidth = 1)
        plt.show()
        5
        return mae_df
    
    # Second Machine Learning: Testing to choose the optimum PCA component for SVR
    def SVR_testing(self):
        # Prepare training and testing data
        X       = self.df.drop(columns = ["total"])
        y       = self.df["total"]
        cutoff  = int(len(self.df)*0.2)
        X_train = X.iloc[:cutoff]
        X_test  = X.iloc[cutoff:]
        y_train = y.iloc[:cutoff]
        y_test  = y.iloc[cutoff:]
        
        #Initiate empty sets
        training_mae = []
        testing_mae  = []
        n_list = []
        kernel = []
        #Initiate the model to test the PCA components
        for k in ["linear", "poly", "rbf"]:
            for n in range(1, X_train.shape[1]):
                test_model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components = n),
                    SVR(kernel = k, degree = 2)
                )
                test_model.fit(X_train, y_train)
                mae_training = mean_absolute_error(y_train, test_model.predict(X_train))
                mae_testing = mean_absolute_error(y_test, test_model.predict(X_test))
                training_mae.append(mae_training)
                testing_mae.append(mae_testing)
                n_list.append(n)
                kernel.append(k)
        dict = {"kernel": kernel, "n_components": n_list, "MAE Training": training_mae, "MAE Testing": testing_mae}
        SVR_result = pd.DataFrame(dict)
        
        #plotting the MAE of different kernel, and PCA component
        linear = SVR_result[SVR_result["kernel"] == "linear"]
        poly   = SVR_result[SVR_result["kernel"] == "poly"]
        rbf   = SVR_result[SVR_result["kernel"] == "rbf"]
        fig = plt.figure(figsize = (16, 5))
        plt.subplots_adjust(wspace = 0.3)
        fig.suptitle("Figure 7: n_componenets vs. MAE for different kernel", fontsize = 18)
        kernels = [linear, poly, rbf]
        kernels_name = ["linear", "poly", "rbf"]
        for idx, kernel in zip(range(4), kernels):
            plt.subplot(1,3, idx+1)
            n_components = kernel["n_components"]
            plt.plot(n_components, kernel["MAE Training"], linestyle = "solid", label = "Training MAE", linewidth = 3)
            plt.plot(n_components, kernel["MAE Testing"], linestyle = "solid", label = "Testing MAE", linewidth = 3)
            plt.xticks(np.arange(min(kernel["n_components"]),max(kernel["n_components"])+1, 2))
            plt.xlabel("Number of Components")
            plt.ylabel("Mean Absolute Error")
            plt.title(kernels_name[idx])
            plt.grid(linestyle = "dashed", linewidth = 1)
            plt.legend()
        plt.show()

        return SVR_result.sort_values("MAE Testing", ascending = True).head(7)
    
    # Second Machine Learning: SVR with 12 PCA components
    def SVR(self):
        # Prepare training and testing data
        X       = self.df.drop(columns = ["total"])
        y       = self.df["total"]
        cutoff  = int(len(self.df)*0.2)
        X_train = X.iloc[:cutoff]
        X_test  = X.iloc[cutoff:]
        y_train = y.iloc[:cutoff]
        y_test  = y.iloc[cutoff:]
        #Initiate the model with 12 PCA component
        SVR_model = make_pipeline(
                    StandardScaler(),
                    PCA(n_components = 12),
                    SVR(kernel = "linear", degree = 2)
                )
        SVR_model.fit(X_train, y_train)
        plt.figure(figsize = (16,9))
        plt.plot(y_test)
        plt.grid(linestyle = "dashed", linewidth = 1)
        plt.title("Figure 8: Time Series Data of Energy of a Household [Test Data]" , fontsize = 18)
        plt.xlabel("Time")
        plt.ylabel("Total Load [Wh]")
        plt.plot(pd.Series(SVR_model.predict(X_test), index = y_test.index))
        plt.show()