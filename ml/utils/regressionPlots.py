import matplotlib.pyplot as plt

'''residual plot shows dependence of error (predicted_target_data - actual_target_data)=residuals vs predicted_target_data 
    -> ideally all points should be at 0, in real scenario, points chould be CHAOTIC - without any 'deterministic shapes'
    (which constitute leaking of explanatory variables' information to prediction)
'''


# function for drawing residual plot
def residual_plot(y_train, y_train_pred, y_test, y_test_pred, title='Figure'):
    plt.figure(num=title)
    plt.scatter(y_train_pred,  y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')


# function for drawing points and linear plot from regression model
def regression_plot(X, y, models, title='Figure'):
    plt.figure(num=title)
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    linePlotColors = ('black', 'red', 'green', 'blue', 'pink')
    for model,color in zip(models,linePlotColors):
        plt.plot(X, model.predict(X), color=color, lw=2, label=model.__str__())
    plt.legend(loc='upper left')