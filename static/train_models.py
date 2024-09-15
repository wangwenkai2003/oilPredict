from io import BytesIO
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import base64
import joblib
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error

def train_linear_regression(features, targets):
    # 确保传入的参数是 numpy 数组
    features = np.array(features)
    targets = np.array(targets)

    # 使用 train_test_split 进行数据划分
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=62)

    # 创建线性回归模型
    model = LinearRegression()
    model.fit(x_train, y_train)

    joblib.dump(model, "model0.pkl")
    # 预测测试集
    predictions = model.predict(x_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, predictions)

    # 获取回归系数
    coefficients = model.coef_

    # 绘制预测结果图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.title("Linear Regression Predictions vs Actual Values")
    plt.grid(True)

    # 将图像转换为 base64 编码的字符串
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return predictions.tolist(), y_test.tolist(), mse, coefficients.tolist(), image_base64

#岭回归
def train_ridge_regression(x_train, y_train, x_test,y_test):
    # 特征标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    alphas_to_test = np.linspace(0.001,1)
    # 训练线性回归模型
    estimator = RidgeCV(alphas=alphas_to_test,store_cv_values=True)
    estimator.fit(x_train, y_train)

    best_alpha = estimator.alpha_
    # print(x_train)

    # 使用最佳参数重新训练模型
    best_estimator = RidgeCV(alphas=[best_alpha], store_cv_values=True)
    best_estimator.fit(x_train, y_train)

   #岭回归
def train_ridge_regression(x_train, y_train, x_test,y_test):
    # 特征标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    alphas_to_test = np.linspace(0.001,1)
    # 训练线性回归模型
    estimator = RidgeCV(alphas=alphas_to_test,store_cv_values=True)
    estimator.fit(x_train, y_train)

    best_alpha = estimator.alpha_
    # print(x_train)

    # 使用最佳参数重新训练模型
    best_estimator = RidgeCV(alphas=[best_alpha], store_cv_values=True)
    best_estimator.fit(x_train, y_train)

#岭回归
def train_ridge_regression(x_train, y_train, x_test,y_test):
    # 特征标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    alphas_to_test = np.linspace(0.001,1)
    # 训练线性回归模型
    estimator = RidgeCV(alphas=alphas_to_test,store_cv_values=True)
    estimator.fit(x_train, y_train)

    best_alpha = estimator.alpha_
    # print(x_train)

    # 使用最佳参数重新训练模型
    best_estimator = RidgeCV(alphas=[best_alpha], store_cv_values=True)
    best_estimator.fit(x_train, y_train)

    # 保存模型
    joblib.dump(best_estimator, "model1.pkl")
    # 计算测试集的拟合分数
    score = best_estimator.score(x_test, y_test)
    # 打印岭回归的结果
    print("岭回归结果:")
    y_predict = np.round(best_estimator.predict(x_test), 2)
    print("预测值：\n", y_predict)
    print("实际值：\n", np.floor(np.round(np.array(y_test), 2) * 100) / 100)
    # 计算均方误差
    error = mean_squared_error(y_test, y_predict)
    # 绘制散点图
    plt.figure()
    plt.scatter(y_test, y_predict)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    # 保存图片
    plt.savefig('RidgeImg')
    plt.close()
    # 读取图片并编码为base64
    with open("RidgeImg.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    print("正规方程-均方误差为：\n", error)
    # 返回预测值，实际值，均方误差，系数，图片数据
    l = [y_predict, np.round(np.array(y_test), 2), error, estimator.coef_, image_data]
    return l

#多项式回归
def train_polynomial_regression(x_train, y_train, x_test,y_test):
    # 5. 使用最优的多项式阶数进行模型训练和预测
    # best_degree = grid_search.best_params_['polynomial_features__degree']
    # 创建pipeline，包括多项式特征转换和线性回归模型
    # bemodel = make_pipeline(PolynomialFeatures(), LinearRegression())
    # #
    # # 设置参数网格
    # param_grid = {'polynomialfeatures__degree': np.arange(1, 10)}
    #
    # # 使用GridSearchCV进行交叉验证
    # grid = GridSearchCV(bemodel, param_grid, cv=5)
    # grid.fit(x_train, y_train)

    poly_features = PolynomialFeatures(degree=5)
    X_train_poly = poly_features.fit_transform(x_train)
    X_test_poly = poly_features.transform(x_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # y_pred = model.predict(X_test_poly)
    # print(y_pred)
    joblib.dump(model, "model2.pkl")
    # 6）模型评估
    score = model.score(X_test_poly, y_test)
    print("多项式回归结果:")
    y_predict = np.round(model.predict(X_test_poly), 2)
    print("预测值：\n", y_predict)
    print("实际值：\n", np.array(y_test))
    plt.figure()
    plt.scatter(y_test, y_predict)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.savefig('PolyImg')
    plt.close()
    with open("PolyImg.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    error = mean_squared_error(y_test, y_predict)
    l = [y_predict, np.round(np.array(y_test),2), error, model.coef_,image_data]
    return l

#使用lasso回归
def train_lasson_regression(x_train, y_train, x_test,y_test):
    # lasso回归
    lasso_clf = LassoCV()
    lasso_clf.fit(x_train, y_train)
    joblib.dump(lasso_clf, "model3.pkl")
    # 预测
    y_predict = np.round(lasso_clf.predict(x_test), 2)
    # 打印预测结果
    print("lasson回归结果:")
    print("预测值：\n", y_predict)
    print("实际值：\n", np.array(y_test))
    # 绘制散点图和直线
    plt.figure()
    plt.scatter(y_test, y_predict)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    # 设置x轴和y轴的标签
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    # 保存图片
    plt.savefig('lassoImg')
    # 关闭图形
    plt.close()

    # 读取图片并编码
    with open("lassoImg.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    # 计算均方误差
    error = mean_squared_error(y_test, y_predict)
    print("正规方程-均方误差为：\n", error)
    print(lasso_clf.score(x_test,y_test))
    # 返回结果
    l = [y_predict, np.round(np.array(y_test),2), error, lasso_clf.coef_,image_data]
    return l

#使用随机森林
def train_metrics(x_train, y_train, x_test,y_test):
    '''
    :param x_train: 训练集特征
    :param y_train: 训练集标签
    :param x_test: 测试集特征
    :param y_test: 测试集标签
    :return: 预测结果，真实结果，均方误差，特征重要性，混淆矩阵图
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import metrics
    # 初始化随机森林回归器
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=25)
    # 训练随机森林回归器
    rf_regressor.fit(x_train, y_train)
    # 将模型保存为pkl文件
    joblib.dump(rf_regressor, "model4.pkl")
    # 预测
    y_pred = rf_regressor.predict(x_test)
    # 计算均方误差
    mse = metrics.mean_squared_error(y_test, y_pred)
    # 获取预测结果，保留两位小数
    predicted_target = np.round(rf_regressor.predict(x_test),2)
    # 绘制散点图
    plt.figure()
    plt.scatter(y_test, y_pred)
    # 绘制连接线
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    # 设置轴标签
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    # 保存混淆矩阵图
    plt.savefig('随机森林')
    plt.close()
    # 将混淆矩阵图转换为base64编码
    with open("随机森林.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # 返回预测结果，真实结果，均方误差，特征重要性，混淆矩阵图
    return [predicted_target, np.round(np.array(y_test),2), mse, np.array(rf_regressor.feature_importances_),image_data]

#使用elasticnet回归
def train_elasticnet_regression(x_train, y_train, x_test,y_test):
    '''
    此函数使用elasticnet回归训练模型，并将结果存储在'model2.pkl'文件中。
    同时，它还计算了模型的均方误差（MSE）并绘制了预测值和实际值的散点图。
    '''
    from sklearn.linear_model import ElasticNet
    #创建elasticnet回归模型
    elastic_net = ElasticNet(random_state=0)
    #使用训练数据拟合模型
    elastic_net.fit(x_train, y_train)
    #将模型存储在'model2.pkl'文件中
    joblib.dump(elastic_net, "model2.pkl")
    #使用测试数据进行预测
    y_predict = np.round(elastic_net.predict(x_test),2)
    #计算均方误差
    error = mean_squared_error(y_test, y_predict)
    print("elasticnet回归结果:")
    print("预测值：\n", y_predict)
    print("实际值：\n", np.array(y_test))
    print("正规方程-均方误差为：\n", error)
    #绘制预测值和实际值的散点图
    plt.figure()
    plt.scatter(y_test, y_predict)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.savefig('elastic')
    plt.close()
    #将散点图存储在'elastic.png'文件中
    with open("elastic.png", "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    #返回预测值、实际值、均方误差和模型系数，以及图像数据
    l = [y_predict, np.round(np.array(y_test), 2), error, np.array(elastic_net.coef_), image_data]
    return l