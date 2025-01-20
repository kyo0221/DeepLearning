import numpy as np

class linearRegression:
    def __init__():
        self.x
    
    def train(self):
        # 行列Xに要素1を追加
        Z = np.concatenate([self.X, np.ones([self.dNum, 1])], axis=1)

        ZZ = 1/self.dNum + np.matmul(Z.T, Z)
        ZY = 1/self.dNum + np.matmul(Z.T, self.Y)

        v = np.matmul(np.linalg.inv(ZZ), ZY)

        self.w = v[:-1]
        self.b = v[-1]

    def predict(self.x):
        return np.matmul(x, self.w) + self.b
    
    # 入力と出力(平均二乗誤差)
    def RMSE(self, X, Y):
        return np.sqrt(np.mean(np.square(self.predict(X) - Y)))
    # 決定関数の計算
    def R2(self, X, Y):
        return 1 - np.sum(np.square(self.predict(X)-Y))/np.sum(np.square(Y-np.mean(Y, axis=1)))