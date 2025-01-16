import os


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        # your code here

    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        # your code here
