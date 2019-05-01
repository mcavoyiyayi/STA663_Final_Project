import pandas as pd
import numpy as np
import PLS.pls_regression as plr
import PLS.pls_1d as pls_1d
import PLS.pls_nd as pls_nd



#This is the test code for PLS


np.random.seed(9856)
x1 = np.random.normal(1, .2, 100)
x2 = np.random.normal(5, .4, 100)
x3 = np.random.normal(12, .8, 100)


def generate_sim(x1, x2, x3):
    sim_data = {'x1': x1,
                'x2': x2,
                'x3': x3,
                'x4': 5 * x1,
                'x5': 2 * x2,
                'x6': 4 * x3,
                'x7': 6 * x1,
                'x8': 5 * x2,
                'x9': 4 * x3,
                'x10': 2 * x1,
                'y0': 3 * x2 + 3 * x3,
                'y1': 6 * x1 + 3 * x3,
                'y2': 7 * x2 + 2 * x1}

    # convert data to csv file
    data = pd.DataFrame(sim_data)

    sim_predictors = data.drop(['y0', 'y1', 'y2'], axis=1).columns.tolist()
    sim_values = ['y0', 'y1', 'y2']

    pred = data[sim_predictors].values
    val = data[sim_values].values

    return pred, val


pred, val = generate_sim(x1, x2, x3)

test_x1 = np.random.normal(1, .2, 100)
test_x2 = np.random.normal(5, .4, 100)
test_x3 = np.random.normal(12, .8, 100)

pred_test, pred_val = generate_sim(test_x1, test_x2, test_x3)


pls = plr(cythoinzed=False)
pls.fit(pred, val)
print(pls.predict(pred_test))


