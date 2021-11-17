import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def _per_categorical(col):
    tbl_per = pd.pivot_table(app_train[['TARGET', col]], index = ['TARGET'], columns = [col], aggfunc = len)
    per_categorical = (tbl_per.iloc[0, :]/tbl_per.iloc[1, :]).sort_values(ascending = True)
    # print(per_categorical)
    # print('-------------------------------------\n')
    return per_categorical

# Nhóm các giá trị rate gần bằng nhau vào 1 nhóm theo schedule_div.
def _divide_group(col, schedule_div = None, n_groups = 3, *kwargs):
    cols = []
    tbl_per_cat = _per_categorical(col)
    
    if schedule_div is None:
        n_cats = len(tbl_per_cat)
        n_val_incat = int(n_cats/n_groups)
        n_odd = n_cats - n_groups*n_val_incat

        for i in range(n_groups):
            if i == n_groups - 1:
                el = tbl_per_cat[(n_val_incat*i):(n_val_incat*(i+1)+n_odd)].index.tolist()
            else:
                el = tbl_per_cat[(n_val_incat*i):n_val_incat*(i+1)].index.tolist()    
            cols.append(el)
    else:
        idx = 0
        for n_cols in schedule_div:
            el_cols = tbl_per_cat[idx:(idx+n_cols)].index.tolist()
            cols.append(el_cols)
            idx += n_cols
                
    return cols

def _map_lambda_cats(cols_list, colname, x): 
    cats = list(map(lambda x:colname + '_' + str(x), np.arange(len(cols_list)).tolist()))
    for i in range(len(cols_ORGANIZATION_TYPE)):
        if x in cols_list[i]:
            return cats[i]
        
def _map_cats(cols_list, colname, dataset):                    
    return list(map(lambda x: _map_lambda_cats(cols_list, colname, x), 
                    dataset[colname]))

def _zoom_3sigma(col, dataset, dataset_apl):
    '''
    col: Tên cột dữ liệu
    dataset: Bảng dữ liệu gốc sử dụng để tính khoảng 3 sigma
    dataset_apl: Bảng dữ liệu mới áp dụng khoảng 3 sigma được lấy từ dataset.
    '''
    xs = dataset[col]
    mu = xs.mean()
    sigma = xs.std()
    low =  mu - 3*sigma
#     low =  0 if low < 0 else low
    high = mu + 3*sigma
    
    def _value(x):
        if x < low: return low
        elif x > high: return high
        else: return x
    xapl = dataset_apl[col]    
    xnew = list(map(lambda x: _value(x), xapl))
    n_low = len([i for i in xnew if i == low])
    n_high = len([i for i in xnew if i == high])
    n = len(xapl)
    # print('Percentage of low: {:.2f}{}'.format(100*n_low/n, '%'))
    # print('Percentage of high: {:.2f}{}'.format(100*n_high/n, '%'))
    # print('Low value: {:.2f}'.format(low))
    # print('High value: {:.2f}'.format(high))
    return xnew

def _count_unique(x):
    return pd.Series.nunique(x)

def missing_data(app_train, app_test):
    if 'TARGET' in app_train.columns:
        TARGET = app_train.pop("TARGET")
    
    train = app_train
    test = app_test

    
    # Khởi tạo inputer theo phương pháp trung bình
    inputer = SimpleImputer(strategy = 'mean')
    inputer.fit(train)

    # Điền các giá trị NA bằng trung bình
    train = inputer.transform(train)
    test = inputer.transform(test)

    # Khởi tạo scaler theo phương pháp MinMaxScaler trong khoảng [-1, 1]
    scaler = MinMaxScaler(feature_range = (-1, 1))
    scaler.fit(train)

    # Scale dữ liệu trên train và test
    train = scaler.transform(train)
    test = scaler.transform(test)

    # Loại bỏ cột SK_ID_CURR đầu tiên do cột này là key. Khi cần lấy từ app_train và app_test sang
    train = train[:, 1:]
    test = test[:, 1:]
    return train, test

if __name__ == "__main__":
    app_train = pd.read_csv('data/application_train.csv')
    app_test = pd.read_csv('data/application_test.csv')
    cols_OCCUPATION_TYPE =  _divide_group(col = 'OCCUPATION_TYPE', schedule_div = [1, 7, 9, 1])
    cols_ORGANIZATION_TYPE = _divide_group(col = 'ORGANIZATION_TYPE')
    
    app_train['ORGANIZATION_TYPE'] = _map_cats(cols_ORGANIZATION_TYPE, 'ORGANIZATION_TYPE', app_train)
    app_test['ORGANIZATION_TYPE'] = _map_cats(cols_ORGANIZATION_TYPE, 'ORGANIZATION_TYPE', app_test)
    app_train['OCCUPATION_TYPE'] = _map_cats(cols_OCCUPATION_TYPE, 'OCCUPATION_TYPE', app_train)
    app_test['OCCUPATION_TYPE'] = _map_cats(cols_OCCUPATION_TYPE, 'OCCUPATION_TYPE', app_test)
    
    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)

    TARGET = app_train['TARGET']

    # Lệnh align theo axis = 1 sẽ lấy những trường xuất hiện đồng thời trong app_train và app_test
    app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
    # Sau lệnh align biến TARGET bị mất, do đó ta cần gán lại biến này
    app_train['TARGET'] = TARGET
    x = _zoom_3sigma('FLAG_MOBIL', app_train, app_train) 

    tbl_dis_val = app_train.apply(_count_unique).sort_values(ascending = False)
    tbl_dis_val[tbl_dis_val > 500]

    cols_3sigma = tbl_dis_val[tbl_dis_val > 500].index.tolist()
    # Loại bỏ biến key là SK_ID_CURR ra khỏi danh sách:
    cols_3sigma = cols_3sigma[1:]

    for col in cols_3sigma:
        app_train[col] = _zoom_3sigma(col, app_train, app_train)

    for col in cols_3sigma:
        app_test[col] = _zoom_3sigma(col, app_train, app_test)  

    train, test = missing_data(app_train=app_train, app_test=app_test)

        
    print('train shape: ', train.shape)
    print('test shape: ', test.shape)