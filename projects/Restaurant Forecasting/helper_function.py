import sys
import operator
import pandas as pd

sys.path.append('/home/ec2-user/SageMaker/Dimas/package')
try:
    from prettytable import PrettyTable
except ImportError:
    from package.prettytable import PrettyTable


class DescriptionDF(object):
    def __init__(self, df):
        self._df = df
        self.describe = df.describe()
        self.row, self.col = df.shape
        self.dtypes = df.dtypes
        self.col_name = list(df.columns.values)
        self.unique_value = {}
        self.len_unique = {}
        self.table = PrettyTable(['Column_Name', 'Type', 'Unique_val', 'Null_val', 'Min_val', 'Mean_val', 'Max_val'])

        for i in self.col_name:
            if i in list(self.describe):
                mean_ = round(self.describe[i].loc['mean'], 5)
                min_ = self.describe[i].loc['min']
                max_ = self.describe[i].loc['max']
            else:
                mean_ = 'Not Float/Int'
                min_ = 'Not Float/Int'
                max_ = 'Not Float/Int'

            if str(df.dtypes[i]) == 'object':
                self.unique_value[i] = df[i].unique()
                self.len_unique[i] = str(len(df[i].unique()))
            else:
                self.unique_value[i] = 'Not Object'
                self.len_unique[i] = 'Not Object'

            self.table.add_row([i, df.dtypes[i], self.len_unique[i], df[i].isnull().sum(), min_, mean_, max_])

    def __str__(self):
        print('Total Column: {}'.format(self.col))
        print('Total Row: {}'.format(self.row))
        print(self.table)
        return ''

    def unique_stat(self, column):
        ''' Get unique count on column'''
        if isinstance(column, str):
            column = [column]
        return self._df.groupby(column).size()


def filter_by_column(df, column, value, operation):
    ''' Filtering Dataframe by value in column '''
    ops = {'eq': operator.eq, 'neq': operator.ne, 'gt': operator.gt, 'ge': operator.ge, 'lt': operator.lt, 'le': operator.le}
    return df[ops[operation](df[column], value)]


def filter_store_id(df, column, store_id=[]):
    filtered_df = filter_by_column(df, column, store_id[0], 'eq')
    if len(store_id) > 1:
        for i in range(0, len(store_id)):
            filtered_df = pd.concat([filtered_df, filter_by_column(df, column, store_id[i], 'eq')])
    final_df = filtered_df.reset_index()
    final_df.rename(columns={'index': 'ldx'}, inplace=True)
    return final_df


def balancing_data(df, column):
    ''' Making data balance with selected column
        Get the lowest count of unique variable
    '''
    g = df.groupby(column)
    df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    return df.reset_index(drop=True)


def transform_submission(idx):
    ''' split sample_submission into air_store_id and date '''
    return idx.rsplit('_', 1)