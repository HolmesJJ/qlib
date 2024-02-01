import os
import json
import requests
import pandas as pd
import akshare as ak

from multiprocessing import Lock
from multiprocessing import Pool


other_file_path = './'
proxies = {"http": None, "https": None}
s = requests.session()
s.trust_env = False
s.proxies = proxies


def delete(root_dir):
    file_list = os.listdir(root_dir)
    for f in file_list:
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)


def delete_empty_file(path):
    files = os.listdir(path)
    for file in files:
        if os.path.getsize(path + file) < 2000:
            os.remove(path + file)
            print(file + " deleted.")
    print('delete_empty_file complete')


def download_basic_data(input_file):
    basic_data_save_path = input_file.split(',')[1]
    input_file = input_file.split(',')[0]
    input_file = str(input_file)
    if len(input_file) < 6:
        input_file = '0' * (6 - len(input_file)) + input_file
    if input_file[0] == '6':
        down_num = '1.' + input_file
        input_file = 'SH' + input_file + '.csv'
    else:
        down_num = '0.' + input_file
        input_file = 'SZ' + input_file + '.csv'
    if input_file not in os.listdir(basic_data_save_path):
        # url = 'http://80.push2his.eastmoney.com/api/qt/stock/kline/get?&secid='+down_num+'&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101'
        # url = 'http://68.push2his.eastmoney.com/api/qt/stock/kline/get?&secid='+down_num+'&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=0&end=20500101&lmt=1000000'
        url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=2&secid=' + down_num + '&beg=0&end=20500000'
        # url = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=2&secid=1.600000&beg=0&end=20500000&_=1606616431926'
        # content=urlopen(url).read()
        content = s.get(url).content
        if len(content) == 0:
            content = s.get(url).content
        if len(content) == 0:
            content = s.get(url).content
        if len(content) == 0:
            print('HTTP, Error ' + input_file)
            return -1
        content = content.decode('utf-8', 'ignore').replace('\n', '')
        content = json.loads(content)
        # lock.acquire()
        f = open(basic_data_save_path + input_file, 'a', encoding='utf-8')
        f.write('date,open,close,high,low,volume,amount,amplitude,pct_chg,change,turnover_rate\n')
        f.write('\n'.join(content['data']['klines']))
        f.close()
        df = pd.read_csv(basic_data_save_path + input_file, encoding='utf-8')
        df['symbol'] = input_file.split('.')[0]
        # lock.release()
        df.to_csv(basic_data_save_path + input_file)


global lock_instance


def init(lock):
    global lock_instance
    lock_instance = lock


def callback(result):
    global lock_instance
    with lock_instance:
        print("Process finished with result:", result)


def download_basic_data_all(data_path, debug=False):
    df = ak.stock_info_a_code_name()['code']
    code_list = list(df)
    download_basic_data('1' + ',' + data_path)
    for i in range(1, len(code_list)):
        code_list[i] = str(code_list[i]) + ',' + data_path
    if debug:
        for code in code_list[1:]:
            download_basic_data(code)
    global lock_instance
    lock_instance = Lock()
    pool = Pool(16, initializer=init, initargs=(lock_instance,))
    pool.map_async(download_basic_data, code_list)
    pool.close()
    pool.join()
    print("All processes are finished.")
    delete_empty_file(data_path)
    print('DownloadComplete')


if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    print(home_dir + '/.qlib/')
    if 'csv_data' not in os.listdir(home_dir + '/.qlib/'):
        os.mkdir(home_dir + '/.qlib/csv_data')
    if 'my_data' not in os.listdir(home_dir + '/.qlib/csv_data'):
        os.mkdir(home_dir + '/.qlib/csv_data/my_data')
    basic_data_save_path = home_dir + '/.qlib/csv_data/my_data/'
    delete(basic_data_save_path)
    download_basic_data_all(basic_data_save_path, debug=True)
    print(len(os.listdir(home_dir + '/.qlib/csv_data/my_data')))

# python scripts/dump_bin.py dump_all --csv_path  ~/.qlib/csv_data/my_data --qlib_dir ~/.qlib/qlib_data/my_data --include_fields open,close,high,low,volume
# python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir ~/.qlib/csv_data/my_data --method parse_instruments
# python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/my_data --method parse_instruments
