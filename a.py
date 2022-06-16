import pandas as pd
from datetime import datetime

# date = datetime.strptime(str(datetime.now().date()), '%Y-%m-%d')
# date = date.strftime('%d %b %Y')

# timestamp = datetime.strptime(str(datetime.now().time()), '%H%M%S%s')
# timestamp = timestamp.strftime('%H%M%S')

date = datetime.strptime(str(datetime.now().isoformat(' ', 'seconds')), "%Y-%m-%d %H:%M:%S")
tanggal = date.strftime('%d %b, %Y')
waktu = date.strftime('%H:%M:%S')

data = {'Mahasiswa': ['paijo','tukinem','parjo'],
        'NIM': [123,234,567],
        'Tanggal': [tanggal, tanggal, tanggal],
        'Waktu': [waktu, waktu, waktu]
        }

df = pd.DataFrame(data, columns= ['Mahasiswa', 'NIM', 'Tanggal', 'Waktu'])

df.to_csv (r'data_presensi.csv', index = False, header=True)

print (df)