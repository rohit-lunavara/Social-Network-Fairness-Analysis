import statistics
import numpy as np

sample1 = [1, 2, 1, 2, 1, 2]
sample2 = [1, 2, 2, 2, 2, 1]
sample3 = [2, 2, 2, 2, 2, 2]

#print(statistics.variance(sample1))
#print(statistics.variance(sample2))
#print(statistics.variance(sample3))
print()

d = {'northwest_antelope_valley': 0.0, 'northeast_antelope_valley': 0.0148616, 'lancaster': 0.13805960000000003, 'quartz_hill': 0.006803399999999999, 'lake_los_angeles': 0.11412759999999961, 'northwest_palmdale': 0.022399999999999986, 'leona_valley': 0.038, 'palmdale': 0.15064660000000005, 'desert_view_highlands': 0.0, 'sun_village': 0.007203599999999999, 'littlerock': 0.087, 'acton': 0.0, 'southeast_antelope_valley': 0.0}
#print(np.var(list(d.values())))

records = [
    ([0], 10, 0.1),
    ([1], 11, 0.2),
    ([2,3,4,5], 12, 0.1)
]
records.append(tuple([0], 10, 0.1))
records.sort(key = lambda x : x[2])
#print(records)
minval = records[0][2]
minindex = 0
for i in range(len(records)) :
    if records[i][2] == minval : minindex = i
#print(minindex)
new_records = records[:minindex+1]
new_records.sort(key = lambda x : x[1], reverse = True)
best_record = new_records.pop(0)
print(best_record)
node = best_record[0].pop(-1)
print('Added node :', node)
spread = best_record[1]
print('Spread with added node :', spread)
#print(np.var(sample1))
#print(np.var(sample2))
#print(np.var(sample3))