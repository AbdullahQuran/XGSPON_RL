import pandas as pd
import os
import glob
import simpy

# str_onu = ''
# space1 = ""
# for i in range(5):
#     str_onu = str_onu + "stime,timestamp,id,report_time,packetsize, , ,"
#     space1 = space1 + "{:f},{:f},{:d},{:f},{:d},{:s},{:s}"
#     q = "{:f},{:f},{:d},{:f},{:d},{:s},{:s}\n".format(1, 1, 2, 1, 1, "" , "")
#     # q = "{:f},{:f},{:d},{:f},{:d},{:s},{:s}\n".format (1.0, 1.0, 1 , 1.0, 1 , "one", "two")
# # print(space1)
# # print (str_onu)
# # print(q)



output_dir = 'C:/Users/Ahmed/Desktop/Thesis_folder/PON/PON_code/anxCopy/output/*total.csv'

results = []
glob.glob(output_dir)
for name in (glob.glob(output_dir)):
    results.append(name)
print(results)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in results ], axis=1, verify_integrity=False)
combined_csv.to_csv( "C:/Users/Ahmed/Desktop/Thesis_folder/PON/PON_code/anxCopy/output/combined.csv")

# a = [1, 2, 3]
# dict = {'1':{1:7, 2:5}, '2':{1:4, 2:6}, '3':{1:10, 2:15}}
# z = (dict.values())

# y = pd.DataFrame.from_dict(dict)
# x = pd.Series(dict).to_frame().T
# print(y)
# # print(x)
# print (z, type(dict))
# tmp5 = ''
# tmp5 = tmp5.ljust(10, '1')
# type(tmp5)
# print(type(tmp5), len(tmp5), tmp5)

# l = [1,2,3]
# l1 = [l, l , l]
# print(sum([sum(i) for i in l1]))
# # print(l1[2][0])
# for i in l1:
#         print("{}".format(i[0]))
# env = simpy.Environment()
# x = env.now
# print(type(x))
x = [1,2,3]
print([2*i for i in x])
x = [2*i for i in x]
print(x)


