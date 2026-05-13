import json
import pandas as pd
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

import json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

wandb_file = "/home/msc-auto/szhao/IDQL/wandb/run-20260511_225012-yt6st58p/run-yt6st58p.wandb"

ds = DataStore()
ds.open_for_scan(wandb_file)

count = 0
history_count = 0

while True:
    data = ds.scan_data()
    if data is None:
        break

    record = pb.Record()
    record.ParseFromString(data)
    count += 1

    if record.history.item:
        history_count += 1
        print("history record:", history_count)

        for item in record.history.item:
            print("  key:", repr(item.key), "value:", item.value_json)

        print("-" * 50)

        if history_count >= 5:
            break

print("total records scanned:", count)
print("history records found:", history_count)

# wandb_file = "/home/msc-auto/szhao/IDQL/wandb/run-20260511_225012-yt6st58p/run-yt6st58p.wandb"

# ds = DataStore()
# ds.open_for_scan(wandb_file)

# rows = []

# while True:
#     data = ds.scan_data()
#     if data is None:
#         break

#     record = pb.Record()
#     record.ParseFromString(data)

#     if record.history.item:
#         row = {}
#         for item in record.history.item:
#             try:
#                 row[item.key] = json.loads(item.value_json)
#             except Exception:
#                 row[item.key] = item.value_json
#         rows.append(row)

# df = pd.DataFrame(rows)
# print(df.columns)
# df.to_csv("wandb_history.csv", index=False)
# print("saved to wandb_history.csv")

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("wandb_history.csv")
# print(df.columns)

# plt.plot(df["_step"], df["train/loss"])
# plt.xlabel("step")
# plt.ylabel("train/loss")
# plt.savefig("/home/msc-auto/szhao/LeveFD/fig3.png", bbox_inches="tight",
#         pad_inches=0,
#         transparent=True)