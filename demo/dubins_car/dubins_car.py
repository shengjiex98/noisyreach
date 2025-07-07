import pandas as pd

from noisyreach import deviation

if __name__ == "__main__":
    df = pd.from_csv("./point_cloud/__SUMMARY__.csv")
    for row in df.rows():
        nn = row["NN"] + str(row["filter %"])
        latency = row["latency ms"]
        accuracy = row["accuracy %"]
        d = deviation(latency, accuracy, "CAR", sampling_file=f"./point_cloud/{nn}.csv")
