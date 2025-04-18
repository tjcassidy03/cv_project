import pandas as pd

df = pd.read_csv("eastAndEasyOCRMetrics.csv")

filtered_df = df[df["iou"] >= 0.1]

averages = filtered_df.drop(columns=["image"]).mean()

averages["image"] = "AVERAGE_IOU>=0.1"

averages = averages[df.columns]

df_with_avg = pd.concat([df, pd.DataFrame([averages])], ignore_index=True)

df_with_avg.to_csv("eastAndEasyOCRMetrics.csv", index=False)

print("Averages appended and saved to 'eastAndEasyOCRMetrics.csv'")
