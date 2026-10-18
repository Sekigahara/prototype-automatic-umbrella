def inspect_precipitation(df):

    cols = [
        "1-minute Precipitation (mm)",
        "Precipitation Presence (Presence/Absence)",
    ]

    print("\n=== PRECIPITATION CHECK ===")

    for col in cols:

        print(f"\n{col}")

        print("dtype:", df[col].dtype)

        print("NaN:", df[col].isna().sum())

        print(
            "Unique:",
            sorted(
                df[col]
                .dropna()
                .unique()
            )[:30]
        )

        print(
            "Non-zero:",
            (df[col] > 0).sum()
        )

        print(
            "Maximum:",
            df[col].max()
        )

        print(
            "Value counts:"
        )

        print(
            df[col]
            .value_counts()
            .head(20)
        )